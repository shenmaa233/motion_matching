from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from motion_matching_common import (
    DEFAULT_COMMAND_SPRING_DAMPING,
    DEFAULT_HEADING_DEGREES,
    DEFAULT_INERTIALIZATION_DAMPING,
    DEFAULT_SEARCH_INTERVAL_FRAMES,
    DEFAULT_SPEED_LEVELS,
    KinematicsHelper,
    MotionClip,
    apply_inertialization_to_qpos_sequence,
    build_runtime_query_feature,
    clip_name_from_path,
    compose_transition,
    export_qpos_trajectory_to_beyond_mimic,
    json_dump,
    load_motion_clip,
    mirror_motion_clip,
    nearest_neighbor_search,
    normalize_features,
    quaternion_to_yaw,
    resolve_repo_path,
    sanitize_skill_name,
    wrap_angle,
    yaw_align_qpos_sequence,
)


@dataclass
class DatabaseBundle:
    normalized_features: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    clip_indices: np.ndarray
    frame_indices: np.ndarray
    locomotion_database_indices: np.ndarray
    manifest: dict[str, Any]


@dataclass(frozen=True)
class StartPoseConfig:
    standing_window_radius_frames: int
    force_root_height_alignment: bool
    root_height_offset_meters: float


@dataclass(frozen=True)
class PreSkillConfig:
    speed_levels_mps: tuple[float, ...]
    heading_degrees: tuple[float, ...]
    start_distance_min_meters: float
    start_distance_max_meters: float


def _load_start_pose_config(config_path: Path) -> StartPoseConfig:
    resolved_path = resolve_repo_path(config_path)
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    start_pose_payload = payload.get("start_pose", {})
    config = StartPoseConfig(
        standing_window_radius_frames=int(start_pose_payload.get("standing_window_radius_frames", 2)),
        force_root_height_alignment=bool(start_pose_payload.get("force_root_height_alignment", True)),
        root_height_offset_meters=float(start_pose_payload.get("root_height_offset_meters", 0.0)),
    )
    if config.standing_window_radius_frames < 0:
        raise ValueError(f"{resolved_path} 的 start_pose.standing_window_radius_frames 不能小于 0")
    return config


def _load_pre_skill_config(config_path: Path) -> PreSkillConfig:
    resolved_path = resolve_repo_path(config_path)
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    pre_skill_payload = payload.get("pre_skill", {})
    start_distance_payload = pre_skill_payload.get("start_distance_meters", {})
    config = PreSkillConfig(
        speed_levels_mps=tuple(float(value) for value in pre_skill_payload.get("speed_levels_mps", DEFAULT_SPEED_LEVELS)),
        heading_degrees=tuple(float(value) for value in pre_skill_payload.get("heading_degrees", DEFAULT_HEADING_DEGREES)),
        start_distance_min_meters=float(start_distance_payload.get("min", 0.5)),
        start_distance_max_meters=float(start_distance_payload.get("max", 6.0)),
    )
    if len(config.speed_levels_mps) == 0:
        raise ValueError(f"{resolved_path} 的 pre_skill.speed_levels_mps 不能为空")
    if len(config.heading_degrees) == 0:
        raise ValueError(f"{resolved_path} 的 pre_skill.heading_degrees 不能为空")
    if config.start_distance_min_meters < 0.0:
        raise ValueError(f"{resolved_path} 的 pre_skill.start_distance_meters.min 不能小于 0")
    if config.start_distance_max_meters < config.start_distance_min_meters:
        raise ValueError(f"{resolved_path} 的 pre_skill.start_distance_meters.max 不能小于 min")
    return config


def _load_database_bundle(database_dir: Path) -> DatabaseBundle:
    database_dir = resolve_repo_path(database_dir)
    database_npz = database_dir / "motion_matching_db.npz"
    manifest_json = database_dir / "motion_matching_db.json"
    if not database_npz.exists():
        raise FileNotFoundError(f"找不到数据库文件: {database_npz}")
    if not manifest_json.exists():
        raise FileNotFoundError(f"找不到数据库清单: {manifest_json}")

    payload = np.load(database_npz, allow_pickle=True)
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    return DatabaseBundle(
        normalized_features=np.asarray(payload["normalized_features"], dtype=np.float64),
        feature_mean=np.asarray(payload["feature_mean"], dtype=np.float64),
        feature_std=np.asarray(payload["feature_std"], dtype=np.float64),
        clip_indices=np.asarray(payload["clip_indices"], dtype=np.int32),
        frame_indices=np.asarray(payload["frame_indices"], dtype=np.int32),
        locomotion_database_indices=np.asarray(payload["locomotion_database_indices"], dtype=np.int32),
        manifest=manifest,
    )


def _load_clip_catalog(manifest: dict[str, Any]) -> list[MotionClip]:
    cache: dict[str, MotionClip] = {}
    clips: list[MotionClip] = []
    for clip_entry in manifest["clips"]:
        source_path = str(clip_entry["path"])
        if source_path not in cache:
            cache[source_path] = load_motion_clip(source_path, clip_name=clip_name_from_path(resolve_repo_path(source_path)))
        base_clip = cache[source_path]
        clip = mirror_motion_clip(base_clip) if clip_entry["mirrored"] else base_clip
        if clip.name != clip_entry["name"]:
            clip = MotionClip(
                name=str(clip_entry["name"]),
                source_path=clip.source_path,
                qpos=clip.qpos,
                joint_pos=clip.joint_pos,
                joint_vel=clip.joint_vel,
                body_pos_w=clip.body_pos_w,
                body_quat_w=clip.body_quat_w,
                body_lin_vel_w=clip.body_lin_vel_w,
                body_ang_vel_w=clip.body_ang_vel_w,
                fps=clip.fps,
                mirrored=clip.mirrored,
                source_name=clip.source_name,
            )
        clips.append(clip)
    return clips


def _find_skill_metadata(manifest: dict[str, Any], skill_name: str) -> dict[str, Any]:
    for skill in manifest["skills"]:
        if skill["skill_name"] == skill_name:
            return skill
    available = [skill["skill_name"] for skill in manifest["skills"]]
    raise KeyError(f"数据库中没有技能 {skill_name}，可选值: {available}")


def _select_candidate_index(
    *,
    query_feature: np.ndarray,
    candidate_indices: np.ndarray,
    database: DatabaseBundle,
    clips: list[MotionClip],
    minimum_future_frames: int,
    maximum_frame_index: int | None = None,
) -> int:
    if len(candidate_indices) == 0:
        raise ValueError("候选集合为空")

    distances = np.linalg.norm(database.normalized_features[candidate_indices] - query_feature[None, :], axis=1)
    ranked_indices = candidate_indices[np.argsort(distances)]
    for database_index in ranked_indices:
        clip_index = int(database.clip_indices[database_index])
        frame_index = int(database.frame_indices[database_index])
        clip = clips[clip_index]
        if maximum_frame_index is not None and frame_index > maximum_frame_index:
            continue
        available_future_frames = clip.num_frames - frame_index - 1
        if available_future_frames >= minimum_future_frames:
            return int(database_index)

    return int(ranked_indices[0])


def _command_vector_local(speed_mps: float, heading_deg: float) -> np.ndarray:
    heading_rad = np.deg2rad(float(heading_deg))
    return np.asarray([speed_mps * np.cos(heading_rad), speed_mps * np.sin(heading_rad)], dtype=np.float64)


def _build_query_from_output(
    *,
    output_qpos: list[np.ndarray],
    kinematics: KinematicsHelper,
    dt: float,
    command_local_velocity_xy: np.ndarray,
    database: DatabaseBundle,
    command_spring_damping: float,
) -> np.ndarray:
    current_qpos = output_qpos[-1]
    previous_qpos = output_qpos[-2] if len(output_qpos) >= 2 else None
    return _build_query_from_pose(
        current_qpos=current_qpos,
        previous_qpos=previous_qpos,
        kinematics=kinematics,
        dt=dt,
        command_local_velocity_xy=command_local_velocity_xy,
        database=database,
        command_spring_damping=command_spring_damping,
    )


def _build_query_from_pose(
    *,
    current_qpos: np.ndarray,
    previous_qpos: np.ndarray | None,
    kinematics: KinematicsHelper,
    dt: float,
    command_local_velocity_xy: np.ndarray,
    database: DatabaseBundle,
    command_spring_damping: float,
) -> np.ndarray:
    pose_state = kinematics.runtime_pose_features(current_qpos, previous_qpos, dt)
    query_feature = build_runtime_query_feature(
        pose_state["left_foot_local_pos"],
        pose_state["right_foot_local_pos"],
        pose_state["left_foot_local_vel"],
        pose_state["right_foot_local_vel"],
        pose_state["root_local_velocity"],
        command_local_velocity_xy,
        damping=command_spring_damping,
    )
    return normalize_features(query_feature, database.feature_mean, database.feature_std)


def _skill_start_candidate_indices(
    *,
    database: DatabaseBundle,
    clips: list[MotionClip],
    skill_name: str,
    skill_start_frame: int,
    allow_mirrored: bool,
) -> np.ndarray:
    skill_frame_indices = np.asarray(database.manifest["skill_frame_indices"][skill_name], dtype=np.int32)
    candidate_mask = database.frame_indices[skill_frame_indices] == int(skill_start_frame)
    if not allow_mirrored:
        candidate_clip_indices = database.clip_indices[skill_frame_indices]
        mirrored_mask = np.asarray([clips[int(clip_index)].mirrored for clip_index in candidate_clip_indices], dtype=bool)
        candidate_mask &= ~mirrored_mask

    candidates = skill_frame_indices[candidate_mask]
    if len(candidates) == 0:
        mirror_hint = "（已过滤 mirrored skill）" if not allow_mirrored else ""
        raise ValueError(f"{skill_name} 在 frame {skill_start_frame} 没有可用的 skill 起始候选 {mirror_hint}")
    return candidates


def _select_standing_start(
    database: DatabaseBundle,
    clips: list[MotionClip],
    *,
    target_root_height: float | None = None,
    window_radius: int = 2,
) -> tuple[MotionClip, int]:
    if len(database.locomotion_database_indices) == 0:
        raise ValueError("locomotion 数据库为空，无法选择 standing 起始帧")

    locomotion_root_heights = np.asarray(
        [
            clips[int(database.clip_indices[int(database_index)])].qpos[int(database.frame_indices[int(database_index)]), 6]
            for database_index in database.locomotion_database_indices
        ],
        dtype=np.float64,
    )
    reference_root_height = float(np.median(locomotion_root_heights)) if target_root_height is None else float(target_root_height)

    best_clip: MotionClip | None = None
    best_frame_index: int | None = None
    best_score: float | None = None

    for database_index in database.locomotion_database_indices:
        clip = clips[int(database.clip_indices[int(database_index)])]
        frame_index = int(database.frame_indices[int(database_index)])
        window_start = max(0, frame_index - window_radius)
        window_end = min(clip.num_frames, frame_index + window_radius + 1)

        root_planar_speed = np.linalg.norm(clip.body_lin_vel_w[window_start:window_end, 0, :2], axis=1)
        mean_planar_speed = float(np.mean(root_planar_speed))
        mean_yaw_rate = float(np.mean(np.abs(clip.body_ang_vel_w[window_start:window_end, 0, 2])))
        root_height = float(clip.qpos[frame_index, 6])

        score = (
            mean_planar_speed
            + 0.5 * mean_yaw_rate
            + 1.0 * abs(root_height - reference_root_height)
            + 1e-3 * float(clip.mirrored)
        )
        if best_score is None or score < best_score:
            best_score = score
            best_clip = clip
            best_frame_index = frame_index

    if best_clip is None or best_frame_index is None:
        raise ValueError("无法从 locomotion 数据库中选择 standing 起始帧")
    return best_clip, best_frame_index


def _skill_entry_target_root_height(
    *,
    database: DatabaseBundle,
    clips: list[MotionClip],
    skill_metadata: dict[str, Any],
) -> float:
    skill_start_frame = int(skill_metadata["skill_start_frame"])
    allow_mirrored_skill = skill_metadata.get("terrain_root_offset") is None
    exact_frame_candidates = _skill_start_candidate_indices(
        database=database,
        clips=clips,
        skill_name=skill_metadata["skill_name"],
        skill_start_frame=skill_start_frame,
        allow_mirrored=allow_mirrored_skill,
    )
    candidate_root_heights = np.asarray(
        [
            clips[int(database.clip_indices[int(database_index)])].qpos[int(database.frame_indices[int(database_index)]), 6]
            for database_index in exact_frame_candidates
        ],
        dtype=np.float64,
    )
    return float(np.median(candidate_root_heights))


def _append_clip_chunk(
    *,
    output_qpos: list[np.ndarray],
    clip: MotionClip,
    source_start_frame: int,
    desired_new_frames: int,
    dt: float,
    inertialization_damping: float,
    align_to_previous_root: bool = True,
) -> np.ndarray:
    if desired_new_frames <= 0:
        return np.zeros((0, clip.qpos.shape[1]), dtype=np.float64)

    previous_qpos = output_qpos[-1] if output_qpos else None
    if previous_qpos is None:
        source_end_frame = min(clip.num_frames, source_start_frame + desired_new_frames)
        source_segment = clip.qpos[source_start_frame:source_end_frame]
        appended = compose_transition(source_segment, None, dt, inertialization_damping)
    else:
        source_end_frame = min(clip.num_frames, source_start_frame + desired_new_frames + 1)
        if source_end_frame - source_start_frame < 2:
            return np.zeros((0, clip.qpos.shape[1]), dtype=np.float64)
        source_segment = clip.qpos[source_start_frame:source_end_frame]
        if align_to_previous_root:
            transitioned = compose_transition(source_segment, previous_qpos, dt, inertialization_damping)
        else:
            transitioned = apply_inertialization_to_qpos_sequence(
                source_segment,
                previous_qpos,
                dt,
                inertialization_damping,
                preserve_root_pose=True,
            )
        appended = transitioned[1:]

    for frame in appended:
        output_qpos.append(frame.copy())
    return appended


def _append_raw_clip_chunk(
    *,
    output_qpos: list[np.ndarray],
    clip: MotionClip,
    source_start_frame: int,
    num_source_frames: int,
) -> np.ndarray:
    if num_source_frames <= 0:
        return np.zeros((0, clip.qpos.shape[1]), dtype=np.float64)

    source_end_frame = min(clip.num_frames, source_start_frame + num_source_frames)
    appended = np.asarray(clip.qpos[source_start_frame:source_end_frame], dtype=np.float64)
    for frame in appended:
        output_qpos.append(frame.copy())
    return appended


def _segment_record(
    *,
    mode: str,
    clip: MotionClip,
    source_start_frame: int,
    num_output_frames: int,
    output_start_frame: int,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "clip_name": clip.name,
        "source_name": clip.source_name,
        "mirrored": clip.mirrored,
        "source_start_frame": int(source_start_frame),
        "source_end_frame": int(source_start_frame + max(num_output_frames - 1, 0)),
        "output_start_frame": int(output_start_frame),
        "output_end_frame": int(output_start_frame + max(num_output_frames - 1, 0)),
    }


def _select_candidate_index_with_history(
    *,
    query_feature: np.ndarray,
    candidate_indices: np.ndarray,
    database: DatabaseBundle,
    minimum_history_frames: int,
    maximum_frame_index: int | None = None,
) -> int:
    if len(candidate_indices) == 0:
        raise ValueError("候选集合为空")

    distances = np.linalg.norm(database.normalized_features[candidate_indices] - query_feature[None, :], axis=1)
    ranked_indices = candidate_indices[np.argsort(distances)]
    for database_index in ranked_indices:
        frame_index = int(database.frame_indices[database_index])
        if maximum_frame_index is not None and frame_index > maximum_frame_index:
            continue
        if frame_index >= minimum_history_frames:
            return int(database_index)

    return int(ranked_indices[0])


def _select_skill_start_index(
    *,
    query_feature: np.ndarray,
    skill_metadata: dict[str, Any],
    database: DatabaseBundle,
    clips: list[MotionClip],
) -> int:
    skill_start_frame = int(skill_metadata["skill_start_frame"])
    skill_end_frame = int(skill_metadata["skill_end_frame"])
    allow_mirrored_skill = skill_metadata.get("terrain_root_offset") is None
    exact_frame_candidates = _skill_start_candidate_indices(
        database=database,
        clips=clips,
        skill_name=skill_metadata["skill_name"],
        skill_start_frame=skill_start_frame,
        allow_mirrored=allow_mirrored_skill,
    )
    return _select_candidate_index(
        query_feature=query_feature,
        candidate_indices=exact_frame_candidates,
        database=database,
        clips=clips,
        minimum_future_frames=skill_end_frame - skill_start_frame,
        maximum_frame_index=skill_start_frame,
    )


def _terrain_pose_from_fixed_skill(
    clip: MotionClip,
    source_frame: int,
    terrain_root_offset: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not terrain_root_offset:
        return None

    translation_local = np.asarray(terrain_root_offset.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64).copy()
    yaw_deg = float(terrain_root_offset.get("yaw_deg", 0.0))
    if clip.mirrored:
        translation_local[1] *= -1.0
        yaw_deg *= -1.0

    root_qpos = clip.qpos[source_frame]
    root_yaw = float(np.rad2deg(quaternion_to_yaw(root_qpos[:4])))
    yaw_rad = np.deg2rad(root_yaw)
    rotation = np.asarray(
        (
            (np.cos(yaw_rad), -np.sin(yaw_rad), 0.0),
            (np.sin(yaw_rad), np.cos(yaw_rad), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    terrain_translation = root_qpos[4:7] + rotation @ translation_local
    terrain_translation = np.where(np.abs(terrain_translation) < 1e-6, 0.0, terrain_translation)
    terrain_yaw_deg = float(np.rad2deg(wrap_angle(np.deg2rad(root_yaw + yaw_deg))))
    if abs(terrain_yaw_deg) < 1e-4:
        terrain_yaw_deg = 0.0

    return {
        "translation": [float(value) for value in terrain_translation.tolist()],
        "yaw_deg": terrain_yaw_deg,
    }


def _generate_single_trajectory(
    *,
    database: DatabaseBundle,
    clips: list[MotionClip],
    skill_metadata: dict[str, Any],
    rng: np.random.Generator,
    pre_skill_config: PreSkillConfig,
    start_pose_config: StartPoseConfig,
    search_interval_frames: int,
    command_spring_damping: float,
    inertialization_damping: float,
    post_skill_frames: int,
    kinematics: KinematicsHelper,
) -> tuple[np.ndarray, dict[str, Any]]:
    speed_mps = float(rng.choice(np.asarray(pre_skill_config.speed_levels_mps, dtype=np.float64)))
    heading_deg = float(rng.choice(np.asarray(pre_skill_config.heading_degrees, dtype=np.float64)))
    pre_skill_distance_meters = float(
        rng.uniform(
            pre_skill_config.start_distance_min_meters,
            pre_skill_config.start_distance_max_meters,
        )
    )
    pre_skill_seconds = pre_skill_distance_meters / max(speed_mps, 1e-6)
    pre_skill_frames = max(search_interval_frames, int(round(pre_skill_seconds * clips[0].fps)))

    approach_command = _command_vector_local(speed_mps, heading_deg)
    skill_command = np.asarray([speed_mps, 0.0], dtype=np.float64)

    initial_query = np.concatenate(
        (
            build_runtime_query_feature(
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                approach_command,
                damping=command_spring_damping,
            )[:27],
        ),
        axis=0,
    )
    initial_query = normalize_features(initial_query, database.feature_mean, database.feature_std)
    initial_database_index = nearest_neighbor_search(initial_query, database.normalized_features, database.locomotion_database_indices)

    output_qpos: list[np.ndarray] = []
    segments: list[dict[str, Any]] = []
    fps = clips[0].fps
    dt = 1.0 / float(fps)

    target_start_root_height = _skill_entry_target_root_height(
        database=database,
        clips=clips,
        skill_metadata=skill_metadata,
    )
    desired_start_root_height = target_start_root_height + float(start_pose_config.root_height_offset_meters)
    standing_clip, standing_frame_idx = _select_standing_start(
        database,
        clips,
        target_root_height=target_start_root_height,
        window_radius=start_pose_config.standing_window_radius_frames,
    )
    standing_qpos = standing_clip.qpos[standing_frame_idx].copy()
    selected_start_root_height = float(standing_qpos[6])
    if start_pose_config.force_root_height_alignment:
        standing_qpos[6] = desired_start_root_height
    output_qpos.append(standing_qpos)
    segments.append(
        _segment_record(
            mode="standing_initial",
            clip=standing_clip,
            source_start_frame=standing_frame_idx,
            num_output_frames=1,
            output_start_frame=0,
        )
    )

    final_approach_frames = min(search_interval_frames, pre_skill_frames)
    free_approach_frames = pre_skill_frames - final_approach_frames

    if free_approach_frames > 0:
        initial_clip = clips[int(database.clip_indices[initial_database_index])]
        initial_frame = int(database.frame_indices[initial_database_index])
        initial_new_frames = min(search_interval_frames, free_approach_frames)
        initial_output_start = len(output_qpos)
        initial_appended = _append_clip_chunk(
            output_qpos=output_qpos,
            clip=initial_clip,
            source_start_frame=initial_frame,
            desired_new_frames=initial_new_frames,
            dt=dt,
            inertialization_damping=inertialization_damping,
        )
        segments.append(
            _segment_record(
                mode="locomotion_approach",
                clip=initial_clip,
                source_start_frame=initial_frame,
                num_output_frames=len(initial_appended),
                output_start_frame=initial_output_start,
            )
        )

        while len(output_qpos) < free_approach_frames:
            remaining_frames = free_approach_frames - len(output_qpos)
            query_feature = _build_query_from_output(
                output_qpos=output_qpos,
                kinematics=kinematics,
                dt=dt,
                command_local_velocity_xy=approach_command,
                database=database,
                command_spring_damping=command_spring_damping,
            )
            database_index = _select_candidate_index(
                query_feature=query_feature,
                candidate_indices=database.locomotion_database_indices,
                database=database,
                clips=clips,
                minimum_future_frames=1,
            )
            clip = clips[int(database.clip_indices[database_index])]
            frame_index = int(database.frame_indices[database_index])
            output_start = len(output_qpos)
            appended = _append_clip_chunk(
                output_qpos=output_qpos,
                clip=clip,
                source_start_frame=frame_index,
                desired_new_frames=min(search_interval_frames, remaining_frames),
                dt=dt,
                inertialization_damping=inertialization_damping,
            )
            if len(appended) == 0:
                break
            segments.append(
                _segment_record(
                    mode="locomotion_approach",
                    clip=clip,
                    source_start_frame=frame_index,
                    num_output_frames=len(appended),
                    output_start_frame=output_start,
                )
            )

    skill_query = (
        _build_query_from_output(
            output_qpos=output_qpos,
            kinematics=kinematics,
            dt=dt,
            command_local_velocity_xy=approach_command,
            database=database,
            command_spring_damping=command_spring_damping,
        )
        if output_qpos
        else initial_query
    )
    skill_database_index = _select_skill_start_index(
        query_feature=skill_query,
        skill_metadata=skill_metadata,
        database=database,
        clips=clips,
    )
    skill_clip = clips[int(database.clip_indices[skill_database_index])]
    skill_frame = int(database.frame_indices[skill_database_index])
    skill_end_frame = int(skill_metadata["skill_end_frame"])
    fixed_skill_anchor_qpos = skill_clip.qpos[skill_frame]
    skill_previous_frame = skill_clip.qpos[skill_frame - 1] if skill_frame > 0 else None
    final_approach_query = _build_query_from_pose(
        current_qpos=skill_clip.qpos[skill_frame],
        previous_qpos=skill_previous_frame,
        kinematics=kinematics,
        dt=dt,
        command_local_velocity_xy=approach_command,
        database=database,
        command_spring_damping=command_spring_damping,
    )
    final_approach_index = _select_candidate_index_with_history(
        query_feature=final_approach_query,
        candidate_indices=database.locomotion_database_indices,
        database=database,
        minimum_history_frames=final_approach_frames - (0 if output_qpos else 1),
    )
    final_approach_clip = clips[int(database.clip_indices[final_approach_index])]
    final_approach_end_frame = int(database.frame_indices[final_approach_index])
    final_approach_start_frame = max(
        0,
        final_approach_end_frame - final_approach_frames + (1 if not output_qpos else 0),
    )
    final_approach_output_start = len(output_qpos)
    final_approach_appended = _append_clip_chunk(
        output_qpos=output_qpos,
        clip=final_approach_clip,
        source_start_frame=final_approach_start_frame,
        desired_new_frames=final_approach_frames,
        dt=dt,
        inertialization_damping=inertialization_damping,
    )
    segments.append(
        _segment_record(
            mode="locomotion_approach",
            clip=final_approach_clip,
            source_start_frame=final_approach_start_frame,
            num_output_frames=len(final_approach_appended),
            output_start_frame=final_approach_output_start,
        )
    )

    aligned_approach_qpos = yaw_align_qpos_sequence(
        np.asarray(output_qpos, dtype=np.float64),
        fixed_skill_anchor_qpos[:4],
        fixed_skill_anchor_qpos[4:7],
        anchor_frame_index=len(output_qpos) - 1,
    )
    if start_pose_config.force_root_height_alignment and aligned_approach_qpos.shape[0] > 0:
        aligned_approach_qpos[0, 6] = desired_start_root_height
    output_qpos = [frame.copy() for frame in aligned_approach_qpos]
    skill_output_start = len(output_qpos)
    terrain_world_pose = _terrain_pose_from_fixed_skill(
        skill_clip,
        skill_frame,
        skill_metadata.get("terrain_root_offset"),
    )
    skill_appended = _append_clip_chunk(
        output_qpos=output_qpos,
        clip=skill_clip,
        source_start_frame=skill_frame,
        desired_new_frames=max(0, skill_end_frame - skill_frame),
        dt=dt,
        inertialization_damping=inertialization_damping,
        align_to_previous_root=False,
    )
    segments.append(
        _segment_record(
            mode="skill_execution",
            clip=skill_clip,
            source_start_frame=skill_frame,
            num_output_frames=len(skill_appended),
            output_start_frame=skill_output_start,
        )
    )

    target_total_frames = len(output_qpos) + post_skill_frames
    while len(output_qpos) < target_total_frames:
        remaining_frames = target_total_frames - len(output_qpos)
        query_feature = _build_query_from_output(
            output_qpos=output_qpos,
            kinematics=kinematics,
            dt=dt,
            command_local_velocity_xy=skill_command,
            database=database,
            command_spring_damping=command_spring_damping,
        )
        database_index = _select_candidate_index(
            query_feature=query_feature,
            candidate_indices=database.locomotion_database_indices,
            database=database,
            clips=clips,
            minimum_future_frames=1,
        )
        clip = clips[int(database.clip_indices[database_index])]
        frame_index = int(database.frame_indices[database_index])
        output_start = len(output_qpos)
        appended = _append_clip_chunk(
            output_qpos=output_qpos,
            clip=clip,
            source_start_frame=frame_index,
            desired_new_frames=min(search_interval_frames, remaining_frames),
            dt=dt,
            inertialization_damping=inertialization_damping,
        )
        if len(appended) == 0:
            break
        segments.append(
            _segment_record(
                mode="locomotion_recovery",
                clip=clip,
                source_start_frame=frame_index,
                num_output_frames=len(appended),
                output_start_frame=output_start,
            )
        )

    manifest = {
        "skill_name": skill_metadata["skill_name"],
        "terrain_path": skill_metadata["terrain_path"],
        "terrain_world_pose": terrain_world_pose,
        "annotation_source": skill_metadata.get("annotation_source", "manual"),
        "needs_review": bool(skill_metadata.get("needs_review", False)),
        "start_pose": {
            "mode": "standing_initial",
            "clip_name": standing_clip.name,
            "source_frame": int(standing_frame_idx),
            "selected_root_height": selected_start_root_height,
            "target_root_height": float(target_start_root_height),
            "desired_root_height": float(desired_start_root_height),
            "final_root_height": float(output_qpos[0][6]),
            "force_root_height_alignment": bool(start_pose_config.force_root_height_alignment),
            "root_height_offset_meters": float(start_pose_config.root_height_offset_meters),
        },
        "skill_world_locked": True,
        "skill_anchor": {
            "fixed_world": True,
            "source_frame": int(skill_frame),
            "mirrored": bool(skill_clip.mirrored),
            "root_translation": [float(value) for value in fixed_skill_anchor_qpos[4:7].tolist()],
            "root_yaw_deg": float(np.rad2deg(quaternion_to_yaw(fixed_skill_anchor_qpos[:4]))),
        },
        "command": {
            "speed_mps": speed_mps,
            "heading_deg": heading_deg,
            "pre_skill_distance_meters": pre_skill_distance_meters,
            "pre_skill_seconds": pre_skill_seconds,
            "pre_skill_frames": pre_skill_frames,
            "post_skill_frames": post_skill_frames,
        },
        "approach": {
            "target_source_frame": skill_frame,
            "target_clip_name": skill_clip.name,
        },
        "search": {
            "interval_frames": search_interval_frames,
            "command_spring_damping": command_spring_damping,
            "inertialization_damping": inertialization_damping,
        },
        "segments": segments,
    }
    return np.asarray(output_qpos, dtype=np.float64), manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="生成单技能 motion matching 批量轨迹。")
    parser.add_argument(
        "--database-dir",
        type=Path,
        default=Path("output/motion_matching/single_skill_db"),
        help="数据库目录，需要包含 motion_matching_db.npz 和 motion_matching_db.json。",
    )
    parser.add_argument(
        "--skill-name",
        type=str,
        default="climb_15_z_scale_1.0",
        help="要生成的技能名。",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=1,
        help="要生成的轨迹数量。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/motion_matching/generated/climb_15_z_scale_1.0"),
        help="轨迹输出目录。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子。",
    )
    parser.add_argument(
        "--generation-config",
        type=Path,
        default=Path("data/motion_matching/trajectory_generation_config.json"),
        help="轨迹生成配置 JSON，目前用于 start_pose 参数。",
    )
    parser.add_argument(
        "--search-interval-frames",
        type=int,
        default=DEFAULT_SEARCH_INTERVAL_FRAMES,
        help="locomotion 检索周期。",
    )
    parser.add_argument(
        "--command-spring-damping",
        type=float,
        default=DEFAULT_COMMAND_SPRING_DAMPING,
        help="命令未来轨迹弹簧阻尼。",
    )
    parser.add_argument(
        "--inertialization-damping",
        type=float,
        default=DEFAULT_INERTIALIZATION_DAMPING,
        help="切换过渡的 inertialization 阻尼。",
    )
    parser.add_argument(
        "--post-skill-seconds",
        type=float,
        default=2.0,
        help="技能结束后继续 locomotion 的时长。",
    )
    args = parser.parse_args()

    database = _load_database_bundle(args.database_dir)
    clips = _load_clip_catalog(database.manifest)
    skill_metadata = _find_skill_metadata(database.manifest, args.skill_name)
    pre_skill_config = _load_pre_skill_config(args.generation_config)
    start_pose_config = _load_start_pose_config(args.generation_config)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    kinematics = KinematicsHelper()
    post_skill_frames = max(1, int(round(float(args.post_skill_seconds) * clips[0].fps)))

    trajectory_manifests: list[dict[str, Any]] = []
    for trajectory_index in range(args.num_trajectories):
        qpos_trajectory, trajectory_manifest = _generate_single_trajectory(
            database=database,
            clips=clips,
            skill_metadata=skill_metadata,
            rng=rng,
            pre_skill_config=pre_skill_config,
            start_pose_config=start_pose_config,
            search_interval_frames=args.search_interval_frames,
            command_spring_damping=args.command_spring_damping,
            inertialization_damping=args.inertialization_damping,
            post_skill_frames=post_skill_frames,
            kinematics=kinematics,
        )

        trajectory_name = f"{sanitize_skill_name(args.skill_name)}_{trajectory_index:04d}"
        trajectory_path = output_dir / f"{trajectory_name}.npz"
        export_qpos_trajectory_to_beyond_mimic(
            qpos_trajectory,
            clips[0].fps,
            trajectory_path,
            kinematics=kinematics,
        )
        trajectory_manifest["trajectory_name"] = trajectory_name
        trajectory_manifest["trajectory_path"] = str(trajectory_path.relative_to(resolve_repo_path(".")))
        trajectory_manifest["num_frames"] = int(qpos_trajectory.shape[0])
        trajectory_manifests.append(trajectory_manifest)
        print(f"[ok] 已生成 {trajectory_path}")

    batch_manifest = {
        "version": 1,
        "database_dir": str(resolve_repo_path(args.database_dir).relative_to(resolve_repo_path("."))),
        "skill_name": args.skill_name,
        "num_trajectories": args.num_trajectories,
        "seed": args.seed,
        "generation_config_path": str(resolve_repo_path(args.generation_config).relative_to(resolve_repo_path("."))),
        "pre_skill_config": {
            "speed_levels_mps": [float(value) for value in pre_skill_config.speed_levels_mps],
            "heading_degrees": [float(value) for value in pre_skill_config.heading_degrees],
            "start_distance_meters": {
                "min": float(pre_skill_config.start_distance_min_meters),
                "max": float(pre_skill_config.start_distance_max_meters),
            },
        },
        "start_pose_config": {
            "standing_window_radius_frames": int(start_pose_config.standing_window_radius_frames),
            "force_root_height_alignment": bool(start_pose_config.force_root_height_alignment),
            "root_height_offset_meters": float(start_pose_config.root_height_offset_meters),
        },
        "trajectories": trajectory_manifests,
    }
    manifest_path = output_dir / "batch_manifest.json"
    json_dump(manifest_path, batch_manifest)
    print(f"[ok] 批量清单已写出到 {manifest_path}")


if __name__ == "__main__":
    main()
