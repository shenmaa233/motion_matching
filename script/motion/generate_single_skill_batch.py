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
    DEFAULT_HORIZONS,
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
    normalize_features,
    quaternion_from_yaw,
    quaternion_multiply,
    quaternion_to_yaw,
    resolve_repo_path,
    sanitize_skill_name,
    wrap_angle,
    world_to_local_vector,
)

MAX_INITIAL_HEADING_DELTA_DEGREES = 45.0
MIN_START_DISTANCE_METERS = 2.0
MIN_TRANSLATION_SPEED_FRACTION = 0.35
MIN_FORWARD_SPEED_FRACTION = 0.45
MAX_LATERAL_SPEED_FRACTION = 0.45
STEERING_QUERY_MAX_HEADING_DEGREES = 45.0


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
    approach_direction_window_frames: int


@dataclass(frozen=True)
class TerrainAnchorConfig:
    translation: tuple[float, float, float]
    yaw_deg: float


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
        approach_direction_window_frames=int(pre_skill_payload.get("approach_direction_window_frames", 30)),
    )
    if len(config.speed_levels_mps) == 0:
        raise ValueError(f"{resolved_path} 的 pre_skill.speed_levels_mps 不能为空")
    if len(config.heading_degrees) == 0:
        raise ValueError(f"{resolved_path} 的 pre_skill.heading_degrees 不能为空")
    if config.start_distance_min_meters < MIN_START_DISTANCE_METERS:
        raise ValueError(
            f"{resolved_path} 的 pre_skill.start_distance_meters.min 不能小于 "
            f"{MIN_START_DISTANCE_METERS:.1f}，否则没有足够距离完成进入 skill 前的方向调整"
        )
    if config.start_distance_max_meters < config.start_distance_min_meters:
        raise ValueError(f"{resolved_path} 的 pre_skill.start_distance_meters.max 不能小于 min")
    if config.approach_direction_window_frames < 1:
        raise ValueError(f"{resolved_path} 的 pre_skill.approach_direction_window_frames 不能小于 1")
    return config


def _load_terrain_anchor_config(config_path: Path) -> TerrainAnchorConfig:
    resolved_path = resolve_repo_path(config_path)
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    terrain_payload = payload.get("terrain_anchor", {})
    translation = tuple(float(value) for value in terrain_payload.get("translation", [0.0, 0.0, 0.0]))
    if len(translation) != 3:
        raise ValueError(f"{resolved_path} 的 terrain_anchor.translation 必须是 3 维")
    return TerrainAnchorConfig(
        translation=(translation[0], translation[1], translation[2]),
        yaw_deg=float(terrain_payload.get("yaw_deg", 0.0)),
    )


def _trajectory_frame_count(path: Path) -> int:
    data = np.load(path, allow_pickle=True)
    if "qpos" in data:
        return int(np.asarray(data["qpos"]).shape[0])
    if "body_pos_w" in data:
        return int(np.asarray(data["body_pos_w"]).shape[0])
    if "body_positions" in data:
        return int(np.asarray(data["body_positions"]).shape[0])
    first_key = data.files[0]
    return int(np.asarray(data[first_key]).shape[0])


def _load_existing_trajectory_manifests(
    output_dir: Path,
    skill_metadata: dict[str, Any],
    target_skill_name: str,
    end_index: int,
) -> list[dict[str, Any]]:
    manifest_path = output_dir / "batch_manifest.json"
    existing_by_name: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for trajectory in payload.get("trajectories", []):
            trajectory_name = str(trajectory.get("trajectory_name", ""))
            if trajectory_name:
                existing_by_name[trajectory_name] = trajectory

    trajectory_manifests: list[dict[str, Any]] = []
    sanitized_skill_name = sanitize_skill_name(target_skill_name)
    repo_root = resolve_repo_path(".")
    for trajectory_index in range(end_index):
        trajectory_name = f"{sanitized_skill_name}_{trajectory_index:04d}"
        existing = existing_by_name.get(trajectory_name)
        if existing is not None:
            trajectory_manifests.append(existing)
            continue

        trajectory_path = output_dir / f"{trajectory_name}.npz"
        if not trajectory_path.exists():
            continue
        trajectory_manifests.append(
            {
                "skill_name": target_skill_name,
                "terrain_path": skill_metadata.get("terrain_path"),
                "trajectory_name": trajectory_name,
                "trajectory_path": str(trajectory_path.relative_to(repo_root)),
                "num_frames": _trajectory_frame_count(trajectory_path),
                "manifest_source": "synthesized_from_existing_file",
            }
        )
    return trajectory_manifests


def _count_existing_trajectories(output_dir: Path, target_skill_name: str) -> int:
    sanitized_skill_name = sanitize_skill_name(target_skill_name)
    count = 0
    while (output_dir / f"{sanitized_skill_name}_{count:04d}.npz").exists():
        count += 1
    return count


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


def _skill_allows_mirrored_candidates(skill_metadata: dict[str, Any]) -> bool:
    for key in ("allow_mirrored_skill", "allow_mirrored", "allow_mirror"):
        if key in skill_metadata:
            return bool(skill_metadata[key])
    return not bool(skill_metadata.get("terrain_path"))


def _sort_skill_names_by_scale(skill_names: list[str]) -> list[str]:
    def scale_key(skill_name: str) -> tuple[int, float, str]:
        marker = "_z_scale_"
        if marker not in skill_name:
            return (1, 0.0, skill_name)
        prefix, scale_text = skill_name.split(marker, maxsplit=1)
        try:
            scale_value = float(scale_text)
        except ValueError:
            return (1, 0.0, skill_name)
        return (0, scale_value, prefix)

    return sorted(skill_names, key=scale_key)


def _resolve_target_skill_names(
    manifest: dict[str, Any],
    *,
    skill_name: str | None,
    skill_prefix: str | None,
    all_scales: bool,
) -> list[str]:
    available = [str(skill["skill_name"]) for skill in manifest["skills"]]

    if all_scales:
        if not skill_prefix:
            raise ValueError("--all-scales 需要与 --skill-prefix 一起使用")
        matched = [name for name in available if name.startswith(f"{skill_prefix}_z_scale_")]
        if not matched:
            raise KeyError(f"数据库中没有前缀为 {skill_prefix} 的 scale 技能，可选值: {available}")
        return _sort_skill_names_by_scale(matched)

    if skill_prefix is not None:
        raise ValueError("仅在使用 --all-scales 时才允许传入 --skill-prefix")
    if skill_name is None:
        raise ValueError("未指定技能，请传入 --skill-name，或使用 --skill-prefix 配合 --all-scales")
    return [skill_name]


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


def _select_locomotion_approach_index(
    *,
    query_feature: np.ndarray,
    command_local_velocity_xy: np.ndarray,
    database: DatabaseBundle,
    clips: list[MotionClip],
    minimum_future_frames: int,
) -> int:
    candidate_indices = database.locomotion_database_indices
    if len(candidate_indices) == 0:
        raise ValueError("locomotion 候选集合为空")

    feature_distances = np.linalg.norm(
        database.normalized_features[candidate_indices] - query_feature[None, :],
        axis=1,
    ).astype(np.float64)
    command_local_velocity_xy = np.asarray(command_local_velocity_xy, dtype=np.float64)
    command_speed = float(np.linalg.norm(command_local_velocity_xy))

    total_costs = feature_distances.copy()
    candidate_clip_indices = database.clip_indices[candidate_indices]
    candidate_frame_indices = database.frame_indices[candidate_indices]
    candidate_local_velocities = np.zeros((len(candidate_indices), 2), dtype=np.float64)
    candidate_yaw_rates = np.zeros((len(candidate_indices),), dtype=np.float64)

    for clip_index in np.unique(candidate_clip_indices):
        mask = candidate_clip_indices == clip_index
        frames = candidate_frame_indices[mask]
        clip = clips[int(clip_index)]
        quaternions = clip.qpos[frames, :4]
        w, x, y, z = np.moveaxis(quaternions, -1, 0)
        root_yaws = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        world_velocities = clip.body_lin_vel_w[frames, 0, :2]
        cos_yaw = np.cos(root_yaws)
        sin_yaw = np.sin(root_yaws)
        candidate_local_velocities[mask, 0] = world_velocities[:, 0] * cos_yaw + world_velocities[:, 1] * sin_yaw
        candidate_local_velocities[mask, 1] = -world_velocities[:, 0] * sin_yaw + world_velocities[:, 1] * cos_yaw
        candidate_yaw_rates[mask] = clip.body_ang_vel_w[frames, 0, 2]

    candidate_speeds = np.linalg.norm(candidate_local_velocities, axis=1)
    if command_speed > 1e-6:
        denominator = np.maximum(candidate_speeds * command_speed, 1e-12)
        direction_similarity = np.sum(candidate_local_velocities * command_local_velocity_xy[None, :], axis=1) / denominator
        direction_costs = 1.0 - np.clip(direction_similarity, -1.0, 1.0)
        direction_costs = np.where(candidate_speeds > 1e-6, direction_costs, 0.0)
    else:
        direction_costs = np.zeros_like(candidate_speeds)
    speed_costs = np.abs(candidate_speeds - command_speed)
    yaw_rate_costs = np.abs(candidate_yaw_rates)
    if command_speed > 1e-6:
        minimum_translation_speed = command_speed * MIN_TRANSLATION_SPEED_FRACTION
        low_translation_costs = np.maximum(0.0, minimum_translation_speed - candidate_speeds)
        in_place_turn_costs = low_translation_costs * np.abs(candidate_yaw_rates)
        minimum_forward_speed = command_speed * MIN_FORWARD_SPEED_FRACTION
        maximum_lateral_speed = command_speed * MAX_LATERAL_SPEED_FRACTION
        forward_progress_costs = np.maximum(0.0, minimum_forward_speed - candidate_local_velocities[:, 0])
        lateral_slide_costs = np.maximum(0.0, np.abs(candidate_local_velocities[:, 1]) - maximum_lateral_speed)
    else:
        low_translation_costs = np.zeros_like(candidate_speeds)
        in_place_turn_costs = np.zeros_like(candidate_speeds)
        forward_progress_costs = np.zeros_like(candidate_speeds)
        lateral_slide_costs = np.zeros_like(candidate_speeds)
    total_costs += (
        1.25 * direction_costs
        + 0.35 * speed_costs
        + 0.35 * yaw_rate_costs
        + 1.50 * low_translation_costs
        + 2.00 * in_place_turn_costs
        + 3.00 * forward_progress_costs
        + 1.25 * lateral_slide_costs
    )

    ranked_indices = candidate_indices[np.argsort(total_costs)]
    for database_index in ranked_indices:
        clip_index = int(database.clip_indices[int(database_index)])
        frame_index = int(database.frame_indices[int(database_index)])
        clip = clips[clip_index]
        available_future_frames = clip.num_frames - frame_index - 1
        if available_future_frames >= minimum_future_frames:
            return int(database_index)

    return int(ranked_indices[0])


def _rotate_command_xy(command_xy: np.ndarray, delta_heading_deg: float) -> np.ndarray:
    delta_heading = float(np.deg2rad(delta_heading_deg))
    cosine = float(np.cos(delta_heading))
    sine = float(np.sin(delta_heading))
    x_value, y_value = np.asarray(command_xy, dtype=np.float64)
    return np.asarray(
        [
            cosine * x_value - sine * y_value,
            sine * x_value + cosine * y_value,
        ],
        dtype=np.float64,
    )


def _limit_command_heading_change(
    previous_command_xy: np.ndarray,
    target_command_xy: np.ndarray,
    max_delta_degrees: float,
) -> np.ndarray:
    previous_command_xy = np.asarray(previous_command_xy, dtype=np.float64)
    target_command_xy = np.asarray(target_command_xy, dtype=np.float64)
    target_speed = float(np.linalg.norm(target_command_xy))
    if target_speed < 1e-6:
        return target_command_xy.copy()

    previous_speed = float(np.linalg.norm(previous_command_xy))
    if previous_speed < 1e-6:
        return target_command_xy.copy()

    previous_heading = float(np.arctan2(previous_command_xy[1], previous_command_xy[0]))
    target_heading = float(np.arctan2(target_command_xy[1], target_command_xy[0]))
    max_delta = float(np.deg2rad(max_delta_degrees))
    heading_delta = float(np.clip(wrap_angle(target_heading - previous_heading), -max_delta, max_delta))
    limited_heading = previous_heading + heading_delta
    return np.asarray(
        [target_speed * np.cos(limited_heading), target_speed * np.sin(limited_heading)],
        dtype=np.float64,
    )


def _clamp_forward_steering_command(command_xy: np.ndarray, speed_mps: float) -> np.ndarray:
    command_xy = np.asarray(command_xy, dtype=np.float64)
    command_speed = float(np.linalg.norm(command_xy))
    if command_speed < 1e-6:
        return np.asarray([speed_mps, 0.0], dtype=np.float64)

    max_heading = float(np.deg2rad(STEERING_QUERY_MAX_HEADING_DEGREES))
    command_heading = float(np.arctan2(command_xy[1], command_xy[0]))
    clamped_heading = float(np.clip(wrap_angle(command_heading), -max_heading, max_heading))
    return np.asarray(
        [speed_mps * np.cos(clamped_heading), speed_mps * np.sin(clamped_heading)],
        dtype=np.float64,
    )


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _build_forward_steering_command_feature(
    command_local_velocity_xy: np.ndarray,
    *,
    horizons: tuple[float, ...] = DEFAULT_HORIZONS,
) -> np.ndarray:
    command_local_velocity_xy = np.asarray(command_local_velocity_xy, dtype=np.float64)
    command_speed = float(np.linalg.norm(command_local_velocity_xy))
    if command_speed < 1e-6:
        return np.asarray([component for _ in horizons for component in (0.0, 0.0, 1.0, 0.0)], dtype=np.float64)

    max_heading = float(np.deg2rad(STEERING_QUERY_MAX_HEADING_DEGREES))
    target_heading = float(np.clip(wrap_angle(np.arctan2(command_local_velocity_xy[1], command_local_velocity_xy[0])), -max_heading, max_heading))
    max_horizon = max(float(max(horizons)), 1e-6)
    feature: list[float] = []
    for horizon in horizons:
        horizon = float(horizon)
        blend = _smoothstep(horizon / max_horizon)
        local_heading = blend * target_heading
        average_heading = 0.5 * local_heading
        local_distance = command_speed * horizon
        local_x = local_distance * np.cos(average_heading)
        local_y = local_distance * np.sin(average_heading)
        feature.extend((float(local_x), float(local_y), float(np.cos(local_heading)), float(np.sin(local_heading))))
    return np.asarray(feature, dtype=np.float64)


def _build_query_from_steering_plan(
    *,
    output_qpos: list[np.ndarray],
    kinematics: KinematicsHelper,
    dt: float,
    command_local_velocity_xy: np.ndarray,
    database: DatabaseBundle,
) -> np.ndarray:
    current_qpos = output_qpos[-1]
    previous_qpos = output_qpos[-2] if len(output_qpos) >= 2 else None
    pose_state = kinematics.runtime_pose_features(current_qpos, previous_qpos, dt)
    command_feature = _build_forward_steering_command_feature(command_local_velocity_xy)
    pose_feature = np.concatenate(
        (
            np.asarray(pose_state["left_foot_local_pos"], dtype=np.float64),
            np.asarray(pose_state["left_foot_local_vel"], dtype=np.float64),
            np.asarray(pose_state["right_foot_local_pos"], dtype=np.float64),
            np.asarray(pose_state["right_foot_local_vel"], dtype=np.float64),
            np.asarray(pose_state["root_local_velocity"], dtype=np.float64),
        ),
        axis=0,
    )
    query_feature = np.concatenate((command_feature, pose_feature), axis=0)
    return normalize_features(query_feature, database.feature_mean, database.feature_std)


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


def _runtime_pose_state_from_output(
    *,
    output_qpos: list[np.ndarray],
    kinematics: KinematicsHelper,
    dt: float,
) -> dict[str, np.ndarray]:
    current_qpos = output_qpos[-1]
    previous_qpos = output_qpos[-2] if len(output_qpos) >= 2 else None
    return kinematics.runtime_pose_features(current_qpos, previous_qpos, dt)


def _current_root_yaw_rate_from_output(output_qpos: list[np.ndarray], dt: float) -> float:
    if len(output_qpos) < 2:
        return 0.0
    current_yaw = float(quaternion_to_yaw(np.asarray(output_qpos[-1][:4], dtype=np.float64)))
    previous_yaw = float(quaternion_to_yaw(np.asarray(output_qpos[-2][:4], dtype=np.float64)))
    return float(wrap_angle(current_yaw - previous_yaw) / max(dt, 1e-6))


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


def _skill_entry_candidate_indices(
    *,
    database: DatabaseBundle,
    clips: list[MotionClip],
    skill_name: str,
    allow_mirrored: bool,
) -> np.ndarray:
    candidates = np.asarray(database.manifest["skill_entry_indices"][skill_name], dtype=np.int32)
    if allow_mirrored:
        return candidates

    candidate_clip_indices = database.clip_indices[candidates]
    mirrored_mask = np.asarray([clips[int(clip_index)].mirrored for clip_index in candidate_clip_indices], dtype=bool)
    filtered = candidates[~mirrored_mask]
    if len(filtered) == 0:
        raise ValueError(f"{skill_name} 的 entry window 在过滤 mirrored skill 后为空")
    return filtered


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


def _append_clip_chunk(
    *,
    output_qpos: list[np.ndarray],
    clip: MotionClip,
    source_start_frame: int,
    desired_new_frames: int,
    dt: float,
    inertialization_damping: float,
    align_to_previous_root: bool = True,
    root_blend_frames: int = 0,
) -> np.ndarray:
    if desired_new_frames <= 0:
        return np.zeros((0, clip.qpos.shape[1]), dtype=np.float64)

    previous_qpos = output_qpos[-1] if output_qpos else None
    pre_previous_qpos = output_qpos[-2] if len(output_qpos) >= 2 else None
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
        if root_blend_frames > 0:
            transitioned = _smooth_root_transition(
                transitioned,
                previous_qpos=previous_qpos,
                pre_previous_qpos=pre_previous_qpos,
                dt=dt,
                blend_frames=root_blend_frames,
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
    allow_mirrored_skill = _skill_allows_mirrored_candidates(skill_metadata)
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


def _select_skill_entry_index(
    *,
    query_feature: np.ndarray,
    skill_metadata: dict[str, Any],
    database: DatabaseBundle,
    clips: list[MotionClip],
    current_root_local_velocity_xy: np.ndarray | None = None,
    current_root_yaw_rate: float | None = None,
    latest_entry_frame: int | None = None,
) -> int:
    skill_start_frame = int(skill_metadata["skill_start_frame"])
    skill_end_frame = int(skill_metadata["skill_end_frame"])
    allow_mirrored_skill = _skill_allows_mirrored_candidates(skill_metadata)
    entry_candidates = _skill_entry_candidate_indices(
        database=database,
        clips=clips,
        skill_name=skill_metadata["skill_name"],
        allow_mirrored=allow_mirrored_skill,
    )
    if latest_entry_frame is not None:
        candidate_frame_indices = database.frame_indices[entry_candidates]
        settled_entry_candidates = entry_candidates[candidate_frame_indices <= int(latest_entry_frame)]
        if len(settled_entry_candidates) > 0:
            entry_candidates = settled_entry_candidates

    feature_distances = np.linalg.norm(database.normalized_features[entry_candidates] - query_feature[None, :], axis=1)
    total_costs = feature_distances.astype(np.float64)
    if current_root_local_velocity_xy is not None:
        candidate_velocity_costs = np.zeros_like(total_costs)
        candidate_yaw_rate_costs = np.zeros_like(total_costs)
        current_root_local_velocity_xy = np.asarray(current_root_local_velocity_xy, dtype=np.float64)
        reference_yaw_rate = 0.0 if current_root_yaw_rate is None else float(current_root_yaw_rate)
        for candidate_offset, database_index in enumerate(entry_candidates):
            clip_index = int(database.clip_indices[database_index])
            frame_index = int(database.frame_indices[database_index])
            clip = clips[clip_index]
            candidate_root_yaw = float(quaternion_to_yaw(clip.qpos[frame_index, :4]))
            candidate_root_local_velocity = world_to_local_vector(
                clip.body_lin_vel_w[frame_index, 0],
                candidate_root_yaw,
            )[:2]
            candidate_yaw_rate = float(clip.body_ang_vel_w[frame_index, 0, 2])
            candidate_velocity_costs[candidate_offset] = float(
                np.linalg.norm(candidate_root_local_velocity - current_root_local_velocity_xy)
            )
            candidate_yaw_rate_costs[candidate_offset] = abs(candidate_yaw_rate - reference_yaw_rate)
        total_costs = total_costs + 0.75 * candidate_velocity_costs + 0.2 * candidate_yaw_rate_costs

    ranked_indices = entry_candidates[np.argsort(total_costs)]
    for database_index in ranked_indices:
        clip_index = int(database.clip_indices[database_index])
        frame_index = int(database.frame_indices[database_index])
        if frame_index > skill_start_frame:
            continue
        clip = clips[clip_index]
        available_future_frames = clip.num_frames - frame_index - 1
        required_future_frames = max(0, skill_end_frame - frame_index)
        if available_future_frames >= required_future_frames:
            return int(database_index)

    return int(ranked_indices[0])


def _smooth_root_transition(
    transitioned_qpos: np.ndarray,
    *,
    previous_qpos: np.ndarray,
    pre_previous_qpos: np.ndarray | None,
    dt: float,
    blend_frames: int,
) -> np.ndarray:
    corrected = np.asarray(transitioned_qpos, dtype=np.float64).copy()
    if corrected.shape[0] <= 1 or blend_frames <= 0:
        return corrected

    blend_count = min(int(blend_frames), corrected.shape[0] - 1)
    previous_root_pos = np.asarray(previous_qpos[4:7], dtype=np.float64)
    previous_root_yaw = float(quaternion_to_yaw(previous_qpos[:4]))
    if pre_previous_qpos is None:
        previous_root_velocity = np.zeros(3, dtype=np.float64)
        previous_yaw_rate = 0.0
    else:
        pre_previous_root_pos = np.asarray(pre_previous_qpos[4:7], dtype=np.float64)
        previous_root_velocity = (previous_root_pos - pre_previous_root_pos) / max(dt, 1e-6)
        pre_previous_yaw = float(quaternion_to_yaw(pre_previous_qpos[:4]))
        previous_yaw_rate = float(wrap_angle(previous_root_yaw - pre_previous_yaw) / max(dt, 1e-6))

    for frame_index in range(1, blend_count + 1):
        normalized_time = frame_index / float(blend_count + 1)
        alpha = normalized_time * normalized_time * (3.0 - 2.0 * normalized_time)
        extrapolated_root_pos = previous_root_pos + previous_root_velocity * (frame_index * dt)
        corrected[frame_index, 4:7] = (
            (1.0 - alpha) * extrapolated_root_pos + alpha * corrected[frame_index, 4:7]
        )

        extrapolated_yaw = previous_root_yaw + previous_yaw_rate * (frame_index * dt)
        target_yaw = float(quaternion_to_yaw(corrected[frame_index, :4]))
        blended_yaw = float(extrapolated_yaw + alpha * wrap_angle(target_yaw - extrapolated_yaw))
        yaw_delta = float(wrap_angle(blended_yaw - target_yaw))
        corrected[frame_index, :4] = quaternion_multiply(
            quaternion_from_yaw(yaw_delta),
            corrected[frame_index, :4],
        )
    return corrected


def _approach_command_from_skill_entry_candidate(
    *,
    database_index: int,
    database: DatabaseBundle,
    clips: list[MotionClip],
    desired_speed_mps: float,
    direction_window_frames: int,
) -> np.ndarray:
    clip_index = int(database.clip_indices[database_index])
    frame_index = int(database.frame_indices[database_index])
    clip = clips[clip_index]

    window_start = max(0, frame_index - direction_window_frames)
    source_root_pos = np.asarray(clip.qpos[frame_index, 4:7], dtype=np.float64)
    source_root_yaw = float(quaternion_to_yaw(clip.qpos[frame_index, :4]))
    source_displacement = source_root_pos - np.asarray(clip.qpos[window_start, 4:7], dtype=np.float64)
    local_displacement = world_to_local_vector(source_displacement, source_root_yaw)[:2]

    if np.linalg.norm(local_displacement) < 1e-4:
        local_velocity = world_to_local_vector(clip.body_lin_vel_w[frame_index, 0], source_root_yaw)[:2]
        local_displacement = local_velocity

    direction_norm = float(np.linalg.norm(local_displacement))
    if direction_norm < 1e-6:
        return np.asarray([desired_speed_mps, 0.0], dtype=np.float64)
    return (local_displacement / direction_norm) * float(desired_speed_mps)


def _terrain_pose_from_local_offset(
    root_qpos: np.ndarray,
    *,
    translation_local: np.ndarray,
    yaw_deg: float,
) -> dict[str, Any]:
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


def _terrain_pose_from_skill_asset_reference(
    *,
    current_root_qpos: np.ndarray,
    skill_clip: MotionClip,
    source_frame: int,
    terrain_world_translation: np.ndarray | None = None,
    terrain_world_yaw_deg: float = 0.0,
) -> dict[str, Any] | None:
    source_frame = int(np.clip(source_frame, 0, skill_clip.num_frames - 1))
    reference_terrain_translation = np.asarray(
        [0.0, 0.0, 0.0] if terrain_world_translation is None else terrain_world_translation,
        dtype=np.float64,
    )
    source_root_qpos = np.asarray(skill_clip.qpos[source_frame], dtype=np.float64)
    source_root_yaw_deg = float(np.rad2deg(quaternion_to_yaw(source_root_qpos[:4])))
    inverse_source_yaw = np.deg2rad(-source_root_yaw_deg)
    inverse_rotation = np.asarray(
        (
            (np.cos(inverse_source_yaw), -np.sin(inverse_source_yaw), 0.0),
            (np.sin(inverse_source_yaw), np.cos(inverse_source_yaw), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    translation_local = inverse_rotation @ (reference_terrain_translation - source_root_qpos[4:7])
    yaw_deg = float(
        np.rad2deg(
            wrap_angle(np.deg2rad(float(terrain_world_yaw_deg) - source_root_yaw_deg))
        )
    )
    return _terrain_pose_from_local_offset(
        current_root_qpos,
        translation_local=translation_local,
        yaw_deg=yaw_deg,
    )


def _generate_single_trajectory(
    *,
    database: DatabaseBundle,
    clips: list[MotionClip],
    skill_metadata: dict[str, Any],
    rng: np.random.Generator,
    pre_skill_config: PreSkillConfig,
    terrain_anchor_config: TerrainAnchorConfig,
    start_pose_config: StartPoseConfig,
    search_interval_frames: int,
    command_spring_damping: float,
    inertialization_damping: float,
    post_skill_frames: int,
    kinematics: KinematicsHelper,
) -> tuple[np.ndarray, dict[str, Any]]:
    speed_mps = float(rng.choice(np.asarray(pre_skill_config.speed_levels_mps, dtype=np.float64)))
    initial_heading_delta_deg = float(
        np.clip(
            rng.choice(np.asarray(pre_skill_config.heading_degrees, dtype=np.float64)),
            -MAX_INITIAL_HEADING_DELTA_DEGREES,
            MAX_INITIAL_HEADING_DELTA_DEGREES,
        )
    )
    pre_skill_distance_meters = float(
        rng.uniform(
            pre_skill_config.start_distance_min_meters,
            pre_skill_config.start_distance_max_meters,
        )
    )
    pre_skill_seconds = pre_skill_distance_meters / max(speed_mps, 1e-6)
    pre_skill_frames = max(search_interval_frames, int(round(pre_skill_seconds * clips[0].fps)))

    approach_command = np.asarray([speed_mps, 0.0], dtype=np.float64)
    skill_command = np.asarray([speed_mps, 0.0], dtype=np.float64)
    skill_transition_root_blend_frames = max(6, search_interval_frames)
    steering_clip_name: str | None = None
    steering_source_frame: int | None = None
    initial_heading_reference_deg: float | None = None
    skill_entry_settle_frames = max(6, search_interval_frames)

    output_qpos: list[np.ndarray] = []
    segments: list[dict[str, Any]] = []
    fps = clips[0].fps
    dt = 1.0 / float(fps)

    standing_clip, standing_frame_idx = _select_standing_start(
        database,
        clips,
        target_root_height=None,
        window_radius=start_pose_config.standing_window_radius_frames,
    )
    standing_qpos = standing_clip.qpos[standing_frame_idx].copy()
    selected_start_root_height = float(standing_qpos[6])
    target_start_root_height = selected_start_root_height
    desired_start_root_height = selected_start_root_height + float(start_pose_config.root_height_offset_meters)
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

    initial_pose_state = _runtime_pose_state_from_output(
        output_qpos=output_qpos,
        kinematics=kinematics,
        dt=dt,
    )
    initial_query = _build_query_from_steering_plan(
        output_qpos=output_qpos,
        kinematics=kinematics,
        dt=dt,
        command_local_velocity_xy=approach_command,
        database=database,
    )
    initial_skill_index = _select_skill_entry_index(
        query_feature=initial_query,
        skill_metadata=skill_metadata,
        database=database,
        clips=clips,
        current_root_local_velocity_xy=initial_pose_state["root_local_velocity"][:2],
        current_root_yaw_rate=_current_root_yaw_rate_from_output(output_qpos, dt),
    )
    initial_entry_command = _approach_command_from_skill_entry_candidate(
        database_index=initial_skill_index,
        database=database,
        clips=clips,
        desired_speed_mps=speed_mps,
        direction_window_frames=pre_skill_config.approach_direction_window_frames,
    )
    initial_heading_reference_deg = float(np.rad2deg(np.arctan2(initial_entry_command[1], initial_entry_command[0])))
    approach_command = _clamp_forward_steering_command(
        _rotate_command_xy(initial_entry_command, initial_heading_delta_deg),
        speed_mps,
    )
    steering_clip = clips[int(database.clip_indices[initial_skill_index])]
    steering_clip_name = steering_clip.name
    steering_source_frame = int(database.frame_indices[initial_skill_index])

    while (len(output_qpos) - 1) < pre_skill_frames:
        remaining_frames = pre_skill_frames - (len(output_qpos) - 1)
        pose_state = _runtime_pose_state_from_output(
            output_qpos=output_qpos,
            kinematics=kinematics,
            dt=dt,
        )
        steering_query = _build_query_from_steering_plan(
            output_qpos=output_qpos,
            kinematics=kinematics,
            dt=dt,
            command_local_velocity_xy=approach_command,
            database=database,
        )
        steering_skill_index = _select_skill_entry_index(
            query_feature=steering_query,
            skill_metadata=skill_metadata,
            database=database,
            clips=clips,
            current_root_local_velocity_xy=pose_state["root_local_velocity"][:2],
            current_root_yaw_rate=_current_root_yaw_rate_from_output(output_qpos, dt),
        )
        target_approach_command = _approach_command_from_skill_entry_candidate(
            database_index=steering_skill_index,
            database=database,
            clips=clips,
            desired_speed_mps=speed_mps,
            direction_window_frames=pre_skill_config.approach_direction_window_frames,
        )
        approach_command = _clamp_forward_steering_command(
            _limit_command_heading_change(
                approach_command,
                target_approach_command,
                MAX_INITIAL_HEADING_DELTA_DEGREES,
            ),
            speed_mps,
        )
        steering_clip = clips[int(database.clip_indices[steering_skill_index])]
        steering_clip_name = steering_clip.name
        steering_source_frame = int(database.frame_indices[steering_skill_index])

        query_feature = _build_query_from_steering_plan(
            output_qpos=output_qpos,
            kinematics=kinematics,
            dt=dt,
            command_local_velocity_xy=approach_command,
            database=database,
        )
        database_index = _select_locomotion_approach_index(
            query_feature=query_feature,
            command_local_velocity_xy=approach_command,
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

    skill_pose_state = _runtime_pose_state_from_output(
        output_qpos=output_qpos,
        kinematics=kinematics,
        dt=dt,
    )
    skill_start_frame = int(skill_metadata["skill_start_frame"])
    skill_query = _build_query_from_steering_plan(
        output_qpos=output_qpos,
        kinematics=kinematics,
        dt=dt,
        command_local_velocity_xy=approach_command,
        database=database,
    )
    skill_database_index = _select_skill_entry_index(
        query_feature=skill_query,
        skill_metadata=skill_metadata,
        database=database,
        clips=clips,
        current_root_local_velocity_xy=skill_pose_state["root_local_velocity"][:2],
        current_root_yaw_rate=_current_root_yaw_rate_from_output(output_qpos, dt),
        latest_entry_frame=max(0, skill_start_frame - skill_entry_settle_frames),
    )
    skill_clip = clips[int(database.clip_indices[skill_database_index])]
    skill_frame = int(database.frame_indices[skill_database_index])
    skill_end_frame = int(skill_metadata["skill_end_frame"])
    skill_transition_qpos = np.asarray(output_qpos[-1], dtype=np.float64)
    terrain_world_pose = _terrain_pose_from_skill_asset_reference(
        current_root_qpos=skill_transition_qpos,
        skill_clip=skill_clip,
        source_frame=skill_frame,
        terrain_world_translation=np.asarray(terrain_anchor_config.translation, dtype=np.float64),
        terrain_world_yaw_deg=float(terrain_anchor_config.yaw_deg),
    )
    skill_output_start = len(output_qpos)
    skill_appended = _append_clip_chunk(
        output_qpos=output_qpos,
        clip=skill_clip,
        source_start_frame=skill_frame,
        desired_new_frames=max(0, skill_end_frame - skill_frame),
        dt=dt,
        inertialization_damping=inertialization_damping,
        root_blend_frames=skill_transition_root_blend_frames,
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
            "root_translation": [float(value) for value in output_qpos[0][4:7].tolist()],
            "root_yaw_deg": float(np.rad2deg(quaternion_to_yaw(output_qpos[0][:4]))),
            "force_root_height_alignment": bool(start_pose_config.force_root_height_alignment),
            "root_height_offset_meters": float(start_pose_config.root_height_offset_meters),
        },
        "skill_world_locked": False,
        "skill_anchor": {
            "fixed_world": False,
            "entry_source_frame": int(skill_frame),
            "source_frame": int(skill_frame),
            "selection_mode": "runtime_entry_match",
            "database_index": int(skill_database_index),
            "mirrored": bool(skill_clip.mirrored),
            "mirroring_allowed": bool(_skill_allows_mirrored_candidates(skill_metadata)),
            "root_translation": [float(value) for value in skill_clip.qpos[skill_frame, 4:7].tolist()],
            "root_yaw_deg": float(np.rad2deg(quaternion_to_yaw(skill_clip.qpos[skill_frame, :4]))),
        },
        "terrain_anchor": {
            "applied_source_frame": int(skill_frame),
            "applied_output_frame": int(skill_output_start - 1),
            "interpreted_as": "terrain_asset_origin_in_skill_clip_world",
            "anchor_policy": "matched_entry_frame_to_transition_root",
        },
        "command": {
            "speed_mps": speed_mps,
            "initial_heading_reference_deg": initial_heading_reference_deg,
            "initial_heading_delta_deg": initial_heading_delta_deg,
            "max_initial_heading_delta_degrees": MAX_INITIAL_HEADING_DELTA_DEGREES,
            "steering_query_max_heading_degrees": STEERING_QUERY_MAX_HEADING_DEGREES,
            "min_forward_speed_fraction": MIN_FORWARD_SPEED_FRACTION,
            "max_lateral_speed_fraction": MAX_LATERAL_SPEED_FRACTION,
            "approach_command_local_xy": [float(value) for value in approach_command.tolist()],
            "pre_skill_distance_meters": pre_skill_distance_meters,
            "min_start_distance_meters": MIN_START_DISTANCE_METERS,
            "pre_skill_seconds": pre_skill_seconds,
            "pre_skill_frames": pre_skill_frames,
            "post_skill_frames": post_skill_frames,
        },
        "approach": {
            "target_clip_name": skill_clip.name,
            "matched_entry_source_frame": int(skill_frame),
            "steering_mode": "forward_arc_plan_to_skill_entry_direction",
            "steering_window_frames": int(pre_skill_config.approach_direction_window_frames),
            "last_steering_clip_name": steering_clip_name,
            "last_steering_source_frame": steering_source_frame,
        },
        "search": {
            "interval_frames": search_interval_frames,
            "command_spring_damping": command_spring_damping,
            "inertialization_damping": inertialization_damping,
            "skill_transition_root_blend_frames": int(skill_transition_root_blend_frames),
            "skill_entry_settle_frames": int(skill_entry_settle_frames),
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
        default=None,
        help="要生成的技能名。",
    )
    parser.add_argument(
        "--skill-prefix",
        type=str,
        default=None,
        help="技能前缀，例如 climb_15。配合 --all-scales 批量生成该前缀下的所有 z_scale。",
    )
    parser.add_argument(
        "--all-scales",
        action="store_true",
        help="按 --skill-prefix 匹配所有 z_scale 技能并批量生成。",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=1,
        help="要生成的轨迹数量。",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=None,
        help="目标总轨迹数；配合 --append-existing 时只生成缺少的尾部编号。",
    )
    parser.add_argument(
        "--append-existing",
        action="store_true",
        help="保留已有轨迹文件和 manifest 条目，从已有连续编号后继续生成。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="轨迹输出目录。单技能模式下默认是 output/motion_matching/generated/<skill_name>。",
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

    default_single_skill = "climb_15_z_scale_1.0"
    effective_skill_name = args.skill_name
    if effective_skill_name is None and not args.all_scales:
        effective_skill_name = default_single_skill

    database = _load_database_bundle(args.database_dir)
    clips = _load_clip_catalog(database.manifest)
    pre_skill_config = _load_pre_skill_config(args.generation_config)
    terrain_anchor_config = _load_terrain_anchor_config(args.generation_config)
    start_pose_config = _load_start_pose_config(args.generation_config)
    target_skill_names = _resolve_target_skill_names(
        database.manifest,
        skill_name=effective_skill_name,
        skill_prefix=args.skill_prefix,
        all_scales=args.all_scales,
    )
    kinematics = KinematicsHelper()
    post_skill_frames = max(1, int(round(float(args.post_skill_seconds) * clips[0].fps)))

    for skill_offset, target_skill_name in enumerate(target_skill_names):
        skill_metadata = _find_skill_metadata(database.manifest, target_skill_name)
        if args.output_dir is None:
            output_dir = resolve_repo_path(Path("output/motion_matching/generated") / target_skill_name)
        elif args.all_scales:
            output_dir = resolve_repo_path(args.output_dir) / target_skill_name
        else:
            output_dir = resolve_repo_path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        existing_count = 0
        generation_start_index = 0
        generation_end_index = args.num_trajectories
        requested_total = args.num_trajectories
        if args.target_total is not None:
            if args.target_total < 0:
                raise ValueError("--target-total 不能小于 0")
            requested_total = args.target_total
            generation_end_index = args.target_total
            if args.append_existing:
                existing_count = _count_existing_trajectories(output_dir, target_skill_name)
                generation_start_index = min(existing_count, args.target_total)
        elif args.append_existing:
            existing_count = _count_existing_trajectories(output_dir, target_skill_name)
            generation_start_index = existing_count
            generation_end_index = existing_count + args.num_trajectories
            requested_total = generation_end_index

        skill_seed = args.seed + skill_offset + generation_start_index * 1_000_003
        rng = np.random.default_rng(skill_seed)
        trajectory_manifests: list[dict[str, Any]] = []
        if args.append_existing and generation_start_index > 0:
            trajectory_manifests = _load_existing_trajectory_manifests(
                output_dir,
                skill_metadata,
                target_skill_name,
                generation_start_index,
            )
            print(
                f"[info] {target_skill_name}: 已有连续轨迹 {generation_start_index} 条，"
                f"本次生成 {generation_end_index - generation_start_index} 条。"
            )

        for trajectory_index in range(generation_start_index, generation_end_index):
            last_error: Exception | None = None
            for attempt_index in range(20):
                try:
                    qpos_trajectory, trajectory_manifest = _generate_single_trajectory(
                        database=database,
                        clips=clips,
                        skill_metadata=skill_metadata,
                        rng=rng,
                        pre_skill_config=pre_skill_config,
                        terrain_anchor_config=terrain_anchor_config,
                        start_pose_config=start_pose_config,
                        search_interval_frames=args.search_interval_frames,
                        command_spring_damping=args.command_spring_damping,
                        inertialization_damping=args.inertialization_damping,
                        post_skill_frames=post_skill_frames,
                        kinematics=kinematics,
                    )
                    break
                except RuntimeError as exc:
                    last_error = exc
                    print(f"[warn] {target_skill_name} #{trajectory_index} 第 {attempt_index + 1} 次采样失败：{exc}")
            else:
                raise RuntimeError(f"{target_skill_name} #{trajectory_index} 连续采样失败") from last_error

            trajectory_name = f"{sanitize_skill_name(target_skill_name)}_{trajectory_index:04d}"
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
            "skill_name": target_skill_name,
            "skill_prefix": args.skill_prefix,
            "all_scales": bool(args.all_scales),
            "num_trajectories": requested_total,
            "seed": skill_seed,
            "generation_config_path": str(resolve_repo_path(args.generation_config).relative_to(resolve_repo_path("."))),
            "pre_skill_config": {
                "speed_levels_mps": [float(value) for value in pre_skill_config.speed_levels_mps],
                "heading_degrees": [float(value) for value in pre_skill_config.heading_degrees],
                "max_initial_heading_delta_degrees": float(MAX_INITIAL_HEADING_DELTA_DEGREES),
                "min_start_distance_meters": float(MIN_START_DISTANCE_METERS),
                "min_translation_speed_fraction": float(MIN_TRANSLATION_SPEED_FRACTION),
                "min_forward_speed_fraction": float(MIN_FORWARD_SPEED_FRACTION),
                "max_lateral_speed_fraction": float(MAX_LATERAL_SPEED_FRACTION),
                "steering_query_max_heading_degrees": float(STEERING_QUERY_MAX_HEADING_DEGREES),
                "start_distance_meters": {
                    "min": float(pre_skill_config.start_distance_min_meters),
                    "max": float(pre_skill_config.start_distance_max_meters),
                },
                "approach_direction_window_frames": int(pre_skill_config.approach_direction_window_frames),
            },
            "terrain_asset_reference_config": {
                "translation": [float(value) for value in terrain_anchor_config.translation],
                "yaw_deg": float(terrain_anchor_config.yaw_deg),
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

    if args.skill_name is None and not args.all_scales:
        print(f"[info] 未显式指定技能，默认建议使用 --skill-name {default_single_skill}")


if __name__ == "__main__":
    main()
