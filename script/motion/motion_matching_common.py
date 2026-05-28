from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HORIZONS = (0.33, 0.67, 1.0)
DEFAULT_SEARCH_INTERVAL_FRAMES = 10
DEFAULT_COMMAND_SPRING_DAMPING = 6.0
DEFAULT_INERTIALIZATION_DAMPING = 10.0
DEFAULT_SPEED_LEVELS = (1.0, 2.0)
DEFAULT_HEADING_DEGREES = (-90.0, -45.0, 0.0, 45.0, 90.0)
ROOT_BODY_INDEX = 0
LEFT_FOOT_BODY_NAME = "left_ankle_roll_link"
RIGHT_FOOT_BODY_NAME = "right_ankle_roll_link"

CANONICAL_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)

CANONICAL_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

BODY_NAME_TO_INDEX = {name: index for index, name in enumerate(CANONICAL_BODY_NAMES)}
JOINT_NAME_TO_INDEX = {name: index for index, name in enumerate(CANONICAL_JOINT_NAMES)}
FEATURE_SIZE = 27


@dataclass
class SkillMetadata:
    skill_name: str
    skill_path: Path
    terrain_path: Path
    skill_start_frame: int
    skill_end_frame: int
    entry_window_length: int
    terrain_root_offset: dict[str, Any] | None
    allow_mirrored_skill: bool | None
    annotation_source: str
    needs_review: bool


@dataclass
class MotionClip:
    name: str
    source_path: str
    qpos: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray
    fps: int
    mirrored: bool = False
    source_name: str | None = None

    @property
    def num_frames(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def dt(self) -> float:
        return 1.0 / float(self.fps)

    @property
    def root_quat_w(self) -> np.ndarray:
        return self.qpos[:, :4]

    @property
    def root_pos_w(self) -> np.ndarray:
        return self.qpos[:, 4:7]


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-12, 1.0, norm)
    return quat / norm


def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    quat = normalize_quaternion(quaternion)
    result = quat.copy()
    result[..., 1:] *= -1.0
    return result


def quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs = normalize_quaternion(lhs)
    rhs = normalize_quaternion(rhs)
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def rotation_matrix_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quat = normalize_quaternion(quaternion)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack(
        (
            np.stack((1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)), axis=-1),
            np.stack((2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)), axis=-1),
            np.stack((2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)), axis=-1),
        ),
        axis=-2,
    )


def quaternion_from_yaw(yaw: float) -> np.ndarray:
    half_yaw = 0.5 * float(yaw)
    return np.asarray([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)], dtype=np.float64)


def quaternion_to_yaw(quaternion: np.ndarray) -> float:
    quat = normalize_quaternion(quaternion)
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(angle) + math.pi) % (2.0 * math.pi) - math.pi


def rotation_vector_to_quaternion(rotation_vector: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotation_vector, dtype=np.float64)
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    axis = np.divide(rotvec, np.where(angle < 1e-12, 1.0, angle))
    half_angle = 0.5 * angle
    sin_half = np.sin(half_angle)
    quat = np.concatenate((np.cos(half_angle), axis * sin_half), axis=-1)
    return normalize_quaternion(quat)


def quaternion_to_rotation_vector(quaternion: np.ndarray) -> np.ndarray:
    quat = normalize_quaternion(quaternion)
    if quat.ndim == 1:
        return quaternion_to_rotation_vector(quat[None, :])[0]

    vector = quat[:, 1:]
    vector_norm = np.linalg.norm(vector, axis=1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(quat[:, :1], -1.0, 1.0))
    angle = np.where(angle > math.pi, angle - 2.0 * math.pi, angle)
    axis = np.divide(vector, np.where(vector_norm < 1e-12, 1.0, vector_norm))
    return axis * angle


def mirror_quaternion_xz(quaternion: np.ndarray) -> np.ndarray:
    quat = normalize_quaternion(quaternion)
    mirrored = quat.copy()
    mirrored[..., 1] *= -1.0
    mirrored[..., 3] *= -1.0
    return mirrored


def mirror_vector_xz(vector: np.ndarray) -> np.ndarray:
    mirrored = np.asarray(vector, dtype=np.float64).copy()
    mirrored[..., 1] *= -1.0
    return mirrored


def mirror_angular_vector_xz(vector: np.ndarray) -> np.ndarray:
    mirrored = np.asarray(vector, dtype=np.float64).copy()
    mirrored[..., 0] *= -1.0
    mirrored[..., 2] *= -1.0
    return mirrored


def yaw_rotation_matrix(yaw: float) -> np.ndarray:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.asarray(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def world_to_local_vector(vector_w: np.ndarray, root_yaw: float) -> np.ndarray:
    rotation = yaw_rotation_matrix(root_yaw)
    return np.asarray(vector_w, dtype=np.float64) @ rotation


def local_to_world_vector(vector_l: np.ndarray, root_yaw: float) -> np.ndarray:
    rotation = yaw_rotation_matrix(root_yaw)
    return np.asarray(vector_l, dtype=np.float64) @ rotation.T


def centered_finite_difference(series: np.ndarray, dt: float) -> np.ndarray:
    series = np.asarray(series, dtype=np.float64)
    if len(series) == 0:
        return np.zeros_like(series)
    if len(series) == 1:
        return np.zeros_like(series)

    velocity = np.zeros_like(series)
    velocity[0] = (series[1] - series[0]) / dt
    velocity[-1] = (series[-1] - series[-2]) / dt
    if len(series) > 2:
        velocity[1:-1] = (series[2:] - series[:-2]) / (2.0 * dt)
    return velocity


def quaternion_velocity_series(quaternions: np.ndarray, dt: float) -> np.ndarray:
    quaternions = normalize_quaternion(quaternions)
    if len(quaternions) <= 1:
        return np.zeros((len(quaternions), 3), dtype=np.float64)

    angular_velocity = np.zeros((len(quaternions), 3), dtype=np.float64)
    relative_forward = quaternion_multiply(quaternions[1:], quaternion_conjugate(quaternions[:-1]))
    forward_rotvec = quaternion_to_rotation_vector(relative_forward) / dt
    angular_velocity[0] = forward_rotvec[0]
    angular_velocity[-1] = forward_rotvec[-1]
    if len(quaternions) > 2:
        relative_central = quaternion_multiply(quaternions[2:], quaternion_conjugate(quaternions[:-2]))
        angular_velocity[1:-1] = quaternion_to_rotation_vector(relative_central) / (2.0 * dt)
    return angular_velocity


def clip_name_from_path(path: Path) -> str:
    return path.stem


def _motion_clip_from_qpos(
    path: Path,
    data: np.lib.npyio.NpzFile,
    *,
    clip_name: str | None = None,
) -> MotionClip:
    if "qpos" not in data or "fps" not in data:
        raise KeyError(f"{path} 缺少 qpos-only clip 所需字段: ['qpos', 'fps']")

    qpos = np.asarray(data["qpos"], dtype=np.float64)
    fps = int(round(float(np.asarray(data["fps"]).reshape(-1)[0])))
    if qpos.ndim != 2 or qpos.shape[1] != 7 + len(CANONICAL_JOINT_NAMES):
        raise ValueError(
            f"{path} 的 qpos 维度是 {qpos.shape}，预期第二维为 {7 + len(CANONICAL_JOINT_NAMES)}"
        )

    kinematics = KinematicsHelper()
    body_pos_w = np.zeros((qpos.shape[0], len(CANONICAL_BODY_NAMES), 3), dtype=np.float64)
    body_quat_w = np.zeros((qpos.shape[0], len(CANONICAL_BODY_NAMES), 4), dtype=np.float64)
    for frame_index, frame_qpos in enumerate(qpos):
        positions, quaternions = kinematics.canonical_body_poses(frame_qpos)
        body_pos_w[frame_index] = positions
        body_quat_w[frame_index] = quaternions

    dt = 1.0 / float(fps)
    joint_pos = qpos[:, 7:]
    joint_vel = centered_finite_difference(joint_pos, dt)
    body_lin_vel_w = centered_finite_difference(body_pos_w, dt)
    body_ang_vel_w = np.zeros((qpos.shape[0], len(CANONICAL_BODY_NAMES), 3), dtype=np.float64)
    for body_index in range(len(CANONICAL_BODY_NAMES)):
        body_ang_vel_w[:, body_index] = quaternion_velocity_series(body_quat_w[:, body_index], dt)

    return MotionClip(
        name=clip_name or clip_name_from_path(path),
        source_path=str(path.relative_to(REPO_ROOT)),
        qpos=qpos,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=normalize_quaternion(body_quat_w),
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        fps=fps,
        mirrored=False,
        source_name=clip_name or clip_name_from_path(path),
    )


def load_motion_clip(path_like: str | Path, *, clip_name: str | None = None) -> MotionClip:
    path = resolve_repo_path(path_like)
    data = np.load(path, allow_pickle=True)
    required_keys = (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "fps",
    )
    missing = [key for key in required_keys if key not in data]
    if missing:
        if "qpos" in data and "fps" in data:
            return _motion_clip_from_qpos(path, data, clip_name=clip_name)
        raise KeyError(f"{path} 缺少必要字段: {missing}")

    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)
    body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float64)
    body_quat_w = normalize_quaternion(np.asarray(data["body_quat_w"], dtype=np.float64))
    body_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float64)
    body_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float64)
    fps = int(round(float(np.asarray(data["fps"]).reshape(-1)[0])))

    if joint_pos.shape[1] != len(CANONICAL_JOINT_NAMES):
        raise ValueError(
            f"{path} 的 joint_pos 维度是 {joint_pos.shape[1]}，预期是 {len(CANONICAL_JOINT_NAMES)}"
        )
    if body_pos_w.shape[1] != len(CANONICAL_BODY_NAMES):
        raise ValueError(
            f"{path} 的 body_pos_w 维度是 {body_pos_w.shape[1]}，预期是 {len(CANONICAL_BODY_NAMES)}"
        )

    qpos = np.concatenate((body_quat_w[:, ROOT_BODY_INDEX], body_pos_w[:, ROOT_BODY_INDEX], joint_pos), axis=1)
    return MotionClip(
        name=clip_name or clip_name_from_path(path),
        source_path=str(path.relative_to(REPO_ROOT)),
        qpos=qpos,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        fps=fps,
        mirrored=False,
        source_name=clip_name or clip_name_from_path(path),
    )


def build_left_right_name_pairs(names: tuple[str, ...]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for name in names:
        if name.startswith("left_"):
            partner = "right_" + name[len("left_") :]
            if partner in names:
                pairs.append((name, partner))
    return pairs


def build_mirror_permutation(names: tuple[str, ...]) -> np.ndarray:
    mirrored_names: list[str] = []
    for name in names:
        if name.startswith("left_"):
            mirrored_names.append("right_" + name[len("left_") :])
        elif name.startswith("right_"):
            mirrored_names.append("left_" + name[len("right_") :])
        else:
            mirrored_names.append(name)
    name_to_index = {name: index for index, name in enumerate(names)}
    return np.asarray([name_to_index[name] for name in mirrored_names], dtype=np.int32)


def joint_axis_sign(joint_name: str) -> float:
    if "pitch" in joint_name or "knee" in joint_name or "elbow" in joint_name:
        return 1.0
    if "roll" in joint_name or "yaw" in joint_name:
        return -1.0
    raise KeyError(f"无法为关节 {joint_name} 推断镜像符号")


JOINT_MIRROR_PERMUTATION = build_mirror_permutation(CANONICAL_JOINT_NAMES)
JOINT_MIRROR_SIGNS = np.asarray([joint_axis_sign(name) for name in CANONICAL_JOINT_NAMES], dtype=np.float64)
BODY_MIRROR_PERMUTATION = build_mirror_permutation(CANONICAL_BODY_NAMES)


def mirror_motion_clip(clip: MotionClip) -> MotionClip:
    mirrored_joint_pos = clip.joint_pos[:, JOINT_MIRROR_PERMUTATION] * JOINT_MIRROR_SIGNS
    mirrored_joint_vel = clip.joint_vel[:, JOINT_MIRROR_PERMUTATION] * JOINT_MIRROR_SIGNS

    mirrored_body_pos = mirror_vector_xz(clip.body_pos_w[:, BODY_MIRROR_PERMUTATION])
    mirrored_body_quat = mirror_quaternion_xz(clip.body_quat_w[:, BODY_MIRROR_PERMUTATION])
    mirrored_body_lin_vel = mirror_vector_xz(clip.body_lin_vel_w[:, BODY_MIRROR_PERMUTATION])
    mirrored_body_ang_vel = mirror_angular_vector_xz(clip.body_ang_vel_w[:, BODY_MIRROR_PERMUTATION])

    mirrored_qpos = np.concatenate(
        (mirror_quaternion_xz(clip.root_quat_w), mirror_vector_xz(clip.root_pos_w), mirrored_joint_pos),
        axis=1,
    )
    return MotionClip(
        name=f"{clip.name}__mirror",
        source_path=clip.source_path,
        qpos=mirrored_qpos,
        joint_pos=mirrored_joint_pos,
        joint_vel=mirrored_joint_vel,
        body_pos_w=mirrored_body_pos,
        body_quat_w=mirrored_body_quat,
        body_lin_vel_w=mirrored_body_lin_vel,
        body_ang_vel_w=mirrored_body_ang_vel,
        fps=clip.fps,
        mirrored=True,
        source_name=clip.source_name or clip.name,
    )


def load_skill_catalog(path_like: str | Path) -> list[SkillMetadata]:
    catalog_path = resolve_repo_path(path_like)
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    if "skills" in payload:
        raw_skills = payload["skills"]
    else:
        raw_skills = [payload]

    result: list[SkillMetadata] = []
    for raw in raw_skills:
        result.append(
            SkillMetadata(
                skill_name=str(raw["skill_name"]),
                skill_path=resolve_repo_path(raw["skill_path"]),
                terrain_path=resolve_repo_path(raw["terrain_path"]),
                skill_start_frame=int(raw["skill_start_frame"]),
                skill_end_frame=int(raw["skill_end_frame"]),
                entry_window_length=int(raw["entry_window_length"]),
                terrain_root_offset=raw.get("terrain_root_offset"),
                allow_mirrored_skill=(
                    bool(raw["allow_mirrored_skill"]) if "allow_mirrored_skill" in raw else None
                ),
                annotation_source=str(raw.get("annotation_source", "manual")),
                needs_review=bool(raw.get("needs_review", False)),
            )
        )
    return result


def skill_catalog_to_json(skills: list[SkillMetadata]) -> dict[str, Any]:
    return {
        "version": 1,
        "skills": [
            {
                "skill_name": skill.skill_name,
                "skill_path": str(skill.skill_path.relative_to(REPO_ROOT)),
                "terrain_path": str(skill.terrain_path.relative_to(REPO_ROOT)),
                "skill_start_frame": skill.skill_start_frame,
                "skill_end_frame": skill.skill_end_frame,
                "entry_window_length": skill.entry_window_length,
                "terrain_root_offset": skill.terrain_root_offset,
                **(
                    {"allow_mirrored_skill": skill.allow_mirrored_skill}
                    if skill.allow_mirrored_skill is not None
                    else {}
                ),
                "annotation_source": skill.annotation_source,
                "needs_review": skill.needs_review,
            }
            for skill in skills
        ],
    }


def compute_feature_vector(
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    body_lin_vel_w: np.ndarray,
    frame_index: int,
    fps: int,
    *,
    horizons: tuple[float, ...] = DEFAULT_HORIZONS,
) -> np.ndarray:
    root_pos_w = body_pos_w[frame_index, ROOT_BODY_INDEX]
    root_yaw = quaternion_to_yaw(body_quat_w[frame_index, ROOT_BODY_INDEX])
    left_foot_index = BODY_NAME_TO_INDEX[LEFT_FOOT_BODY_NAME]
    right_foot_index = BODY_NAME_TO_INDEX[RIGHT_FOOT_BODY_NAME]

    feature: list[float] = []
    for horizon in horizons:
        target_index = min(frame_index + int(round(horizon * fps)), body_pos_w.shape[0] - 1)
        relative_root_pos = body_pos_w[target_index, ROOT_BODY_INDEX] - root_pos_w
        local_xy = world_to_local_vector(relative_root_pos, root_yaw)[:2]
        target_yaw = quaternion_to_yaw(body_quat_w[target_index, ROOT_BODY_INDEX])
        local_heading = wrap_angle(target_yaw - root_yaw)
        feature.extend((float(local_xy[0]), float(local_xy[1]), math.cos(float(local_heading)), math.sin(float(local_heading))))

    for foot_index in (left_foot_index, right_foot_index):
        local_pos = world_to_local_vector(body_pos_w[frame_index, foot_index] - root_pos_w, root_yaw)
        local_vel = world_to_local_vector(body_lin_vel_w[frame_index, foot_index], root_yaw)
        feature.extend(local_pos.tolist())
        feature.extend(local_vel.tolist())

    root_local_velocity = world_to_local_vector(body_lin_vel_w[frame_index, ROOT_BODY_INDEX], root_yaw)
    feature.extend(root_local_velocity.tolist())
    return np.asarray(feature, dtype=np.float64)


def build_clip_feature_matrix(clip: MotionClip, *, horizons: tuple[float, ...] = DEFAULT_HORIZONS) -> np.ndarray:
    features = np.zeros((clip.num_frames, FEATURE_SIZE), dtype=np.float64)
    for frame_index in range(clip.num_frames):
        features[frame_index] = compute_feature_vector(
            clip.body_pos_w,
            clip.body_quat_w,
            clip.body_lin_vel_w,
            frame_index,
            clip.fps,
            horizons=horizons,
        )
    return features


def fit_feature_normalization(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float64), std.astype(np.float64)


def normalize_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(features, dtype=np.float64) - mean) / std


def critically_damped_spring_value(
    position0: np.ndarray,
    velocity0: np.ndarray,
    goal: np.ndarray,
    horizon: float,
    damping: float,
) -> np.ndarray:
    position0 = np.asarray(position0, dtype=np.float64)
    velocity0 = np.asarray(velocity0, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    j0 = position0 - goal
    j1 = velocity0 + damping * j0
    decay = math.exp(-damping * horizon)
    return decay * (j0 + horizon * j1) + goal


def critically_damped_integrated_velocity(
    velocity0: np.ndarray,
    acceleration0: np.ndarray,
    goal_velocity: np.ndarray,
    horizon: float,
    damping: float,
) -> np.ndarray:
    velocity0 = np.asarray(velocity0, dtype=np.float64)
    acceleration0 = np.asarray(acceleration0, dtype=np.float64)
    goal_velocity = np.asarray(goal_velocity, dtype=np.float64)
    j0 = velocity0 - goal_velocity
    j1 = acceleration0 + damping * j0
    decay = math.exp(-damping * horizon)
    return (
        goal_velocity * horizon
        + (j0 * (1.0 - decay)) / damping
        + (j1 * (1.0 - decay * (1.0 + damping * horizon))) / (damping * damping)
    )


def build_command_feature(
    root_local_velocity_xy: np.ndarray,
    command_local_velocity_xy: np.ndarray,
    *,
    horizons: tuple[float, ...] = DEFAULT_HORIZONS,
    damping: float = DEFAULT_COMMAND_SPRING_DAMPING,
) -> np.ndarray:
    current_velocity = np.asarray(root_local_velocity_xy, dtype=np.float64)
    target_velocity = np.asarray(command_local_velocity_xy, dtype=np.float64)
    current_acceleration = np.zeros(2, dtype=np.float64)
    current_heading = np.zeros(1, dtype=np.float64)
    current_heading_rate = np.zeros(1, dtype=np.float64)
    if np.linalg.norm(target_velocity) > 1e-6:
        target_heading = np.asarray([math.atan2(float(target_velocity[1]), float(target_velocity[0]))], dtype=np.float64)
    else:
        target_heading = np.zeros(1, dtype=np.float64)

    feature: list[float] = []
    for horizon in horizons:
        local_position = critically_damped_integrated_velocity(
            current_velocity,
            current_acceleration,
            target_velocity,
            horizon,
            damping,
        )
        local_heading = critically_damped_spring_value(
            current_heading,
            current_heading_rate,
            target_heading,
            horizon,
            damping,
        )[0]
        feature.extend((float(local_position[0]), float(local_position[1]), math.cos(float(local_heading)), math.sin(float(local_heading))))
    return np.asarray(feature, dtype=np.float64)


def build_runtime_query_feature(
    left_foot_local_pos: np.ndarray,
    right_foot_local_pos: np.ndarray,
    left_foot_local_vel: np.ndarray,
    right_foot_local_vel: np.ndarray,
    root_local_velocity: np.ndarray,
    command_local_velocity_xy: np.ndarray,
    *,
    horizons: tuple[float, ...] = DEFAULT_HORIZONS,
    damping: float = DEFAULT_COMMAND_SPRING_DAMPING,
) -> np.ndarray:
    command_feature = build_command_feature(
        root_local_velocity[:2],
        command_local_velocity_xy,
        horizons=horizons,
        damping=damping,
    )
    pose_feature = np.concatenate(
        (
            np.asarray(left_foot_local_pos, dtype=np.float64),
            np.asarray(left_foot_local_vel, dtype=np.float64),
            np.asarray(right_foot_local_pos, dtype=np.float64),
            np.asarray(right_foot_local_vel, dtype=np.float64),
            np.asarray(root_local_velocity, dtype=np.float64),
        ),
        axis=0,
    )
    return np.concatenate((command_feature, pose_feature), axis=0)


def nearest_neighbor_search(query_feature: np.ndarray, database_features: np.ndarray, candidate_indices: np.ndarray) -> int:
    if len(candidate_indices) == 0:
        raise ValueError("候选集合为空，无法执行 motion matching 检索")
    candidate_features = database_features[candidate_indices]
    distances = np.linalg.norm(candidate_features - query_feature[None, :], axis=1)
    return int(candidate_indices[int(np.argmin(distances))])


def yaw_align_qpos_sequence(
    sequence_qpos: np.ndarray,
    target_root_quat: np.ndarray,
    target_root_pos: np.ndarray,
    anchor_frame_index: int = 0,
) -> np.ndarray:
    aligned = np.asarray(sequence_qpos, dtype=np.float64).copy()
    if aligned.shape[0] == 0:
        return aligned

    if anchor_frame_index < 0:
        anchor_frame_index += aligned.shape[0]
    if anchor_frame_index < 0 or anchor_frame_index >= aligned.shape[0]:
        raise IndexError(f"anchor_frame_index {anchor_frame_index} 超出序列范围 {aligned.shape[0]}")

    source_root_quat = aligned[anchor_frame_index, :4]
    source_root_pos = aligned[anchor_frame_index, 4:7]

    yaw_delta = wrap_angle(quaternion_to_yaw(target_root_quat) - quaternion_to_yaw(source_root_quat))
    yaw_quaternion = quaternion_from_yaw(float(yaw_delta))
    rotation = yaw_rotation_matrix(float(yaw_delta))
    translation = np.asarray(target_root_pos, dtype=np.float64) - rotation @ source_root_pos

    aligned[:, :4] = quaternion_multiply(yaw_quaternion[None, :], aligned[:, :4])
    aligned[:, 4:7] = aligned[:, 4:7] @ rotation.T + translation
    return aligned


def inertialization_offset(delta0: np.ndarray, time_seconds: float, damping: float) -> np.ndarray:
    delta0 = np.asarray(delta0, dtype=np.float64)
    decay = math.exp(-damping * time_seconds)
    return decay * (delta0 + damping * time_seconds * delta0)


def apply_inertialization_to_qpos_sequence(
    aligned_qpos: np.ndarray,
    previous_qpos: np.ndarray | None,
    dt: float,
    damping: float,
    preserve_root_pose: bool = False,
) -> np.ndarray:
    if previous_qpos is None:
        return np.asarray(aligned_qpos, dtype=np.float64)

    corrected = np.asarray(aligned_qpos, dtype=np.float64).copy()
    delta_joint = previous_qpos[7:] - corrected[0, 7:]
    if preserve_root_pose:
        delta_root_pos = np.zeros(3, dtype=np.float64)
        delta_root_rot = np.zeros(3, dtype=np.float64)
    else:
        delta_root_pos = previous_qpos[4:7] - corrected[0, 4:7]
        delta_root_rot = quaternion_to_rotation_vector(
            quaternion_multiply(previous_qpos[:4], quaternion_conjugate(corrected[0, :4]))
        )

    for frame_index in range(corrected.shape[0]):
        time_seconds = frame_index * dt
        corrected[frame_index, 7:] += inertialization_offset(delta_joint, time_seconds, damping)
        if not preserve_root_pose:
            corrected[frame_index, 4:7] += inertialization_offset(delta_root_pos, time_seconds, damping)
            correction_rotvec = inertialization_offset(delta_root_rot, time_seconds, damping)
            correction_quat = rotation_vector_to_quaternion(correction_rotvec[None, :])[0]
            corrected[frame_index, :4] = quaternion_multiply(correction_quat, corrected[frame_index, :4])
            corrected[frame_index, :4] = normalize_quaternion(corrected[frame_index, :4])
    return corrected


def compose_transition(
    source_qpos: np.ndarray,
    previous_qpos: np.ndarray | None,
    dt: float,
    damping: float,
) -> np.ndarray:
    if len(source_qpos) == 0:
        return np.zeros((0, 36), dtype=np.float64)
    if previous_qpos is None:
        return np.asarray(source_qpos, dtype=np.float64)

    aligned = yaw_align_qpos_sequence(source_qpos, previous_qpos[:4], previous_qpos[4:7])
    return apply_inertialization_to_qpos_sequence(aligned, previous_qpos, dt, damping)


class KinematicsHelper:
    def __init__(self, robot_model_path: str | Path = "data/models/g1/g1_29dof.urdf") -> None:
        from pydrake.all import (  # pylint: disable=import-outside-toplevel
            AddMultibodyPlantSceneGraph,
            BodyIndex,
            DiagramBuilder,
            MultibodyPlant,
            Parser,
            RigidTransform,
        )

        builder = DiagramBuilder()
        plant = MultibodyPlant(1e-3)
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, plant=plant)
        parser = Parser(plant=plant, scene_graph=scene_graph)
        parser.AddModels(str(resolve_repo_path(robot_model_path)))
        parser.AddModels(str(resolve_repo_path("data/models/ground_box.sdf")))
        plant.WeldFrames(
            plant.world_frame(),
            plant.GetFrameByName("ground_link"),
            RigidTransform(np.asarray([0.0, 0.0, -0.5], dtype=np.float64)),
        )
        plant.Finalize()

        self._body_index_ctor = BodyIndex
        self.plant = plant
        self.context = plant.CreateDefaultContext()
        self.canonical_body_indices = [plant.GetBodyByName(name).index() for name in CANONICAL_BODY_NAMES]

    def canonical_body_poses(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.plant.SetPositions(self.context, np.asarray(qpos, dtype=np.float64))
        positions = np.zeros((len(self.canonical_body_indices), 3), dtype=np.float64)
        quaternions = np.zeros((len(self.canonical_body_indices), 4), dtype=np.float64)
        for index, body_index in enumerate(self.canonical_body_indices):
            body = self.plant.get_body(body_index)
            pose = self.plant.EvalBodyPoseInWorld(self.context, body)
            positions[index] = pose.translation()
            rotation = pose.rotation().ToQuaternion()
            quaternions[index] = np.asarray([rotation.w(), rotation.x(), rotation.y(), rotation.z()], dtype=np.float64)
        return positions, normalize_quaternion(quaternions)

    def runtime_pose_features(self, current_qpos: np.ndarray, previous_qpos: np.ndarray | None, dt: float) -> dict[str, np.ndarray]:
        current_positions, current_quaternions = self.canonical_body_poses(current_qpos)
        if previous_qpos is None:
            previous_positions = current_positions.copy()
        else:
            previous_positions, _ = self.canonical_body_poses(previous_qpos)

        current_root_quat = current_quaternions[ROOT_BODY_INDEX]
        current_root_pos = current_positions[ROOT_BODY_INDEX]
        current_root_yaw = quaternion_to_yaw(current_root_quat)
        body_linear_velocity = (current_positions - previous_positions) / max(dt, 1e-6)

        left_foot_index = BODY_NAME_TO_INDEX[LEFT_FOOT_BODY_NAME]
        right_foot_index = BODY_NAME_TO_INDEX[RIGHT_FOOT_BODY_NAME]
        left_foot_local_pos = world_to_local_vector(current_positions[left_foot_index] - current_root_pos, current_root_yaw)
        right_foot_local_pos = world_to_local_vector(current_positions[right_foot_index] - current_root_pos, current_root_yaw)
        left_foot_local_vel = world_to_local_vector(body_linear_velocity[left_foot_index], current_root_yaw)
        right_foot_local_vel = world_to_local_vector(body_linear_velocity[right_foot_index], current_root_yaw)
        root_local_velocity = world_to_local_vector(body_linear_velocity[ROOT_BODY_INDEX], current_root_yaw)
        return {
            "root_pos_w": current_root_pos,
            "root_quat_w": current_root_quat,
            "left_foot_local_pos": left_foot_local_pos,
            "right_foot_local_pos": right_foot_local_pos,
            "left_foot_local_vel": left_foot_local_vel,
            "right_foot_local_vel": right_foot_local_vel,
            "root_local_velocity": root_local_velocity,
        }


def export_qpos_trajectory_to_beyond_mimic(
    qpos_trajectory: np.ndarray,
    fps: int,
    output_path: str | Path,
    *,
    kinematics: KinematicsHelper | None = None,
) -> dict[str, np.ndarray]:
    if kinematics is None:
        kinematics = KinematicsHelper()

    qpos = np.asarray(qpos_trajectory, dtype=np.float64)
    dt = 1.0 / float(fps)
    body_pos_w = np.zeros((len(qpos), len(CANONICAL_BODY_NAMES), 3), dtype=np.float64)
    body_quat_w = np.zeros((len(qpos), len(CANONICAL_BODY_NAMES), 4), dtype=np.float64)

    for frame_index, frame_qpos in enumerate(qpos):
        positions, quaternions = kinematics.canonical_body_poses(frame_qpos)
        body_pos_w[frame_index] = positions
        body_quat_w[frame_index] = quaternions

    joint_pos = qpos[:, 7:]
    joint_vel = centered_finite_difference(joint_pos, dt)
    body_lin_vel_w = centered_finite_difference(body_pos_w, dt)

    body_ang_vel_w = np.zeros((len(qpos), len(CANONICAL_BODY_NAMES), 3), dtype=np.float64)
    for body_index in range(len(CANONICAL_BODY_NAMES)):
        body_ang_vel_w[:, body_index] = quaternion_velocity_series(body_quat_w[:, body_index], dt)

    payload = {
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
        "body_pos_w": body_pos_w.astype(np.float32),
        "body_quat_w": body_quat_w.astype(np.float32),
        "body_lin_vel_w": body_lin_vel_w.astype(np.float32),
        "body_ang_vel_w": body_ang_vel_w.astype(np.float32),
        "fps": np.asarray([fps], dtype=np.int64),
    }
    output_path = resolve_repo_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return payload


def sample_body_mapping_report(
    clip: MotionClip,
    *,
    sample_frames: int = 5,
    kinematics: KinematicsHelper | None = None,
) -> dict[str, Any]:
    if kinematics is None:
        kinematics = KinematicsHelper()

    if sample_frames <= 1:
        frame_indices = np.asarray([0], dtype=np.int32)
    else:
        frame_indices = np.linspace(0, clip.num_frames - 1, num=min(sample_frames, clip.num_frames), dtype=np.int32)

    canonical_positions = np.zeros((len(frame_indices), len(CANONICAL_BODY_NAMES), 3), dtype=np.float64)
    canonical_quaternions = np.zeros((len(frame_indices), len(CANONICAL_BODY_NAMES), 4), dtype=np.float64)
    for sample_index, frame_index in enumerate(frame_indices):
        positions, quaternions = kinematics.canonical_body_poses(clip.qpos[frame_index])
        canonical_positions[sample_index] = positions
        canonical_quaternions[sample_index] = quaternions

    sampled_body_positions = clip.body_pos_w[frame_indices]
    sampled_body_quaternions = clip.body_quat_w[frame_indices]
    pairwise_position_error = np.linalg.norm(
        sampled_body_positions[:, :, None, :] - canonical_positions[:, None, :, :],
        axis=-1,
    ).mean(axis=0)
    relative_quaternions = quaternion_multiply(
        sampled_body_quaternions[:, :, None, :],
        quaternion_conjugate(canonical_quaternions[:, None, :, :]),
    )
    flattened_relative_quaternions = relative_quaternions.reshape(-1, 4)
    flattened_orientation_error = np.linalg.norm(quaternion_to_rotation_vector(flattened_relative_quaternions), axis=-1)
    pairwise_orientation_error = flattened_orientation_error.reshape(relative_quaternions.shape[:-1]).mean(axis=0)
    pairwise_score = pairwise_position_error + 0.1 * pairwise_orientation_error

    entries: list[dict[str, Any]] = []
    sorted_indices = np.argsort(pairwise_score, axis=1)
    for body_index, body_name in enumerate(CANONICAL_BODY_NAMES):
        best_index = int(sorted_indices[body_index, 0])
        second_index = int(sorted_indices[body_index, 1]) if pairwise_score.shape[1] > 1 else best_index
        entries.append(
            {
                "data_body_index": body_index,
                "best_name": CANONICAL_BODY_NAMES[best_index],
                "best_error": float(pairwise_score[body_index, best_index]),
                "best_position_error": float(pairwise_position_error[body_index, best_index]),
                "best_orientation_error": float(pairwise_orientation_error[body_index, best_index]),
                "second_name": CANONICAL_BODY_NAMES[second_index],
                "second_error": float(pairwise_score[body_index, second_index]),
                "second_position_error": float(pairwise_position_error[body_index, second_index]),
                "second_orientation_error": float(pairwise_orientation_error[body_index, second_index]),
                "canonical_name": body_name,
            }
        )

    return {
        "clip_name": clip.name,
        "source_path": clip.source_path,
        "sample_frames": frame_indices.tolist(),
        "entries": entries,
    }


def clip_catalog_entry(clip: MotionClip, category: str, skill_name: str | None) -> dict[str, Any]:
    return {
        "name": clip.name,
        "source_name": clip.source_name,
        "path": clip.source_path,
        "category": category,
        "skill_name": skill_name,
        "mirrored": clip.mirrored,
        "fps": clip.fps,
        "num_frames": clip.num_frames,
    }


def sanitize_skill_name(skill_name: str) -> str:
    return skill_name.replace("/", "__")
