from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from motion_matching_common import (
    CANONICAL_BODY_NAMES,
    CANONICAL_JOINT_NAMES,
    MotionClip,
    build_clip_feature_matrix,
    clip_catalog_entry,
    fit_feature_normalization,
    json_dump,
    load_motion_clip,
    load_skill_catalog,
    mirror_motion_clip,
    normalize_features,
    resolve_repo_path,
    sample_body_mapping_report,
    skill_catalog_to_json,
)


def _sorted_motion_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.glob("*.npz") if path.is_file())


def _append_clip_to_database(
    *,
    clip: MotionClip,
    category: str,
    skill_name: str | None,
    feature_blocks: list[np.ndarray],
    clip_index_blocks: list[np.ndarray],
    frame_index_blocks: list[np.ndarray],
    clip_catalog: list[dict],
    locomotion_indices: list[int],
    skill_frame_indices: dict[str, list[int]],
    skill_entry_indices: dict[str, list[int]],
    skill_start_frame: int | None,
    skill_end_frame: int | None,
    entry_window_length: int | None,
) -> None:
    database_offset = sum(block.shape[0] for block in feature_blocks)
    features = build_clip_feature_matrix(clip).astype(np.float32)
    feature_blocks.append(features)
    clip_catalog.append(clip_catalog_entry(clip, category, skill_name))

    clip_id = len(clip_catalog) - 1
    clip_index_blocks.append(np.full((clip.num_frames,), clip_id, dtype=np.int32))
    frame_index_blocks.append(np.arange(clip.num_frames, dtype=np.int32))

    database_rows = list(range(database_offset, database_offset + clip.num_frames))
    if category == "locomotion":
        locomotion_indices.extend(database_rows)
        return

    if skill_name is None or skill_start_frame is None or skill_end_frame is None or entry_window_length is None:
        raise ValueError("技能 clip 缺少必要的标注元数据")

    skill_frame_indices.setdefault(skill_name, []).extend(database_rows)
    entry_start = max(0, skill_start_frame - entry_window_length)
    entry_end = min(skill_start_frame, clip.num_frames - 1)
    if entry_end < entry_start:
        raise ValueError(f"{clip.name} 的 entry window 无效: [{entry_start}, {entry_end}]")
    skill_entry_rows = database_rows[entry_start : entry_end + 1]
    skill_entry_indices.setdefault(skill_name, []).extend(skill_entry_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建单技能 motion matching 检索数据库。")
    parser.add_argument(
        "--locomotion-dir",
        type=Path,
        default=Path("data/locomotion"),
        help="locomotion 片段目录。",
    )
    parser.add_argument(
        "--skill-catalog",
        type=Path,
        default=Path("data/motion_matching/single_skill_catalog.json"),
        help="技能标注清单 JSON。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/motion_matching/single_skill_db"),
        help="数据库输出目录。",
    )
    parser.add_argument(
        "--disable-mirror",
        action="store_true",
        help="关闭左右镜像扩增。",
    )
    args = parser.parse_args()

    locomotion_dir = resolve_repo_path(args.locomotion_dir)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    locomotion_paths = _sorted_motion_files(locomotion_dir)
    if not locomotion_paths:
        raise FileNotFoundError(f"{locomotion_dir} 下没有找到 locomotion `.npz` 文件")

    skills = load_skill_catalog(args.skill_catalog)
    if not skills:
        raise ValueError("技能清单为空，无法构建单技能数据库")

    include_mirror = not args.disable_mirror
    clip_catalog: list[dict] = []
    feature_blocks: list[np.ndarray] = []
    clip_index_blocks: list[np.ndarray] = []
    frame_index_blocks: list[np.ndarray] = []
    locomotion_indices: list[int] = []
    skill_frame_indices: dict[str, list[int]] = {}
    skill_entry_indices: dict[str, list[int]] = {}
    mapping_reports: list[dict] = []

    locomotion_clips = [load_motion_clip(path) for path in locomotion_paths]
    if locomotion_clips:
        mapping_reports.append(sample_body_mapping_report(locomotion_clips[0]))

    for clip in locomotion_clips:
        _append_clip_to_database(
            clip=clip,
            category="locomotion",
            skill_name=None,
            feature_blocks=feature_blocks,
            clip_index_blocks=clip_index_blocks,
            frame_index_blocks=frame_index_blocks,
            clip_catalog=clip_catalog,
            locomotion_indices=locomotion_indices,
            skill_frame_indices=skill_frame_indices,
            skill_entry_indices=skill_entry_indices,
            skill_start_frame=None,
            skill_end_frame=None,
            entry_window_length=None,
        )
        if include_mirror:
            mirrored = mirror_motion_clip(clip)
            _append_clip_to_database(
                clip=mirrored,
                category="locomotion",
                skill_name=None,
                feature_blocks=feature_blocks,
                clip_index_blocks=clip_index_blocks,
                frame_index_blocks=frame_index_blocks,
                clip_catalog=clip_catalog,
                locomotion_indices=locomotion_indices,
                skill_frame_indices=skill_frame_indices,
                skill_entry_indices=skill_entry_indices,
                skill_start_frame=None,
                skill_end_frame=None,
                entry_window_length=None,
            )

    for skill in skills:
        clip = load_motion_clip(skill.skill_path, clip_name=skill.skill_name)
        mapping_reports.append(sample_body_mapping_report(clip))
        _append_clip_to_database(
            clip=clip,
            category="skill",
            skill_name=skill.skill_name,
            feature_blocks=feature_blocks,
            clip_index_blocks=clip_index_blocks,
            frame_index_blocks=frame_index_blocks,
            clip_catalog=clip_catalog,
            locomotion_indices=locomotion_indices,
            skill_frame_indices=skill_frame_indices,
            skill_entry_indices=skill_entry_indices,
            skill_start_frame=skill.skill_start_frame,
            skill_end_frame=skill.skill_end_frame,
            entry_window_length=skill.entry_window_length,
        )
        if include_mirror:
            mirrored = mirror_motion_clip(clip)
            _append_clip_to_database(
                clip=mirrored,
                category="skill",
                skill_name=skill.skill_name,
                feature_blocks=feature_blocks,
                clip_index_blocks=clip_index_blocks,
                frame_index_blocks=frame_index_blocks,
                clip_catalog=clip_catalog,
                locomotion_indices=locomotion_indices,
                skill_frame_indices=skill_frame_indices,
                skill_entry_indices=skill_entry_indices,
                skill_start_frame=skill.skill_start_frame,
                skill_end_frame=skill.skill_end_frame,
                entry_window_length=skill.entry_window_length,
            )

    features = np.concatenate(feature_blocks, axis=0).astype(np.float32)
    clip_indices = np.concatenate(clip_index_blocks, axis=0).astype(np.int32)
    frame_indices = np.concatenate(frame_index_blocks, axis=0).astype(np.int32)
    locomotion_database_indices = np.asarray(locomotion_indices, dtype=np.int32)
    feature_mean, feature_std = fit_feature_normalization(features)
    normalized_features = normalize_features(features, feature_mean, feature_std).astype(np.float32)

    database_npz_path = output_dir / "motion_matching_db.npz"
    np.savez_compressed(
        database_npz_path,
        features=features,
        normalized_features=normalized_features,
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        clip_indices=clip_indices,
        frame_indices=frame_indices,
        locomotion_database_indices=locomotion_database_indices,
    )

    manifest = {
        "version": 1,
        "database_npz": str(database_npz_path.relative_to(resolve_repo_path("."))),
        "include_mirror": include_mirror,
        "body_names": list(CANONICAL_BODY_NAMES),
        "joint_names": list(CANONICAL_JOINT_NAMES),
        "clips": clip_catalog,
        "skills": skill_catalog_to_json(skills)["skills"],
        "skill_frame_indices": {name: indices for name, indices in skill_frame_indices.items()},
        "skill_entry_indices": {name: indices for name, indices in skill_entry_indices.items()},
        "body_mapping_reports": mapping_reports,
    }
    manifest_path = output_dir / "motion_matching_db.json"
    json_dump(manifest_path, manifest)

    print(f"[ok] 数据库已写出到 {database_npz_path}")
    print(f"[ok] 清单已写出到 {manifest_path}")
    print(f"[info] 总帧数: {features.shape[0]}")
    print(f"[info] locomotion 候选数: {len(locomotion_database_indices)}")
    for skill_name, indices in skill_entry_indices.items():
        print(f"[info] {skill_name} 的 entry window 候选数: {len(indices)}")


if __name__ == "__main__":
    main()
