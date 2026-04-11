from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from motion_matching_common import load_motion_clip, resolve_repo_path


def suggest_skill_annotations(clip_path: Path) -> dict[str, int]:
    clip = load_motion_clip(clip_path)
    root_height = clip.body_pos_w[:, 0, 2]
    root_speed = np.linalg.norm(clip.body_lin_vel_w[:, 0, :2], axis=1)
    height_velocity = np.gradient(root_height)

    peak_frame = int(np.argmax(root_height))
    start_candidates = np.where((height_velocity > 0.01) & (root_speed > 0.8))[0]
    skill_start_frame = int(start_candidates[0]) if len(start_candidates) else max(0, peak_frame - 30)

    settle_candidates = np.where(
        (np.arange(clip.num_frames) > peak_frame)
        & (np.abs(root_height - root_height[-1]) < 0.03)
        & (root_speed < 0.35)
    )[0]
    skill_end_frame = int(settle_candidates[0]) if len(settle_candidates) else min(clip.num_frames - 1, peak_frame + 45)
    entry_window_length = 24

    return {
        "skill_start_frame": skill_start_frame,
        "skill_end_frame": skill_end_frame,
        "entry_window_length": entry_window_length,
        "peak_frame": peak_frame,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="根据轨迹统计给出 (s_k, e_k, H_k) 的保守候选值。")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/skills/climb_15_z_scale_1.0.npz"),
        help="技能轨迹 `.npz` 文件。",
    )
    args = parser.parse_args()

    clip = load_motion_clip(args.input)
    suggestion = suggest_skill_annotations(args.input)
    print(f"clip: {clip.name}")
    print(f"frames: {clip.num_frames}")
    print(f"skill_start_frame: {suggestion['skill_start_frame']}")
    print(f"skill_end_frame: {suggestion['skill_end_frame']}")
    print(f"entry_window_length: {suggestion['entry_window_length']}")
    print(f"peak_frame: {suggestion['peak_frame']}")


if __name__ == "__main__":
    main()
