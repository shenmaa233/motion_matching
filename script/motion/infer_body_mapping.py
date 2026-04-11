from __future__ import annotations

import argparse
from pathlib import Path

from motion_matching_common import json_dump, load_motion_clip, sample_body_mapping_report


def main() -> None:
    parser = argparse.ArgumentParser(description="输出当前 `.npz` 与 G1 URDF 的 body 对齐报告。")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/skills/climb_15_z_scale_1.0.npz"),
        help="要检查的轨迹 `.npz` 文件。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/motion_matching/body_mapping_report.json"),
        help="报告输出路径。",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=5,
        help="采样多少帧来计算对齐误差。",
    )
    args = parser.parse_args()

    clip = load_motion_clip(args.input)
    report = sample_body_mapping_report(clip, sample_frames=args.sample_frames)
    json_dump(args.output, report)
    print(f"[ok] body 对齐报告已写出到 {args.output}")
    for entry in report["entries"]:
        print(
            f"body[{entry['data_body_index']:02d}] -> {entry['best_name']:<24s} "
            f"best={entry['best_error']:.6f} second={entry['second_name']:<24s} "
            f"second_err={entry['second_error']:.6f}"
        )


if __name__ == "__main__":
    main()
