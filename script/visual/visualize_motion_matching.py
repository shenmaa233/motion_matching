from __future__ import annotations

import argparse
from pathlib import Path

from visualize_lafan import visualize_lafan


def main() -> None:
    parser = argparse.ArgumentParser(description="可视化 motion matching 生成的 BeyondMimic 轨迹。")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("output/motion_matching/generated_final"),
        help="生成结果目录，或单个 `.npz` 轨迹文件。",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="只播放文件名包含该子串的轨迹。",
    )
    args = parser.parse_args()
    visualize_lafan(base_path=str(args.input_path), filter=args.filter)
    input("Press Enter to quit...")

if __name__ == "__main__":
    main()
