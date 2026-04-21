"""
通用目标点定位命令。

读取场景 JSON 中的 targets，在裁剪缓存或源 DEM 中输出目标点像元坐标。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from article_planner.scenario_config import load_scenario_config, scenario_output_dir
from article_planner.target_locator import locate_targets, write_target_locations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据场景配置定位目标点像元坐标。")
    parser.add_argument("--scenario-config", type=str, default="scenarios/huashan.json", help="场景 JSON 路径。")
    parser.add_argument("--workdir", type=str, default=".", help="项目根目录。")
    parser.add_argument(
        "--source",
        choices=["auto", "crop", "dem"],
        default="auto",
        help="auto 优先使用裁剪缓存；crop 只使用裁剪缓存；dem 直接读取源 DEM。",
    )
    parser.add_argument("--search-radius-px", type=int, default=30, help="声明高程存在时的邻域搜索半径。")
    parser.add_argument("--output", type=str, default="", help="输出 JSON；为空时写入场景输出目录。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_scenario_config(args.scenario_config, args.workdir)
    payload = locate_targets(
        args.scenario_config,
        workdir=args.workdir,
        source=args.source,
        search_radius_px=args.search_radius_px,
    )

    output = Path(args.output) if args.output else scenario_output_dir(cfg, args.workdir) / "target_locations.json"
    if not output.is_absolute():
        output = Path(args.workdir).resolve() / output
    path = write_target_locations(payload, output)

    print(f"[完成] 目标定位结果: {path}")
    for item in payload["targets"]:
        err = item["elevation_error_m"]
        err_text = "无声明高程" if err is None else f"高程误差={err:.1f}m"
        print(
            f"  - {item['name']}: row={item['row']}, col={item['col']}, "
            f"lon={item['lon']:.6f}, lat={item['lat']:.6f}, elev={item['elevation_m']:.1f}m, {err_text}"
        )


if __name__ == "__main__":
    main()
