"""
华山旧裁剪中心诊断工具。

该脚本保留历史五峰像元中心的诊断逻辑，避免 `init_graph.py` 主流程继续携带华山专用常量。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from article_planner.scenario_config import load_scenario_config, resolve_path
from init_graph import EPSG_SRC, EPSG_WGS84, pixel_to_xy, read_tiff_with_georef, xy_to_pixel


LEGACY_PEAK_PIXELS = {
    "南峰": {"row": 4609, "col": 1938},
    "东峰": {"row": 4642, "col": 1985},
    "西峰": {"row": 4600, "col": 1949},
    "北峰": {"row": 4468, "col": 2004},
    "中峰": {"row": 4594, "col": 1951},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="诊断华山历史五峰像元中心与当前场景裁剪中心的偏移。")
    parser.add_argument("--scenario-config", type=str, default="scenarios/huashan.json")
    parser.add_argument("--workdir", type=str, default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.workdir).resolve()
    cfg = load_scenario_config(args.scenario_config, root)
    crop_cfg = cfg["crop"]
    center_lon = float(crop_cfg["center_lon"])
    center_lat = float(crop_cfg["center_lat"])

    tif_path = resolve_path(str(cfg["dem_path"]), root)
    _, x0, y0, sx, sy = read_tiff_with_georef(tif_path)
    tf_to_wgs = Transformer.from_crs(EPSG_SRC, EPSG_WGS84, always_xy=True)
    tf_to_utm = Transformer.from_crs(EPSG_WGS84, EPSG_SRC, always_xy=True)

    legacy_row = int(np.mean([v["row"] for v in LEGACY_PEAK_PIXELS.values()]))
    legacy_col = int(np.mean([v["col"] for v in LEGACY_PEAK_PIXELS.values()]))
    x_old, y_old = pixel_to_xy(legacy_row, legacy_col, x0, y0, sx, sy)
    lon_old, lat_old = tf_to_wgs.transform(x_old, y_old)

    x_new, y_new = tf_to_utm.transform(center_lon, center_lat)
    center_row, center_col = xy_to_pixel(x_new, y_new, x0, y0, sx, sy)
    x_new_chk, y_new_chk = pixel_to_xy(center_row, center_col, x0, y0, sx, sy)
    lon_new_chk, lat_new_chk = tf_to_wgs.transform(x_new_chk, y_new_chk)

    print(f"[旧中心像元] row={legacy_row}, col={legacy_col}")
    print(f"[旧中心经纬度] lon={lon_old:.6f}, lat={lat_old:.6f}")
    print(f"[当前裁剪中心] lon={center_lon:.6f}, lat={center_lat:.6f}")
    print(f"[当前中心像元] row={center_row}, col={center_col}")
    print(f"[吸附后经纬度] lon={lon_new_chk:.6f}, lat={lat_new_chk:.6f}")
    print(f"[相对旧中心偏移] dlon={lon_new_chk - lon_old:+.6f}, dlat={lat_new_chk - lat_old:+.6f}")


if __name__ == "__main__":
    main()
