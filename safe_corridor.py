"""
根据场景 DEM、人群风险和终端目标构建安全走廊与三层飞行高度。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

from article_planner.scenario_config import (
    adaptive_corridor_params,
    depot_params,
    display_names,
    load_scenario_config,
    resolve_resolution_m,
    scenario_output_dir,
    target_specs,
)
from virtual_depots import generate_virtual_depots


CACHE_DEM = "Z_crop.npy"

H_MIN_OFFSET = 30.0
H_MAX_OFFSET = 120.0

LAYERS = [
    {"name": "Terminal Approach Layer", "low": 30.0, "high": 60.0, "color": "#2196F3", "cmap": "Blues"},
    {"name": "Regional Branch Layer", "low": 60.0, "high": 90.0, "color": "#4CAF50", "cmap": "Greens"},
    {"name": "Backbone Layer", "low": 90.0, "high": 120.0, "color": "#FF5722", "cmap": "Oranges"},
]

def load_peak_positions(
    rows: int,
    cols: int,
    z: np.ndarray,
    geo_path: Path,
    targets: dict,
    names: dict,
    resolution_m: float,
) -> dict:
    if not geo_path.exists():
        return {}
    geo = np.load(geo_path)
    lon_grid = np.asarray(geo["lon_grid"], dtype=float)
    lat_grid = np.asarray(geo["lat_grid"], dtype=float)
    out = {}
    for name, p in targets.items():
        lon = float(p["lon"])
        lat = float(p["lat"])
        d2 = (lon_grid - lon) ** 2 + (lat_grid - lat) ** 2
        idx = int(np.argmin(d2))
        r, c = np.unravel_index(idx, lon_grid.shape)
        x_km = c * resolution_m / 1000.0
        y_km = (rows - 1 - r) * resolution_m / 1000.0
        label = names.get(name, name)
        out[label] = {
            "x_km": float(x_km),
            "y_km": float(y_km),
            "z_m": float(z[r, c]),
            "row": int(r),
            "col": int(c),
        }
    return out


def _normalise(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=float)
    lo = float(np.nanmin(values[finite]))
    hi = float(np.nanmax(values[finite]))
    if hi <= lo + 1e-12:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def nearest_rc_from_lonlat(lon_grid: np.ndarray, lat_grid: np.ndarray, lon: float, lat: float) -> tuple[int, int]:
    d2 = (lon_grid - lon) ** 2 + (lat_grid - lat) ** 2
    idx = int(np.argmin(d2))
    r, c = np.unravel_index(idx, lon_grid.shape)
    return int(r), int(c)


def terminal_distance_km(rows: int, cols: int, terminal_rcs: list[tuple[int, int]], resolution_m: float) -> np.ndarray:
    rr, cc = np.indices((rows, cols))
    out = np.full((rows, cols), np.inf, dtype=float)
    for tr, tc in terminal_rcs:
        d = np.sqrt((rr - tr) ** 2 + (cc - tc) ** 2) * float(resolution_m) / 1000.0
        out = np.minimum(out, d)
    return out


def build_adaptive_corridor(
    z: np.ndarray,
    risk_human: np.ndarray,
    terminal_rcs: list[tuple[int, int]],
    params: dict,
    resolution_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    rows, cols = z.shape
    zf = np.asarray(z, dtype=float)
    z_smooth = gaussian_filter(zf, sigma=3)
    gy, gx = np.gradient(z_smooth, float(resolution_m), float(resolution_m))
    slope_deg = np.degrees(np.arctan(np.sqrt(gx * gx + gy * gy)))
    local_max = maximum_filter(z_smooth, size=21)
    local_min = minimum_filter(z_smooth, size=21)
    relief = np.maximum(local_max - local_min, 1e-6)
    ridge_score = np.clip((z_smooth - local_min) / relief, 0.0, 1.0)
    open_score = (1.0 - _normalise(slope_deg)) * (1.0 - risk_human)

    base_floor = float(params.get("base_floor_offset_m", H_MIN_OFFSET))
    base_ceiling = float(params.get("base_ceiling_offset_m", H_MAX_OFFSET))
    slope_threshold = float(params.get("slope_threshold_deg", 18.0))
    slope_high = float(params.get("slope_high_deg", 42.0))
    slope_extra = float(params.get("slope_floor_extra_m", 35.0))
    ridge_extra = float(params.get("ridge_floor_extra_m", 20.0))
    open_extra = float(params.get("open_ceiling_extra_m", 35.0))
    terminal_radius = float(params.get("terminal_radius_km", 0.75))
    terminal_thickness = float(params.get("terminal_thickness_m", 72.0))
    min_thickness = float(params.get("min_thickness_m", 60.0))
    high_risk_threshold = float(params.get("high_risk_threshold", 0.65))
    layer_positions = [float(v) for v in params.get("layer_positions", [0.22, 0.52, 0.82])]
    if len(layer_positions) != 3:
        layer_positions = [0.22, 0.52, 0.82]

    slope_factor = np.clip((slope_deg - slope_threshold) / max(slope_high - slope_threshold, 1e-6), 0.0, 1.0)
    floor_offset = base_floor + slope_extra * slope_factor + ridge_extra * np.clip(ridge_score - 0.55, 0.0, 1.0) / 0.45
    ceiling_offset = base_ceiling + open_extra * np.clip(open_score, 0.0, 1.0)

    d_terminal = terminal_distance_km(rows, cols, terminal_rcs, resolution_m)
    terminal_mask = d_terminal <= terminal_radius
    ceiling_offset = np.where(terminal_mask, np.minimum(ceiling_offset, floor_offset + terminal_thickness), ceiling_offset)
    ceiling_offset = np.maximum(ceiling_offset, floor_offset + min_thickness)

    floor = zf + floor_offset
    ceiling = zf + ceiling_offset
    thickness = ceiling - floor
    layer_mid_arr = np.stack([floor + pos * thickness for pos in layer_positions], axis=0).astype(np.float32)

    layer_allowed = np.ones((3, rows, cols), dtype=bool)
    high_risk = risk_human >= high_risk_threshold
    # 高人群风险区限制低层和支路层，骨干层保留高净空越障能力。
    layer_allowed[0, high_risk & (~terminal_mask)] = False
    layer_allowed[1, high_risk & (~terminal_mask)] = False

    meta = {
        "base_floor_offset_m": base_floor,
        "base_ceiling_offset_m": base_ceiling,
        "floor_offset_min_m": float(np.nanmin(floor_offset)),
        "floor_offset_max_m": float(np.nanmax(floor_offset)),
        "ceiling_offset_min_m": float(np.nanmin(ceiling_offset)),
        "ceiling_offset_max_m": float(np.nanmax(ceiling_offset)),
        "thickness_min_m": float(np.nanmin(thickness)),
        "thickness_max_m": float(np.nanmax(thickness)),
        "terminal_area_ratio": float(np.mean(terminal_mask)),
        "high_risk_area_ratio": float(np.mean(high_risk)),
        "layer_allowed_ratio": [float(np.mean(layer_allowed[i])) for i in range(3)],
        "layer_positions": layer_positions,
    }
    return floor.astype(np.float32), ceiling.astype(np.float32), layer_mid_arr, layer_allowed, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="根据 DEM 构建安全走廊和分层飞行高度。")
    parser.add_argument("--scenario-config", type=str, default="")
    parser.add_argument("--workdir", type=str, default=".")
    args = parser.parse_args()

    root = Path(args.workdir).resolve()
    scene_cfg = load_scenario_config(args.scenario_config or None, root)
    scene_name = str(scene_cfg.get("scene_name", "default"))
    out_dir = scenario_output_dir(scene_cfg, root)
    dem_path = out_dir / CACHE_DEM
    geo_path = out_dir / "Z_crop_geo.npz"

    if not dem_path.exists():
        raise FileNotFoundError(f"缺少 {dem_path}，请先运行 init_graph.py。")

    resolution_m = resolve_resolution_m(scene_cfg, out_dir)
    z = np.asarray(np.load(dem_path), dtype=float)
    rows, cols = z.shape
    print(f"[读取] 场景={scene_name}, DEM shape={z.shape}, 高程={np.nanmin(z):.0f}~{np.nanmax(z):.0f} m")

    risk_human = np.zeros((rows, cols), dtype=float)
    risk_path = out_dir / "risk_human.npy"
    if risk_path.exists():
        risk_arr = np.asarray(np.load(risk_path), dtype=float)
        if risk_arr.shape == z.shape:
            risk_human = np.clip(risk_arr, 0.0, 1.0)

    terminal_rcs: list[tuple[int, int]] = []
    if geo_path.exists():
        geo = np.load(geo_path)
        lon_grid = np.asarray(geo["lon_grid"], dtype=float)
        lat_grid = np.asarray(geo["lat_grid"], dtype=float)
        for p in target_specs(scene_cfg).values():
            terminal_rcs.append(nearest_rc_from_lonlat(lon_grid, lat_grid, float(p["lon"]), float(p["lat"])))
        depot_path = out_dir / "generated_depots.json"
        depots = []
        if not depot_path.exists():
            depots = generate_virtual_depots(
                z,
                lon_grid,
                lat_grid,
                target_specs(scene_cfg),
                depot_params(scene_cfg),
                risk_human=risk_human,
                resolution_m=resolution_m,
            )
            depot_payload = {
                "scene_name": scene_name,
                "rule": "低坡度、低海拔、低风险、远离目标点且位于边缘或山脚过渡区",
                "depots": depots,
            }
            depot_path.write_text(json.dumps(depot_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[配送站] 已预生成 {len(depots)} 个虚拟配送站，用于终端进近走廊收缩。")
        elif depot_path.exists():
            try:
                depots = json.loads(depot_path.read_text(encoding="utf-8")).get("depots", [])
            except Exception:
                depots = []
        for d in depots:
            if "row" in d and "col" in d:
                terminal_rcs.append((int(d["row"]), int(d["col"])))

    corridor_params = adaptive_corridor_params(scene_cfg)
    floor, ceiling, layer_mid_arr, layer_allowed, corridor_meta = build_adaptive_corridor(
        z,
        risk_human,
        terminal_rcs,
        corridor_params,
        resolution_m,
    )
    np.save(out_dir / "floor.npy", floor.astype(np.float32))
    np.save(out_dir / "ceiling.npy", ceiling.astype(np.float32))
    np.save(out_dir / "layer_allowed.npy", layer_allowed.astype(np.bool_))
    (out_dir / "corridor_meta.json").write_text(
        json.dumps(corridor_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[步骤2] 下边界={np.min(floor):.0f}~{np.max(floor):.0f} m")
    print(f"[步骤2] 上边界={np.min(ceiling):.0f}~{np.max(ceiling):.0f} m")

    for idx, layer_cfg in enumerate(LAYERS):
        mid = layer_mid_arr[idx]
        print(
            f"[步骤3] {layer_cfg['name']}: 自适应层面，"
            f"绝对高度={np.min(mid):.0f}~{np.max(mid):.0f} m"
        )
    np.save(out_dir / "layer_mid.npy", layer_mid_arr)
    print(f"[步骤3] layer_mid.npy 已保存, shape={layer_mid_arr.shape}")

    x_km = np.arange(cols) * resolution_m / 1000.0
    y_km = np.arange(rows) * resolution_m / 1000.0
    extent = [0.0, cols * resolution_m / 1000.0, 0.0, rows * resolution_m / 1000.0]
    targets = target_specs(scene_cfg)
    names = display_names(scene_cfg)
    peak_pos = load_peak_positions(rows, cols, z, geo_path, targets, names, resolution_m)

    fig = plt.figure(figsize=(22, 12), dpi=300)
    fig.suptitle(f"{scene_name} Flyable Corridor and Layered Decks", fontsize=15, y=0.98)

    # 1) East-west profile
    ax1 = fig.add_subplot(2, 3, 1)
    mid_row = rows // 2
    ax1.fill_between(x_km, z[mid_row], floor[mid_row], color="#8B4513", alpha=0.75, label="Terrain (No-Fly)")
    ax1.fill_between(x_km, floor[mid_row], ceiling[mid_row], color="#87CEEB", alpha=0.55, label="Flyable Corridor")
    for cfg in LAYERS:
        ax1.plot(
            x_km,
            z[mid_row] + cfg["high"],
            color=cfg["color"],
            lw=1.3,
            ls="--",
            label=f"{cfg['name']} upper bound (+{cfg['high']:.0f}m)",
        )
    ax1.set_xlabel("East-West (km)")
    ax1.set_ylabel("Absolute Elevation (m)")
    ax1.set_title("Corridor Cross-Section (Middle Row)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=7, loc="upper right")
    if peak_pos:
        y_top = float(np.max(ceiling[mid_row]))
        for name, p in peak_pos.items():
            xpk = float(p["x_km"])
            ax1.axvline(xpk, color="black", lw=0.8, alpha=0.35, ls=":")
            ax1.text(
                xpk,
                y_top + 10.0,
                name.replace(" Peak", ""),
                rotation=90,
                fontsize=6.8,
                ha="center",
                va="bottom",
                color="black",
            )

    # 2) Corridor thickness
    ax2 = fig.add_subplot(2, 3, 2)
    thickness = ceiling - floor
    im2 = ax2.imshow(thickness, cmap="YlOrRd", extent=extent, origin="lower", aspect="equal")
    plt.colorbar(im2, ax=ax2, label="Corridor Thickness (m)", shrink=0.82)
    ax2.set_xlabel("East-West (km)")
    ax2.set_ylabel("South-North (km)")
    ax2.set_title("Adaptive Corridor Thickness")
    ax2.grid(True, alpha=0.25, ls="--")
    if peak_pos:
        for name, p in peak_pos.items():
            ax2.scatter(
                [p["x_km"]],
                [p["y_km"]],
                s=34,
                c="#0D47A1",
                marker="*",
                edgecolors="white",
                linewidths=0.45,
                zorder=6,
            )
            ax2.text(
                p["x_km"] + 0.05,
                p["y_km"] + 0.05,
                name.replace(" Peak", ""),
                fontsize=6.8,
                color="#0D47A1",
                zorder=7,
            )

    # 3) 3D corridor
    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    step = 10
    z_s = z[::step, ::step]
    f_s = floor[::step, ::step]
    c_s = ceiling[::step, ::step]
    rs, cs = z_s.shape
    xg = np.arange(cs) * step * resolution_m / 1000.0
    yg = np.arange(rs) * step * resolution_m / 1000.0
    xg, yg = np.meshgrid(xg, yg)
    ax3.plot_surface(xg, yg, z_s, color="#8B4513", alpha=0.68, linewidth=0)
    ax3.plot_surface(xg, yg, f_s, color="#2196F3", alpha=0.20, linewidth=0)
    ax3.plot_surface(xg, yg, c_s, color="#FF5722", alpha=0.14, linewidth=0)
    ax3.view_init(elev=25, azim=225)
    ax3.set_xlabel("East-West (km)", labelpad=6)
    ax3.set_ylabel("South-North (km)", labelpad=6)
    ax3.set_zlabel("Elevation (m)", labelpad=6)
    ax3.set_title("3D Corridor Envelope")
    if peak_pos:
        for name, p in peak_pos.items():
            ax3.scatter(
                [p["x_km"]],
                [p["y_km"]],
                [p["z_m"] + 85.0],
                c="#0D47A1",
                marker="*",
                s=80,
                edgecolors="white",
                linewidths=0.6,
                zorder=8,
            )
            ax3.text(
                p["x_km"],
                p["y_km"],
                p["z_m"] + 115.0,
                name.replace(" Peak", ""),
                fontsize=6.2,
                color="black",
            )

    # 4-6) Layer mid-height maps
    for i, cfg in enumerate(LAYERS):
        ax = fig.add_subplot(2, 3, 4 + i)
        im = ax.imshow(layer_mid_arr[i], cmap=cfg["cmap"], extent=extent, origin="lower", aspect="equal")
        plt.colorbar(im, ax=ax, label="Absolute Elevation (m)", shrink=0.82)
        ax.set_xlabel("East-West (km)")
        ax.set_ylabel("South-North (km)")
        ax.set_title(
            f"Layer {i + 1}: {cfg['name']}\nAGL +{cfg['low']:.0f}~+{cfg['high']:.0f} m",
            fontsize=9,
        )
        ax.grid(True, alpha=0.25, ls="--")
        if peak_pos:
            for name, p in peak_pos.items():
                ax.scatter(
                    [p["x_km"]],
                    [p["y_km"]],
                    s=30,
                    c="#0D47A1",
                    marker="*",
                    edgecolors="white",
                    linewidths=0.42,
                    zorder=6,
                )
                ax.text(
                    p["x_km"] + 0.05,
                    p["y_km"] + 0.05,
                    name.replace(" Peak", ""),
                    fontsize=6.4,
                    color="#0D47A1",
                    zorder=7,
                )

    plt.tight_layout()
    corridor_png = out_dir / "corridor_vis.png"
    plt.savefig(corridor_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[完成] {corridor_png} 已保存")
    print(
        f"[完成] 输出: {out_dir / 'floor.npy'} / {out_dir / 'ceiling.npy'} / "
        f"{out_dir / 'layer_mid.npy'} / {out_dir / 'layer_allowed.npy'}"
    )
    print("[下一步] 运行 layered_graph.py")


if __name__ == "__main__":
    main()
