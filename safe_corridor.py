"""
Build flyable corridor bounds and three layered flight decks from DEM.

Inputs:
    Z_crop.npy

Outputs:
    floor.npy
    ceiling.npy
    layer_mid.npy
    corridor_vis.png
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


CACHE_DEM = "Z_crop.npy"
RESOLUTION = 12.5  # m/pixel

H_MIN_OFFSET = 30.0
H_MAX_OFFSET = 120.0

LAYERS = [
    {"name": "Terminal Approach Layer", "low": 30.0, "high": 60.0, "color": "#2196F3", "cmap": "Blues"},
    {"name": "Regional Branch Layer", "low": 60.0, "high": 90.0, "color": "#4CAF50", "cmap": "Greens"},
    {"name": "Backbone Layer", "low": 90.0, "high": 120.0, "color": "#FF5722", "cmap": "Oranges"},
]


def main() -> None:
    if not os.path.exists(CACHE_DEM):
        raise FileNotFoundError(f"Missing {CACHE_DEM}. Run init_graph.py first.")

    z = np.asarray(np.load(CACHE_DEM), dtype=float)
    rows, cols = z.shape
    print(f"[load] DEM shape={z.shape}, elevation={np.nanmin(z):.0f}~{np.nanmax(z):.0f} m")

    floor = z + H_MIN_OFFSET
    ceiling = z + H_MAX_OFFSET
    np.save("floor.npy", floor.astype(np.float32))
    np.save("ceiling.npy", ceiling.astype(np.float32))
    print(f"[step2] floor={np.min(floor):.0f}~{np.max(floor):.0f} m")
    print(f"[step2] ceiling={np.min(ceiling):.0f}~{np.max(ceiling):.0f} m")

    layer_mid = []
    for cfg in LAYERS:
        mid = z + 0.5 * (cfg["low"] + cfg["high"])
        layer_mid.append(mid)
        print(
            f"[step3] {cfg['name']}: +{cfg['low']:.0f}~+{cfg['high']:.0f} m AGL, "
            f"abs={np.min(mid):.0f}~{np.max(mid):.0f} m"
        )
    layer_mid_arr = np.stack(layer_mid, axis=0).astype(np.float32)
    np.save("layer_mid.npy", layer_mid_arr)
    print(f"[step3] layer_mid.npy saved, shape={layer_mid_arr.shape}")

    x_km = np.arange(cols) * RESOLUTION / 1000.0
    y_km = np.arange(rows) * RESOLUTION / 1000.0
    extent = [0.0, cols * RESOLUTION / 1000.0, 0.0, rows * RESOLUTION / 1000.0]

    fig = plt.figure(figsize=(22, 12), dpi=160)
    fig.suptitle("Huashan Flyable Corridor and Layered Decks", fontsize=15, y=0.98)

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

    # 2) Corridor thickness
    ax2 = fig.add_subplot(2, 3, 2)
    thickness = ceiling - floor
    im2 = ax2.imshow(thickness, cmap="YlOrRd", extent=extent, origin="lower", aspect="equal")
    plt.colorbar(im2, ax=ax2, label="Corridor Thickness (m)", shrink=0.82)
    ax2.set_xlabel("East-West (km)")
    ax2.set_ylabel("South-North (km)")
    ax2.set_title(f"Corridor Thickness (constant {H_MAX_OFFSET - H_MIN_OFFSET:.0f} m)")
    ax2.grid(True, alpha=0.25, ls="--")

    # 3) 3D corridor
    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    step = 10
    z_s = z[::step, ::step]
    f_s = floor[::step, ::step]
    c_s = ceiling[::step, ::step]
    rs, cs = z_s.shape
    xg = np.arange(cs) * step * RESOLUTION / 1000.0
    yg = np.arange(rs) * step * RESOLUTION / 1000.0
    xg, yg = np.meshgrid(xg, yg)
    ax3.plot_surface(xg, yg, z_s, color="#8B4513", alpha=0.68, linewidth=0)
    ax3.plot_surface(xg, yg, f_s, color="#2196F3", alpha=0.20, linewidth=0)
    ax3.plot_surface(xg, yg, c_s, color="#FF5722", alpha=0.14, linewidth=0)
    ax3.view_init(elev=25, azim=225)
    ax3.set_xlabel("East-West (km)", labelpad=6)
    ax3.set_ylabel("South-North (km)", labelpad=6)
    ax3.set_zlabel("Elevation (m)", labelpad=6)
    ax3.set_title("3D Corridor Envelope")

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

    plt.tight_layout()
    plt.savefig("corridor_vis.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("[done] corridor_vis.png saved")
    print("[done] outputs: floor.npy / ceiling.npy / layer_mid.npy")
    print("[next] run layered_graph.py")


if __name__ == "__main__":
    main()
