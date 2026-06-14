from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DPI = 600
OUT_STEM = "fig_1_overall_framework_proposed_method"
FINAL_DIR = PROJECT_ROOT / "final_results" / "paper_revision" / "figures"
INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate_artifacts" / "figures" / "huashan"
TERRAIN_IMAGE = FINAL_DIR / "fig_2_1a_huashan_pyvista_realistic.png"
GRAPH_IMAGE = FINAL_DIR / "fig_3_2_safe_corridor_layers.png"


COLORS = {
    "blue": "#2B74C7",
    "blue_dark": "#164C91",
    "terminal": "#2C7FB8",
    "regional": "#2CA25F",
    "backbone": "#E69F00",
    "terrain": "#7A5A48",
    "risk": "#D95F02",
    "flow": "#285FAD",
    "axis_band": "#F3F6F8",
    "axis_edge": "#D9E3EC",
    "support": "#E6EEF6",
    "support_edge": "#B8CEE5",
    "path": "#D95F02",
    "smooth": "#0072B2",
    "danger": "#D94841",
    "shadow": "#C7D1DA",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
            "savefig.dpi": DPI,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def add_text(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    size: float,
    weight: str = "normal",
    color: str = "#1F2933",
    ha: str = "center",
    va: str = "center",
    linespacing: float = 1.05,
    zorder: int = 6,
) -> None:
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        linespacing=linespacing,
        zorder=zorder,
    )


def project_iso(x0: float, y0: float, w: float, h: float, u: float, v: float, z: float = 0.0) -> tuple[float, float]:
    """将归一化三维坐标投影为等轴测二维坐标。"""
    px = x0 + w * (0.50 + 0.58 * (u - 0.5) + 0.31 * (v - 0.5))
    py = y0 + h * (0.36 + 0.24 * (v - 0.5) - 0.11 * (u - 0.5) + 0.43 * z)
    return px, py


def add_shadow(ax: plt.Axes, x: float, y: float, w: float, h: float, alpha: float = 0.14) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.006, y - 0.006),
            w,
            h,
            boxstyle="round,pad=0.003,rounding_size=0.014",
            facecolor=COLORS["shadow"],
            edgecolor="none",
            alpha=alpha,
            zorder=1,
        )
    )


def transparent_crop(path: Path, crop_box: tuple[int, int, int, int], threshold: int = 248) -> Image.Image | None:
    """裁剪已有高质量图像，并将白色背景转为透明。"""
    if not path.exists():
        return None
    img = Image.open(path).convert("RGBA").crop(crop_box)
    arr = np.asarray(img)
    mask = np.any(arr[:, :, :3] < threshold, axis=2)
    ys, xs = np.where(mask)
    if xs.size and ys.size:
        margin = 22
        left = max(int(xs.min()) - margin, 0)
        right = min(int(xs.max()) + margin, img.width)
        top = max(int(ys.min()) - margin, 0)
        bottom = min(int(ys.max()) + margin, img.height)
        img = img.crop((left, top, right, bottom))
    arr = np.array(img)
    white = np.all(arr[:, :, :3] > threshold, axis=2)
    arr[:, :, 3] = np.where(white, 0, arr[:, :, 3])
    return Image.fromarray(arr)


def draw_pipeline_band(ax: plt.Axes) -> None:
    x, y, w, h = 0.044, 0.370, 0.912, 0.482
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.010,rounding_size=0.030",
            facecolor=COLORS["axis_band"],
            edgecolor=COLORS["axis_edge"],
            lw=0.55,
            alpha=0.78,
            zorder=0,
        )
    )
    ax.plot([x + 0.018, x + w - 0.018], [y + h - 0.106, y + h - 0.106], color="#E0E7EE", lw=0.75, zorder=1)


def draw_step_title(ax: plt.Axes, cx: float, y: float, text: str) -> None:
    add_text(ax, cx, y, text, 6.8, weight="bold", color="#1F2933")


def draw_flow(ax: plt.Axes, x0: float, x1: float, y: float, label: str) -> None:
    arrow = FancyArrowPatch(
        (x0, y),
        (x1, y),
        arrowstyle="-|>",
        mutation_scale=10.5,
        lw=1.05,
        color=COLORS["flow"],
        shrinkA=1.0,
        shrinkB=1.0,
        zorder=6,
    )
    ax.add_patch(arrow)
    add_text(ax, (x0 + x1) / 2, y + 0.034, label, 6.0, color="#173E71")


def draw_input_model(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    terrain = transparent_crop(TERRAIN_IMAGE, (620, 305, 3060, 2070), threshold=248)
    add_shadow(ax, x + 0.012 * w, y + 0.026 * h, 0.84 * w, 0.72 * h, alpha=0.12)
    if terrain is not None:
        ax.imshow(
            terrain,
            extent=(x - 0.010 * w, x + 0.950 * w, y + 0.005 * h, y + 0.805 * h),
            interpolation="lanczos",
            zorder=3,
            aspect="auto",
        )
    else:
        base = np.array([project_iso(x, y, w, h, 0.05, 0.12, 0.0), project_iso(x, y, w, h, 0.95, 0.14, 0.0), project_iso(x, y, w, h, 0.88, 0.88, 0.0), project_iso(x, y, w, h, 0.12, 0.84, 0.0)])
        ax.add_patch(Polygon(base, closed=True, facecolor="#C8D3B8", edgecolor="#7B8B70", lw=0.65, zorder=3))
    add_text(ax, x + 0.50 * w, y + 0.005 * h, "Huashan DEM relief and geospatial inputs", 4.9, color="#425466")


def draw_risk_model(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    layers = [
        (0.05, "#A8C66C", "Terrain"),
        (0.40, "#F1B44C", "Human"),
        (0.75, "#7FA6D6", "LOS"),
    ]
    for z, color, label in layers:
        corners = np.array(
            [
                project_iso(x, y, w, h, 0.08, 0.10, z),
                project_iso(x, y, w, h, 0.92, 0.12, z),
                project_iso(x, y, w, h, 0.90, 0.86, z),
                project_iso(x, y, w, h, 0.12, 0.84, z),
            ]
        )
        thickness = corners.copy()
        thickness[:, 1] -= 0.016 * h
        ax.add_patch(Polygon(np.vstack([thickness[0], thickness[1], corners[1], corners[0]]), closed=True, facecolor="#67717B", edgecolor="none", alpha=0.16, zorder=2))
        ax.add_patch(Polygon(corners, closed=True, facecolor=color, edgecolor="#65727E", lw=0.55, alpha=0.70, zorder=3))
        for t in np.linspace(0.18, 0.82, 5):
            p0 = project_iso(x, y, w, h, 0.12, t, z)
            p1 = project_iso(x, y, w, h, 0.88, t + 0.020 * np.sin(8 * t), z)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="white", lw=0.42, alpha=0.62, zorder=4)
        add_text(ax, corners[2, 0] + 0.012, corners[2, 1], label, 4.8, color="#334155", ha="left")
    for u, v in [(0.18, 0.18), (0.48, 0.54), (0.78, 0.34)]:
        pts = [project_iso(x, y, w, h, u, v, z) for z, _, _ in layers]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#596873", lw=0.70, alpha=0.55, zorder=5)
    add_text(ax, x + 0.50 * w, y + 0.005 * h, "exploded risk tensor view", 4.9, color="#425466")


def draw_corridor_model(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    s = np.linspace(0.08, 0.92, 90)
    center_v = 0.44 + 0.10 * np.sin(2.3 * np.pi * s) - 0.035 * np.cos(6.2 * np.pi * s)
    lower = np.array([project_iso(x, y, w, h, u, v, 0.18) for u, v in zip(s, center_v)])
    upper = np.array([project_iso(x, y, w, h, u, v + 0.03, 0.82) for u, v in zip(s, center_v)])
    shell = np.vstack([lower, upper[::-1]])
    ax.add_patch(Polygon(shell, closed=True, facecolor="#C9D7E2", edgecolor="none", alpha=0.42, zorder=2))
    for idx in np.linspace(0, len(s) - 1, 7, dtype=int):
        cx = (lower[idx, 0] + upper[idx, 0]) / 2
        cy = (lower[idx, 1] + upper[idx, 1]) / 2
        ax.add_patch(Ellipse((cx, cy), 0.050 * w, 0.135 * h, angle=-20, facecolor="none", edgecolor="#7890A2", lw=0.50, alpha=0.55, zorder=4))
    for frac in np.linspace(0.16, 0.84, 5):
        a = np.array([project_iso(x, y, w, h, u, v + 0.025, 0.18 + frac * 0.64) for u, v in zip(s, center_v)])
        ax.plot(a[:, 0], a[:, 1], color="#8EA2B2", lw=0.35, alpha=0.45, zorder=3)
    ax.plot(lower[:, 0], lower[:, 1], color="#4E6471", lw=0.90, ls=(0, (3, 2)), zorder=5)
    ax.plot(upper[:, 0], upper[:, 1], color="#4E6471", lw=0.90, ls=(0, (3, 2)), zorder=5)
    for z, color in [(0.32, COLORS["terminal"]), (0.52, COLORS["regional"]), (0.70, COLORS["backbone"])]:
        pts = np.array([project_iso(x, y, w, h, u, v + 0.015 * np.sin(5 * u), z) for u, v in zip(s, center_v)])
        ax.plot(pts[:, 0] + 0.003, pts[:, 1] - 0.004, color="#6B7883", lw=1.25, alpha=0.16, zorder=4)
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.42, zorder=6)
    traj_u = np.linspace(0.12, 0.88, 8)
    traj_v = np.interp(traj_u, s, center_v)
    traj_z = np.linspace(0.28, 0.66, 8)
    trajectory = np.array([project_iso(x, y, w, h, u, v + 0.02, z) for u, v, z in zip(traj_u, traj_v, traj_z)])
    ax.plot(trajectory[:, 0], trajectory[:, 1], color="#26323D", lw=1.0, alpha=0.62, zorder=7)
    add_text(ax, x + 0.50 * w, y + 0.005 * h, "transparent corridor tube", 4.9, color="#425466")


def draw_graph_model(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    graph = transparent_crop(GRAPH_IMAGE, (2100, 40, 3910, 1210), threshold=248)
    if graph is not None:
        ax.imshow(
            graph,
            extent=(x - 0.055 * w, x + 1.055 * w, y - 0.008 * h, y + 0.850 * h),
            interpolation="lanczos",
            zorder=3,
            aspect="auto",
        )
    else:
        layers = [(0.26, COLORS["terminal"]), (0.54, COLORS["regional"]), (0.80, COLORS["backbone"])]
        u_vals = np.array([0.14, 0.36, 0.58, 0.82])
        v_vals = np.array([0.28, 0.56, 0.42, 0.68])
        all_pts = []
        for z, color in layers:
            pts = [project_iso(x, y, w, h, float(u), float(v), z) for u, v in zip(u_vals, v_vals)]
            all_pts.append(pts)
            for p0, p1 in zip(pts[:-1], pts[1:]):
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=1.1, zorder=4)
            for px, py in pts:
                ax.scatter([px], [py], s=20, facecolor=color, edgecolor="white", linewidth=0.45, zorder=6)
        for idx in [0, 2, 3]:
            column = [layer[idx] for layer in all_pts]
            ax.plot([p[0] for p in column], [p[1] for p in column], color="#53606A", lw=0.82, alpha=0.72, zorder=3)
    add_text(ax, x + 0.50 * w, y + 0.005 * h, "three-layer topology with access edges", 4.8, color="#425466")


def draw_postprocess_model(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    plane = np.array(
        [
            project_iso(x, y, w, h, 0.05, 0.10, 0.12),
            project_iso(x, y, w, h, 0.95, 0.12, 0.12),
            project_iso(x, y, w, h, 0.94, 0.82, 0.70),
            project_iso(x, y, w, h, 0.10, 0.86, 0.70),
        ]
    )
    ax.add_patch(Polygon(plane, closed=True, facecolor="#EFF5F9", edgecolor="#D1DCE6", lw=0.55, alpha=0.88, zorder=2))
    raw = np.array([[0.10, 0.20, 0.16], [0.24, 0.46, 0.28], [0.38, 0.34, 0.34], [0.50, 0.64, 0.44], [0.62, 0.48, 0.50], [0.78, 0.70, 0.62], [0.90, 0.56, 0.72]])
    pts = np.array([project_iso(x, y, w, h, float(u), float(v), float(z)) for u, v, z in raw])
    ax.plot(pts[:, 0], pts[:, 1], color=COLORS["path"], lw=1.10, zorder=5)
    ax.scatter(pts[:, 0], pts[:, 1], s=14, color=COLORS["path"], edgecolor="white", linewidth=0.35, zorder=6)
    pruned = pts[[0, 3, 5, 6]]
    ax.plot(pruned[:, 0], pruned[:, 1], color="#5B636A", lw=1.05, ls=(0, (3, 2)), alpha=0.88, zorder=5)
    verts = [
        tuple(pts[0]),
        (x + 0.35 * w, y + 0.35 * h),
        (x + 0.57 * w, y + 0.60 * h),
        tuple(pts[5]),
        (x + 0.84 * w, y + 0.72 * h),
        (x + 0.88 * w, y + 0.62 * h),
        tuple(pts[6]),
    ]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    ax.add_patch(PathPatch(MplPath(verts, codes), fill=False, lw=1.75, color=COLORS["smooth"], capstyle="round", zorder=7))
    add_text(ax, x + 0.50 * w, y + 0.005 * h, "LOS pruning to B-spline trajectory", 4.8, color="#425466")


def draw_replan_model(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    base_center = project_iso(x, y, w, h, 0.50, 0.45, 0.10)
    top_center = project_iso(x, y, w, h, 0.50, 0.45, 0.78)
    ax.add_patch(Rectangle((base_center[0] - 0.070 * w, base_center[1]), 0.140 * w, top_center[1] - base_center[1], facecolor=COLORS["danger"], edgecolor="none", alpha=0.18, zorder=3))
    ax.add_patch(Ellipse(base_center, 0.150 * w, 0.050 * h, angle=-14, facecolor=COLORS["danger"], edgecolor="#A73531", lw=0.65, alpha=0.28, zorder=4))
    ax.add_patch(Ellipse(top_center, 0.150 * w, 0.050 * h, angle=-14, facecolor=COLORS["danger"], edgecolor="#A73531", lw=0.75, alpha=0.36, zorder=6))
    add_text(ax, top_center[0], top_center[1] + 0.026 * h, "dynamic\nno-fly zone", 4.8, color="#8C2F2B", linespacing=1.0)

    old = np.array([[0.10, 0.20, 0.14], [0.30, 0.32, 0.22], [0.50, 0.45, 0.36], [0.74, 0.58, 0.50], [0.90, 0.64, 0.58]])
    old_pts = np.array([project_iso(x, y, w, h, float(u), float(v), float(z)) for u, v, z in old])
    ax.plot(old_pts[:, 0], old_pts[:, 1], color=COLORS["path"], lw=1.25, alpha=0.72, zorder=4)
    ax.plot(old_pts[1:3, 0], old_pts[1:3, 1], color=COLORS["danger"], lw=2.0, alpha=0.90, zorder=7)

    new = np.array([[0.10, 0.20, 0.14], [0.28, 0.26, 0.22], [0.42, 0.22, 0.30], [0.58, 0.24, 0.40], [0.78, 0.47, 0.52], [0.90, 0.64, 0.58]])
    new_pts = np.array([project_iso(x, y, w, h, float(u), float(v), float(z)) for u, v, z in new])
    ax.plot(new_pts[:, 0], new_pts[:, 1], color=COLORS["smooth"], lw=1.75, zorder=8)
    ax.scatter(new_pts[:, 0], new_pts[:, 1], s=12, facecolor=COLORS["smooth"], edgecolor="white", linewidth=0.35, zorder=9)

    graph_nodes = np.array([[0.66, 0.72, 0.42], [0.82, 0.78, 0.50], [0.86, 0.58, 0.48], [0.70, 0.50, 0.36]])
    gp = np.array([project_iso(x, y, w, h, float(u), float(v), float(z)) for u, v, z in graph_nodes])
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]:
        ax.plot([gp[i, 0], gp[j, 0]], [gp[i, 1], gp[j, 1]], color="#607080", lw=0.62, alpha=0.62, zorder=4)
    for px, py in gp:
        ax.scatter([px], [py], s=16, facecolor="white", edgecolor=COLORS["blue"], linewidth=0.70, zorder=6)
    add_text(ax, x + 0.50 * w, y + 0.005 * h, "local edge invalidation and path bypass", 4.8, color="#425466")


def draw_support_base(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, formula: str, color: str) -> None:
    add_shadow(ax, x, y, w, h, alpha=0.10)
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.006",
            facecolor=COLORS["support"],
            edgecolor=COLORS["support_edge"],
            lw=0.65,
            zorder=2,
        )
    )
    ax.add_patch(Rectangle((x, y + h - 0.020), w, 0.020, facecolor=color, edgecolor="none", alpha=0.22, zorder=3))
    add_text(ax, x + w / 2, y + 0.061, title, 5.7, weight="bold", color="#26323D")
    add_text(ax, x + w / 2, y + 0.030, formula, 6.2, color="#334155")


def add_outcome(ax: plt.Axes) -> None:
    x, y, w, h = 0.105, 0.054, 0.790, 0.074
    add_shadow(ax, x, y, w, h, alpha=0.09)
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.014",
            facecolor="#F7FAFD",
            edgecolor="#CBD8E4",
            lw=0.68,
            zorder=2,
        )
    )
    cx = x + 0.055
    cy = y + h / 2
    for r, lw in [(0.027, 1.1), (0.017, 0.95)]:
        ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor=COLORS["blue"], lw=lw, zorder=4))
    ax.scatter([cx], [cy], s=36, facecolor=COLORS["blue"], edgecolor="white", linewidth=0.4, zorder=5)
    ax.add_patch(FancyArrowPatch((cx + 0.002, cy + 0.002), (cx + 0.040, cy + 0.031), arrowstyle="-|>", mutation_scale=7.5, lw=1.1, color=COLORS["blue"], zorder=5))
    add_text(ax, x + 0.50 * w, y + 0.050, "Outcome", 6.2, weight="bold", color="#1F2933")
    add_text(ax, x + 0.52 * w, y + 0.026, "A continuous mountainous airspace is converted into a risk-aware graph,\nthen updated locally and smoothed into a flyable UAV trajectory.", 5.45, color="#1F2933")


def build_figure() -> plt.Figure:
    setup_style()
    fig = plt.figure(figsize=(12.2, 6.65))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_text(ax, 0.5, 0.955, "Overall Framework of the Proposed Method", 11.1, weight="bold")
    draw_pipeline_band(ax)

    x0, x1 = 0.080, 0.920
    centers = np.linspace(x0, x1, 6)
    stage_w, stage_h = 0.132, 0.330
    stage_y = 0.444
    titles = [
        "1. Input",
        "2. Risk Modeling",
        "3. Corridor Generation",
        "4. Airway Graph",
        "5. Path Post-processing",
        "6. Incremental Replanning",
    ]
    draw_funcs = [draw_input_model, draw_risk_model, draw_corridor_model, draw_graph_model, draw_postprocess_model, draw_replan_model]

    for cx, title, func in zip(centers, titles, draw_funcs):
        draw_step_title(ax, cx, 0.815, title)
        func(ax, cx - stage_w / 2, stage_y, stage_w, stage_h)

    flow_y = 0.664
    labels = [
        r"$\mathcal{M}_{DEM},\ \mathcal{P}_{OSM}$",
        r"$\mathcal{F}_{risk}(x,y,z)$",
        r"$\Omega_{safe},\ \Gamma_{mid}$",
        r"$\mathcal{G}=(V,E)$",
        r"$\tau_{smooth}(t)$",
    ]
    for i, label in enumerate(labels):
        draw_flow(ax, centers[i] + stage_w * 0.49, centers[i + 1] - stage_w * 0.49, flow_y, label)

    add_text(ax, 0.076, 0.334, "Algorithmic support base", 5.8, color="#52616F", ha="left")
    base_y, base_h = 0.226, 0.092
    bases = [
        (2, "Adaptive corridor", r"$z_f,\ z_c,\ \Gamma_{mid}$", COLORS["terminal"]),
        (3, "Graph construction", r"$\mathcal{G}=(V,E),\ C_{edge}=0$", COLORS["regional"]),
        (4, "LOS pruning and B-spline", r"$\Pi_{LOS}\ \rightarrow\ \mathbf{B}(t)$", COLORS["smooth"]),
        (5, "LPA* algorithm", r"$g(v),\ rhs(v),\ U$", COLORS["blue"]),
    ]
    base_w = stage_w * 1.12
    for idx, title, formula, color in bases:
        bx = centers[idx] - base_w / 2
        draw_support_base(ax, bx, base_y, base_w, base_h, title, formula, color)
        ax.plot([centers[idx], centers[idx]], [base_y + base_h, 0.374], color="#C2D2E2", lw=0.62, zorder=1)

    add_outcome(ax)
    return fig


def export_figure(fig: plt.Figure) -> dict[str, Path]:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "svg": FINAL_DIR / f"{OUT_STEM}.svg",
        "pdf": FINAL_DIR / f"{OUT_STEM}.pdf",
        "png": FINAL_DIR / f"{OUT_STEM}.png",
        "tiff": FINAL_DIR / f"{OUT_STEM}.tiff",
    }
    fig.savefig(paths["svg"], bbox_inches="tight", pad_inches=0.015)
    fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.015)
    fig.savefig(paths["png"], dpi=DPI, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)

    with Image.open(paths["png"]) as img:
        img.save(paths["tiff"], dpi=(DPI, DPI), compression="raw")

    for path in paths.values():
        shutil.copy2(path, INTERMEDIATE_DIR / path.name)
    return paths


def main() -> None:
    fig = build_figure()
    paths = export_figure(fig)
    print("Generated Nature-style overall framework figure:")
    for kind, path in paths.items():
        print(f"{kind.upper()}: {path}")


if __name__ == "__main__":
    main()
