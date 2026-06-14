from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Polygon
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "intermediate_artifacts" / "figures" / "area_event_path_impact"
FIG_STEM = OUT_DIR / "area_event_path_impact_nature_polished"
WIDTH_MM = 183
HEIGHT_MM = 96


PALETTE = {
    "ink": "#242424",
    "muted": "#747B83",
    "network": "#CBD1D8",
    "network_light": "#E2E6EA",
    "blue": "#1265B7",
    "blue_light": "#D8E8F8",
    "green": "#2C8A46",
    "red": "#C7372F",
    "red_fill": "#F5B2A8",
    "orange": "#D77825",
    "paper": "#FFFFFF",
}


def configure_matplotlib() -> None:
    """设置期刊图常用字体、线宽和可编辑文本导出参数。"""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
            "font.size": 7,
            "axes.linewidth": 0.6,
            "figure.facecolor": PALETTE["paper"],
            "savefig.facecolor": PALETTE["paper"],
        }
    )


def event_polygon() -> np.ndarray:
    """生成确定性的非规则事件区域边界，避免圆形区域显得过于示意化。"""
    center = np.array([5.65, 3.10])
    theta = np.linspace(0, 2 * np.pi, 19, endpoint=False)
    radius = 1.32 + 0.20 * np.sin(3 * theta + 0.4) + 0.12 * np.cos(5 * theta)
    x = center[0] + 1.22 * radius * np.cos(theta)
    y = center[1] + 0.86 * radius * np.sin(theta)
    return np.column_stack([x, y])


def draw_polyline_with_arrows(
    ax: plt.Axes,
    points: np.ndarray,
    color: str,
    linewidth: float,
    arrow_segments: tuple[int, ...],
    linestyle: str = "-",
    zorder: int = 5,
) -> None:
    """绘制折线并在指定线段上加方向箭头。"""
    ax.plot(
        points[:, 0],
        points[:, 1],
        color=color,
        lw=linewidth,
        linestyle=linestyle,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=zorder,
    )
    for idx in arrow_segments:
        start = points[idx]
        end = points[idx + 1]
        vec = end - start
        arrow_start = start + 0.47 * vec
        arrow_end = start + 0.68 * vec
        ax.add_patch(
            FancyArrowPatch(
                arrow_start,
                arrow_end,
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=0,
                facecolor=color,
                edgecolor=color,
                zorder=zorder + 1,
            )
        )


def annotate_with_leader(
    ax: plt.Axes,
    label: str,
    xy: tuple[float, float],
    xytext: tuple[float, float],
    color: str,
    ha: str = "center",
) -> None:
    """用细引线添加直接标签，减少图例依赖。"""
    ax.annotate(
        label,
        xy=xy,
        xytext=xytext,
        ha=ha,
        va="center",
        color=color,
        fontsize=7,
        arrowprops={
            "arrowstyle": "-",
            "lw": 0.55,
            "color": color,
            "shrinkA": 1,
            "shrinkB": 2,
        },
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7, "alpha": 0.86},
        zorder=10,
    )


def draw_background_network(ax: plt.Axes) -> None:
    """绘制低对比度航路网络和地形约束线，作为空间参照而不抢占主体。"""
    routes = [
        [(0.3, 3.1), (1.1, 3.0), (2.1, 3.35), (3.2, 4.0), (4.4, 4.55), (5.7, 4.70), (7.6, 4.1), (9.1, 3.65), (10.6, 2.6), (11.7, 1.9)],
        [(0.6, 1.0), (1.5, 1.25), (2.8, 1.02), (4.1, 1.35), (5.3, 1.05), (6.5, 1.35), (7.8, 1.0), (9.2, 1.1), (10.8, 1.55)],
        [(1.1, 4.9), (2.2, 4.45), (3.2, 4.0), (3.9, 5.45)],
        [(7.2, 5.4), (6.0, 4.7), (5.7, 4.70)],
        [(8.6, 4.3), (9.5, 4.25), (10.6, 4.0), (11.8, 4.2)],
        [(1.0, 3.0), (1.25, 1.9), (1.5, 1.25), (1.0, 0.35)],
        [(2.1, 3.35), (1.6, 4.05), (2.9, 4.45)],
        [(4.4, 4.55), (4.0, 3.4), (3.8, 2.2), (4.1, 1.35)],
        [(6.5, 1.35), (6.8, 2.25), (7.6, 4.1)],
        [(9.2, 1.1), (9.1, 3.65), (9.5, 4.25)],
        [(10.6, 2.6), (10.8, 1.55), (11.6, 1.15)],
    ]
    for route in routes:
        arr = np.array(route)
        ax.plot(
            arr[:, 0],
            arr[:, 1],
            color=PALETTE["network"],
            lw=1.05,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=1,
        )

    ridge = np.array(
        [
            [7.65, 0.75],
            [7.9, 1.35],
            [8.05, 2.0],
            [8.0, 2.65],
            [7.75, 3.2],
            [7.85, 3.85],
            [8.25, 4.5],
            [8.55, 5.25],
        ]
    )
    ax.plot(
        ridge[:, 0],
        ridge[:, 1],
        color="#9CCAF2",
        lw=3.2,
        alpha=0.58,
        solid_capstyle="round",
        zorder=0,
    )
    ax.text(
        8.18,
        0.55,
        "地形或走廊边界",
        color="#6EA7D8",
        fontsize=5.8,
        ha="center",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.45, "alpha": 0.82},
        zorder=10,
    )


def draw_event_region(ax: plt.Axes, poly: np.ndarray) -> None:
    """绘制事件足迹、受影响边和风险提示符号。"""
    path_vertices = np.vstack([poly, poly[0]])
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(poly) - 1) + [MplPath.CLOSEPOLY]
    patch = PathPatch(
        MplPath(path_vertices, codes),
        facecolor=PALETTE["red_fill"],
        edgecolor=PALETTE["red"],
        lw=0.9,
        alpha=0.55,
        linestyle=(0, (3, 2)),
        zorder=3,
    )
    ax.add_patch(patch)

    affected_edges = [
        np.array([[4.2, 3.55], [4.85, 3.33], [5.45, 3.10]]),
        np.array([[5.0, 4.1], [5.25, 3.45], [5.7, 2.85]]),
        np.array([[5.35, 2.0], [5.95, 2.35], [6.65, 2.65]]),
    ]
    for edge in affected_edges:
        ax.plot(edge[:, 0], edge[:, 1], color=PALETTE["orange"], lw=1.55, alpha=0.92, zorder=4)
        for x, y in edge[1:-1]:
            ax.plot([x - 0.05, x + 0.05], [y - 0.05, y + 0.05], color=PALETTE["red"], lw=0.9, zorder=6)
            ax.plot([x - 0.05, x + 0.05], [y + 0.05, y - 0.05], color=PALETTE["red"], lw=0.9, zorder=6)

    warning = Polygon(
        [[5.86, 3.21], [5.57, 2.72], [6.15, 2.72]],
        closed=True,
        facecolor="#FFF4EA",
        edgecolor=PALETTE["red"],
        lw=1.0,
        zorder=6,
    )
    ax.add_patch(warning)
    ax.text(5.86, 2.89, "!", color=PALETTE["red"], fontsize=10, ha="center", va="center", fontweight="bold", zorder=7)
    ax.text(
        5.63,
        3.58,
        "事件足迹",
        color=PALETTE["red"],
        fontsize=7.2,
        ha="center",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7, "alpha": 0.86},
        zorder=8,
    )


def draw_endpoints(ax: plt.Axes, start: tuple[float, float], goal: tuple[float, float]) -> None:
    """绘制起点和终点，使用同心圆强调路径端点。"""
    for xy, color, label, align in [
        (start, PALETTE["blue"], "起点", "right"),
        (goal, PALETTE["green"], "终点", "left"),
    ]:
        ax.add_patch(Circle(xy, 0.16, facecolor="white", edgecolor=color, lw=1.2, zorder=8))
        ax.add_patch(Circle(xy, 0.075, facecolor=color, edgecolor=color, lw=0, zorder=9))
        offset = -0.28 if align == "right" else 0.28
        ha = "right" if align == "right" else "left"
        ax.text(
            xy[0] + offset,
            xy[1] + 0.30,
            label,
            ha=ha,
            va="center",
            fontsize=8,
            color=PALETTE["ink"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.55, "alpha": 0.88},
            zorder=10,
        )


def write_source_data(
    flown_prefix: np.ndarray,
    planned_future: np.ndarray,
    replanned_future: np.ndarray,
    poly: np.ndarray,
) -> None:
    """写出示意图几何源数据，方便论文图源文件归档。"""
    csv_path = OUT_DIR / "area_event_path_impact_source_geometry.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "sequence", "x", "y", "note"])
        for idx, (x, y) in enumerate(flown_prefix):
            writer.writerow(["flown_prefix", idx, f"{x:.3f}", f"{y:.3f}", "事件发生前已航行并保持不变的路径前缀"])
        for idx, (x, y) in enumerate(planned_future):
            writer.writerow(["planned_future", idx, f"{x:.3f}", f"{y:.3f}", "事件发生前尚未航行的原计划后续路径"])
        for idx, (x, y) in enumerate(replanned_future):
            writer.writerow(["local_replanned_future", idx, f"{x:.3f}", f"{y:.3f}", "从当前位置开始替换受影响边的局部重规划路径"])
        for idx, (x, y) in enumerate(poly):
            writer.writerow(["event_footprint", idx, f"{x:.3f}", f"{y:.3f}", "区域事件影响范围"])


def write_qa_notes() -> None:
    """写出与 Nature figure 工作流对应的中文质检记录。"""
    qa_path = OUT_DIR / "area_event_path_impact_qa.md"
    qa_path.write_text(
        "\n".join(
            [
                "# 区域事件对路径影响示意图 QA",
                "",
                "核心结论：区域事件只改变当前位置之后的受影响边，已经航行过的路径前缀保持不变。",
                "",
                "图型：schematic-led composite，单个主示意面板。",
                "",
                "证据链：深蓝色实线表示已经航行并锁定的前缀，灰色虚线表示原计划未来段，红色虚线表示被事件影响的未来边，蓝色实线表示从当前位置开始替换受影响边的局部重规划段。",
                "",
                "导出契约：双栏宽 183 mm，高 96 mm；SVG 和 PDF 保留可编辑文本；TIFF 以 600 dpi 输出；PNG 作为快速预览。",
                "",
                "审稿风险：该图为机制示意，不呈现定量效果；图注中应说明几何为示意化表达，不能替代实验地图或统计结果。",
                "",
                "源数据：`area_event_path_impact_source_geometry.csv` 记录已航行前缀、原计划未来段、局部重规划段和事件足迹的示意几何坐标。",
            ]
        ),
        encoding="utf-8",
    )


def build_figure() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    flown_prefix = np.array([[0.75, 3.05], [2.10, 3.25], [3.55, 3.55]])
    planned_future = np.array([[3.55, 3.55], [5.25, 3.02], [7.12, 2.35], [9.40, 2.06], [11.20, 2.00]])
    replanned_future = np.array([[3.55, 3.55], [4.55, 4.10], [5.80, 4.52], [6.95, 4.45], [8.10, 3.92], [8.85, 3.05], [10.05, 2.35], [11.20, 2.00]])
    affected_future = np.array([[4.18, 3.35], [5.25, 3.02], [6.76, 2.48]])
    poly = event_polygon()

    fig, ax = plt.subplots(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=150)
    ax.set_xlim(0.2, 11.85)
    ax.set_ylim(0.35, 5.55)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    draw_background_network(ax)
    draw_event_region(ax, poly)

    draw_polyline_with_arrows(
        ax,
        planned_future,
        color="#565B62",
        linewidth=1.15,
        arrow_segments=(),
        linestyle=(0, (3.2, 3.2)),
        zorder=4,
    )
    ax.plot(
        affected_future[:, 0],
        affected_future[:, 1],
        color=PALETTE["red"],
        lw=1.7,
        linestyle=(0, (3.2, 3.2)),
        zorder=6,
    )
    draw_polyline_with_arrows(
        ax,
        flown_prefix,
        color="#285C8F",
        linewidth=1.85,
        arrow_segments=(0,),
        zorder=8,
    )
    draw_polyline_with_arrows(
        ax,
        replanned_future,
        color=PALETTE["blue"],
        linewidth=1.75,
        arrow_segments=(1, 3, 4, 5),
        zorder=7,
    )
    draw_endpoints(ax, tuple(flown_prefix[0]), tuple(planned_future[-1]))
    ax.add_patch(Circle(tuple(flown_prefix[-1]), 0.13, facecolor="white", edgecolor=PALETTE["ink"], lw=1.0, zorder=10))
    ax.add_patch(Circle(tuple(flown_prefix[-1]), 0.055, facecolor=PALETTE["ink"], edgecolor=PALETTE["ink"], lw=0, zorder=11))

    annotate_with_leader(ax, "已航行边，保留", xy=(2.15, 3.25), xytext=(1.75, 3.86), color="#285C8F", ha="center")
    annotate_with_leader(ax, "当前位置，前缀锁定", xy=tuple(flown_prefix[-1]), xytext=(3.45, 4.28), color=PALETTE["ink"], ha="center")
    annotate_with_leader(ax, "原计划未来段", xy=(8.02, 2.22), xytext=(8.95, 1.62), color="#4D5258", ha="left")
    annotate_with_leader(ax, "局部重规划段", xy=(8.85, 3.05), xytext=(9.42, 3.72), color=PALETTE["blue"], ha="left")
    annotate_with_leader(ax, "受影响未来边", xy=(5.15, 3.06), xytext=(4.10, 2.42), color=PALETTE["red"], ha="right")

    ax.text(
        0.45,
        5.26,
        "区域事件仅触发当前位置后的局部重规划",
        ha="left",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=PALETTE["ink"],
        zorder=10,
    )
    ax.text(
        0.45,
        4.98,
        "已航行前缀保持不变，受影响未来边被替换为绕行段。",
        ha="left",
        va="center",
        fontsize=6.3,
        color=PALETTE["muted"],
        zorder=10,
    )

    legend_handles = [
        mpl.lines.Line2D([0], [0], color="#285C8F", lw=1.8, label="已航行，保持"),
        mpl.lines.Line2D([0], [0], color="#565B62", lw=1.1, linestyle=(0, (3.2, 3.2)), label="原计划未来段"),
        mpl.lines.Line2D([0], [0], color=PALETTE["red"], lw=1.5, linestyle=(0, (3.2, 3.2)), label="受影响边"),
        mpl.lines.Line2D([0], [0], color=PALETTE["blue"], lw=1.8, label="局部重规划段"),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=False,
        fontsize=5.8,
        handlelength=1.8,
        borderaxespad=0.2,
    )
    for text in legend.get_texts():
        text.set_color(PALETTE["muted"])

    for ext in ("svg", "pdf", "tiff", "png"):
        kwargs = {"facecolor": "white"}
        if ext in {"tiff", "png"}:
            kwargs["dpi"] = 600 if ext == "tiff" else 300
        fig.savefig(f"{FIG_STEM}.{ext}", **kwargs)
    plt.close(fig)

    write_source_data(flown_prefix, planned_future, replanned_future, poly)
    write_qa_notes()


if __name__ == "__main__":
    build_figure()
