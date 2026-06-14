from __future__ import annotations

from pathlib import Path
import shutil
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "intermediate_artifacts" / "figures" / "method_framework"
FINAL_DIR = ROOT / "final_results" / "paper_revision" / "figures"
STEM = "fig_1_method_framework_draft_style"
WIDTH_MM = 183
HEIGHT_MM = 132


COLORS = {
    "ink": "#1F2933",
    "muted": "#65727F",
    "line": "#B9C7D4",
    "panel": "#F7FAFC",
    "panel_edge": "#D5E1EC",
    "input": "#DCEBFA",
    "risk": "#F8E7DC",
    "corridor": "#E1F0DD",
    "graph": "#E4E8F6",
    "path": "#DDEEF4",
    "event": "#F4DDDD",
    "blue": "#2369B3",
    "blue_dark": "#194F86",
    "green": "#3D8B57",
    "orange": "#D4822B",
    "red": "#C43D35",
    "gray": "#8B96A3",
}


def setup_matplotlib() -> None:
    """设置论文图导出参数，保留 SVG 和 PDF 中的可编辑文字。"""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial",
                "DejaVu Sans",
                "sans-serif",
            ],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
            "font.size": 7,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    face: str,
    edge: str = COLORS["panel_edge"],
    lw: float = 1.0,
    radius: float = 0.018,
    zorder: int = 2,
) -> patches.FancyBboxPatch:
    """绘制低饱和圆角模块框。"""
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.010,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = COLORS["blue"],
    lw: float = 1.2,
    rad: float = 0.0,
    zorder: int = 8,
) -> None:
    """绘制模块之间的方向箭头。"""
    ax.add_patch(
        patches.FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
            connectionstyle=f"arc3,rad={rad}",
            zorder=zorder,
        )
    )


def add_text(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    size: float = 7,
    color: str = COLORS["ink"],
    weight: str = "normal",
    ha: str = "center",
    va: str = "center",
) -> None:
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=20)


def add_wrapped_text(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    width: int,
    size: float = 5.8,
    color: str = COLORS["muted"],
    ha: str = "center",
    va: str = "top",
) -> None:
    wrapped = "\n".join(textwrap.wrap(text, width=width, break_long_words=False, replace_whitespace=False))
    add_text(ax, x, y, wrapped, size=size, color=color, ha=ha, va=va)


def draw_card_header(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    idx: int,
    title: str,
    face: str,
) -> None:
    """绘制编号、标题和淡色顶部标识条。"""
    ax.add_patch(
        patches.Rectangle((x + 0.012, y + h - 0.030), w - 0.024, 0.012, facecolor=face, edgecolor="none", zorder=5)
    )
    ax.add_patch(patches.Circle((x + 0.027, y + h - 0.055), 0.017, facecolor=face, edgecolor="white", linewidth=0.7, zorder=7))
    add_text(ax, x + 0.027, y + h - 0.055, str(idx), size=6.0, weight="bold", color=COLORS["ink"])
    add_text(ax, x + 0.055, y + h - 0.054, title, size=7.1, weight="bold", ha="left", color=COLORS["ink"])


def draw_input_icon(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    """绘制输入数据示意，包括地形、风险点和任务端点。"""
    base_y = y + 0.040
    pts = [
        (x + 0.035, base_y),
        (x + 0.070, base_y + 0.045),
        (x + 0.115, base_y + 0.020),
        (x + 0.155, base_y + 0.075),
        (x + 0.205, base_y + 0.030),
        (x + 0.245, base_y + 0.055),
    ]
    path = MplPath(pts, [MplPath.MOVETO] + [MplPath.LINETO] * (len(pts) - 1))
    ax.add_patch(patches.PathPatch(path, fill=False, edgecolor=COLORS["green"], linewidth=1.3, zorder=8))
    ax.plot([p[0] for p in pts], [p[1] - 0.025 for p in pts], color="#A2B377", linewidth=0.9, alpha=0.75, zorder=7)
    for px, py in [(x + 0.075, base_y + 0.065), (x + 0.175, base_y + 0.087), (x + 0.228, base_y + 0.050)]:
        ax.add_patch(patches.Circle((px, py), 0.006, facecolor=COLORS["red"], edgecolor="white", linewidth=0.3, zorder=10))
    ax.plot([x + 0.040, x + 0.245], [y + h - 0.105, y + h - 0.105], color=COLORS["line"], linewidth=0.6, zorder=7)
    ax.plot([x + 0.040, x + 0.245], [y + h - 0.122, y + h - 0.122], color=COLORS["line"], linewidth=0.6, zorder=7)


def draw_risk_icon(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    """绘制多源风险张量和通信视距约束。"""
    colors = ["#BFD7EE", "#F2C978", "#CDE2A4"]
    labels = ["通信", "人群", "地形"]
    for i, (c, label) in enumerate(zip(colors, labels)):
        yy = y + 0.050 + i * 0.035
        poly = patches.Polygon(
            [
                (x + 0.052, yy),
                (x + 0.235, yy + 0.018),
                (x + 0.200, yy + 0.052),
                (x + 0.020, yy + 0.033),
            ],
            closed=True,
            facecolor=c,
            edgecolor="#8AA1B2",
            linewidth=0.7,
            alpha=0.82,
            zorder=7 + i,
        )
        ax.add_patch(poly)
        add_text(ax, x + 0.252, yy + 0.026, label, size=5.2, color=COLORS["muted"], ha="left")
    for xx in [x + 0.070, x + 0.145, x + 0.205]:
        ax.plot([xx, xx], [y + 0.060, y + 0.160], color="#778899", linewidth=0.55, zorder=11)
    ax.plot([x + 0.030, x + 0.245], [y + 0.176, y + 0.176], color=COLORS["blue"], linewidth=1.0, alpha=0.8, zorder=11)
    add_text(ax, x + 0.142, y + 0.151, "风险场 f(x,y,z)", size=5.4, color=COLORS["blue_dark"])


def draw_corridor_icon(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    """绘制安全走廊包络和三层中面。"""
    xs = [x + 0.030, x + 0.075, x + 0.120, x + 0.170, x + 0.230]
    lower = [y + 0.055, y + 0.060, y + 0.048, y + 0.062, y + 0.070]
    upper = [yy + 0.110 for yy in lower]
    ax.plot(xs, lower, color="#7FA6C2", linewidth=1.0, linestyle=(0, (3, 2)), zorder=8)
    ax.plot(xs, upper, color="#7FA6C2", linewidth=1.0, linestyle=(0, (3, 2)), zorder=8)
    ax.fill_between(xs, lower, upper, color="#DCEBF7", alpha=0.58, zorder=6)
    for ratio, color in [(0.25, "#2A7CB8"), (0.50, "#3D8B57"), (0.75, "#D4822B")]:
        ys = [lo + ratio * (up - lo) for lo, up in zip(lower, upper)]
        ax.plot(xs, ys, color=color, linewidth=1.4, zorder=9)
    add_text(ax, x + 0.138, y + 0.150, "Ωsafe，Γmid", size=5.5, color=COLORS["blue_dark"])


def draw_graph_icon(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    """绘制三层航线图拓扑。"""
    layers = [y + 0.065, y + 0.110, y + 0.155]
    node_x = [x + 0.052, x + 0.115, x + 0.182, x + 0.235]
    layer_cols = ["#2A7CB8", "#3D8B57", "#D4822B"]
    for li, yy in enumerate(layers):
        ax.plot([node_x[0], node_x[-1]], [yy, yy], color=layer_cols[li], linewidth=1.0, alpha=0.7, zorder=7)
        for xx in node_x:
            ax.add_patch(patches.Circle((xx, yy), 0.008, facecolor=layer_cols[li], edgecolor="white", linewidth=0.4, zorder=9))
    for i, xx in enumerate(node_x[1:3]):
        ax.plot([xx, xx + 0.016], [layers[0], layers[1]], color=COLORS["gray"], linewidth=0.7, zorder=8)
        ax.plot([xx + 0.016, xx + 0.030], [layers[1], layers[2]], color=COLORS["gray"], linewidth=0.7, zorder=8)
    ax.add_patch(patches.Circle((x + 0.034, y + 0.045), 0.010, facecolor="white", edgecolor=COLORS["blue"], linewidth=1.0, zorder=10))
    ax.add_patch(patches.Circle((x + 0.253, y + 0.178), 0.010, facecolor="white", edgecolor=COLORS["green"], linewidth=1.0, zorder=10))
    add_text(ax, x + 0.142, y + 0.146, "G = (V, E)", size=5.5, color=COLORS["blue_dark"])


def draw_path_icon(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    """绘制离散路径、LOS剪枝和B样条平滑。"""
    pts = [(x + 0.040, y + 0.058), (x + 0.077, y + 0.087), (x + 0.118, y + 0.103), (x + 0.166, y + 0.139), (x + 0.232, y + 0.167)]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLORS["gray"], linewidth=1.2, linestyle=(0, (3, 2)), zorder=7)
    for p in pts:
        ax.add_patch(patches.Circle(p, 0.007, facecolor=COLORS["orange"], edgecolor="white", linewidth=0.4, zorder=9))
    curve = MplPath(
        [pts[0], (x + 0.090, y + 0.050), (x + 0.170, y + 0.190), pts[-1]],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(patches.PathPatch(curve, fill=False, edgecolor=COLORS["blue"], linewidth=2.1, capstyle="round", zorder=10))
    add_text(ax, x + 0.142, y + 0.146, "τsmooth(t)", size=5.5, color=COLORS["blue_dark"])


def draw_event_icon(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    """绘制区域事件驱动的局部重规划。"""
    prefix = [(x + 0.035, y + 0.062), (x + 0.082, y + 0.086), (x + 0.122, y + 0.104)]
    old = [(x + 0.122, y + 0.104), (x + 0.164, y + 0.128), (x + 0.215, y + 0.146)]
    new = [(x + 0.122, y + 0.104), (x + 0.148, y + 0.064), (x + 0.206, y + 0.076), (x + 0.242, y + 0.145)]
    ax.add_patch(patches.Ellipse((x + 0.168, y + 0.122), 0.064, 0.095, facecolor="#F2C9C4", edgecolor=COLORS["red"], linewidth=0.8, alpha=0.62, zorder=6))
    ax.plot([p[0] for p in prefix], [p[1] for p in prefix], color=COLORS["blue_dark"], linewidth=1.8, zorder=9)
    ax.plot([p[0] for p in old], [p[1] for p in old], color=COLORS["red"], linewidth=1.3, linestyle=(0, (3, 2)), zorder=8)
    ax.plot([p[0] for p in new], [p[1] for p in new], color=COLORS["blue"], linewidth=2.0, zorder=10)
    for p in prefix + new[1:]:
        ax.add_patch(patches.Circle(p, 0.007, facecolor=COLORS["blue"], edgecolor="white", linewidth=0.4, zorder=11))
    ax.add_patch(patches.Circle(prefix[-1], 0.009, facecolor=COLORS["ink"], edgecolor="white", linewidth=0.4, zorder=12))
    add_text(ax, x + 0.176, y + 0.148, "事件更新", size=5.5, color=COLORS["red"])


def draw_module(
    ax: plt.Axes,
    box: tuple[float, float, float, float],
    idx: int,
    title: str,
    details: str,
    face: str,
    icon_func,
) -> None:
    x, y, w, h = box
    rounded_box(ax, x, y, w, h, COLORS["panel"], edge=COLORS["panel_edge"], lw=0.9, radius=0.020)
    draw_card_header(ax, x, y, w, h, idx, title, face)
    icon_func(ax, x + 0.012, y + 0.038, w - 0.024, h - 0.074)
    add_wrapped_text(ax, x + w / 2, y + 0.035, details, width=16, size=5.25, color=COLORS["muted"])


def draw_support_layer(ax: plt.Axes) -> None:
    """绘制草稿底部的算法支撑层与输出结果。"""
    rounded_box(ax, 0.060, 0.075, 0.880, 0.150, "#F8FBFD", edge=COLORS["panel_edge"], lw=1.0, radius=0.022, zorder=1)
    add_text(ax, 0.090, 0.196, "算法支撑与结果输出", size=7.0, weight="bold", ha="left", color=COLORS["ink"])
    support = [
        ("自适应走廊", "zf，zc，Γmid", COLORS["corridor"]),
        ("分层图构建", "节点，边，层间接入", COLORS["graph"]),
        ("增量搜索", "LPA*，局部边更新", COLORS["input"]),
        ("轨迹连续化", "LOS剪枝，B样条平滑", COLORS["path"]),
    ]
    x0 = 0.150
    for i, (title, sub, color) in enumerate(support):
        bx = x0 + i * 0.145
        rounded_box(ax, bx, 0.112, 0.118, 0.070, "#EEF5FA", edge="#C5D8EA", lw=0.8, radius=0.012, zorder=3)
        ax.add_patch(patches.Rectangle((bx + 0.011, 0.165), 0.096, 0.010, facecolor=color, edgecolor="none", zorder=4))
        add_text(ax, bx + 0.059, 0.145, title, size=5.8, weight="bold")
        add_text(ax, bx + 0.059, 0.125, sub, size=4.7, color=COLORS["muted"])
    rounded_box(ax, 0.760, 0.112, 0.145, 0.070, "#FDF8F2", edge="#E8D7C1", lw=0.8, radius=0.012, zorder=3)
    add_text(ax, 0.832, 0.152, "规划结果", size=6.0, weight="bold", color=COLORS["ink"])
    add_text(ax, 0.832, 0.130, "初始路径，事件后路径", size=4.7, color=COLORS["muted"])


def build_figure() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()

    fig, ax = plt.subplots(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=160)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_text(ax, 0.5, 0.965, "山地无人机动态航路规划方法总框架", size=10.2, weight="bold")
    add_text(ax, 0.5, 0.938, "由环境输入构建安全走廊与三层航线网络，并在区域事件下执行局部增量重规划", size=6.2, color=COLORS["muted"])

    rounded_box(ax, 0.045, 0.270, 0.910, 0.600, "#FBFCFE", edge="#D9E5EF", lw=1.0, radius=0.030, zorder=0)
    add_text(ax, 0.067, 0.838, "高维环境约束下的航路规划系统", size=7.0, weight="bold", ha="left", color=COLORS["blue_dark"])
    ax.plot([0.070, 0.930], [0.812, 0.812], color="#E2EAF1", linewidth=0.9, zorder=2)

    w = 0.260
    h = 0.190
    xs = [0.085, 0.370, 0.655]
    top_y = 0.590
    bot_y = 0.345
    cards = {
        1: (xs[0], top_y, w, h),
        2: (xs[1], top_y, w, h),
        3: (xs[2], top_y, w, h),
        4: (xs[2], bot_y, w, h),
        5: (xs[1], bot_y, w, h),
        6: (xs[0], bot_y, w, h),
    }
    modules = [
        (1, "山地任务输入", "DEM，OSM风险要素，任务端点，通信源", COLORS["input"], draw_input_icon),
        (2, "多源风险建模", "地形遮挡，人群暴露，通信可达性融合", COLORS["risk"], draw_risk_icon),
        (3, "安全走廊生成", "走廊上下边界，三层飞行中面，可通行域", COLORS["corridor"], draw_corridor_icon),
        (4, "三层航线网络", "终端层，区域支路层，骨干层及层间接入", COLORS["graph"], draw_graph_icon),
        (5, "路径规划与后处理", "离散图路径，LOS剪枝，B样条连续化", COLORS["path"], draw_path_icon),
        (6, "事件驱动重规划", "保留已航行前缀，仅更新受影响未来边", COLORS["event"], draw_event_icon),
    ]
    for idx, title, details, face, icon_func in modules:
        draw_module(ax, cards[idx], idx, title, details, face, icon_func)

    arrow(ax, (cards[1][0] + w + 0.006, top_y + h * 0.54), (cards[2][0] - 0.006, top_y + h * 0.54))
    arrow(ax, (cards[2][0] + w + 0.006, top_y + h * 0.54), (cards[3][0] - 0.006, top_y + h * 0.54))
    arrow(ax, (cards[3][0] + w * 0.50, top_y - 0.006), (cards[4][0] + w * 0.50, bot_y + h + 0.006))
    arrow(ax, (cards[4][0] - 0.006, bot_y + h * 0.54), (cards[5][0] + w + 0.006, bot_y + h * 0.54))
    arrow(ax, (cards[5][0] - 0.006, bot_y + h * 0.54), (cards[6][0] + w + 0.006, bot_y + h * 0.54))

    arrow(ax, (cards[5][0] + w * 0.50, bot_y - 0.012), (0.835, 0.188), color=COLORS["orange"], lw=1.0, rad=-0.12)
    arrow(ax, (cards[6][0] + w * 0.50, bot_y - 0.012), (0.835, 0.188), color=COLORS["red"], lw=1.0, rad=0.10)
    draw_support_layer(ax)

    rounded_box(ax, 0.055, 0.030, 0.890, 0.026, "#FFFFFF", edge=COLORS["line"], lw=0.7, radius=0.006, zorder=1)
    add_text(
        ax,
        0.500,
        0.043,
        "输出：安全约束下的初始航路，以及区域事件发生后的局部增量重规划航路",
        size=5.6,
        color=COLORS["muted"],
    )

    qa_path = OUT_DIR / f"{STEM}_qa.md"
    qa_path.write_text(
        "\n".join(
            [
                "# 方法总框架图 QA",
                "",
                "核心结论：本文方法将山地连续空域转换为风险感知的安全走廊与三层航线图，并在区域事件发生后仅对受影响未来边执行局部增量重规划。",
                "",
                "图型：schematic-led composite，草稿式六模块蛇形流程。",
                "",
                "面板逻辑：1环境输入，2风险建模，3安全走廊，4三层航线网络，5路径规划与后处理，6事件驱动重规划。",
                "",
                "导出契约：双栏宽 183 mm，高 132 mm；SVG 和 PDF 保留可编辑文字；TIFF 以 600 dpi 输出；PNG 用于快速预览。",
                "",
                "审稿风险：该图是方法总览，不展示定量结果；应与第 3 章方法描述配合使用，第四章仍保留实验对比图。",
            ]
        ),
        encoding="utf-8",
    )

    out_stem = OUT_DIR / STEM
    for ext in ("svg", "pdf", "tiff", "png"):
        save_kwargs = {"facecolor": "white"}
        if ext == "tiff":
            save_kwargs["dpi"] = 600
        elif ext == "png":
            save_kwargs["dpi"] = 300
        fig.savefig(f"{out_stem}.{ext}", **save_kwargs)
        shutil.copy2(f"{out_stem}.{ext}", FINAL_DIR / f"{STEM}.{ext}")
    shutil.copy2(qa_path, FINAL_DIR / f"{STEM}_qa.md")
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
