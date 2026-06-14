"""合成华山游客暴露风险与通信视距风险双联图。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "final_results" / "paper_revision" / "figures"
INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate_artifacts" / "figures" / "huashan"

LEFT_SOURCE = FIGURE_DIR / "fig_2_1b_huashan_human_risk_pyvista_3d.png"
RIGHT_SOURCE = FIGURE_DIR / "fig_2_1c_huashan_communication_reachability.png"
OUTPUT_STEM = "fig_2_1bc_huashan_tourist_comm_risk_dual_panel"

DPI = 600
FIGSIZE = (7.16, 2.98)
PANEL_CAPTIONS = (
    "(a) Tourist Exposure Risk",
    "(b) Communication LOS Risk",
)
LEFT_PANEL_TAG = "(a)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合成华山人员暴露风险与通信视距风险双联图。")
    parser.add_argument("--left-source", type=Path, default=LEFT_SOURCE, help="左侧人员暴露风险图。")
    parser.add_argument("--right-source", type=Path, default=RIGHT_SOURCE, help="右侧通信视距风险图。")
    parser.add_argument("--output-stem", type=str, default=OUTPUT_STEM, help="输出文件名前缀。")
    parser.add_argument("--left-caption", type=str, default=PANEL_CAPTIONS[0], help="左侧面板标题。")
    parser.add_argument("--right-caption", type=str, default=PANEL_CAPTIONS[1], help="右侧面板标题。")
    return parser.parse_args()


def trim_white_margin(
    image: Image.Image,
    threshold: int = 3,
    padding: int = 3,
    left_padding: int | None = None,
    bottom_padding: int | None = None,
) -> Image.Image:
    """裁去图片外缘白边，保留少量留白以免贴边。"""
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, background).convert("L")
    mask = diff.point(lambda value: 255 if value > threshold else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return rgb
    x0, y0, x1, y1 = bbox
    lp = padding if left_padding is None else left_padding
    bp = padding if bottom_padding is None else bottom_padding
    x0 = max(0, x0 - lp)
    y0 = max(0, y0 - padding)
    x1 = min(rgb.width, x1 + padding)
    y1 = min(rgb.height, y1 + bp)
    return rgb.crop((x0, y0, x1, y1))


def prepare_panel(path: Path, *, is_left: bool = False) -> Image.Image:
    """读取源图并转换为白底 RGB 面板。"""
    if not path.exists():
        raise FileNotFoundError(f"缺少源图：{path}")
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.alpha_composite(image)
        image = background.convert("RGB")
    else:
        image = image.convert("RGB")
    if is_left:
        return trim_white_margin(image, left_padding=52, bottom_padding=180)
    return trim_white_margin(image)


def compose(
    *,
    left_source: Path = LEFT_SOURCE,
    right_source: Path = RIGHT_SOURCE,
    output_stem: str = OUTPUT_STEM,
    panel_captions: tuple[str, str] = PANEL_CAPTIONS,
) -> list[Path]:
    """生成 PNG 与 PDF，并同步到中间结果目录。"""
    panels = [prepare_panel(left_source, is_left=True), prepare_panel(right_source)]

    plt.rcParams.update(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGSIZE,
        gridspec_kw={"width_ratios": [panels[0].width / panels[0].height, panels[1].width / panels[1].height]},
    )

    for ax, panel, caption in zip(axes, panels, panel_captions):
        ax.imshow(panel, interpolation='none')
        ax.set_axis_off()
        ax.text(
            0.5,
            -0.055,
            caption,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.6,
            fontfamily="Arial",
            clip_on=False,
        )


    fig.subplots_adjust(left=0.002, right=0.998, bottom=0.088, top=0.998, wspace=0.014)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_DIR / f"{output_stem}.png"
    pdf_path = FIGURE_DIR / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", pad_inches=0.002)
    fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight", pad_inches=0.002)
    plt.close(fig)

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in (png_path, pdf_path):
        target = INTERMEDIATE_DIR / path.name
        shutil.copy2(path, target)
        copied.append(target)

    return [png_path, pdf_path, *copied]


def main() -> None:
    args = parse_args()
    paths = compose(
        left_source=args.left_source,
        right_source=args.right_source,
        output_stem=args.output_stem,
        panel_captions=(args.left_caption, args.right_caption),
    )
    print("已生成双联图：")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
