"""使用 PyVista 渲染华山 DEM 的学术白盒三维地形图。

脚本只读取已有的华山 DEM 裁剪缓存，不重新运行地形裁剪或 benchmark。
输出为纯 DEM 场景，包含地形光照、盒式坐标网格和高程色标，不添加专题要素图例。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy.ndimage import gaussian_filter

from huashan_peak_annotations import draw_peak_annotations, peak_world_points, project_peak_points


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEM = PROJECT_ROOT / "intermediate_artifacts" / "data" / "huashan" / "Z_crop.npy"
DEFAULT_META = PROJECT_ROOT / "intermediate_artifacts" / "data" / "huashan" / "Z_crop_meta.json"
DEFAULT_GEO = PROJECT_ROOT / "intermediate_artifacts" / "data" / "huashan" / "Z_crop_geo.npz"
DEFAULT_SCENE = PROJECT_ROOT / "scenarios" / "huashan.json"
DEFAULT_OUT = PROJECT_ROOT / "intermediate_artifacts" / "figures" / "huashan" / "fig_2_1a_huashan_pyvista_realistic.png"


REFERENCE_TERRAIN_CMAP = LinearSegmentedColormap.from_list(
    "huashan_reference_terrain",
    [
        (0.00, "#0B321F"),
        (0.16, "#1F5631"),
        (0.34, "#4D7041"),
        (0.50, "#8B8058"),
        (0.66, "#A38E63"),
        (0.80, "#8D8677"),
        (0.92, "#CFC8B8"),
        (1.00, "#F4F1EA"),
    ],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 PyVista 渲染参考图风格的华山 DEM 三维地形图。")
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM, help="华山裁剪 DEM 缓存，默认读取 Z_crop.npy。")
    parser.add_argument("--meta", type=Path, default=DEFAULT_META, help="裁剪元数据 JSON，用于读取分辨率。")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出 PNG 路径。")
    parser.add_argument("--pdf-out", type=Path, default=None, help="同步输出 PDF 路径。默认与 PNG 同目录同名。")
    parser.add_argument("--pdf-dpi", type=int, default=300, help="把 PNG 嵌入 PDF 时使用的分辨率。")
    parser.add_argument("--stride", type=int, default=2, help="网格采样步长，2 表示约 400×400 曲面。")
    parser.add_argument("--vertical-exag", type=float, default=1.45, help="垂向夸张系数。")
    parser.add_argument("--smooth-sigma", type=float, default=0.45, help="显示曲面高斯平滑系数。")
    parser.add_argument("--width", type=int, default=3600, help="输出图像宽度。")
    parser.add_argument("--height", type=int, default=2200, help="输出图像高度。")
    parser.add_argument("--hide-grid", action="store_true", help="隐藏盒式坐标网格。")
    parser.add_argument("--hide-scalar-bar", action="store_true", help="隐藏高程色标。")
    parser.add_argument("--geo", type=Path, default=DEFAULT_GEO, help="华山裁剪区域经纬度网格。")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="华山场景配置文件。")
    parser.add_argument("--hide-peak-labels", action="store_true", help="隐藏华山五峰单字母标注。")
    return parser.parse_args()


def resolve_pdf_out(png_out: Path, pdf_out: Path | None) -> Path:
    """根据 PNG 输出路径推导 PDF 输出路径。"""
    if pdf_out is not None:
        return pdf_out
    return png_out.with_suffix(".pdf")


def save_png_as_pdf(png_path: Path, pdf_path: Path, dpi: int) -> Path:
    """把 PyVista 截图同步保存为论文可用的单页 PDF。"""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(png_path) as image:
        image.convert("RGB").save(pdf_path, "PDF", resolution=max(1, int(dpi)))
    return pdf_path


def read_resolution(meta_path: Path) -> float:
    if not meta_path.exists():
        return 12.5
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return float(meta.get("resolution_m") or meta.get("pixel_size_x_m") or 12.5)


def build_structured_grid(
    z: np.ndarray,
    resolution_m: float,
    stride: int,
    vertical_exag: float,
    smooth_sigma: float,
) -> pv.StructuredGrid:
    """把 DEM 栅格转换为以中心点为原点的 PyVista 结构化曲面。"""
    stride = max(1, int(stride))
    z_clean = np.nan_to_num(z, nan=float(np.nanmedian(z)))
    z_smooth = gaussian_filter(z_clean, sigma=max(0.0, float(smooth_sigma)), mode="nearest")

    z_sample = z_smooth[::stride, ::stride]
    elev_sample = z_clean[::stride, ::stride]
    rows, cols = z_sample.shape
    base_elev = float(np.nanmin(z_clean))

    x = (np.arange(cols, dtype=float) - 0.5 * (cols - 1)) * resolution_m * stride
    y = (np.arange(rows - 1, -1, -1, dtype=float) - 0.5 * (rows - 1)) * resolution_m * stride
    xx, yy = np.meshgrid(x, y)
    zz = np.maximum(z_sample - base_elev, 0.0) * vertical_exag

    grid = pv.StructuredGrid(xx, yy, zz)
    grid["Elevation"] = elev_sample.ravel(order="F")
    return grid


def build_ground_shadow(bounds: tuple[float, float, float, float, float, float], span: float) -> pv.StructuredGrid:
    """生成地形底部柔和阴影，强化参考图中的落地感。"""
    x_min, x_max, y_min, y_max, z_min, _ = bounds
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    x_extent = x_max - x_min
    y_extent = y_max - y_min

    pad = 0.015 * span
    xs = np.linspace(x_min - pad, x_max + pad, 150)
    ys = np.linspace(y_min - pad, y_max + pad, 150)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.full_like(xx, z_min - 0.020 * span)

    dx = (xx - (x_mid - 0.035 * span)) / (0.54 * x_extent)
    dy = (yy - (y_mid - 0.055 * span)) / (0.48 * y_extent)
    alpha = 0.20 * np.exp(-2.20 * (dx**2 + dy**2))
    alpha = np.clip(alpha, 0.0, 0.20)

    rgba = np.zeros((xx.size, 4), dtype=np.uint8)
    rgba[:, 0:3] = 30
    rgba[:, 3] = np.round(alpha.ravel(order="F") * 255).astype(np.uint8)

    shadow = pv.StructuredGrid(xx, yy, zz)
    shadow["Shadow_rgba"] = rgba
    return shadow


def configure_reference_lights(plotter: pv.Plotter, center: tuple[float, float, float], span: float, z_top: float) -> None:
    """配置接近参考图的西北侧主光和弱补光。"""
    plotter.remove_all_lights()
    x_mid, y_mid, z_mid = center
    key_light = pv.Light(
        position=(x_mid - 0.95 * span, y_mid - 1.15 * span, z_top + 1.25 * span),
        focal_point=(x_mid, y_mid, z_mid),
        color="white",
        intensity=1.25,
        positional=True,
    )
    key_light.cone_angle = 55
    fill_light = pv.Light(
        position=(x_mid + 0.90 * span, y_mid + 0.65 * span, z_top + 0.55 * span),
        focal_point=(x_mid, y_mid, z_mid),
        color="#F7F4ED",
        intensity=0.28,
        positional=True,
    )
    ambient_light = pv.Light(light_type="headlight", intensity=0.16)
    plotter.add_light(key_light)
    plotter.add_light(fill_light)
    plotter.add_light(ambient_light)


def style_bounds_actor(actor: pv.CubeAxesActor) -> None:
    """微调盒式坐标轴，使网格线、边框和文字接近参考图。"""
    for prop_getter in (
        actor.GetXAxesLinesProperty,
        actor.GetYAxesLinesProperty,
        actor.GetZAxesLinesProperty,
    ):
        prop = prop_getter()
        prop.SetColor(0.0, 0.0, 0.0)
        prop.SetLineWidth(1.2)

    for prop_getter in (
        actor.GetXAxesGridlinesProperty,
        actor.GetYAxesGridlinesProperty,
        actor.GetZAxesGridlinesProperty,
    ):
        prop = prop_getter()
        prop.SetColor(0.68, 0.68, 0.68)
        prop.SetLineWidth(0.7)

    for prop_getter in (
        actor.GetXAxesTitleProperty,
        actor.GetYAxesTitleProperty,
        actor.GetZAxesTitleProperty,
        actor.GetXAxesLabelProperty,
        actor.GetYAxesLabelProperty,
        actor.GetZAxesLabelProperty,
    ):
        prop = prop_getter()
        prop.SetColor(0.0, 0.0, 0.0)
        prop.SetFontFamilyToTimes()
        prop.SetBold(False)


def render_scene(args: argparse.Namespace) -> tuple[Path, Path]:
    z = np.load(args.dem)
    resolution_m = read_resolution(args.meta)
    grid = build_structured_grid(z, resolution_m, args.stride, args.vertical_exag, args.smooth_sigma)
    bounds = grid.bounds

    x_mid = 0.5 * (bounds[0] + bounds[1])
    y_mid = 0.5 * (bounds[2] + bounds[3])
    z_mid = 0.45 * bounds[5]
    span = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    # 华山 DEM 存在极少量尖峰值，色标沿用论文图的 99.7% 高程上限。
    elev_display_max = float(np.nanpercentile(z, 99.7))

    pv.global_theme.smooth_shading = True
    pv.global_theme.multi_samples = 8
    plotter = pv.Plotter(off_screen=True, window_size=(int(args.width), int(args.height)))
    plotter.set_background("white")
    plotter.add_mesh(
        build_ground_shadow(bounds, span),
        scalars="Shadow_rgba",
        rgb=True,
        opacity=1.0,
        show_scalar_bar=False,
        lighting=False,
        use_transparency=True,
    )
    plotter.add_mesh(
        grid,
        scalars="Elevation",
        cmap=REFERENCE_TERRAIN_CMAP,
        clim=(0.0, elev_display_max),
        show_scalar_bar=not args.hide_scalar_bar,
        scalar_bar_args={
            "title": "Elevation (m)",
            "color": "black",
            "title_font_size": 22,
            "label_font_size": 18,
            "font_family": "times",
            "vertical": True,
            "position_x": 0.755,
            "position_y": 0.12,
            "width": 0.030,
            "height": 0.65,
            "fmt": "%.0f",
            "outline": True,
            "n_labels": 7,
        },
        smooth_shading=True,
        split_sharp_edges=False,
        ambient=0.20,
        diffuse=0.84,
        specular=0.10,
        specular_power=18,
        pbr=False,
    )

    configure_reference_lights(plotter, (x_mid, y_mid, z_mid), span, bounds[5])
    plotter.enable_anti_aliasing("ssaa")
    plotter.enable_ssao(radius=0.22, bias=0.012, kernel_size=256, blur=True)

    if not args.hide_grid:
        actor = plotter.show_bounds(
            bounds=(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]),
            axes_ranges=(bounds[0], bounds[1], bounds[2], bounds[3], 0.0, elev_display_max),
            show_xlabels=True,
            show_ylabels=True,
            show_zlabels=True,
            xtitle="East-West (m)",
            ytitle="North-South (m)",
            ztitle="Elevation (m)",
            font_family="times",
            font_size=14,
            fmt="%.0f",
            n_xlabels=5,
            n_ylabels=5,
            n_zlabels=5,
            grid="back",
            location="outer",
            ticks="outside",
            all_edges=True,
            corner_factor=1.0,
            padding=0.0,
            use_3d_text=True,
            color="black",
        )
        style_bounds_actor(actor)

    camera_pos = (
        (x_mid - 1.34 * span, y_mid - 1.12 * span, bounds[5] + 1.10 * span),
        (x_mid + 0.01 * span, y_mid + 0.00 * span, 0.32 * bounds[5]),
        (0.0, 0.0, 1.0),
    )
    plotter.camera_position = camera_pos
    plotter.camera.view_angle = 32.0
    plotter.camera.clipping_range = (10.0, 6.5 * span)
    plotter.camera.zoom(0.68)

    peak_labels = []
    if not args.hide_peak_labels:
        plotter.render()
        peak_labels = project_peak_points(
            plotter,
            peak_world_points(
                config_or_path=args.scene,
                geo_path=args.geo,
                z=z,
                resolution_m=resolution_m,
                vertical_exag=args.vertical_exag,
                lift_scene_m=125.0,
            ),
            (int(args.width), int(args.height)),
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(plotter.screenshot(transparent_background=False, return_img=True)).convert("RGB")
    plotter.close()
    if peak_labels:
        image = draw_peak_annotations(image, peak_labels, reference_width=float(args.width))
    image.save(args.out)
    pdf_out = resolve_pdf_out(args.out, args.pdf_out)
    save_png_as_pdf(args.out, pdf_out, args.pdf_dpi)
    return args.out, pdf_out


def main() -> None:
    args = parse_args()
    png_out, pdf_out = render_scene(args)
    print(f"[完成] PyVista 华山 DEM 参考风格渲染图 PNG：{png_out}")
    print(f"[完成] PyVista 华山 DEM 参考风格渲染图 PDF：{pdf_out}")


if __name__ == "__main__":
    main()
