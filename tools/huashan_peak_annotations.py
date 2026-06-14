"""华山五峰单字母标注工具。

该模块只负责把场景配置中的五峰经纬度转换为图面位置，并在截图上绘制轻量标注。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree


DEFAULT_3D_OFFSETS: Mapping[str, tuple[float, float]] = {
    "S": (86.0, -118.0),
    "E": (138.0, -34.0),
    "W": (-138.0, -44.0),
    "N": (-96.0, -126.0),
    "C": (18.0, -154.0),
}

MAP_OFFSETS: Mapping[str, tuple[float, float]] = {
    "S": (44.0, -52.0),
    "E": (58.0, 20.0),
    "W": (-62.0, -10.0),
    "N": (-54.0, -58.0),
    "C": (8.0, 58.0),
}


@dataclass(frozen=True)
class PeakTarget:
    """场景配置中的单个峰顶目标。"""

    name: str
    display_name: str
    letter: str
    lon: float
    lat: float
    elev: float | None


@dataclass(frozen=True)
class PeakWorldPoint:
    """三维渲染坐标系中的峰顶标注锚点。"""

    target: PeakTarget
    point: tuple[float, float, float]


@dataclass(frozen=True)
class PeakScreenPoint:
    """二维截图坐标系中的峰顶标注锚点。"""

    letter: str
    xy: tuple[float, float]


def _letter_from_target(name: str, display_name: str) -> str | None:
    text = f"{name} {display_name}".lower()
    if "south" in text or "南" in text:
        return "S"
    if "east" in text or "东" in text or "東" in text:
        return "E"
    if "west" in text or "西" in text:
        return "W"
    if "north" in text or "北" in text:
        return "N"
    if "central" in text or "center" in text or "中" in text:
        return "C"
    return None


def load_peak_targets(config_or_path: Mapping[str, object] | Path) -> list[PeakTarget]:
    """读取华山五峰目标，并转换为 S、E、W、N、C 五个单字母。"""
    if isinstance(config_or_path, Path):
        config = json.loads(config_or_path.read_text(encoding="utf-8"))
    else:
        config = dict(config_or_path)

    targets = config.get("targets", {})
    if not isinstance(targets, Mapping):
        return []

    records: list[PeakTarget] = []
    for raw_name, raw_meta in targets.items():
        if not isinstance(raw_meta, Mapping):
            continue
        name = str(raw_name)
        display_name = str(raw_meta.get("display_name", name))
        letter = _letter_from_target(name, display_name)
        if letter is None:
            continue
        try:
            lon = float(raw_meta["lon"])
            lat = float(raw_meta["lat"])
        except Exception:
            continue
        elev_raw = raw_meta.get("elev")
        elev = float(elev_raw) if elev_raw is not None else None
        records.append(PeakTarget(name=name, display_name=display_name, letter=letter, lon=lon, lat=lat, elev=elev))

    order = {"S": 0, "E": 1, "W": 2, "N": 3, "C": 4}
    deduped: dict[str, PeakTarget] = {}
    for record in records:
        deduped.setdefault(record.letter, record)
    return sorted(deduped.values(), key=lambda item: order.get(item.letter, 99))


def peak_world_points(
    *,
    config_or_path: Mapping[str, object] | Path,
    geo_path: Path,
    z: np.ndarray,
    resolution_m: float,
    vertical_exag: float,
    lift_scene_m: float = 120.0,
) -> list[PeakWorldPoint]:
    """把五峰经纬度转换为 PyVista 场景坐标。"""
    geo = np.load(geo_path)
    lon_grid = np.asarray(geo["lon_grid"], dtype=float)
    lat_grid = np.asarray(geo["lat_grid"], dtype=float)
    rows, cols = z.shape
    tree = cKDTree(np.column_stack([lon_grid.ravel(), lat_grid.ravel()]))
    base_elev = float(np.nanmin(z))

    points: list[PeakWorldPoint] = []
    for target in load_peak_targets(config_or_path):
        _, flat_idx = tree.query(np.asarray([target.lon, target.lat], dtype=float), k=1)
        row = int(flat_idx) // cols
        col = int(flat_idx) % cols
        row = int(np.clip(row, 0, rows - 1))
        col = int(np.clip(col, 0, cols - 1))
        elev = float(z[row, col])
        if not math.isfinite(elev) and target.elev is not None:
            elev = float(target.elev)
        x = (float(col) - 0.5 * float(cols - 1)) * float(resolution_m)
        y = (float(rows - 1 - row) - 0.5 * float(rows - 1)) * float(resolution_m)
        z_scene = max(elev - base_elev, 0.0) * float(vertical_exag) + float(lift_scene_m)
        points.append(PeakWorldPoint(target=target, point=(x, y, z_scene)))
    return points


def project_peak_points(plotter, peak_points: Sequence[PeakWorldPoint], window_size: tuple[int, int]) -> list[PeakScreenPoint]:
    """把三维峰顶锚点投影到截图像素坐标。"""
    renderer = plotter.renderer
    height = float(window_size[1])
    projected: list[PeakScreenPoint] = []
    for peak in peak_points:
        x, y, z = peak.point
        renderer.SetWorldPoint(float(x), float(y), float(z), 1.0)
        renderer.WorldToDisplay()
        display = renderer.GetDisplayPoint()
        projected.append(PeakScreenPoint(letter=peak.target.letter, xy=(float(display[0]), height - float(display[1]))))
    return projected


def map_peak_screen_points(
    config_or_path: Mapping[str, object] | Path,
    lonlat_to_pixel,
) -> list[PeakScreenPoint]:
    """把五峰经纬度转换为二维地图截图坐标。"""
    points: list[PeakScreenPoint] = []
    for target in load_peak_targets(config_or_path):
        x, y = lonlat_to_pixel(float(target.lon), float(target.lat))
        points.append(PeakScreenPoint(letter=target.letter, xy=(float(x), float(y))))
    return points


def _load_font(size_px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/timesbd.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size_px)
    return ImageFont.load_default()


def _draw_thin_arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], width: int, scale: float) -> None:
    color = (30, 30, 30, 205)
    draw.line([start, end], fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head_len = max(5.0, 9.0 * scale)
    head_w = max(2.5, 4.2 * scale)
    back_x = end[0] - head_len * math.cos(angle)
    back_y = end[1] - head_len * math.sin(angle)
    left = (back_x + head_w * math.sin(angle), back_y - head_w * math.cos(angle))
    right = (back_x - head_w * math.sin(angle), back_y + head_w * math.cos(angle))
    draw.polygon([end, left, right], fill=color)


def draw_peak_annotations(
    image: Image.Image | np.ndarray,
    peaks: Iterable[PeakScreenPoint],
    *,
    offsets: Mapping[str, tuple[float, float]] | None = None,
    reference_width: float = 3600.0,
    font_size_px: int | None = None,
) -> Image.Image:
    """在截图上绘制五峰单字母和细箭头。"""
    base = Image.fromarray(image).convert("RGBA") if isinstance(image, np.ndarray) else image.convert("RGBA")
    width, height = base.size
    scale = max(0.45, min(1.25, width / float(reference_width)))
    font = _load_font(font_size_px or max(22, int(round(48 * scale))))
    line_width = max(1, int(round(1.45 * scale)))
    offset_table = offsets or DEFAULT_3D_OFFSETS

    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    for peak in peaks:
        x, y = peak.xy
        if x < -40 or x > width + 40 or y < -40 or y > height + 40:
            continue
        dx, dy = offset_table.get(peak.letter, (68.0, -78.0))
        lx = float(np.clip(x + dx * scale, 24.0, width - 24.0))
        ly = float(np.clip(y + dy * scale, 24.0, height - 24.0))
        _draw_thin_arrow(draw, (lx, ly), (x, y), line_width, scale)
        bbox = draw.textbbox((0, 0), peak.letter, font=font, stroke_width=max(1, int(round(1.8 * scale))))
        tx = lx - 0.5 * (bbox[2] - bbox[0])
        ty = ly - 0.5 * (bbox[3] - bbox[1])
        draw.text(
            (tx, ty),
            peak.letter,
            font=font,
            fill=(12, 12, 12, 235),
            stroke_width=max(1, int(round(1.8 * scale))),
            stroke_fill=(255, 255, 255, 220),
        )

    return Image.alpha_composite(base, overlay).convert("RGB")
