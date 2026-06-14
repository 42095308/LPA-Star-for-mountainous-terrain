"""根据参考图生成一组清晰透明的矢量图标资产。

图标采用手写 SVG 几何元素构造，不依赖栅格描摹。每个图标导出 SVG
和带 alpha 通道的 PNG，便于论文、PPT、Word 和网页场景中继续缩放使用。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import fitz
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "intermediate_artifacts" / "figures" / "reference_vector_icons"
SVG_DIR = OUT_DIR / "svg"
PNG_DIR = OUT_DIR / "png"
CONTACT_SHEET = OUT_DIR / "reference_vector_icons_contact_sheet.png"
MANIFEST = OUT_DIR / "reference_vector_icons_manifest.md"

INK = "#263746"
INK_SOFT = "#596B7B"
BLUE = "#2475CC"
BLUE_DARK = "#185799"
BLUE_LIGHT = "#DCEFFF"
BLUE_GRID = "#91BFE8"
GREEN = "#4D8C57"
GREEN_LIGHT = "#DDECD8"
ORANGE = "#E58A2A"
ORANGE_LIGHT = "#FFE2B8"
RED = "#E5534B"
RED_LIGHT = "#FFE1DC"
MAP_LIGHT = "#EDF3E7"
MAP_MID = "#D6E6CF"
MOUNTAIN = "#A7B0A9"
MOUNTAIN_LIGHT = "#ECE9DE"
MOUNTAIN_DARK = "#7E8A83"
PAPER = "#F5FAFF"
WHITE = "#FFFFFF"


@dataclass(frozen=True)
class IconSpec:
    file_stem: str
    title: str
    description: str
    width: int
    height: int
    builder: Callable[[], str]


def ensure_dirs() -> None:
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)


def tag(name: str, attrs: dict[str, object], body: str | None = None) -> str:
    attr_text = " ".join(f'{key}="{escape(str(value))}"' for key, value in attrs.items() if value is not None)
    if body is None:
        return f"<{name} {attr_text}/>"
    return f"<{name} {attr_text}>{body}</{name}>"


def group(body: list[str], opacity: float | None = None) -> str:
    return tag("g", {"opacity": opacity}, "\n".join(body))


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = INK,
    width: float = 8,
    dash: str | None = None,
    opacity: float | None = None,
) -> str:
    return tag(
        "line",
        {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "stroke": color,
            "stroke-width": width,
            "stroke-linecap": "round",
            "stroke-dasharray": dash,
            "opacity": opacity,
        },
    )


def polyline(
    points: list[tuple[float, float]],
    color: str = INK,
    width: float = 8,
    fill: str = "none",
    dash: str | None = None,
    opacity: float | None = None,
) -> str:
    return tag(
        "polyline",
        {
            "points": " ".join(f"{x},{y}" for x, y in points),
            "fill": fill,
            "stroke": color,
            "stroke-width": width,
            "stroke-linejoin": "round",
            "stroke-linecap": "round",
            "stroke-dasharray": dash,
            "opacity": opacity,
        },
    )


def polygon(
    points: list[tuple[float, float]],
    fill: str,
    color: str = INK,
    width: float = 6,
    opacity: float | None = None,
) -> str:
    return tag(
        "polygon",
        {
            "points": " ".join(f"{x},{y}" for x, y in points),
            "fill": fill,
            "stroke": color,
            "stroke-width": width,
            "stroke-linejoin": "round",
            "opacity": opacity,
        },
    )


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str,
    color: str = INK,
    width: float = 6,
    rx: float = 16,
    opacity: float | None = None,
) -> str:
    return tag(
        "rect",
        {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "rx": rx,
            "fill": fill,
            "stroke": color,
            "stroke-width": width,
            "opacity": opacity,
        },
    )


def circle(
    cx: float,
    cy: float,
    r: float,
    fill: str,
    color: str = INK,
    width: float = 6,
    opacity: float | None = None,
) -> str:
    return tag(
        "circle",
        {
            "cx": cx,
            "cy": cy,
            "r": r,
            "fill": fill,
            "stroke": color,
            "stroke-width": width,
            "opacity": opacity,
        },
    )


def path(
    d: str,
    fill: str = "none",
    color: str = INK,
    width: float = 8,
    dash: str | None = None,
    opacity: float | None = None,
) -> str:
    return tag(
        "path",
        {
            "d": d,
            "fill": fill,
            "stroke": color,
            "stroke-width": width,
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
            "stroke-dasharray": dash,
            "opacity": opacity,
        },
    )


def pin_path(cx: float, cy: float, scale: float, fill: str, stroke: str = INK) -> str:
    d = (
        f"M {cx} {cy + 116 * scale} "
        f"C {cx - 12 * scale} {cy + 96 * scale}, {cx - 72 * scale} {cy + 32 * scale}, {cx - 72 * scale} {cy - 22 * scale} "
        f"C {cx - 72 * scale} {cy - 66 * scale}, {cx - 40 * scale} {cy - 98 * scale}, {cx} {cy - 98 * scale} "
        f"C {cx + 40 * scale} {cy - 98 * scale}, {cx + 72 * scale} {cy - 66 * scale}, {cx + 72 * scale} {cy - 22 * scale} "
        f"C {cx + 72 * scale} {cy + 32 * scale}, {cx + 12 * scale} {cy + 96 * scale}, {cx} {cy + 116 * scale} Z"
    )
    return path(d, fill, stroke, 9)


def isometric_grid(origin_x: float, origin_y: float, width: float, height: float, color: str) -> list[str]:
    parts: list[str] = []
    left = (origin_x, origin_y + height * 0.5)
    top = (origin_x + width * 0.5, origin_y)
    right = (origin_x + width, origin_y + height * 0.5)
    bottom = (origin_x + width * 0.5, origin_y + height)
    for t in (0.22, 0.40, 0.58, 0.76):
        x1 = left[0] + (top[0] - left[0]) * t
        y1 = left[1] + (top[1] - left[1]) * t
        x2 = bottom[0] + (right[0] - bottom[0]) * t
        y2 = bottom[1] + (right[1] - bottom[1]) * t
        parts.append(line(x1, y1, x2, y2, color, 4, opacity=0.85))
        x3 = top[0] + (right[0] - top[0]) * t
        y3 = top[1] + (right[1] - top[1]) * t
        x4 = left[0] + (bottom[0] - left[0]) * t
        y4 = left[1] + (bottom[1] - left[1]) * t
        parts.append(line(x3, y3, x4, y4, color, 4, opacity=0.85))
    return parts


def mountain_strip(x: float, y: float, scale: float = 1.0) -> list[str]:
    def pts(items: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return [(x + px * scale, y + py * scale) for px, py in items]

    parts = [
        polygon(pts([(0, 150), (82, 40), (156, 158)]), "#EEEDE4", MOUNTAIN_DARK, 4),
        polygon(pts([(82, 40), (156, 158), (104, 142)]), "#B6C0B6", MOUNTAIN_DARK, 3),
        polygon(pts([(82, 40), (52, 100), (80, 88), (92, 120), (104, 75)]), "#FFFDF2", "none", 0),
        polygon(pts([(118, 162), (226, 0), (356, 170)]), "#E8E7DC", MOUNTAIN_DARK, 5),
        polygon(pts([(226, 0), (356, 170), (260, 134)]), "#AEB8AF", MOUNTAIN_DARK, 3),
        polygon(pts([(226, 0), (180, 82), (218, 66), (246, 116), (256, 48)]), "#FFFDF2", "none", 0),
        polygon(pts([(316, 160), (410, 24), (512, 162)]), "#E3E4DA", MOUNTAIN_DARK, 4),
        polygon(pts([(410, 24), (512, 162), (436, 142)]), "#A8B4AA", MOUNTAIN_DARK, 3),
        polygon(pts([(410, 24), (376, 88), (404, 76), (424, 112), (438, 60)]), "#F8F6EB", "none", 0),
        polygon(pts([(70, 166), (182, 112), (300, 174), (210, 210)]), "#CCD5CA", "#89958D", 3, 0.95),
        polyline(pts([(226, 0), (228, 70), (224, 132), (246, 196)]), "#7E8A83", 3, opacity=0.72),
        polyline(pts([(82, 40), (86, 92), (76, 154)]), "#7E8A83", 3, opacity=0.62),
        polyline(pts([(410, 24), (398, 84), (420, 168)]), "#7E8A83", 3, opacity=0.62),
        path(
            f"M {x + 10 * scale} {y + 168 * scale} "
            f"C {x + 92 * scale} {y + 190 * scale}, {x + 172 * scale} {y + 160 * scale}, {x + 250 * scale} {y + 190 * scale} "
            f"C {x + 330 * scale} {y + 220 * scale}, {x + 430 * scale} {y + 170 * scale}, {x + 520 * scale} {y + 154 * scale}",
            "none",
            "#6F7F78",
            3,
            opacity=0.62,
        ),
    ]
    return parts


def iso_cell(origin: tuple[float, float], i: int, j: int, sx: float, sy: float) -> list[tuple[float, float]]:
    ox, oy = origin
    x0 = ox + (i - j) * sx
    y0 = oy + (i + j) * sy
    return [(x0, y0), (x0 + sx, y0 + sy), (x0, y0 + 2 * sy), (x0 - sx, y0 + sy)]


def person_icon(cx: float, cy: float, scale: float = 1.0, color: str = INK) -> list[str]:
    return [
        circle(cx, cy - 24 * scale, 11 * scale, color, color, 3),
        line(cx, cy - 10 * scale, cx, cy + 34 * scale, color, 7 * scale),
        line(cx - 25 * scale, cy + 4 * scale, cx + 25 * scale, cy + 4 * scale, color, 6 * scale),
        line(cx, cy + 34 * scale, cx - 18 * scale, cy + 62 * scale, color, 6 * scale),
        line(cx, cy + 34 * scale, cx + 20 * scale, cy + 60 * scale, color, 6 * scale),
    ]


def icon_terrain_surface() -> str:
    defs = """
    <defs>
      <clipPath id="terrain_tile_clip">
        <polygon points="48,220 256,88 464,220 260,342"/>
      </clipPath>
    </defs>
    """
    tile = [
        polygon([(48, 220), (256, 88), (464, 220), (260, 342)], "#EDF2E7", INK, 7),
        polygon([(48, 220), (260, 342), (260, 362), (46, 242)], "#CAD8C9", INK_SOFT, 4, 0.86),
        polygon([(260, 342), (464, 220), (464, 242), (260, 362)], "#BFD0BD", INK_SOFT, 4, 0.86),
    ]
    clipped_grid = [
        *isometric_grid(48, 88, 416, 254, "#BAC8B6"),
        path("M 72 236 C 126 204, 178 214, 230 185 C 278 158, 340 174, 420 145", "none", "#9AAC9E", 4, "10 10", 0.55),
        path("M 78 262 C 136 234, 182 246, 244 214 C 306 184, 360 206, 440 180", "none", "#A9BAA9", 4, "10 11", 0.55),
        path("M 96 286 C 164 264, 218 272, 276 238 C 328 208, 390 232, 446 210", "none", "#9FB29E", 4, "10 11", 0.55),
    ]
    terrain = [
        polygon([(92, 248), (154, 124), (206, 274), (146, 262)], "#ECEDE2", MOUNTAIN_DARK, 5),
        polygon([(154, 124), (206, 274), (174, 248)], "#B8C2B8", MOUNTAIN_DARK, 4),
        polygon([(154, 124), (120, 202), (148, 187), (164, 226), (174, 169)], "#FAF8EF", "none", 0),
        polygon([(168, 276), (254, 42), (364, 278), (270, 322)], "#E9E7DC", MOUNTAIN_DARK, 7),
        polygon([(254, 42), (364, 278), (292, 244)], "#AEB9AE", MOUNTAIN_DARK, 5),
        polygon([(168, 276), (254, 42), (250, 238), (222, 266)], "#F1EFE4", MOUNTAIN_DARK, 4),
        polygon([(254, 42), (214, 160), (250, 142), (276, 210), (286, 116)], "#FFFDF2", "none", 0),
        polygon([(250, 238), (292, 244), (270, 322), (222, 266)], "#C9D1C7", "#8B968F", 3),
        polygon([(302, 270), (370, 142), (430, 258), (378, 290)], "#DADFD4", MOUNTAIN_DARK, 5),
        polygon([(370, 142), (430, 258), (382, 244)], "#A8B5AA", MOUNTAIN_DARK, 4),
        polygon([(370, 142), (336, 208), (364, 196), (378, 226), (390, 178)], "#F4F2E8", "none", 0),
        polygon([(140, 280), (204, 228), (286, 298), (238, 330)], "#D4DCD0", "#8D998F", 4),
        polyline([(254, 42), (252, 118), (250, 196), (250, 238), (270, 322)], "#7D8981", 4, opacity=0.75),
        polyline([(154, 124), (156, 184), (146, 262)], "#7D8981", 3, opacity=0.65),
        polyline([(370, 142), (360, 202), (378, 290)], "#7D8981", 3, opacity=0.65),
        path("M 106 284 C 158 304, 214 292, 262 314 C 314 336, 374 292, 428 270", "none", "#6F7F78", 4, opacity=0.70),
        path("M 180 210 C 222 190, 270 202, 316 184", "none", "#C1CABF", 3, "8 9", 0.7),
        path("M 188 236 C 236 220, 294 230, 344 206", "none", "#B5C1B4", 3, "8 9", 0.7),
        path("M 214 264 C 260 250, 314 260, 372 232", "none", "#AAB8AA", 3, "8 9", 0.7),
    ]
    parts = [
        defs,
        group(tile),
        tag("g", {"clip-path": "url(#terrain_tile_clip)"}, "\n".join(clipped_grid)),
        group(terrain),
    ]
    return "\n".join(parts)


def icon_map_location() -> str:
    parts = [
        polygon([(72, 304), (214, 210), (434, 294), (288, 392)], MAP_LIGHT, INK, 7),
        polygon([(72, 304), (214, 210), (235, 270), (110, 350)], "#E3EBDD", "none", 0),
        polygon([(236, 270), (352, 238), (434, 294), (286, 392)], "#D7E7D2", "none", 0),
        line(126, 268, 326, 348, "#D8E0D3", 34),
        line(204, 226, 246, 382, "#D8E0D3", 30),
        line(90, 320, 255, 232, INK_SOFT, 9),
        line(254, 232, 430, 296, INK_SOFT, 9),
        line(124, 352, 250, 274, INK_SOFT, 9),
        line(250, 274, 386, 334, INK_SOFT, 9),
        line(206, 224, 250, 384, INK_SOFT, 9),
        path("M 116 306 C 162 274, 184 298, 212 276 C 246 250, 286 258, 328 276", "none", GREEN, 11),
        polygon([(98, 316), (158, 278), (194, 296), (132, 336)], GREEN_LIGHT, GREEN, 4),
        polygon([(286, 320), (368, 288), (418, 306), (336, 360)], "#E4F1DE", GREEN, 4),
        pin_path(257, 135, 0.92, "#57A5E8", BLUE_DARK),
        circle(257, 116, 28, WHITE, BLUE_DARK, 7),
    ]
    return "\n".join(parts)


def icon_communication_tower() -> str:
    base = [(96, 352), (256, 270), (416, 352), (256, 438)]
    parts = [
        polygon(base, BLUE_LIGHT, BLUE, 7),
        *isometric_grid(96, 270, 320, 168, BLUE_GRID),
        line(256, 112, 192, 344, INK, 9),
        line(256, 112, 320, 344, INK, 9),
        line(214, 344, 298, 344, INK, 9),
        line(238, 178, 278, 178, INK, 7),
        line(220, 244, 292, 244, INK, 7),
        line(236, 178, 286, 244, INK, 5),
        line(276, 178, 226, 244, INK, 5),
        line(220, 244, 316, 344, INK, 5),
        line(292, 244, 196, 344, INK, 5),
        circle(256, 100, 13, WHITE, INK, 7),
        line(256, 113, 256, 344, INK, 6),
        path("M 198 112 C 164 148, 164 208, 198 242", "none", BLUE, 8),
        path("M 314 112 C 348 148, 348 208, 314 242", "none", BLUE, 8),
        path("M 166 78 C 110 138, 110 218, 166 278", "none", BLUE, 6, opacity=0.70),
        path("M 346 78 C 402 138, 402 218, 346 278", "none", BLUE, 6, opacity=0.70),
    ]
    return "\n".join(parts)


def icon_route_uav() -> str:
    parts = [
        pin_path(74, 88, 0.72, "#2B3D4D", INK),
        circle(74, 73, 22, WHITE, INK, 6),
        path("M 92 184 C 178 142, 224 216, 304 166 C 384 116, 448 126, 520 98 C 590 70, 662 94, 720 64", "none", BLUE_DARK, 9),
        path("M 148 162 C 220 128, 262 196, 334 145 C 404 96, 464 112, 530 88", "none", INK_SOFT, 5, "13 16", 0.65),
        circle(302, 166, 7, WHITE, BLUE_DARK, 5),
        circle(448, 122, 7, WHITE, BLUE_DARK, 5),
        line(538, 86, 616, 86, INK, 8),
        line(577, 66, 577, 108, INK, 8),
        rect(548, 76, 58, 22, "#EAF4FF", INK, 6, 8),
        circle(532, 86, 13, WHITE, INK, 6),
        circle(622, 86, 13, WHITE, INK, 6),
        circle(577, 58, 13, WHITE, INK, 6),
        circle(577, 114, 13, WHITE, INK, 6),
        line(606, 86, 676, 70, INK, 6),
        line(606, 90, 674, 104, INK, 6),
        line(672, 70, 718, 54, INK, 5),
        line(674, 104, 724, 104, INK, 5),
        circle(724, 54, 10, WHITE, INK, 5),
        circle(728, 104, 10, WHITE, INK, 5),
        pin_path(710, 166, 0.42, GREEN, GREEN),
        circle(710, 156, 12, WHITE, GREEN, 5),
    ]
    return "\n".join(parts)


def icon_parameter_checklist() -> str:
    parts = [
        rect(138, 92, 236, 340, PAPER, BLUE, 10, 22),
        rect(198, 66, 116, 52, WHITE, BLUE, 10, 14),
        rect(210, 86, 92, 18, "#E7F3FF", "none", 0, 8),
        rect(170, 164, 34, 34, WHITE, BLUE, 6, 6),
        polyline([(177, 181), (187, 192), (202, 170)], GREEN, 7),
        line(228, 181, 320, 181, INK_SOFT, 7),
        rect(170, 222, 34, 34, WHITE, BLUE, 6, 6),
        polyline([(177, 239), (187, 250), (202, 228)], GREEN, 7),
        line(228, 239, 326, 239, INK_SOFT, 7),
        rect(170, 280, 34, 34, WHITE, BLUE, 6, 6),
        polyline([(177, 297), (187, 308), (202, 286)], GREEN, 7),
        line(228, 297, 316, 297, INK_SOFT, 7),
        line(226, 356, 306, 356, INK_SOFT, 7),
    ]
    return "\n".join(parts)


def icon_terrain_voxel_block() -> str:
    roof = [(72, 104), (224, 24), (592, 40), (724, 118), (574, 198), (202, 184)]
    parts = [
        polygon([(72, 104), (202, 184), (202, 288), (72, 210)], "#D4D8D7", INK_SOFT, 5, 0.92),
        polygon([(202, 184), (574, 198), (574, 306), (202, 288)], "#C6CCCB", INK_SOFT, 5, 0.92),
        polygon([(574, 198), (724, 118), (724, 222), (574, 306)], "#B8C0BF", INK_SOFT, 5, 0.92),
        polygon(roof, "#EEF0ED", INK, 6, 0.94),
        line(118, 80, 246, 158, "#AEB6B5", 4),
        line(174, 54, 314, 172, "#AEB6B5", 4),
        line(248, 26, 392, 188, "#AEB6B5", 4),
        line(340, 28, 488, 192, "#AEB6B5", 4),
        line(456, 34, 608, 178, "#AEB6B5", 4),
        line(174, 156, 320, 78, "#AEB6B5", 4),
        line(252, 182, 424, 92, "#AEB6B5", 4),
        line(360, 188, 536, 94, "#AEB6B5", 4),
        line(482, 194, 650, 104, "#AEB6B5", 4),
        polygon([(592, 40), (652, 8), (706, 40), (644, 72)], "#F4F5F2", INK, 5),
        polygon([(644, 72), (706, 40), (706, 128), (644, 162)], "#C3CBC9", INK_SOFT, 5),
        polygon([(592, 40), (644, 72), (644, 162), (592, 132)], "#D8DDDB", INK_SOFT, 5),
        line(616, 54, 670, 24, "#AEB6B5", 3),
        line(620, 90, 686, 58, "#AEB6B5", 3),
        *mountain_strip(54, 144, 1.06),
        path("M 62 316 C 170 340, 286 306, 400 338 C 512 368, 616 314, 720 300", "none", "#73817A", 4, opacity=0.58),
    ]
    return "\n".join(parts)


def icon_mid_surface_waves() -> str:
    parts: list[str] = []
    for y, color in ((70, BLUE), (158, GREEN), (246, RED)):
        parts.extend(
            [
                path(
                    f"M 48 {y} C 98 {y - 18}, 142 {y + 18}, 194 {y} "
                    f"C 246 {y - 18}, 292 {y + 18}, 344 {y} C 392 {y - 16}, 430 {y - 8}, 464 {y - 12}",
                    "none",
                    color,
                    10,
                ),
                path(
                    f"M 48 {y + 34} C 98 {y + 16}, 142 {y + 52}, 194 {y + 34} "
                    f"C 246 {y + 16}, 292 {y + 52}, 344 {y + 34} C 392 {y + 18}, 430 {y + 26}, 464 {y + 22}",
                    "none",
                    color,
                    8,
                    opacity=0.82,
                ),
            ]
        )
    return "\n".join(parts)


def icon_blue_corridor_envelope() -> str:
    parts = [
        *mountain_strip(42, 112, 0.86),
        path("M 96 78 C 188 42, 286 72, 376 48 C 462 25, 536 54, 596 42 L 606 160 C 536 178, 462 148, 368 178 C 274 208, 180 174, 84 212 Z", BLUE_LIGHT, "none", 0, opacity=0.34),
        path("M 96 78 C 188 42, 286 72, 376 48 C 462 25, 536 54, 596 42", "none", BLUE, 6, "12 10"),
        path("M 84 212 C 180 174, 274 208, 368 178 C 462 148, 536 178, 606 160", "none", BLUE, 6, "12 10"),
        path("M 96 78 L 84 212", "none", BLUE, 5, "9 10"),
        path("M 596 42 L 606 160", "none", BLUE, 5, "9 10"),
        path("M 154 98 L 146 224", "none", BLUE, 4, "8 10", 0.65),
        path("M 318 72 L 310 206", "none", BLUE, 4, "8 10", 0.65),
        path("M 476 62 L 486 184", "none", BLUE, 4, "8 10", 0.65),
    ]
    return "\n".join(parts)


def icon_red_corridor_envelope() -> str:
    parts = [
        *mountain_strip(42, 112, 0.86),
        path("M 96 78 C 188 42, 286 72, 376 48 C 462 25, 536 54, 596 42", "none", RED, 6, "12 10"),
        path("M 84 212 C 180 174, 274 208, 368 178 C 462 148, 536 178, 606 160", "none", RED, 6, "12 10"),
        path("M 96 78 L 84 212", "none", RED, 5, "9 10"),
        path("M 596 42 L 606 160", "none", RED, 5, "9 10"),
        path("M 96 78 C 188 42, 286 72, 376 48 C 462 25, 536 54, 596 42 L 606 160 C 536 178, 462 148, 368 178 C 274 208, 180 174, 84 212 Z", RED_LIGHT, "none", 0, opacity=0.36),
        path("M 154 98 L 146 224", "none", RED, 4, "8 10", 0.65),
        path("M 318 72 L 310 206", "none", RED, 4, "8 10", 0.65),
        path("M 476 62 L 486 184", "none", RED, 4, "8 10", 0.65),
    ]
    return "\n".join(parts)


def icon_communication_risk_fan() -> str:
    base = [(96, 358), (256, 278), (416, 358), (256, 440)]
    parts = [
        polygon(base, BLUE_LIGHT, BLUE, 7),
        *isometric_grid(96, 278, 320, 162, BLUE_GRID),
        line(254, 118, 190, 350, INK, 9),
        line(254, 118, 318, 350, INK, 9),
        line(214, 350, 298, 350, INK, 9),
        line(236, 184, 276, 184, INK, 7),
        line(220, 250, 292, 250, INK, 7),
        line(236, 184, 286, 250, INK, 5),
        line(276, 184, 226, 250, INK, 5),
        line(254, 132, 254, 350, INK, 6),
        circle(254, 104, 13, WHITE, INK, 7),
        path("M 190 118 C 154 154, 154 214, 190 250", "none", BLUE, 8),
        path("M 158 86 C 102 146, 102 226, 158 286", "none", BLUE, 6, opacity=0.70),
        path("M 272 220 L 424 146 L 424 292 Z", RED_LIGHT, "none", 0, opacity=0.62),
        path("M 272 220 L 420 156", "none", RED, 8),
        path("M 272 220 L 420 286", "none", RED, 5, opacity=0.48),
        path("M 320 200 C 348 214, 348 232, 320 246", "none", RED, 5, opacity=0.48),
    ]
    return "\n".join(parts)


def icon_human_exposure_heatmap() -> str:
    origin = (258, 84)
    colors = [
        ["#FFE7B8", "#FFD991", "#FFD07B", "#F6BA64", "#EB9950"],
        ["#F9D793", "#F6C46F", "#F0A95D", "#EA8650", "#DF6C4C"],
        ["#F1BF73", "#EFA65F", "#EA884F", "#E86549", "#D94F43"],
        ["#EAA35F", "#E9884F", "#E26C49", "#D85744", "#CB4940"],
        ["#DD8754", "#DC704B", "#D95C45", "#CA4D3E", "#BA4238"],
    ]
    parts: list[str] = []
    for i in range(5):
        for j in range(5):
            parts.append(polygon(iso_cell(origin, i, j, 42, 22), colors[j][i], "#9B8A62", 3))
    outline = [
        iso_cell(origin, 0, 0, 42, 22)[0],
        iso_cell(origin, 4, 0, 42, 22)[1],
        iso_cell(origin, 4, 4, 42, 22)[2],
        iso_cell(origin, 0, 4, 42, 22)[3],
    ]
    parts.extend(
        [
            polygon(outline, "none", INK, 7),
            line(258, 84, 426, 172, "#8E835E", 4, opacity=0.7),
            line(216, 106, 384, 194, "#8E835E", 4, opacity=0.7),
            line(174, 128, 342, 216, "#8E835E", 4, opacity=0.7),
            line(300, 106, 132, 194, "#8E835E", 4, opacity=0.7),
            line(342, 128, 174, 216, "#8E835E", 4, opacity=0.7),
            *person_icon(236, 160, 0.75, INK),
            *person_icon(300, 190, 0.78, INK),
            *person_icon(354, 232, 0.72, INK),
        ]
    )
    return "\n".join(parts)


def icon_terrain_risk_surface() -> str:
    origin = (256, 76)
    base_outline = [
        iso_cell(origin, 0, 0, 31, 16)[0],
        iso_cell(origin, 6, 0, 31, 16)[1],
        iso_cell(origin, 6, 6, 31, 16)[2],
        iso_cell(origin, 0, 6, 31, 16)[3],
    ]
    parts: list[str] = [polygon(base_outline, "#91CB68", "none", 0)]
    for i in range(7):
        for j in range(7):
            distance = abs(i - 3) + abs(j - 4)
            if distance <= 1:
                color = "#E86A42"
            elif distance == 2:
                color = "#F3B850"
            elif distance == 3:
                color = "#D7DA63"
            else:
                color = "#91CB68"
            cell = iso_cell(origin, i, j, 31, 16)
            lift = max(0, 42 - 7 * ((i - 2.5) ** 2 + (j - 3.5) ** 2))
            cell = [(px, py - lift * 0.28) for px, py in cell]
            parts.append(polygon(cell, color, "#668A59", 3))
    outline = [(px, py + 3) for px, py in base_outline]
    parts.extend(
        [
            polygon(outline, "none", INK, 7),
            path("M 104 210 C 180 180, 246 206, 306 166 C 366 126, 428 164, 462 150", "none", "#5F8E54", 5, opacity=0.7),
            path("M 132 242 C 210 214, 278 234, 342 198 C 390 170, 430 192, 470 182", "none", "#5F8E54", 4, opacity=0.55),
        ]
    )
    return "\n".join(parts)


ICONS = [
    IconSpec("reference_terrain_surface", "地形网格", "参考图一，对应等距地形与高程面", 512, 384, icon_terrain_surface),
    IconSpec("reference_map_location", "地图定位", "参考图二，对应地图分区与任务位置", 512, 512, icon_map_location),
    IconSpec("reference_communication_tower", "通信源", "参考图三，对应通信塔和覆盖信号", 512, 512, icon_communication_tower),
    IconSpec("reference_route_uav", "航线无人机", "参考图四，对应起点、飞行轨迹和无人机", 768, 256, icon_route_uav),
    IconSpec("reference_parameter_checklist", "参数清单", "参考图五，对应参数核验和配置清单", 512, 512, icon_parameter_checklist),
    IconSpec("reference_terrain_voxel_block", "地形体素块", "新增参考图一，对应分层体素地形与山体", 768, 384, icon_terrain_voxel_block),
    IconSpec("reference_mid_surface_waves", "三层中面", "新增参考图二，对应三层飞行中面", 512, 320, icon_mid_surface_waves),
    IconSpec("reference_blue_corridor_envelope", "走廊上包络", "新增参考图三，对应蓝色安全走廊包络", 640, 320, icon_blue_corridor_envelope),
    IconSpec("reference_red_corridor_envelope", "走廊下包络", "新增参考图四，对应红色安全走廊边界", 640, 320, icon_red_corridor_envelope),
    IconSpec("reference_communication_risk_fan", "通信风险扇区", "新增参考图五，对应通信源遮挡风险扇区", 512, 512, icon_communication_risk_fan),
    IconSpec("reference_human_exposure_heatmap", "人员暴露热区", "新增参考图六，对应人员暴露风险场", 512, 512, icon_human_exposure_heatmap),
    IconSpec("reference_terrain_risk_surface", "地形风险热场", "新增参考图七，对应地形风险热力面", 512, 384, icon_terrain_risk_surface),
]


def wrap_svg(spec: IconSpec, body: str) -> str:
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{spec.width}" height="{spec.height}" '
                f'viewBox="0 0 {spec.width} {spec.height}" shape-rendering="geometricPrecision">'
            ),
            body,
            "</svg>",
        ]
    )


def render_png(svg_path: Path, png_path: Path) -> None:
    doc = fitz.open(str(svg_path))
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(3, 3), alpha=True)
    pix.save(str(png_path))
    doc.close()


def generate_icons() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in ICONS:
        svg_path = SVG_DIR / f"{spec.file_stem}.svg"
        png_path = PNG_DIR / f"{spec.file_stem}.png"
        svg_path.write_text(wrap_svg(spec, spec.builder()), encoding="utf-8")
        render_png(svg_path, png_path)
        rows.append(
            {
                "file_stem": spec.file_stem,
                "title": spec.title,
                "description": spec.description,
                "svg": str(svg_path),
                "png": str(png_path),
                "width": str(spec.width),
                "height": str(spec.height),
            }
        )
    return rows


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for item in font_candidates:
        path_obj = Path(item)
        if path_obj.exists():
            return ImageFont.truetype(str(path_obj), size=size)
    return ImageFont.load_default()


def make_contact_sheet(rows: list[dict[str, str]]) -> None:
    cell_w = 330
    cell_h = 260
    cols = 3
    sheet_rows = (len(rows) + cols - 1) // cols
    image = Image.new("RGBA", (cell_w * cols, cell_h * sheet_rows), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    title_font = load_font(24)
    name_font = load_font(15)
    for i, row in enumerate(rows):
        col = i % cols
        row_index = i // cols
        x = col * cell_w
        y = row_index * cell_h
        icon = Image.open(row["png"]).convert("RGBA")
        icon.thumbnail((220, 150))
        image.paste(icon, (x + (cell_w - icon.width) // 2, y + 20), icon)
        draw.text((x + 26, y + 184), row["title"], fill=INK, font=title_font)
        draw.text((x + 26, y + 218), row["file_stem"], fill=INK_SOFT, font=name_font)
    image.save(CONTACT_SHEET)


def write_manifest(rows: list[dict[str, str]]) -> None:
    lines = [
        "# 参考矢量图标清单",
        "",
        "本目录根据用户提供的参考图生成一套独立图标。每个图标均包含 SVG 和透明 PNG，SVG 可继续编辑，PNG 以三倍分辨率从 SVG 渲染，适合直接插入文档和幻灯片。",
        "",
        f"图标总览：`{CONTACT_SHEET.name}`",
        "",
        "| 文件名 | 图标含义 | 说明 | SVG | PNG | 尺寸 |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        svg_rel = Path(row["svg"]).relative_to(OUT_DIR)
        png_rel = Path(row["png"]).relative_to(OUT_DIR)
        lines.append(
            f"| `{row['file_stem']}` | {row['title']} | {row['description']} | "
            f"`{svg_rel}` | `{png_rel}` | {row['width']} x {row['height']} |"
        )
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    rows = generate_icons()
    make_contact_sheet(rows)
    write_manifest(rows)
    print(f"输出目录：{OUT_DIR}")
    print(f"图标数量：{len(rows)}")
    print(f"总览图：{CONTACT_SHEET}")
    print(f"清单：{MANIFEST}")


if __name__ == "__main__":
    main()
