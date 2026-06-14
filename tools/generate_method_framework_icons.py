"""生成论文方法框架图的小图标资产。

图标服务于山地无人机动态航路规划方法框架图，覆盖环境输入、风险场、
安全飞行走廊、三层航路网络、事件驱动增量更新和路径后处理等模块。
每个图标同时导出 SVG 和透明 PNG，并生成一张总览图便于人工挑选。
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import fitz
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "intermediate_artifacts" / "figures" / "method_framework_icons"
SVG_DIR = OUT_DIR / "svg"
PNG_DIR = OUT_DIR / "png"
CONTACT_SHEET = OUT_DIR / "method_framework_icons_contact_sheet.png"
MANIFEST = OUT_DIR / "method_framework_icons_manifest.md"

SIZE = 512

BLUE = "#1F65B7"
BLUE_LIGHT = "#DCEBFA"
BLUE_MID = "#87B8E8"
GREEN = "#3F8F57"
GREEN_LIGHT = "#DFF0DD"
ORANGE = "#E08A21"
ORANGE_LIGHT = "#F8E3BE"
RED = "#D33E35"
RED_LIGHT = "#F5D6D3"
INK = "#213040"
MUTED = "#6B7887"
GRID = "#C9D7E5"
MOUNTAIN = "#9FA8A3"
MOUNTAIN_LIGHT = "#E8E6DC"
WHITE = "#FFFFFF"


def ensure_dirs() -> None:
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)


def tag(name: str, attrs: dict[str, object], body: str | None = None) -> str:
    attr_text = " ".join(f'{key}="{escape(str(value))}"' for key, value in attrs.items() if value is not None)
    if body is None:
        return f"<{name} {attr_text}/>"
    return f"<{name} {attr_text}>{body}</{name}>"


def line(x1: float, y1: float, x2: float, y2: float, color: str = INK, width: float = 10, dash: str | None = None) -> str:
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
        },
    )


def polyline(points: list[tuple[float, float]], color: str = INK, width: float = 10, fill: str = "none", dash: str | None = None) -> str:
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
        },
    )


def polygon(points: list[tuple[float, float]], fill: str, color: str = INK, width: float = 7, opacity: float | None = None) -> str:
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


def circle(cx: float, cy: float, r: float, fill: str, color: str = WHITE, width: float = 6, opacity: float | None = None) -> str:
    return tag("circle", {"cx": cx, "cy": cy, "r": r, "fill": fill, "stroke": color, "stroke-width": width, "opacity": opacity})


def rect(x: float, y: float, w: float, h: float, fill: str, color: str = INK, width: float = 7, rx: float = 16, opacity: float | None = None) -> str:
    return tag("rect", {"x": x, "y": y, "width": w, "height": h, "rx": rx, "fill": fill, "stroke": color, "stroke-width": width, "opacity": opacity})


def text(x: float, y: float, content: str, size: float = 48, color: str = INK, weight: int = 700, anchor: str = "middle") -> str:
    return tag(
        "text",
        {
            "x": x,
            "y": y,
            "text-anchor": anchor,
            "font-family": "Arial, Helvetica, sans-serif",
            "font-size": size,
            "font-weight": weight,
            "fill": color,
        },
        escape(content),
    )


def path(d: str, fill: str = "none", color: str = INK, width: float = 10, dash: str | None = None, opacity: float | None = None) -> str:
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


def arrow_defs() -> str:
    return """
    <defs>
      <marker id="arrow_blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#1F65B7"/>
      </marker>
      <marker id="arrow_ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#213040"/>
      </marker>
    </defs>
    """


def arrow(x1: float, y1: float, x2: float, y2: float, color: str = BLUE, width: float = 9) -> str:
    marker = "url(#arrow_blue)" if color == BLUE else "url(#arrow_ink)"
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
            "marker-end": marker,
        },
    )


def mountain_base(y_offset: float = 80, scale: float = 1.0) -> str:
    pts = [
        (60, 390), (120, 260), (160, 330), (215, 190), (270, 345),
        (326, 240), (388, 365), (454, 286), (488, 390)
    ]
    pts = [(x, y_offset + (y - 80) * scale) for x, y in pts]
    shadow = [(56, y_offset + 326 * scale), (488, y_offset + 326 * scale), (456, y_offset + 360 * scale), (78, y_offset + 360 * scale)]
    ridges = [
        [(120, 260), (145, 380)],
        [(215, 190), (230, 392)],
        [(326, 240), (310, 390)],
        [(388, 365), (356, 392)],
    ]
    parts = [polygon(shadow, MOUNTAIN_LIGHT, "none", 0, 0.5), polygon(pts, "#F2F0E7", MOUNTAIN, 6)]
    for ridge in ridges:
        scaled = [(x, y_offset + (y - 80) * scale) for x, y in ridge]
        parts.append(polyline(scaled, "#8F9792", 5))
    return "\n".join(parts)


def grid_plane(x: float, y: float, w: float, h: float, fill: str, stroke: str = GRID, opacity: float = 0.88) -> str:
    pts = [(x, y + h * 0.30), (x + w * 0.82, y), (x + w, y + h * 0.62), (x + w * 0.18, y + h)]
    parts = [polygon(pts, fill, stroke, 5, opacity)]
    for i in range(1, 5):
        t = i / 5
        p1 = (x + w * 0.82 * t, y + h * 0.30 * (1 - t))
        p2 = (x + w * (0.18 + 0.82 * t), y + h * (1 - 0.38 * t))
        parts.append(line(p1[0], p1[1], p2[0], p2[1], stroke, 3))
        q1 = (x + w * 0.18 * t, y + h * (0.30 + 0.70 * t))
        q2 = (x + w * (0.82 + 0.18 * t), y + h * (0.62 * t))
        parts.append(line(q1[0], q1[1], q2[0], q2[1], stroke, 3))
    return "\n".join(parts)


def icon_dem_terrain() -> str:
    return grid_plane(60, 145, 380, 235, "#E9EFE3", "#95A58E") + mountain_base(110, 0.70)


def icon_osm_features() -> str:
    parts = [grid_plane(72, 145, 368, 240, "#EFF4EA", "#B3C2AD")]
    roads = [
        [(110, 250), (190, 225), (272, 270), (378, 230)],
        [(142, 330), (230, 295), (304, 360)],
        [(230, 166), (230, 390)],
    ]
    for road in roads:
        parts.append(polyline(road, "#AEBB9D", 12))
    parts.append(path("M256 120 C218 120 192 148 192 186 C192 235 256 300 256 300 C256 300 320 235 320 186 C320 148 294 120 256 120 Z", BLUE, BLUE, 5))
    parts.append(circle(256, 184, 28, WHITE, BLUE, 7))
    return "\n".join(parts)


def icon_communication_source() -> str:
    parts = [grid_plane(90, 310, 320, 115, BLUE_LIGHT, "#8BB4DE")]
    parts += [
        line(210, 340, 256, 150, INK, 9),
        line(302, 340, 256, 150, INK, 9),
        line(224, 285, 288, 285, INK, 7),
        line(236, 230, 276, 230, INK, 7),
        line(246, 182, 266, 182, INK, 7),
        circle(256, 145, 15, WHITE, INK, 7),
        path("M190 140 C152 178 152 232 190 270", "none", BLUE, 10),
        path("M322 140 C360 178 360 232 322 270", "none", BLUE, 10),
        path("M160 104 C96 166 96 244 160 306", "none", BLUE, 8, opacity=0.55),
        path("M352 104 C416 166 416 244 352 306", "none", BLUE, 8, opacity=0.55),
    ]
    return "\n".join(parts)


def icon_task_terminals_uav() -> str:
    parts = [
        path("M86 330 C150 240 248 372 340 226", "none", INK, 8, "12 13"),
        path("M350 210 C386 190 430 194 456 224", "none", INK, 8, "12 13"),
        path("M84 135 C54 135 34 157 34 188 C34 228 84 282 84 282 C84 282 134 228 134 188 C134 157 114 135 84 135 Z", BLUE, BLUE, 4),
        circle(84, 188, 20, WHITE, BLUE, 5),
        path("M428 120 C398 120 378 142 378 173 C378 213 428 267 428 267 C428 267 478 213 478 173 C478 142 458 120 428 120 Z", GREEN, GREEN, 4),
        circle(428, 173, 20, WHITE, GREEN, 5),
    ]
    parts += [
        line(245, 185, 335, 185, INK, 8),
        line(290, 145, 290, 225, INK, 8),
        circle(220, 185, 16, WHITE, BLUE, 7),
        circle(360, 185, 16, WHITE, BLUE, 7),
        circle(290, 125, 16, WHITE, BLUE, 7),
        circle(290, 245, 16, WHITE, BLUE, 7),
        rect(264, 166, 52, 38, INK, INK, 0, 8),
    ]
    return "\n".join(parts)


def icon_uav_parameters() -> str:
    parts = [rect(130, 88, 252, 336, "#F7FAFD", BLUE, 10, 18), rect(190, 60, 132, 58, WHITE, BLUE, 9, 14)]
    y_positions = [168, 238, 308]
    for y in y_positions:
        parts.append(rect(164, y - 22, 36, 36, WHITE, BLUE, 6, 6))
        parts.append(path(f"M170 {y-2} L184 {y+12} L204 {y-18}", "none", GREEN, 8))
        parts.append(line(228, y, 342, y, MUTED, 9))
    parts.append(line(228, 374, 310, 374, MUTED, 9))
    return "\n".join(parts)


def icon_terrain_risk_field() -> str:
    parts = [grid_plane(64, 130, 384, 260, GREEN_LIGHT, "#9DBB92")]
    colors = [GREEN_LIGHT, "#C7E5A4", "#F1D26E", ORANGE, RED]
    for i, col in enumerate(colors):
        parts.append(circle(250 + i * 18, 274 - i * 10, 56 - i * 5, col, "none", 0, 0.65))
    parts.append(polyline([(116, 310), (180, 220), (232, 276), (292, 174), (394, 282)], GREEN, 11))
    return "\n".join(parts)


def icon_human_exposure_risk() -> str:
    parts = [grid_plane(70, 132, 370, 260, ORANGE_LIGHT, "#D9AB7C")]
    for cx, cy, r in [(300, 250, 90), (232, 286, 78), (348, 330, 62)]:
        parts.append(circle(cx, cy, r, RED_LIGHT, "none", 0, 0.65))
    for x, y in [(178, 210), (270, 260), (342, 206), (232, 348)]:
        parts.append(circle(x, y, 15, INK, WHITE, 3))
        parts.append(line(x, y + 18, x, y + 62, INK, 9))
        parts.append(line(x - 22, y + 40, x + 22, y + 40, INK, 7))
        parts.append(line(x, y + 62, x - 20, y + 92, INK, 7))
        parts.append(line(x, y + 62, x + 20, y + 92, INK, 7))
    return "\n".join(parts)


def icon_communication_risk_field() -> str:
    parts = [grid_plane(68, 318, 376, 120, BLUE_LIGHT, "#91B9DF")]
    parts.append(icon_communication_source())
    parts.append(polygon([(250, 220), (430, 152), (430, 292)], RED_LIGHT, RED, 4, 0.46))
    parts.append(path("M254 218 C324 194 372 188 430 152", "none", RED, 8, "12 10"))
    return "\n".join(parts)


def icon_corridor_floor_boundary() -> str:
    parts = [mountain_base(88, 0.78), path("M86 222 C162 202 222 250 304 225 C365 205 420 230 462 210", "none", RED, 9, "14 10")]
    parts.append(path("M86 222 L86 304 M462 210 L462 300", "none", RED, 5, "10 10"))
    return "\n".join(parts)


def icon_corridor_ceiling_boundary() -> str:
    parts = [mountain_base(88, 0.78), path("M72 128 C160 98 224 144 304 118 C374 94 438 116 482 96", "none", BLUE, 9, "14 10")]
    parts.append(path("M72 128 L72 302 M482 96 L482 294", "none", BLUE, 5, "10 10"))
    return "\n".join(parts)


def icon_three_mid_surfaces() -> str:
    parts = []
    curves = [(BLUE, 150), (GREEN, 250), (RED, 350)]
    for color, y in curves:
        parts.append(path(f"M74 {y} C142 {y-38} 206 {y+34} 278 {y-10} C344 {y-50} 396 {y+14} 452 {y-18}", "none", color, 11, "16 10"))
        parts.append(path(f"M84 {y+32} C152 {y-6} 216 {y+66} 288 {y+22} C354 {y-18} 406 {y+46} 462 {y+14}", "none", color, 5, "10 10", 0.45))
    return "\n".join(parts)


def icon_flyable_corridor_mask() -> str:
    parts = [mountain_base(110, 0.70)]
    parts.append(polygon([(84, 196), (184, 150), (326, 170), (444, 122), (444, 278), (320, 325), (188, 300), (84, 344)], "#E9EDF2", "#7D8791", 7, 0.68))
    for x in [148, 218, 286, 356, 424]:
        parts.append(line(x, 160, x, 330, "#7D8791", 4, "8 9"))
    parts.append(path("M86 196 L188 236 L320 218 L444 278", "none", "#7D8791", 5))
    return "\n".join(parts)


def network_layers(include_mountain: bool = True) -> str:
    parts = []
    if include_mountain:
        parts.append(mountain_base(150, 0.56))
    layers = [(BLUE, 140), (GREEN, 248), (RED, 356)]
    nodes = [(130, 0), (210, -28), (286, 0), (370, -20), (420, 18)]
    for color, y in layers:
        pts = [(x, y + dy) for x, dy in nodes]
        for a, b in zip(pts, pts[1:]):
            parts.append(line(a[0], a[1], b[0], b[1], color, 6))
        parts.append(line(pts[0][0], pts[0][1], pts[2][0], pts[2][1], color, 5, opacity if False else None))
        for x, yy in pts:
            parts.append(circle(x, yy, 15, color, WHITE, 4))
    for i in range(len(nodes)):
        x = nodes[i][0]
        parts.append(line(x, 140 + nodes[i][1], x, 356 + nodes[i][1], INK, 5, "10 12"))
    return "\n".join(parts)


def icon_three_layer_network() -> str:
    return network_layers(True)


def icon_mid_surface_input() -> str:
    return icon_three_mid_surfaces() + arrow(356, 256, 456, 256, BLUE, 10)


def icon_node_sampling() -> str:
    parts = [icon_three_mid_surfaces()]
    for color, y in [(BLUE, 150), (GREEN, 250), (RED, 350)]:
        for x in [132, 226, 326, 426]:
            parts.append(circle(x, y + ((x // 50) % 3 - 1) * 16, 12, color, WHITE, 4))
    return "\n".join(parts)


def icon_candidate_edges() -> str:
    return network_layers(False)


def icon_safety_check() -> str:
    parts = [network_layers(False)]
    parts.append(path("M256 105 L362 145 L342 278 C326 352 256 404 256 404 C256 404 186 352 170 278 L150 145 Z", GREEN_LIGHT, BLUE, 10))
    parts.append(path("M205 258 L242 294 L312 212", "none", GREEN, 18))
    return "\n".join(parts)


def icon_inter_layer_links() -> str:
    parts = []
    for y, color in [(130, BLUE), (256, GREEN), (382, RED)]:
        for x in [150, 256, 362]:
            parts.append(circle(x, y, 15, color, WHITE, 4))
        parts.append(line(150, y, 362, y, color, 7))
    for x in [150, 256, 362]:
        parts.append(line(x, 130, x, 382, INK, 7, "12 12"))
    return "\n".join(parts)


def icon_cost_assignment() -> str:
    parts = [
        circle(122, 194, 44, BLUE_LIGHT, BLUE, 8),
        circle(256, 194, 44, ORANGE_LIGHT, ORANGE, 8),
        circle(390, 194, 44, RED_LIGHT, RED, 8),
        path("M112 190 L132 190 M122 180 L122 210", "none", BLUE, 7),
        path("M246 204 L266 204 L256 176 Z", ORANGE, ORANGE, 4),
        path("M372 208 C390 174 408 208 390 226 C382 218 376 214 372 208 Z", RED, RED, 4),
        arrow(132, 300, 210, 300, BLUE, 8),
        arrow(302, 300, 380, 300, BLUE, 8),
        rect(194, 262, 124, 76, WHITE, BLUE, 8, 18),
        text(256, 314, "c(e)", 45, INK, 700),
    ]
    return "\n".join(parts)


def icon_graph_output() -> str:
    parts = [network_layers(False), rect(120, 56, 272, 76, WHITE, BLUE, 8, 20), text(256, 108, "G=(V,E)", 42, INK, 700)]
    return "\n".join(parts)


def icon_regional_event() -> str:
    parts = [
        circle(202, 230, 70, "#D8E2E8", "none", 0),
        circle(270, 200, 88, "#D8E2E8", "none", 0),
        circle(334, 242, 66, "#D8E2E8", "none", 0),
        rect(154, 230, 220, 82, "#D8E2E8", "#D8E2E8", 0, 24),
        path("M250 244 L218 330 L262 330 L238 418 L314 294 L270 294 L294 244 Z", ORANGE, ORANGE, 4),
        path("M116 314 C76 252 116 172 190 180", "none", MUTED, 7, "11 10"),
    ]
    return "\n".join(parts)


def icon_affected_edges() -> str:
    nodes = [(130, 150), (260, 110), (382, 180), (188, 290), (332, 340), (250, 250)]
    edges = [(0, 1), (1, 2), (0, 3), (3, 5), (5, 4), (5, 2), (1, 5)]
    parts = []
    for a, b in edges:
        color = RED if (a, b) in [(5, 2), (1, 5)] else MUTED
        parts.append(line(nodes[a][0], nodes[a][1], nodes[b][0], nodes[b][1], color, 9))
    for i, (x, y) in enumerate(nodes):
        parts.append(circle(x, y, 18, RED if i in [2, 5] else "#4E5965", WHITE, 4))
    return "\n".join(parts)


def icon_lpa_state_reuse() -> str:
    parts = [
        path("M156 146 C156 104 356 104 356 146 L356 346 C356 388 156 388 156 346 Z", "#EFF5FA", BLUE, 8),
        path("M156 146 C156 188 356 188 356 146", "none", BLUE, 8),
        path("M156 246 C156 288 356 288 356 246", "none", BLUE, 8),
        path("M390 170 C444 222 444 302 392 354", "none", GREEN, 10, None),
        path("M390 354 L414 350 L396 326", "none", GREEN, 9),
        text(256, 450, "g/rhs", 38, INK, 700),
    ]
    return "\n".join(parts)


def icon_updated_path() -> str:
    parts = [
        path("M74 366 C142 314 198 330 250 282 C314 222 374 214 440 150", "none", MUTED, 7, "12 12", 0.55),
        path("M74 366 C158 250 270 330 440 150", "none", BLUE, 13),
        path("M74 330 C48 330 32 348 32 374 C32 408 74 452 74 452 C74 452 116 408 116 374 C116 348 100 330 74 330 Z", BLUE, BLUE, 4),
        circle(74, 374, 15, WHITE, BLUE, 5),
        path("M438 102 C412 102 396 120 396 146 C396 180 438 224 438 224 C438 224 480 180 480 146 C480 120 464 102 438 102 Z", GREEN, GREEN, 4),
        circle(438, 146, 15, WHITE, GREEN, 5),
    ]
    return "\n".join(parts)


def icon_los_pruning() -> str:
    nodes = [(86, 248), (158, 304), (242, 260), (324, 304), (426, 238)]
    parts = []
    for a, b in zip(nodes, nodes[1:]):
        parts.append(line(a[0], a[1], b[0], b[1], MUTED, 8, "12 10"))
    parts.append(line(nodes[0][0], nodes[0][1], nodes[-1][0], nodes[-1][1], BLUE, 10))
    for x, y in nodes:
        parts.append(circle(x, y, 15, "#4E5965", WHITE, 4))
    parts.append(line(234, 224, 280, 270, RED, 12))
    parts.append(line(280, 224, 234, 270, RED, 12))
    return "\n".join(parts)


def icon_bspline_smoothing() -> str:
    pts = [(78, 336), (146, 268), (220, 300), (296, 210), (388, 282), (442, 230)]
    parts = []
    for a, b in zip(pts, pts[1:]):
        parts.append(line(a[0], a[1], b[0], b[1], MUTED, 6, "11 9"))
    for x, y in pts:
        parts.append(circle(x, y, 14, BLUE, WHITE, 4))
    parts.append(path("M78 336 C156 220 240 342 318 238 C360 186 400 236 442 230", "none", BLUE, 15))
    return "\n".join(parts)


def icon_safety_recheck() -> str:
    parts = [
        path("M92 342 C176 242 282 326 420 188", "none", BLUE, 12),
        path("M256 92 L378 138 L352 286 C334 366 256 420 256 420 C256 420 178 366 160 286 L134 138 Z", GREEN_LIGHT, GREEN, 10),
        path("M204 258 L242 296 L318 200", "none", GREEN, 18),
    ]
    return "\n".join(parts)


def icon_continuous_trajectory() -> str:
    return icon_updated_path()


def icon_adaptive_corridor_generation() -> str:
    return icon_corridor_ceiling_boundary() + path("M84 258 C164 218 230 270 312 236 C384 205 438 226 480 205", "none", GREEN, 8, "12 10")


def icon_structured_graph_compression() -> str:
    parts = []
    dense = [(105 + (i % 4) * 42, 110 + (i // 4) * 42) for i in range(16)]
    for i, (x, y) in enumerate(dense):
        for j, (xx, yy) in enumerate(dense):
            if j > i and abs(x - xx) <= 45 and abs(y - yy) <= 45:
                parts.append(line(x, y, xx, yy, "#A8B8C8", 3))
        parts.append(circle(x, y, 7, BLUE, WHITE, 2))
    parts.append(arrow(278, 200, 340, 200, BLUE, 8))
    compact = [(360, 140), (430, 190), (384, 280), (448, 330)]
    for a, b in [(0, 1), (1, 2), (2, 3), (0, 2)]:
        parts.append(line(compact[a][0], compact[a][1], compact[b][0], compact[b][1], BLUE, 7))
    for x, y in compact:
        parts.append(circle(x, y, 13, GREEN, WHITE, 3))
    return "\n".join(parts)


def icon_local_update_postprocessing() -> str:
    return icon_lpa_state_reuse() + path("M72 410 C156 326 250 410 430 252", "none", BLUE, 10)


def icon_outcome_target() -> str:
    parts = [
        circle(256, 256, 166, WHITE, BLUE, 11),
        circle(256, 256, 104, WHITE, BLUE, 11),
        circle(256, 256, 42, BLUE, BLUE, 8),
        arrow(270, 242, 414, 98, BLUE, 13),
    ]
    return "\n".join(parts)


ICONS: list[tuple[str, str, str, object]] = [
    ("stage1_dem", "DEM 地形输入", "地形高程与坡度、起伏等地形特征来源", icon_dem_terrain),
    ("stage1_osm", "OSM 要素", "人员暴露风险的空间要素来源", icon_osm_features),
    ("stage1_communication_source", "通信源", "通信视距可达性和遮挡关系的来源", icon_communication_source),
    ("stage1_task_terminals_uav", "任务端点与无人机", "配送站、补给点、目标点和无人机任务接入", icon_task_terminals_uav),
    ("stage1_uav_parameters", "无人机参数", "速度、能耗、爬升等规划参数", icon_uav_parameters),
    ("stage2_terrain_risk", "地形风险场", "由地形邻近性和地形起伏形成的规划层风险", icon_terrain_risk_field),
    ("stage2_human_exposure_risk", "人员暴露风险", "由 OSM 风险要素形成的人员暴露代理", icon_human_exposure_risk),
    ("stage2_communication_risk", "通信风险场", "由 DEM 视距和通信源形成的通信可达性风险", icon_communication_risk_field),
    ("stage3_floor_boundary", "走廊下边界", "安全飞行走廊的下边界约束", icon_corridor_floor_boundary),
    ("stage3_ceiling_boundary", "走廊上边界", "安全飞行走廊的上边界约束", icon_corridor_ceiling_boundary),
    ("stage3_mid_surfaces", "三层飞行中面", "端点接入层、区域支路层、骨干通行层的空间支撑", icon_three_mid_surfaces),
    ("stage3_flyable_mask", "可飞走廊掩码", "层可通行域和安全走廊包络", icon_flyable_corridor_mask),
    ("stage4_three_layer_network", "三层航路网络", "地形感知三层航线图核心结构", icon_three_layer_network),
    ("stage4_mid_surface_input", "中面输入", "三层中面进入图构建流程", icon_mid_surface_input),
    ("stage4_node_sampling", "节点采样", "在中面上执行地形驱动节点采样", icon_node_sampling),
    ("stage4_candidate_edges", "候选边生成", "同层边和候选连接生成", icon_candidate_edges),
    ("stage4_safety_check", "安全校验", "净空、风险和走廊约束下的候选边过滤", icon_safety_check),
    ("stage4_inter_layer_links", "层间连接", "三层网络中的垂直和斜向层间接驳", icon_inter_layer_links),
    ("stage4_cost_assignment", "边代价赋值", "时间、能耗和综合风险共同形成 c(e)", icon_cost_assignment),
    ("stage4_graph_output", "图结构输出", "生成可搜索图 G=(V,E)", icon_graph_output),
    ("stage5_regional_event", "区域事件", "临时禁飞、风场扰动或通信退化", icon_regional_event),
    ("stage5_affected_edges", "受影响边", "事件映射到局部受影响边集合", icon_affected_edges),
    ("stage5_lpa_state_reuse", "LPA* 状态复用", "复用 g/rhs 状态进行局部一致性传播", icon_lpa_state_reuse),
    ("stage5_updated_path", "更新路径", "局部增量重规划后的离散路径", icon_updated_path),
    ("stage6_los_pruning", "LOS 剪枝", "删除满足直视约束的冗余离散节点", icon_los_pruning),
    ("stage6_bspline_smoothing", "B 样条平滑", "将剪枝折线连续化为平滑轨迹", icon_bspline_smoothing),
    ("stage6_safety_recheck", "安全复检", "平滑轨迹采样点再次检查安全约束", icon_safety_recheck),
    ("stage6_continuous_trajectory", "连续轨迹输出", "路径更新后的连续飞行轨迹表达", icon_continuous_trajectory),
    ("callout_adaptive_corridor", "自适应走廊生成", "DEM 驱动的安全走廊贡献说明", icon_adaptive_corridor_generation),
    ("callout_structured_graph", "结构化图压缩", "连续空域压缩为紧凑三层图", icon_structured_graph_compression),
    ("callout_local_update_postprocess", "局部更新与后处理", "事件局部更新和轨迹连续化的联合说明", icon_local_update_postprocessing),
    ("outcome_target", "结果目标", "安全约束、紧凑图搜索和增量可更新问题表述", icon_outcome_target),
]


def wrap_svg(body: str) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE}" viewBox="0 0 {SIZE} {SIZE}">',
            arrow_defs(),
            body,
            "</svg>",
        ]
    )


def render_png(svg_path: Path, png_path: Path) -> None:
    doc = fitz.open(str(svg_path))
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=True)
    pix.save(str(png_path))


def generate_icons() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for file_stem, title, description, builder in ICONS:
        svg_path = SVG_DIR / f"{file_stem}.svg"
        png_path = PNG_DIR / f"{file_stem}.png"
        svg_path.write_text(wrap_svg(builder()), encoding="utf-8")
        render_png(svg_path, png_path)
        rows.append({"file_stem": file_stem, "title": title, "description": description, "svg": str(svg_path), "png": str(png_path)})
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
    thumb = 150
    cell_w = 260
    cell_h = 220
    cols = 4
    sheet_rows = (len(rows) + cols - 1) // cols
    image = Image.new("RGB", (cell_w * cols, cell_h * sheet_rows), WHITE)
    draw = ImageDraw.Draw(image)
    title_font = load_font(22)
    name_font = load_font(15)
    for i, row in enumerate(rows):
        col = i % cols
        r = i // cols
        x = col * cell_w
        y = r * cell_h
        icon = Image.open(row["png"]).convert("RGBA")
        icon.thumbnail((thumb, thumb))
        image.paste(icon, (x + (cell_w - icon.width) // 2, y + 10), icon)
        draw.text((x + 16, y + 165), row["title"], fill=INK, font=title_font)
        draw.text((x + 16, y + 194), row["file_stem"], fill=MUTED, font=name_font)
    image.save(CONTACT_SHEET)


def write_manifest(rows: list[dict[str, str]]) -> None:
    lines = [
        "# 方法框架图小图标清单",
        "",
        "本目录为论文方法框架图生成了一套统一风格的小图标。每个图标均包含 SVG 和透明 PNG 两种格式，SVG 适合论文和 PPT 中继续缩放或编辑，PNG 适合直接插入 Word 或临时预览。",
        "",
        f"图标总览：`{CONTACT_SHEET.name}`",
        "",
        "| 文件名 | 图标含义 | 对应论文内容 | SVG | PNG |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        svg_rel = Path(row["svg"]).relative_to(OUT_DIR)
        png_rel = Path(row["png"]).relative_to(OUT_DIR)
        lines.append(f"| `{row['file_stem']}` | {row['title']} | {row['description']} | `{svg_rel}` | `{png_rel}` |")
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
