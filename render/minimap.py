from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np

from tracking.court import COURT_LENGTH_M, COURT_PATHS, COURT_WIDTH_M


_COURT_BACKGROUND_COLOR = (29, 36, 47)
_MARKER_OUTLINE_COLOR = (235, 238, 242)
_UNKNOWN_MARKER_COLOR = (20, 17, 22)
_WHITE = (255, 255, 255)


def _relative_luminance(color):
    rgb = (channel / 255 for channel in reversed(color))
    linear_rgb = (channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4 for channel in rgb)
    return sum(channel * weight for channel, weight in zip(linear_rgb, (0.2126, 0.7152, 0.0722)))


def _number_color(marker):
    luminance = _relative_luminance(marker)
    white_contrast = 1.05 / (luminance + 0.05)
    dark_contrast = (luminance + 0.05) / (_relative_luminance(_COURT_BACKGROUND_COLOR) + 0.05)
    return _WHITE if white_contrast > dark_contrast else _COURT_BACKGROUND_COLOR


class CourtMiniMap:
    """Render court positions and centered jersey numbers using team colors."""

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.frame_scale = frame_height / 720.0
        self.margin = max(1, round(9 * self.frame_scale))
        requested = round(frame_height * 0.19 * COURT_LENGTH_M / COURT_WIDTH_M)
        self.width = min(max(1, requested), max(1, frame_width - 2 * self.margin))
        self.height = max(1, round(self.width * COURT_WIDTH_M / COURT_LENGTH_M))
        self.render_scale = 3
        self.pad = max(1, round(self.width * 0.035))
        self.marker_radius = max(1, round(self.width * 0.024))
        self.text_scale = 0.8 * self.frame_scale
        self.line_thickness = max(1, round(self.frame_scale))
        self.marker_thickness = max(1, round(2 * self.frame_scale))
        self.scale = np.asarray((self.width - 2 * self.pad, self.height - 2 * self.pad), dtype=np.float32)
        self.scale /= (COURT_LENGTH_M, COURT_WIDTH_M)
        self.base = self._build_base()

    def _pixel(self, points):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        return np.rint(points * self.scale + self.pad).astype(np.int32) * self.render_scale

    def _build_base(self):
        scale = self.render_scale
        canvas = np.full((self.height * scale, self.width * scale, 3), _COURT_BACKGROUND_COLOR, dtype=np.uint8)
        cv2.polylines(
            canvas,
            [self._pixel(path) for path in COURT_PATHS],
            False,
            [142, 150, 161],
            self.line_thickness * scale,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            canvas,
            (0, 0),
            ((self.width - 1) * scale, (self.height - 1) * scale),
            [188, 193, 201],
            self.line_thickness * scale,
            cv2.LINE_AA,
        )
        return canvas

    def _draw_marker(self, panel, center, marker, light_marker, holder):
        scale = self.render_scale
        radius = self.marker_radius * scale
        line = self.line_thickness * scale
        outline = _WHITE if light_marker else _MARKER_OUTLINE_COLOR
        inner_radius = max(scale, radius - round(1.25 * self.frame_scale) * scale)
        if holder:
            cv2.circle(panel, center, radius + line + scale, [0, 220, 255], line, cv2.LINE_AA)
        cv2.circle(panel, center, radius, outline, -1, cv2.LINE_AA)
        cv2.circle(panel, center, inner_radius, marker, -1, cv2.LINE_AA)
        if not light_marker:
            cv2.circle(panel, center, inner_radius, _COURT_BACKGROUND_COLOR, scale, cv2.LINE_AA)
        cv2.circle(panel, center, radius, outline, line, cv2.LINE_AA)

    def _draw_jersey_numbers(self, panel, labels):
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = self.render_scale
        thickness = max(1, round(1.75 * self.frame_scale))
        outline_thickness = (thickness + max(1, round(self.frame_scale))) * scale
        inner_radius = max(1, self.marker_radius - round(1.25 * self.frame_scale))
        max_width = max(1, 2 * inner_radius - 1) * scale
        max_height = max(1, 2 * inner_radius - 3) * scale
        (_, reference_height), baseline = cv2.getTextSize("88", font, self.text_scale * scale, outline_thickness)
        font_scale = self.text_scale * scale * min(1.0, max_height / (reference_height + baseline))
        for center, text, marker in labels:
            text_color = _number_color(marker)
            bold_number = text_color == _COURT_BACKGROUND_COLOR
            text_thickness = min(thickness + 2 * int(bold_number), outline_thickness // scale) * scale
            (width, height), baseline = cv2.getTextSize(text.zfill(2), font, font_scale, outline_thickness)
            fit = min(1.0, max_width / max(1, width), max_height / max(1, height + baseline))
            glyph_scale = font_scale * fit
            text_thickness = max(1, round(text_thickness * fit))
            stroke_thickness = max(text_thickness, round(outline_thickness * fit))
            (width, height), baseline = cv2.getTextSize(text, font, glyph_scale, text_thickness)
            padding = stroke_thickness + 2
            mask = np.zeros((height + baseline + 2 * padding, width + 2 * padding), dtype=np.uint8)
            cv2.putText(mask, text, (padding, padding + height), font, glyph_scale, 255, text_thickness, cv2.LINE_AA)
            # Center the visible glyph, since OpenCV's text metrics include baseline space.
            x, y, width, glyph_height = cv2.boundingRect(mask)
            origin = (
                round(center[0] - (x - padding) - (width - 1) / 2),
                round(center[1] - (y - padding - height) - (glyph_height - 1) / 2),
            )
            if not bold_number:
                cv2.putText(
                    panel,
                    text,
                    origin,
                    font,
                    glyph_scale,
                    _COURT_BACKGROUND_COLOR,
                    stroke_thickness,
                    cv2.LINE_AA,
                )
            cv2.putText(panel, text, origin, font, glyph_scale, text_color, text_thickness, cv2.LINE_AA)

    def draw(self, frame: np.ndarray, players: Iterable[dict[str, Any]]) -> None:
        """Draw the mini-map in place, keeping jersey numbers above all markers."""
        panel = self.base.copy()
        labels = []
        court_limit = np.asarray((COURT_LENGTH_M, COURT_WIDTH_M), dtype=np.float32)
        for item in players:
            position = item.get("court_xy")
            if position is None:
                continue
            position = np.asarray(position, dtype=np.float32).reshape(2)
            if not np.isfinite(position).all() or np.any(position < -2.0) or np.any(position > court_limit + 2.0):
                continue
            center = tuple(self._pixel(np.clip(position, (0, 0), court_limit))[0])
            marker_color = item.get("color")
            marker = _UNKNOWN_MARKER_COLOR if marker_color is None else tuple(map(int, marker_color))
            team_semantic = item.get("team_semantic")
            if team_semantic in ("dark", "light"):
                light_marker = team_semantic == "light"
            else:
                blue, green, red = marker
                light_marker = marker_color is not None and 0.114 * blue + 0.587 * green + 0.299 * red >= 128
            self._draw_marker(panel, center, marker, light_marker, item.get("holder", False))
            jersey_number = item.get("jersey_number")
            if jersey_number is not None:
                text = str(jersey_number).strip()
                if text.isascii() and text.isdigit():
                    labels.append((center, text, marker))
        frame_h, frame_w = frame.shape[:2]
        x0 = max(0, frame_w - self.margin - self.width)
        y0 = self.margin
        roi = frame[y0 : y0 + self.height, x0 : x0 + self.width]
        if not roi.size:
            return
        self._draw_jersey_numbers(panel, labels)
        panel = cv2.resize(panel, (self.width, self.height), interpolation=cv2.INTER_AREA)
        roi[:] = panel[: roi.shape[0], : roi.shape[1]]
        cv2.rectangle(
            frame,
            (x0, y0),
            (x0 + roi.shape[1] - 1, y0 + roi.shape[0] - 1),
            [188, 193, 201],
            self.line_thickness,
            cv2.LINE_AA,
        )
