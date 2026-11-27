"""Active ROI geometry, viewport aspect-ratio fitting, and coordinate normalization."""

from __future__ import annotations

from dataclasses import dataclass

from airwriting.config import ROIConfig


@dataclass(frozen=True)
class ROIRect:
    """Pixel-space rectangle `(x, y, width, height)` inside a frame or widget."""

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


def _require_positive_dimensions(width: int, height: int, label: str) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(f"{label} dimensions must be positive, got ({width}, {height})")


def roi_to_pixel_rect(roi: ROIConfig, frame_width: int, frame_height: int) -> ROIRect:
    """Map normalized `[0, 1]` ROI bounds to pixel coordinates on a frame."""
    _require_positive_dimensions(frame_width, frame_height, "Frame")
    x1 = round(roi.x_min * frame_width)
    y1 = round(roi.y_min * frame_height)
    x2 = round(roi.x_max * frame_width)
    y2 = round(roi.y_max * frame_height)
    return ROIRect(x=x1, y=y1, width=max(1, x2 - x1), height=max(1, y2 - y1))


def fit_aspect_rect(
    frame_width: int,
    frame_height: int,
    viewport_width: int,
    viewport_height: int,
) -> ROIRect:
    """Fit a video frame inside a viewport while preserving aspect ratio (centered)."""
    _require_positive_dimensions(frame_width, frame_height, "Frame")
    _require_positive_dimensions(viewport_width, viewport_height, "Viewport")

    scale = min(viewport_width / frame_width, viewport_height / frame_height)
    fitted_w = max(1, round(frame_width * scale))
    fitted_h = max(1, round(frame_height * scale))
    offset_x = (viewport_width - fitted_w) // 2
    offset_y = (viewport_height - fitted_h) // 2
    return ROIRect(x=offset_x, y=offset_y, width=fitted_w, height=fitted_h)


def roi_in_viewport(roi: ROIConfig, video_rect: ROIRect) -> ROIRect:
    """Project normalized ROI bounds onto the rendered video rectangle inside a widget."""
    local = roi_to_pixel_rect(roi, video_rect.width, video_rect.height)
    return ROIRect(
        x=video_rect.x + local.x,
        y=video_rect.y + local.y,
        width=local.width,
        height=local.height,
    )


def is_point_in_roi(x: float, y: float, roi: ROIConfig) -> bool:
    """Return True if a normalized frame point `(x, y)` lies within `roi`."""
    return roi.x_min <= x <= roi.x_max and roi.y_min <= y <= roi.y_max


def normalize_point_to_roi(
    x: float,
    y: float,
    roi: ROIConfig,
    *,
    clamp: bool = False,
) -> tuple[float, float]:
    """Convert normalized frame coordinates `(x, y)` into ROI-local `[0, 1]` coordinates."""
    nx = (x - roi.x_min) / (roi.x_max - roi.x_min)
    ny = (y - roi.y_min) / (roi.y_max - roi.y_min)
    if clamp:
        nx = min(1.0, max(0.0, nx))
        ny = min(1.0, max(0.0, ny))
    return (nx, ny)


def denormalize_point_from_roi(
    nx: float,
    ny: float,
    roi: ROIConfig,
) -> tuple[float, float]:
    """Convert ROI-local `[0, 1]` coordinates back into normalized frame coordinates."""
    return (
        roi.x_min + nx * (roi.x_max - roi.x_min),
        roi.y_min + ny * (roi.y_max - roi.y_min),
    )
