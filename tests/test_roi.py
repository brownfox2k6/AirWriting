"""Tests for Active ROI geometry, viewport fitting, and coordinate normalization."""

import pytest

from airwriting.config import ROIConfig
from airwriting.core.roi import (
    ROIRect,
    fit_aspect_rect,
    is_point_in_roi,
    normalize_point_to_roi,
    roi_in_viewport,
    roi_to_pixel_rect,
)


def test_roi_to_pixel_rect_maps_normalized_bounds_to_frame_pixels() -> None:
    roi = ROIConfig(x_min=0.15, y_min=0.15, x_max=0.85, y_max=0.85)

    rect = roi_to_pixel_rect(roi, frame_width=1000, frame_height=800)

    assert rect == ROIRect(x=150, y=120, width=700, height=560)
    assert rect.right == 850
    assert rect.bottom == 680


@pytest.mark.parametrize(("width", "height"), [(0, 480), (640, 0), (-10, 480)])
def test_roi_to_pixel_rect_rejects_non_positive_dimensions(width: int, height: int) -> None:
    roi = ROIConfig(x_min=0.15, y_min=0.15, x_max=0.85, y_max=0.85)

    with pytest.raises(ValueError):
        roi_to_pixel_rect(roi, frame_width=width, frame_height=height)


def test_fit_aspect_rect_centers_video_with_pillarbox_when_viewport_is_wider() -> None:
    # 4:3 frame (640x480) inside 1000x600 viewport -> height-limited to 800x600, x offset = 100
    rect = fit_aspect_rect(
        frame_width=640, frame_height=480, viewport_width=1000, viewport_height=600
    )

    assert rect == ROIRect(x=100, y=0, width=800, height=600)


def test_fit_aspect_rect_centers_video_with_letterbox_when_viewport_is_taller() -> None:
    # 16:9 frame (1280x720) inside 800x600 viewport -> width-limited to 800x450, y offset = 75
    rect = fit_aspect_rect(
        frame_width=1280, frame_height=720, viewport_width=800, viewport_height=600
    )

    assert rect == ROIRect(x=0, y=75, width=800, height=450)


def test_roi_in_viewport_offsets_roi_by_video_rect_origin() -> None:
    roi = ROIConfig(x_min=0.15, y_min=0.15, x_max=0.85, y_max=0.85)
    video_rect = ROIRect(x=100, y=50, width=800, height=600)

    rect = roi_in_viewport(roi, video_rect)

    assert rect == ROIRect(x=220, y=140, width=560, height=420)


def test_is_point_in_roi_and_normalize_point_to_roi() -> None:
    roi = ROIConfig(x_min=0.2, y_min=0.1, x_max=0.7, y_max=0.6)

    assert is_point_in_roi(0.2, 0.1, roi) is True
    assert is_point_in_roi(0.7, 0.6, roi) is True
    assert is_point_in_roi(0.19, 0.3, roi) is False

    nx, ny = normalize_point_to_roi(0.45, 0.35, roi)
    assert nx == pytest.approx(0.5)
    assert ny == pytest.approx(0.5)

    cx, cy = normalize_point_to_roi(0.0, 0.9, roi, clamp=True)
    assert cx == pytest.approx(0.0)
    assert cy == pytest.approx(1.0)
