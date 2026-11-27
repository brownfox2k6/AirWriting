"""Core camera, Active ROI, and tracking utilities for AirWriting."""

from airwriting.core.camera import (
    CameraWorker,
    FPSCounter,
    FrameSource,
    Handedness,
    OpenCVFrameSource,
    bgr_to_qimage,
    mirror_frame,
)
from airwriting.core.roi import (
    ROIRect,
    fit_aspect_rect,
    is_point_in_roi,
    normalize_point_to_roi,
    roi_in_viewport,
    roi_to_pixel_rect,
)

__all__ = [
    "CameraWorker",
    "FPSCounter",
    "FrameSource",
    "Handedness",
    "OpenCVFrameSource",
    "ROIRect",
    "bgr_to_qimage",
    "fit_aspect_rect",
    "is_point_in_roi",
    "mirror_frame",
    "normalize_point_to_roi",
    "roi_in_viewport",
    "roi_to_pixel_rect",
]
