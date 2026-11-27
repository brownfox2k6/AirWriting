"""Aspect-ratio-preserving webcam viewport with crisp Active ROI overlay."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from airwriting.config import ROIConfig
from airwriting.core.roi import ROIRect, fit_aspect_rect, roi_in_viewport
from airwriting.core.tracking import TrackingResult, TrackingState
from airwriting.interaction.hand_pose import HandPose

OverlayRenderer = Callable[[QPainter, ROIRect, ROIRect], None]


class CameraViewWidget(QWidget):
    """Displays mirrored webcam frames with a proportional Active ROI overlay."""

    def __init__(self, roi: ROIConfig, *, parent: QWidget | None = None) -> None:
        super().__init__(cast(Any, parent))
        self._roi = roi
        self._frame: QImage | None = None
        self._status_message: str = "Đang khởi động webcam..."
        self._overlay_renderer: OverlayRenderer | None = None
        self._tracking_result: TrackingResult | None = None
        self._hand_pose: HandPose = HandPose.UNKNOWN
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    @property
    def roi(self) -> ROIConfig:
        return self._roi

    @property
    def status_message(self) -> str:
        return self._status_message

    @property
    def tracking_result(self) -> TrackingResult | None:
        return self._tracking_result

    @property
    def hand_pose(self) -> HandPose:
        return self._hand_pose

    def set_roi(self, roi: ROIConfig) -> None:
        self._roi = roi
        self.update()

    def set_frame(self, image: QImage) -> None:
        self._frame = image
        self._status_message = ""
        self.update()

    def set_tracking_result(self, result: TrackingResult | None) -> None:
        """Update the rendered fingertip tracking state and hover cursor position."""
        self._tracking_result = result
        self.update()

    def set_hand_pose(self, pose: HandPose) -> None:
        """Update the displayed M2 hand pose badge."""
        if pose is self._hand_pose:
            return
        self._hand_pose = pose
        self.update()

    def set_status_message(self, message: str) -> None:
        self._frame = None
        self._tracking_result = None
        self._hand_pose = HandPose.UNKNOWN
        self._status_message = message
        self.update()

    def set_overlay_renderer(self, renderer: OverlayRenderer | None) -> None:
        """Attach an optional overlay hook `(painter, video_rect, roi_rect)` for future stages."""
        self._overlay_renderer = renderer
        self.update()

    def video_rect(self) -> ROIRect | None:
        """Return the letterboxed video rectangle inside this widget, or None if no frame."""
        if self._frame is None or self._frame.isNull():
            return None
        if self.width() <= 0 or self.height() <= 0:
            return None
        return fit_aspect_rect(
            self._frame.width(),
            self._frame.height(),
            self.width(),
            self.height(),
        )

    def active_roi_rect(self) -> ROIRect | None:
        """Return the Active ROI rectangle in widget pixel coordinates, or None if no frame."""
        v_rect = self.video_rect()
        if v_rect is None:
            return None
        return roi_in_viewport(self._roi, v_rect)

    def cursor_widget_pos(self) -> tuple[int, int] | None:
        """Return widget pixel `(x, y)` of the hover cursor when in `State 1`, else None."""
        v_rect = self.video_rect()
        res = self._tracking_result
        if v_rect is None or res is None:
            return None
        if res.state is not TrackingState.HOVER or res.frame_point is None:
            return None
        fx, fy = res.frame_point
        return (round(v_rect.x + fx * v_rect.width), round(v_rect.y + fy * v_rect.height))

    def paintEvent(self, event: QPaintEvent | None) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(18, 20, 24))

        v_rect = self.video_rect()
        if self._frame is not None and v_rect is not None:
            target = QRect(v_rect.x, v_rect.y, v_rect.width, v_rect.height)
            painter.drawImage(target, self._frame)

            r_rect = roi_in_viewport(self._roi, v_rect)
            self._draw_roi_overlay(painter, v_rect, r_rect)
            self._draw_hover_cursor(painter)
            self._draw_hand_pose_badge(painter, v_rect)

            if self._overlay_renderer is not None:
                self._overlay_renderer(painter, v_rect, r_rect)
        else:
            painter.setPen(QColor(220, 224, 232))
            font = QFont()
            font.setPointSize(12)
            painter.setFont(font)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self._status_message or "Chưa có tín hiệu webcam.",
            )

        painter.end()

    def _draw_roi_overlay(self, painter: QPainter, video_rect: ROIRect, roi_rect: ROIRect) -> None:
        # Subtle dimming outside the Active ROI within the video bounds
        shade = QColor(0, 0, 0, 85)
        vx, vy, vw = video_rect.x, video_rect.y, video_rect.width
        rx, ry, rw, rh = roi_rect.x, roi_rect.y, roi_rect.width, roi_rect.height

        painter.fillRect(QRect(vx, vy, vw, max(0, ry - vy)), shade)
        painter.fillRect(
            QRect(vx, roi_rect.bottom, vw, max(0, video_rect.bottom - roi_rect.bottom)),
            shade,
        )
        painter.fillRect(QRect(vx, ry, max(0, rx - vx), rh), shade)
        painter.fillRect(
            QRect(roi_rect.right, ry, max(0, video_rect.right - roi_rect.right), rh),
            shade,
        )

        # Crisp Active ROI border
        pen = QPen(QColor(0, 220, 130), 2)
        painter.setPen(pen)
        painter.drawRect(QRect(rx, ry, rw, rh))

        # Corner label badge
        badge_font = QFont()
        badge_font.setPointSize(9)
        badge_font.setBold(True)
        painter.setFont(badge_font)
        label_rect = QRect(rx + 6, ry + 4, 110, 20)
        painter.fillRect(label_rect, QColor(0, 0, 0, 140))
        painter.setPen(QColor(0, 220, 130))
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, "ACTIVE ROI")

    def _draw_hover_cursor(self, painter: QPainter) -> None:
        cursor = self.cursor_widget_pos()
        if cursor is None:
            return
        cx, cy = cursor

        # Outer hollow ring + center dot signifying State 1 (Hover)
        ring_pen = QPen(QColor(225, 232, 240), 2)
        painter.setPen(ring_pen)
        painter.setBrush(QColor(180, 195, 210, 50))
        painter.drawEllipse(cx - 12, cy - 12, 24, 24)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(cx - 4, cy - 4, 8, 8)

        badge_font = QFont()
        badge_font.setPointSize(8)
        badge_font.setBold(True)
        painter.setFont(badge_font)
        tag_rect = QRect(cx + 14, cy - 10, 62, 18)
        painter.fillRect(tag_rect, QColor(18, 20, 24, 180))
        painter.setPen(QColor(225, 232, 240))
        painter.drawText(tag_rect, Qt.AlignmentFlag.AlignCenter, "HOVER")

    def _draw_hand_pose_badge(self, painter: QPainter, video_rect: ROIRect) -> None:
        badge_font = QFont()
        badge_font.setPointSize(9)
        badge_font.setBold(True)
        painter.setFont(badge_font)

        badge_width = min(250, max(120, video_rect.width - 16))
        badge_rect = QRect(video_rect.x + 8, video_rect.bottom - 30, badge_width, 22)
        painter.fillRect(badge_rect, QColor(18, 20, 24, 185))

        pose_colors = {
            HandPose.ONE_FINGER: QColor(0, 220, 130),
            HandPose.TWO_FINGERS: QColor(250, 204, 21),
            HandPose.OPEN_PALM: QColor(96, 165, 250),
            HandPose.UNKNOWN: QColor(180, 188, 200),
        }
        painter.setPen(pose_colors[self._hand_pose])
        painter.drawText(
            badge_rect,
            Qt.AlignmentFlag.AlignCenter,
            f"DÁNG TAY: {self._hand_pose.display_label}",
        )
