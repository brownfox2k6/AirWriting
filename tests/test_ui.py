"""Headless PyQt6 UI tests for CameraViewWidget and MainWindow."""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtGui import QImage

from airwriting.config import ROIConfig, load_config
from airwriting.core.camera import Handedness
from airwriting.core.roi import ROIRect
from airwriting.core.tracking import INDEX_FINGER_TIP, DetectedHand, TrackingState
from airwriting.ui.camera_view import CameraViewWidget
from airwriting.ui.main_window import MainWindow
from conftest import pump_until


def _fake_hand(
    handedness: Handedness,
    index_tip: tuple[float, float],
    score: float = 0.95,
) -> DetectedHand:
    points = [(0.5, 0.5, 0.0) for _ in range(21)]
    points[INDEX_FINGER_TIP] = (index_tip[0], index_tip[1], 0.0)
    return DetectedHand(handedness=handedness, score=score, landmarks=tuple(points))


class ControllableHandDetector:
    """Thread-safe fake HandDetector for testing live MainWindow tracking states."""

    def __init__(self, initial_hands: list[DetectedHand] | None = None) -> None:
        self.hands: list[DetectedHand] = list(initial_hands or [])
        self.closed = False

    def detect(self, rgb_frame: NDArray[np.uint8], timestamp_ms: int) -> list[DetectedHand]:
        return list(self.hands)

    def close(self) -> None:
        self.closed = True


class RepeatingFrameSource:
    """Fake FrameSource that produces frames until released or fails to open."""

    def __init__(self, *, open_ok: bool = True, width: int = 320, height: int = 240) -> None:
        self._open_ok = open_ok
        self._frame = np.full((height, width, 3), 128, dtype=np.uint8)
        self._opened = False
        self.released = False
        self.read_count = 0

    def open(self) -> bool:
        self._opened = self._open_ok
        return self._open_ok

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        if not self._opened:
            return (False, None)
        self.read_count += 1
        time.sleep(0.01)
        return (True, self._frame)

    def release(self) -> None:
        self.released = True
        self._opened = False


def test_camera_view_computes_aspect_fitted_video_and_active_roi_rects(
    qt_app: QCoreApplication,
) -> None:
    roi = ROIConfig(x_min=0.15, y_min=0.15, x_max=0.85, y_max=0.85)
    widget = CameraViewWidget(roi)
    widget.resize(1000, 600)

    # 4:3 image inside 1000x600 -> fitted video rect is 800x600 centered at x=100
    image = QImage(640, 480, QImage.Format.Format_RGB888)
    image.fill(0)
    widget.set_frame(image)
    widget.grab()  # Exercises paintEvent offscreen

    assert widget.video_rect() == ROIRect(x=100, y=0, width=800, height=600)
    assert widget.active_roi_rect() == ROIRect(x=220, y=90, width=560, height=420)
    qt_app.processEvents()


def test_main_window_toggles_handedness_immediately_and_emits_signal(
    qt_app: QCoreApplication,
) -> None:
    config = load_config()
    window = MainWindow(config, auto_start=False)
    changes: list[Handedness] = []
    window.handedness_changed.connect(changes.append)

    assert window.handedness is Handedness.RIGHT

    window.left_hand_button.click()
    assert window.handedness is Handedness.LEFT
    assert changes == [Handedness.LEFT]

    window.right_hand_button.click()
    assert window.handedness is Handedness.RIGHT
    assert changes == [Handedness.LEFT, Handedness.RIGHT]

    window.close()
    qt_app.processEvents()


def test_main_window_streams_frames_updates_fps_and_releases_on_close(
    qt_app: QCoreApplication,
) -> None:
    config = load_config()
    source = RepeatingFrameSource(open_ok=True)
    window = MainWindow(config, frame_source_factory=lambda: source, auto_start=True)

    assert pump_until(qt_app, lambda: window.fps > 0.0)
    assert window.camera_view.active_roi_rect() is not None

    window.close()
    qt_app.processEvents()
    assert source.released is True


def test_main_window_handles_camera_failure_gracefully_and_supports_retry(
    qt_app: QCoreApplication,
) -> None:
    config = load_config()
    failing = RepeatingFrameSource(open_ok=False)
    working = RepeatingFrameSource(open_ok=True)
    sources = iter([failing, working])

    window = MainWindow(config, frame_source_factory=lambda: next(sources), auto_start=True)

    assert pump_until(qt_app, lambda: "không mở được" in window.status_message.lower())
    assert failing.released is True

    window.retry_button.click()
    assert pump_until(qt_app, lambda: window.fps > 0.0)
    window.close()
    qt_app.processEvents()
    assert working.released is True


def test_main_window_displays_state_1_hover_cursor_and_transitions_to_state_0(
    qt_app: QCoreApplication,
) -> None:
    config = load_config()
    source = RepeatingFrameSource(open_ok=True, width=640, height=480)
    detector = ControllableHandDetector([_fake_hand(Handedness.RIGHT, (0.5, 0.5))])

    window = MainWindow(
        config,
        frame_source_factory=lambda: source,
        hand_detector_factory=lambda: detector,
        auto_start=True,
    )
    window.resize(1000, 600)

    assert pump_until(qt_app, lambda: window.tracking_state is TrackingState.HOVER)
    roi_rect = window.camera_view.active_roi_rect()
    cursor_pos = window.camera_view.cursor_widget_pos()
    assert roi_rect is not None
    assert cursor_pos is not None
    assert roi_rect.x <= cursor_pos[0] <= roi_rect.right
    assert roi_rect.y <= cursor_pos[1] <= roi_rect.bottom
    window.camera_view.grab()

    detector.hands = [_fake_hand(Handedness.RIGHT, (0.05, 0.05))]
    assert pump_until(qt_app, lambda: window.tracking_state is TrackingState.OUT_OF_RANGE)
    assert window.camera_view.cursor_widget_pos() is None

    window.close()
    qt_app.processEvents()
    assert detector.closed is True


def test_main_window_handedness_toggle_updates_live_tracking_immediately(
    qt_app: QCoreApplication,
) -> None:
    config = load_config()
    source = RepeatingFrameSource(open_ok=True, width=640, height=480)
    detector = ControllableHandDetector([_fake_hand(Handedness.RIGHT, (0.5, 0.5))])

    window = MainWindow(
        config,
        frame_source_factory=lambda: source,
        hand_detector_factory=lambda: detector,
        auto_start=True,
    )

    assert pump_until(qt_app, lambda: window.tracking_state is TrackingState.HOVER)

    window.left_hand_button.click()
    assert pump_until(qt_app, lambda: window.tracking_state is TrackingState.OUT_OF_RANGE)

    detector.hands = [_fake_hand(Handedness.LEFT, (0.5, 0.5))]
    assert pump_until(qt_app, lambda: window.tracking_state is TrackingState.HOVER)
    assert window.tracking_result is not None
    assert window.tracking_result.handedness is Handedness.LEFT

    window.close()
    qt_app.processEvents()
    assert detector.closed is True
