"""Webcam capture thread, horizontal mirroring, FPS counter, and handedness state."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtGui import QImage

if TYPE_CHECKING:
    from airwriting.core.tracking import HandTracker


class Handedness(StrEnum):
    """Dominant writing hand selected by the user."""

    RIGHT = "right"
    LEFT = "left"

    @property
    def display_label(self) -> str:
        return "Phải" if self is Handedness.RIGHT else "Trái"


class FPSCounter:
    """Sliding-window frame-rate estimator using a monotonic clock."""

    def __init__(
        self,
        window_size: int = 30,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {window_size}")
        self._timestamps: deque[float] = deque(maxlen=window_size)
        self._clock = clock

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0.0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def tick(self, timestamp: float | None = None) -> float:
        now = self._clock() if timestamp is None else timestamp
        self._timestamps.append(now)
        return self.fps

    def reset(self) -> None:
        self._timestamps.clear()


class FrameSource(Protocol):
    """Minimal frame source protocol for webcam capture and headless test doubles."""

    def open(self) -> bool: ...

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]: ...

    def release(self) -> None: ...


def mirror_frame(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Return a horizontally mirrored copy of a BGR/RGB image frame."""
    flipped = cv2.flip(frame, 1)
    return np.ascontiguousarray(flipped, dtype=np.uint8)


def bgr_to_rgb(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert an OpenCV BGR frame to a contiguous RGB uint8 array."""
    return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def rgb_to_qimage(rgb: NDArray[np.uint8]) -> QImage:
    """Convert a contiguous RGB uint8 frame to an owned RGB888 QImage safe across threads."""
    height, width, channels = rgb.shape
    bytes_per_line = int(channels * width)
    image = QImage(
        rgb.tobytes(),
        int(width),
        int(height),
        bytes_per_line,
        QImage.Format.Format_RGB888,
    )
    return image.copy()


def bgr_to_qimage(frame: NDArray[np.uint8]) -> QImage:
    """Convert an OpenCV BGR frame to an owned RGB888 QImage safe across threads."""
    return rgb_to_qimage(bgr_to_rgb(frame))


class OpenCVFrameSource:
    """OpenCV VideoCapture wrapper implementing FrameSource."""

    def __init__(
        self,
        camera_index: int = 0,
        *,
        capture_factory: Callable[[int], Any] = cv2.VideoCapture,
    ) -> None:
        self._camera_index = camera_index
        self._capture_factory = capture_factory
        self._capture: Any | None = None

    @property
    def camera_index(self) -> int:
        return self._camera_index

    def open(self) -> bool:
        self.release()
        capture = self._capture_factory(self._camera_index)
        if not capture.isOpened():
            capture.release()
            return False
        self._capture = capture
        return True

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        if self._capture is None:
            return (False, None)
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return (False, None)
        return (True, np.asarray(frame, dtype=np.uint8))

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class CameraWorker(QThread):
    """Background QThread that reads, mirrors, tracks, and emits webcam frames."""

    frame_ready = pyqtSignal(QImage, float, arguments=["image", "fps"])
    tracking_ready = pyqtSignal(object, arguments=["result"])
    error_occurred = pyqtSignal(str, arguments=["message"])

    def __init__(
        self,
        source: FrameSource,
        *,
        mirror: bool = True,
        fps_window: int = 30,
        hand_tracker: HandTracker | None = None,
        clock: Callable[[], float] = time.perf_counter,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._mirror = mirror
        self._hand_tracker = hand_tracker
        self._clock = clock
        self._fps_counter = FPSCounter(window_size=fps_window, clock=clock)
        self._stop_event = threading.Event()

    def run(self) -> None:
        self._stop_event.clear()
        self._fps_counter.reset()
        try:
            if not self._source.open():
                self.error_occurred.emit(
                    "Không mở được webcam. Hãy kiểm tra kết nối hoặc quyền camera."
                )
                return

            while not self._stop_event.is_set():
                ok, frame = self._source.read()
                if not ok or frame is None:
                    if not self._stop_event.is_set():
                        self.error_occurred.emit("Mất tín hiệu từ webcam.")
                    break

                processed = mirror_frame(frame) if self._mirror else frame
                rgb = bgr_to_rgb(processed)
                image = rgb_to_qimage(rgb)
                now = self._clock()
                tracking_result = (
                    self._hand_tracker.process_frame(rgb, now)
                    if self._hand_tracker is not None and not self._stop_event.is_set()
                    else None
                )
                fps = self._fps_counter.tick(now)
                if not self._stop_event.is_set():
                    self.frame_ready.emit(image, fps)
                    if tracking_result is not None:
                        self.tracking_ready.emit(tracking_result)
        except Exception as exc:
            if not self._stop_event.is_set():
                self.error_occurred.emit(f"Lỗi webcam: {exc}")
        finally:
            if self._hand_tracker is not None:
                self._hand_tracker.close()
            self._source.release()

    def stop(self, timeout_ms: int = 3000) -> None:
        """Signal the capture loop to exit and wait for clean camera release."""
        self._stop_event.set()
        if self.isRunning():
            self.wait(timeout_ms)
