"""Tests for Handedness, FPSCounter, mirror_frame, and CameraWorker."""

from __future__ import annotations

import time

import numpy as np
import pytest
from numpy.typing import NDArray
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtGui import QImage

from airwriting.core.camera import CameraWorker, FPSCounter, Handedness, mirror_frame
from conftest import pump_until


class FakeFrameSource:
    """In-memory FrameSource for deterministic headless tests."""

    def __init__(self, frames: list[NDArray[np.uint8]], *, open_ok: bool = True) -> None:
        self._frames = frames
        self._open_ok = open_ok
        self._index = 0
        self.opened = False
        self.released = False

    def open(self) -> bool:
        self.opened = self._open_ok
        return self._open_ok

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        if not self.opened or self._index >= len(self._frames):
            return (False, None)
        frame = self._frames[self._index]
        self._index += 1
        time.sleep(0.005)
        return (True, frame)

    def release(self) -> None:
        self.released = True
        self.opened = False


def test_handedness_exposes_values_and_vietnamese_labels() -> None:
    assert Handedness.RIGHT.value == "right"
    assert Handedness.LEFT.value == "left"
    assert Handedness.RIGHT.display_label == "Phải"
    assert Handedness.LEFT.display_label == "Trái"


def test_fps_counter_computes_smoothed_rate_over_sliding_window() -> None:
    counter = FPSCounter(window_size=4)

    assert counter.fps == 0.0
    assert counter.tick(1.0) == 0.0
    assert counter.tick(1.1) == pytest.approx(10.0)
    assert counter.tick(1.2) == pytest.approx(10.0)
    assert counter.tick(1.3) == pytest.approx(10.0)
    # Sliding window drops t=1.0, now spans 1.1 -> 1.35 (3 intervals over 0.25s = 12 FPS)
    assert counter.tick(1.35) == pytest.approx(12.0)

    counter.reset()
    assert counter.fps == 0.0


def test_fps_counter_rejects_window_smaller_than_two() -> None:
    with pytest.raises(ValueError):
        FPSCounter(window_size=1)


def test_mirror_frame_flips_horizontally_without_mutating_input() -> None:
    # 2x3 BGR frame: left column red (0, 0, 255), right column blue (255, 0, 0)
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frame[:, 0] = [0, 0, 255]
    frame[:, 2] = [255, 0, 0]

    mirrored = mirror_frame(frame)

    assert np.array_equal(mirrored[0, 0], [255, 0, 0])
    assert np.array_equal(mirrored[0, 2], [0, 0, 255])
    # Original remains untouched
    assert np.array_equal(frame[0, 0], [0, 0, 255])


def test_camera_worker_emits_mirrored_qimages_and_releases_cleanly(
    qt_app: QCoreApplication,
) -> None:
    # BGR frame with bright red pixel on the left edge (x=0)
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[:, 0] = [0, 0, 255]  # BGR red -> RGB (255, 0, 0)
    source = FakeFrameSource([frame.copy(), frame.copy(), frame.copy()])

    received: list[tuple[QImage, float]] = []
    worker = CameraWorker(source, mirror=True)
    worker.frame_ready.connect(lambda img, fps: received.append((img, fps)))

    worker.start()
    assert pump_until(qt_app, lambda: not worker.isRunning())
    worker.stop()

    assert len(received) == 3
    first_img, _ = received[0]
    assert first_img.width() == 6
    assert first_img.height() == 4
    # After horizontal mirror, the red pixel originally at x=0 is now at x=5
    mirrored_pixel = first_img.pixelColor(5, 0)
    assert (mirrored_pixel.red(), mirrored_pixel.green(), mirrored_pixel.blue()) == (255, 0, 0)
    assert received[-1][1] > 0.0
    assert source.released is True


def test_camera_worker_reports_error_when_source_fails_to_open(
    qt_app: QCoreApplication,
) -> None:
    source = FakeFrameSource([], open_ok=False)
    errors: list[str] = []
    worker = CameraWorker(source)
    worker.error_occurred.connect(errors.append)

    worker.start()
    assert pump_until(qt_app, lambda: not worker.isRunning())
    worker.stop()

    assert len(errors) == 1
    assert source.released is True
