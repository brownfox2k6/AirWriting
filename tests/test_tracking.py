"""Unit tests for HandTracker and MediaPipeHandDetector boundaries."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from mediapipe.tasks.python.vision import RunningMode
from numpy.typing import NDArray

from airwriting.config import OneEuroConfig, ROIConfig
from airwriting.core.camera import Handedness
from airwriting.core.tracking import (
    INDEX_FINGER_TIP,
    DetectedHand,
    HandTracker,
    MediaPipeHandDetector,
    TrackingState,
    build_hand_landmarker_options,
    ensure_hand_model,
    interpret_handedness,
)


def _make_landmarks(
    index_tip: tuple[float, float], z: float = 0.0
) -> tuple[tuple[float, float, float], ...]:
    points = [(0.5, 0.5, z) for _ in range(21)]
    points[INDEX_FINGER_TIP] = (index_tip[0], index_tip[1], z)
    return tuple(points)


class FakeHandDetector:
    """In-memory HandDetector test double for deterministic tracking tests."""

    def __init__(self) -> None:
        self.hands: list[DetectedHand] = []
        self.closed = False
        self.timestamps_ms: list[int] = []

    def detect(self, rgb_frame: NDArray[np.uint8], timestamp_ms: int) -> list[DetectedHand]:
        self.timestamps_ms.append(timestamp_ms)
        return list(self.hands)

    def close(self) -> None:
        self.closed = True


def _blank_frame() -> NDArray[np.uint8]:
    return np.zeros((32, 32, 3), dtype=np.uint8)


def test_hand_tracker_normalizes_index_tip_inside_roi_to_state_1_hover() -> None:
    roi = ROIConfig(x_min=0.15, y_min=0.15, x_max=0.85, y_max=0.85)
    one_euro = OneEuroConfig(enabled=True, min_cutoff=1.0, beta=0.05, d_cutoff=1.0)
    detector = FakeHandDetector()
    landmarks = _make_landmarks((0.5, 0.5))
    detector.hands = [DetectedHand(handedness=Handedness.RIGHT, score=0.95, landmarks=landmarks)]

    tracker = HandTracker(detector, roi, one_euro, dominant_hand=Handedness.RIGHT)
    result = tracker.process_frame(_blank_frame(), timestamp=1.0)

    assert result.state is TrackingState.HOVER
    assert TrackingState.HOVER.display_label == "State 1: Hover"
    assert result.handedness is Handedness.RIGHT
    assert result.raw_frame_point == (0.5, 0.5)
    assert result.frame_point == (0.5, 0.5)
    assert result.roi_point == (0.5, 0.5)
    assert result.landmarks == landmarks
    assert detector.timestamps_ms == [1000]


def test_hand_tracker_ignores_non_dominant_hand_and_low_score_detections() -> None:
    roi = ROIConfig(x_min=0.15, y_min=0.15, x_max=0.85, y_max=0.85)
    one_euro = OneEuroConfig(enabled=False, min_cutoff=1.0, beta=0.05, d_cutoff=1.0)
    detector = FakeHandDetector()
    detector.hands = [
        DetectedHand(
            handedness=Handedness.LEFT,
            score=0.99,
            landmarks=_make_landmarks((0.5, 0.5)),
        ),
        DetectedHand(
            handedness=Handedness.RIGHT,
            score=0.25,
            landmarks=_make_landmarks((0.4, 0.4)),
        ),
    ]

    tracker = HandTracker(
        detector,
        roi,
        one_euro,
        dominant_hand=Handedness.RIGHT,
        min_hand_score=0.5,
    )
    result = tracker.process_frame(_blank_frame(), timestamp=1.0)

    assert result.state is TrackingState.OUT_OF_RANGE
    assert TrackingState.OUT_OF_RANGE.display_label == "State 0: Ngoài vùng"
    assert result.frame_point is None
    assert result.roi_point is None
    assert result.landmarks is None


def test_hand_tracker_transitions_to_state_0_outside_roi_and_resets_filter_on_reentry() -> None:
    roi = ROIConfig(x_min=0.15, y_min=0.15, x_max=0.85, y_max=0.85)
    one_euro = OneEuroConfig(enabled=True, min_cutoff=1.0, beta=0.05, d_cutoff=1.0)
    detector = FakeHandDetector()
    tracker = HandTracker(detector, roi, one_euro, dominant_hand=Handedness.RIGHT)

    # Frame 1: enter at bottom-left inside ROI (0.29, 0.29 -> roi (0.2, 0.2))
    detector.hands = [
        DetectedHand(
            handedness=Handedness.RIGHT,
            score=0.9,
            landmarks=_make_landmarks((0.29, 0.29)),
        )
    ]
    r1 = tracker.process_frame(_blank_frame(), timestamp=1.0)
    assert r1.state is TrackingState.HOVER
    assert r1.frame_point == (0.29, 0.29)

    # Frame 2: exit ROI (0.05, 0.05) -> State 0, preserving raw_frame_point and landmarks
    outside_landmarks = _make_landmarks((0.05, 0.05))
    detector.hands = [
        DetectedHand(handedness=Handedness.RIGHT, score=0.9, landmarks=outside_landmarks)
    ]
    r2 = tracker.process_frame(_blank_frame(), timestamp=1.03)
    assert r2.state is TrackingState.OUT_OF_RANGE
    assert r2.raw_frame_point == (0.05, 0.05)
    assert r2.landmarks == outside_landmarks
    assert r2.frame_point is None
    assert r2.roi_point is None

    # Frame 3: re-enter far away at (0.71, 0.71 -> roi (0.8, 0.8)); reset avoids drag from 0.29
    detector.hands = [
        DetectedHand(
            handedness=Handedness.RIGHT,
            score=0.9,
            landmarks=_make_landmarks((0.71, 0.71)),
        )
    ]
    r3 = tracker.process_frame(_blank_frame(), timestamp=1.06)
    assert r3.state is TrackingState.HOVER
    assert r3.frame_point is not None
    assert r3.roi_point is not None
    assert r3.roi_point[0] == pytest.approx(0.8)
    assert r3.roi_point[1] == pytest.approx(0.8)


def test_hand_tracker_switches_dominant_hand_immediately_and_resets_filter() -> None:
    roi = ROIConfig(x_min=0.15, y_min=0.15, x_max=0.85, y_max=0.85)
    one_euro = OneEuroConfig(enabled=True, min_cutoff=1.0, beta=0.05, d_cutoff=1.0)
    detector = FakeHandDetector()
    tracker = HandTracker(detector, roi, one_euro, dominant_hand=Handedness.RIGHT)

    detector.hands = [
        DetectedHand(
            handedness=Handedness.RIGHT, score=0.95, landmarks=_make_landmarks((0.22, 0.22))
        ),
        DetectedHand(
            handedness=Handedness.LEFT, score=0.92, landmarks=_make_landmarks((0.78, 0.78))
        ),
    ]
    right_res = tracker.process_frame(_blank_frame(), timestamp=1.0)
    assert right_res.handedness is Handedness.RIGHT
    assert right_res.roi_point is not None and right_res.roi_point[0] < 0.2

    tracker.set_dominant_hand(Handedness.LEFT)
    assert tracker.dominant_hand is Handedness.LEFT
    left_res = tracker.process_frame(_blank_frame(), timestamp=1.03)
    assert left_res.state is TrackingState.HOVER
    assert left_res.handedness is Handedness.LEFT
    assert left_res.roi_point is not None
    assert left_res.roi_point[0] == pytest.approx(0.9)


def test_hand_tracker_one_euro_enabled_flag_toggles_jitter_filtering() -> None:
    roi = ROIConfig(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)
    detector = FakeHandDetector()

    raw_tracker = HandTracker(
        detector,
        roi,
        OneEuroConfig(enabled=False, min_cutoff=1.0, beta=0.05, d_cutoff=1.0),
    )
    smoothed_tracker = HandTracker(
        detector,
        roi,
        OneEuroConfig(enabled=True, min_cutoff=1.0, beta=0.05, d_cutoff=1.0),
    )

    raw_xs: list[float] = []
    smooth_xs: list[float] = []
    for idx in range(30):
        x = 0.5 + (0.02 if idx % 2 == 0 else -0.02)
        detector.hands = [
            DetectedHand(
                handedness=Handedness.RIGHT,
                score=0.95,
                landmarks=_make_landmarks((x, 0.5)),
            )
        ]
        raw_pt = raw_tracker.process_frame(_blank_frame(), timestamp=idx / 30.0).roi_point
        smooth_pt = smoothed_tracker.process_frame(_blank_frame(), timestamp=idx / 30.0).roi_point
        assert raw_pt is not None and smooth_pt is not None
        raw_xs.append(raw_pt[0])
        smooth_xs.append(smooth_pt[0])

    assert raw_xs[-1] == pytest.approx(0.48)
    assert max(abs(val - 0.5) for val in smooth_xs[15:]) < 0.008

    raw_tracker.close()
    assert detector.closed is True


def test_interpret_handedness_accounts_for_mirrored_frames() -> None:
    assert interpret_handedness("Left", mirrored=True) is Handedness.RIGHT
    assert interpret_handedness("Right", mirrored=True) is Handedness.LEFT
    assert interpret_handedness("Right", mirrored=False) is Handedness.RIGHT
    assert interpret_handedness("Left", mirrored=False) is Handedness.LEFT

    with pytest.raises(ValueError):
        interpret_handedness("Unknown", mirrored=True)


def test_build_hand_landmarker_options_enforces_single_hand_video_mode(tmp_path: Path) -> None:
    model_file = tmp_path / "hand_landmarker.task"
    model_file.write_bytes(b"stub")

    options = build_hand_landmarker_options(
        model_file,
        min_detection_confidence=0.6,
        min_presence_confidence=0.55,
        min_tracking_confidence=0.65,
    )

    assert options.num_hands == 1
    assert options.running_mode == RunningMode.VIDEO
    assert options.base_options.model_asset_path == str(model_file)
    assert options.min_hand_detection_confidence == pytest.approx(0.6)
    assert options.min_hand_presence_confidence == pytest.approx(0.55)
    assert options.min_tracking_confidence == pytest.approx(0.65)


def test_ensure_hand_model_returns_existing_path_or_raises_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.task"
    with pytest.raises(FileNotFoundError):
        ensure_hand_model(missing, auto_download=False)

    existing = tmp_path / "hand_landmarker.task"
    existing.write_bytes(b"model-bytes")
    assert ensure_hand_model(existing, auto_download=False) == existing


def test_mediapipe_hand_detector_enforces_monotonic_timestamps_and_maps_results() -> None:
    class FakeLandmarker:
        def __init__(self) -> None:
            self.timestamps_ms: list[int] = []
            self.closed = False

        def detect_for_video(self, mp_image: object, timestamp_ms: int) -> SimpleNamespace:
            self.timestamps_ms.append(timestamp_ms)
            landmarks = [
                SimpleNamespace(x=0.35 if idx == INDEX_FINGER_TIP else 0.5, y=0.65, z=-0.01)
                for idx in range(21)
            ]
            return SimpleNamespace(
                handedness=[[SimpleNamespace(category_name="Left", score=0.91)]],
                hand_landmarks=[landmarks],
            )

        def close(self) -> None:
            self.closed = True

    fake_lm = FakeLandmarker()
    detector = MediaPipeHandDetector(mirrored=True, landmarker=fake_lm)
    frame = np.zeros((24, 24, 3), dtype=np.uint8)

    hands_1 = detector.detect(frame, timestamp_ms=100)
    detector.detect(frame, timestamp_ms=100)
    detector.detect(frame, timestamp_ms=95)

    assert fake_lm.timestamps_ms == [100, 101, 102]
    assert len(hands_1) == 1
    assert hands_1[0].handedness is Handedness.RIGHT
    assert hands_1[0].score == pytest.approx(0.91)
    assert hands_1[0].index_tip == pytest.approx((0.35, 0.65))

    detector.close()
    assert fake_lm.closed is True
