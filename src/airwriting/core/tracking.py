"""Real-time dominant-hand index-fingertip tracking and State 0/1 transitions."""

from __future__ import annotations

import os
import shutil
import threading
import urllib.request
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from numpy.typing import NDArray

from airwriting.config import OneEuroConfig, ROIConfig
from airwriting.core.camera import Handedness
from airwriting.core.filters import OneEuroFilter2D
from airwriting.core.roi import denormalize_point_from_roi, is_point_in_roi, normalize_point_to_roi

INDEX_FINGER_TIP = 8
EXPECTED_HAND_LANDMARKS = 21
DEFAULT_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_HAND_MODEL_PATH = Path.home() / ".cache" / "airwriting" / "hand_landmarker.task"


class TrackingState(StrEnum):
    """Hover-level tracking state before stroke interaction (Buxton State 0 vs State 1)."""

    OUT_OF_RANGE = "state_0"
    HOVER = "state_1"

    @property
    def display_label(self) -> str:
        return "State 1: Hover" if self is TrackingState.HOVER else "State 0: Ngoài vùng"


@dataclass(frozen=True)
class DetectedHand:
    """Single detected hand returned by a `HandDetector` implementation."""

    handedness: Handedness
    score: float
    landmarks: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        if len(self.landmarks) != EXPECTED_HAND_LANDMARKS:
            raise ValueError(
                f"Expected {EXPECTED_HAND_LANDMARKS} hand landmarks, got {len(self.landmarks)}"
            )

    @property
    def index_tip(self) -> tuple[float, float]:
        """Normalized `(x, y)` coordinates of landmark #8 (index finger tip)."""
        x, y, _ = self.landmarks[INDEX_FINGER_TIP]
        return (x, y)


class HandDetector(Protocol):
    """Boundary interface for single-hand landmark detection models."""

    def detect(self, rgb_frame: NDArray[np.uint8], timestamp_ms: int) -> list[DetectedHand]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class TrackingResult:
    """Output of tracking the dominant hand's index fingertip on one frame."""

    state: TrackingState
    handedness: Handedness | None
    raw_frame_point: tuple[float, float] | None
    frame_point: tuple[float, float] | None
    roi_point: tuple[float, float] | None
    landmarks: tuple[tuple[float, float, float], ...] | None
    timestamp: float


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


class HandTracker:
    """Tracks the selected dominant hand, checks Active ROI bounds, and applies One Euro Filter."""

    def __init__(
        self,
        detector: HandDetector,
        roi: ROIConfig,
        one_euro: OneEuroConfig,
        *,
        dominant_hand: Handedness = Handedness.RIGHT,
        min_hand_score: float = 0.5,
    ) -> None:
        self._detector = detector
        self._roi = roi
        self._filter = OneEuroFilter2D(one_euro)
        self._dominant_hand = dominant_hand
        self._min_hand_score = float(min_hand_score)
        self._lock = threading.Lock()

    @property
    def dominant_hand(self) -> Handedness:
        with self._lock:
            return self._dominant_hand

    def set_dominant_hand(self, handedness: Handedness) -> None:
        """Switch the tracked writing hand and reset filter state to avoid cross-hand jumps."""
        with self._lock:
            if handedness is self._dominant_hand:
                return
            self._dominant_hand = handedness
            self._filter.reset()

    def set_one_euro_config(self, config: OneEuroConfig) -> None:
        """Update One Euro Filter parameters safely."""
        with self._lock:
            self._filter.update_config(config)

    def _select_dominant_hand(
        self,
        hands: list[DetectedHand],
        dominant: Handedness,
    ) -> DetectedHand | None:
        candidates = [
            hand
            for hand in hands
            if hand.handedness is dominant and hand.score >= self._min_hand_score
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda hand: hand.score)

    def process_frame(self, rgb_frame: NDArray[np.uint8], timestamp: float) -> TrackingResult:
        """Detect dominant hand, enforce Active ROI, and return filtered `TrackingResult`."""
        with self._lock:
            dominant = self._dominant_hand

        timestamp_ms = int(round(timestamp * 1000.0))
        detections = self._detector.detect(rgb_frame, timestamp_ms)
        hand = self._select_dominant_hand(detections, dominant)

        if hand is None:
            with self._lock:
                self._filter.reset()
            return TrackingResult(
                state=TrackingState.OUT_OF_RANGE,
                handedness=None,
                raw_frame_point=None,
                frame_point=None,
                roi_point=None,
                landmarks=None,
                timestamp=timestamp,
            )

        raw_x, raw_y = hand.index_tip
        in_frame = 0.0 <= raw_x <= 1.0 and 0.0 <= raw_y <= 1.0
        if not in_frame or not is_point_in_roi(raw_x, raw_y, self._roi):
            with self._lock:
                self._filter.reset()
            return TrackingResult(
                state=TrackingState.OUT_OF_RANGE,
                handedness=hand.handedness,
                raw_frame_point=(raw_x, raw_y),
                frame_point=None,
                roi_point=None,
                landmarks=hand.landmarks,
                timestamp=timestamp,
            )

        roi_x, roi_y = normalize_point_to_roi(raw_x, raw_y, self._roi, clamp=True)
        with self._lock:
            smooth_x, smooth_y = self._filter.filter(roi_x, roi_y, timestamp)

        clamped_roi_x = _clamp01(smooth_x)
        clamped_roi_y = _clamp01(smooth_y)
        frame_x, frame_y = denormalize_point_from_roi(clamped_roi_x, clamped_roi_y, self._roi)

        return TrackingResult(
            state=TrackingState.HOVER,
            handedness=hand.handedness,
            raw_frame_point=(raw_x, raw_y),
            frame_point=(frame_x, frame_y),
            roi_point=(clamped_roi_x, clamped_roi_y),
            landmarks=hand.landmarks,
            timestamp=timestamp,
        )

    def close(self) -> None:
        """Release underlying detector resources."""
        self._detector.close()


def interpret_handedness(raw_label: str, *, mirrored: bool) -> Handedness:
    """Convert MediaPipe's subject-relative handedness label for mirrored or unmirrored frames."""
    normalized = raw_label.strip().lower()
    if normalized not in {"left", "right"}:
        raise ValueError(f"Unsupported handedness label: {raw_label!r}")
    if mirrored:
        return Handedness.RIGHT if normalized == "left" else Handedness.LEFT
    return Handedness.RIGHT if normalized == "right" else Handedness.LEFT


def build_hand_landmarker_options(
    model_path: Path | str,
    *,
    min_detection_confidence: float = 0.5,
    min_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> HandLandmarkerOptions:
    """Build MediaPipe `HandLandmarkerOptions` pinned to `num_hands=1` in VIDEO mode."""
    return HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def ensure_hand_model(
    model_path: Path | str | None = None,
    *,
    download_url: str = DEFAULT_HAND_MODEL_URL,
    auto_download: bool = True,
) -> Path:
    """Resolve `hand_landmarker.task` path, downloading atomically when allowed."""
    if model_path is not None:
        target = Path(model_path)
    else:
        env_path = os.environ.get("AIRWRITING_HAND_MODEL_PATH")
        target = Path(env_path) if env_path else DEFAULT_HAND_MODEL_PATH

    if target.is_file():
        return target
    if not auto_download:
        raise FileNotFoundError(f"MediaPipe hand model asset not found: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_suffix(f"{target.suffix}.part")
    try:
        with (
            urllib.request.urlopen(download_url, timeout=30) as response,
            tmp_target.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle)
        tmp_target.replace(target)
    except Exception:
        if tmp_target.exists():
            tmp_target.unlink()
        raise
    return target


class MediaPipeHandDetector:
    """MediaPipe Tasks `HandLandmarker` wrapper configured with `num_hands=1`."""

    def __init__(
        self,
        *,
        mirrored: bool = True,
        model_path: Path | str | None = None,
        auto_download: bool = True,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        landmarker: Any | None = None,
    ) -> None:
        self._mirrored = mirrored
        self._model_path = model_path
        self._auto_download = auto_download
        self._min_detection_confidence = min_detection_confidence
        self._min_presence_confidence = min_presence_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._landmarker: Any | None = landmarker
        self._last_timestamp_ms: int | None = None

    def _get_landmarker(self) -> Any:
        if self._landmarker is None:
            resolved_path = ensure_hand_model(
                self._model_path,
                auto_download=self._auto_download,
            )
            options = build_hand_landmarker_options(
                resolved_path,
                min_detection_confidence=self._min_detection_confidence,
                min_presence_confidence=self._min_presence_confidence,
                min_tracking_confidence=self._min_tracking_confidence,
            )
            self._landmarker = HandLandmarker.create_from_options(options)
        return self._landmarker

    def detect(self, rgb_frame: NDArray[np.uint8], timestamp_ms: int) -> list[DetectedHand]:
        """Run single-hand landmark detection on a contiguous RGB uint8 frame."""
        landmarker = self._get_landmarker()
        ts = int(timestamp_ms)
        if self._last_timestamp_ms is not None and ts <= self._last_timestamp_ms:
            ts = self._last_timestamp_ms + 1
        self._last_timestamp_ms = ts

        contiguous = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=contiguous)
        result = landmarker.detect_for_video(mp_image, ts)

        detected: list[DetectedHand] = []
        hand_landmarks_list = getattr(result, "hand_landmarks", ()) or ()
        handedness_list = getattr(result, "handedness", ()) or ()
        for hand_landmarks, hand_cats in zip(hand_landmarks_list, handedness_list, strict=False):
            if not hand_cats or len(hand_landmarks) != EXPECTED_HAND_LANDMARKS:
                continue
            top_cat = hand_cats[0]
            try:
                handedness = interpret_handedness(
                    str(top_cat.category_name),
                    mirrored=self._mirrored,
                )
            except ValueError:
                continue
            points = tuple((float(lm.x), float(lm.y), float(lm.z)) for lm in hand_landmarks)
            detected.append(
                DetectedHand(
                    handedness=handedness,
                    score=float(top_cat.score),
                    landmarks=points,
                )
            )
        return detected

    def close(self) -> None:
        """Release underlying MediaPipe `HandLandmarker` if initialized."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
