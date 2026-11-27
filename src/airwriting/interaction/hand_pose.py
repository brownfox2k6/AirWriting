"""Scale-invariant 2D hand pose classifier for M2 touchless air-writing gestures."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum

from airwriting.core.tracking import EXPECTED_HAND_LANDMARKS

Landmark3D = tuple[float, float, float]

WRIST = 0
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17

# (mcp, pip, tip) triplets for index, middle, ring, and pinky fingers.
_FINGER_TRIPLETS: tuple[tuple[int, int, int], ...] = (
    (5, 6, 8),
    (9, 10, 12),
    (13, 14, 16),
    (17, 18, 20),
)

_EPS = 1e-6


class HandPose(StrEnum):
    """Discrete static hand poses recognized for the M2 explicit gesture mechanism."""

    ONE_FINGER = "one_finger"
    TWO_FINGERS = "two_fingers"
    OPEN_PALM = "open_palm"
    UNKNOWN = "unknown"

    @property
    def display_label(self) -> str:
        labels = {
            HandPose.ONE_FINGER: "1 ngón (Viết)",
            HandPose.TWO_FINGERS: "2 ngón (Nhấc bút)",
            HandPose.OPEN_PALM: "Xòe 5 ngón (Chốt)",
            HandPose.UNKNOWN: "Không xác định",
        }
        return labels[self]


def _dist2d(a: Landmark3D, b: Landmark3D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _cos_angle(a: Landmark3D, vertex: Landmark3D, b: Landmark3D) -> float | None:
    v1x, v1y = vertex[0] - a[0], vertex[1] - a[1]
    v2x, v2y = b[0] - vertex[0], b[1] - vertex[1]
    norm1 = math.hypot(v1x, v1y)
    norm2 = math.hypot(v2x, v2y)
    if norm1 < _EPS or norm2 < _EPS:
        return None
    return (v1x * v2x + v1y * v2y) / (norm1 * norm2)


def _classify_finger(
    landmarks: Sequence[Landmark3D],
    mcp_idx: int,
    pip_idx: int,
    tip_idx: int,
    palm_scale: float,
) -> bool | None:
    wrist = landmarks[WRIST]
    mcp = landmarks[mcp_idx]
    pip = landmarks[pip_idx]
    tip = landmarks[tip_idx]

    wrist_to_pip = _dist2d(wrist, pip)
    if wrist_to_pip < _EPS:
        return None
    ratio = _dist2d(wrist, tip) / wrist_to_pip
    tip_to_mcp_norm = _dist2d(mcp, tip) / palm_scale
    cos_bend = _cos_angle(mcp, pip, tip)
    if cos_bend is None:
        return None

    if ratio >= 1.15 and tip_to_mcp_norm >= 0.45 and cos_bend >= 0.25:
        return True
    if ratio <= 1.05 or cos_bend <= -0.15:
        return False
    return None


def _classify_thumb(landmarks: Sequence[Landmark3D], palm_scale: float) -> bool | None:
    ip_to_pinky = _dist2d(landmarks[THUMB_IP], landmarks[PINKY_MCP])
    if ip_to_pinky < _EPS:
        return None
    thumb_ratio = _dist2d(landmarks[THUMB_TIP], landmarks[PINKY_MCP]) / ip_to_pinky
    tip_to_index_norm = _dist2d(landmarks[THUMB_TIP], landmarks[INDEX_MCP]) / palm_scale

    if thumb_ratio >= 1.12 and tip_to_index_norm >= 0.40:
        return True
    if thumb_ratio <= 1.05:
        return False
    return None


def classify_hand_pose(
    landmarks: Sequence[Landmark3D] | None,
    *,
    aspect_ratio: float = 1.0,
) -> HandPose:
    """Classify 21 MediaPipe 2D landmarks into `ONE_FINGER`, `TWO_FINGERS`, `OPEN_PALM`."""
    if aspect_ratio <= 0.0:
        raise ValueError(f"aspect_ratio must be > 0.0, got {aspect_ratio}")
    if landmarks is None or len(landmarks) != EXPECTED_HAND_LANDMARKS:
        return HandPose.UNKNOWN

    isotropic: Sequence[Landmark3D] = (
        [(x * aspect_ratio, y, z) for x, y, z in landmarks] if aspect_ratio != 1.0 else landmarks
    )
    palm_scale = _dist2d(isotropic[WRIST], isotropic[MIDDLE_MCP])
    if palm_scale < _EPS:
        return HandPose.UNKNOWN

    index_ext, middle_ext, ring_ext, pinky_ext = (
        _classify_finger(isotropic, mcp, pip, tip, palm_scale) for mcp, pip, tip in _FINGER_TRIPLETS
    )
    thumb_ext = _classify_thumb(isotropic, palm_scale)

    if (
        thumb_ext is not True
        and index_ext is True
        and middle_ext is False
        and ring_ext is False
        and pinky_ext is False
    ):
        return HandPose.ONE_FINGER
    if (
        thumb_ext is not True
        and index_ext is True
        and middle_ext is True
        and ring_ext is False
        and pinky_ext is False
    ):
        return HandPose.TWO_FINGERS
    if (
        thumb_ext is True
        and index_ext is True
        and middle_ext is True
        and ring_ext is True
        and pinky_ext is True
    ):
        return HandPose.OPEN_PALM
    return HandPose.UNKNOWN


class PoseDebouncer:
    """Temporal stability filter requiring a pose to hold for `debounce_s` seconds."""

    def __init__(
        self,
        debounce_s: float = 0.12,
        *,
        initial_pose: HandPose = HandPose.UNKNOWN,
    ) -> None:
        self._debounce_s = 0.0
        self.set_debounce_s(debounce_s)
        self._stable_pose = initial_pose
        self._candidate_pose: HandPose | None = None
        self._candidate_since: float | None = None

    @property
    def debounce_s(self) -> float:
        return self._debounce_s

    @property
    def stable_pose(self) -> HandPose:
        return self._stable_pose

    def set_debounce_s(self, debounce_s: float) -> None:
        """Update the required hold duration (`>= 0.0` seconds)."""
        if debounce_s < 0.0:
            raise ValueError(f"debounce_s must be >= 0.0, got {debounce_s}")
        self._debounce_s = float(debounce_s)

    def reset(self, pose: HandPose = HandPose.UNKNOWN) -> None:
        """Reset debouncer state to `pose` and clear any pending transition."""
        self._stable_pose = pose
        self._candidate_pose = None
        self._candidate_since = None

    def update(self, raw_pose: HandPose, timestamp: float) -> HandPose:
        """Feed one raw classification at `timestamp` seconds and return stable `HandPose`."""
        t = float(timestamp)
        if raw_pose is self._stable_pose:
            self._candidate_pose = None
            self._candidate_since = None
            return self._stable_pose

        if self._debounce_s == 0.0:
            self._stable_pose = raw_pose
            self._candidate_pose = None
            self._candidate_since = None
            return self._stable_pose

        if self._candidate_pose is not raw_pose or self._candidate_since is None:
            self._candidate_pose = raw_pose
            self._candidate_since = t
            return self._stable_pose

        elapsed = t - self._candidate_since
        if elapsed + 1e-9 >= self._debounce_s:
            self._stable_pose = raw_pose
            self._candidate_pose = None
            self._candidate_since = None
        return self._stable_pose
