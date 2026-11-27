"""Unit tests for M2 2D hand pose classification and temporal debounce."""

from __future__ import annotations

import math

import pytest

from airwriting.interaction.hand_pose import HandPose, PoseDebouncer, classify_hand_pose


def _build_hand_landmarks(
    *,
    thumb: bool = False,
    index: bool = False,
    middle: bool = False,
    ring: bool = False,
    pinky: bool = False,
    index_override: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    | None = None,
) -> list[tuple[float, float, float]]:
    """Build a realistic normalized 21-landmark right hand fixture."""
    pts: list[tuple[float, float, float]] = [(0.50, 0.70, 0.0)]  # 0: WRIST

    # 1-4: THUMB (CMC, MCP, IP, TIP)
    pts.append((0.46, 0.66, 0.0))
    pts.append((0.43, 0.61, 0.0))
    if thumb:
        pts.append((0.40, 0.57, 0.0))
        pts.append((0.37, 0.54, 0.0))
    else:
        pts.append((0.47, 0.59, 0.0))
        pts.append((0.52, 0.60, 0.0))

    fingers = (
        ((0.46, 0.56), index, index_override),  # 5-8: INDEX
        ((0.50, 0.55), middle, None),  # 9-12: MIDDLE
        ((0.54, 0.56), ring, None),  # 13-16: RING
        ((0.57, 0.58), pinky, None),  # 17-20: PINKY
    )
    for (mx, my), extended, override in fingers:
        pts.append((mx, my, 0.0))  # MCP
        if override is not None:
            pip_xy, dip_xy, tip_xy = override
            pts.append((pip_xy[0], pip_xy[1], 0.0))
            pts.append((dip_xy[0], dip_xy[1], 0.0))
            pts.append((tip_xy[0], tip_xy[1], 0.0))
        elif extended:
            pts.append((mx, my - 0.06, 0.0))
            pts.append((mx, my - 0.10, 0.0))
            pts.append((mx, my - 0.13, 0.0))
        else:
            pts.append((mx, my - 0.04, 0.0))
            pts.append((mx, my + 0.01, 0.0))
            pts.append((mx, my + 0.04, 0.0))

    return pts


def _transform_landmarks(
    landmarks: list[tuple[float, float, float]],
    *,
    scale: float = 1.0,
    dx: float = 0.0,
    dy: float = 0.0,
    angle_deg: float = 0.0,
    mirror_x: bool = False,
) -> list[tuple[float, float, float]]:
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    cx, cy = landmarks[0][0], landmarks[0][1]
    transformed: list[tuple[float, float, float]] = []
    for x, y, z in landmarks:
        rx = (x - cx) * scale
        ry = (y - cy) * scale
        nx = cx + (rx * cos_a - ry * sin_a) + dx
        ny = cy + (rx * sin_a + ry * cos_a) + dy
        if mirror_x:
            nx = 1.0 - nx
        transformed.append((nx, ny, z))
    return transformed


def test_classify_hand_pose_distinguishes_three_m2_gestures_and_labels() -> None:
    one_finger = _build_hand_landmarks(index=True)
    two_fingers = _build_hand_landmarks(index=True, middle=True)
    open_palm = _build_hand_landmarks(thumb=True, index=True, middle=True, ring=True, pinky=True)

    assert classify_hand_pose(one_finger) is HandPose.ONE_FINGER
    assert HandPose.ONE_FINGER.display_label == "1 ngón (Viết)"
    assert classify_hand_pose(two_fingers) is HandPose.TWO_FINGERS
    assert HandPose.TWO_FINGERS.display_label == "2 ngón (Nhấc bút)"
    assert classify_hand_pose(open_palm) is HandPose.OPEN_PALM
    assert HandPose.OPEN_PALM.display_label == "Xòe 5 ngón (Chốt)"
    assert HandPose.UNKNOWN.display_label == "Không xác định"


def test_classify_hand_pose_returns_unknown_for_intermediate_and_invalid_inputs() -> None:
    fist = _build_hand_landmarks()
    three_fingers = _build_hand_landmarks(index=True, middle=True, ring=True)
    thumb_and_index = _build_hand_landmarks(thumb=True, index=True)
    thumb_index_middle = _build_hand_landmarks(thumb=True, index=True, middle=True)
    four_fingers_no_thumb = _build_hand_landmarks(
        thumb=False, index=True, middle=True, ring=True, pinky=True
    )
    deadzone_index = _build_hand_landmarks(
        index=True,
        index_override=((0.46, 0.50), (0.47, 0.49), (0.48, 0.478)),
    )
    degenerate_palm = [(0.5, 0.5, 0.0) for _ in range(21)]

    assert classify_hand_pose(fist) is HandPose.UNKNOWN
    assert classify_hand_pose(three_fingers) is HandPose.UNKNOWN
    assert classify_hand_pose(thumb_and_index) is HandPose.UNKNOWN
    assert classify_hand_pose(thumb_index_middle) is HandPose.UNKNOWN
    assert classify_hand_pose(four_fingers_no_thumb) is HandPose.UNKNOWN
    assert classify_hand_pose(deadzone_index) is HandPose.UNKNOWN
    assert classify_hand_pose(None) is HandPose.UNKNOWN
    assert classify_hand_pose(fist[:20]) is HandPose.UNKNOWN
    assert classify_hand_pose(degenerate_palm) is HandPose.UNKNOWN

    with pytest.raises(ValueError):
        classify_hand_pose(fist, aspect_ratio=0.0)


def test_classify_hand_pose_compensates_for_widescreen_aspect_ratio() -> None:
    aspect = 16.0 / 9.0
    for base_landmarks, expected in (
        (_build_hand_landmarks(index=True), HandPose.ONE_FINGER),
        (_build_hand_landmarks(index=True, middle=True), HandPose.TWO_FINGERS),
        (
            _build_hand_landmarks(thumb=True, index=True, middle=True, ring=True, pinky=True),
            HandPose.OPEN_PALM,
        ),
    ):
        rotated = _transform_landmarks(base_landmarks, angle_deg=35.0)
        # Simulate MediaPipe normalized x compression on a 16:9 webcam frame
        widescreen = [(0.5 + (x - 0.5) / aspect, y, z) for x, y, z in rotated]
        assert classify_hand_pose(widescreen, aspect_ratio=aspect) is expected


def test_classify_hand_pose_is_invariant_to_scale_translation_rotation_and_mirroring() -> None:
    fixtures = [
        (_build_hand_landmarks(index=True), HandPose.ONE_FINGER),
        (_build_hand_landmarks(index=True, middle=True), HandPose.TWO_FINGERS),
        (
            _build_hand_landmarks(thumb=True, index=True, middle=True, ring=True, pinky=True),
            HandPose.OPEN_PALM,
        ),
    ]

    for base_landmarks, expected in fixtures:
        # Far distance (~70 cm, 0.5x scale) and close distance (~50 cm, 1.6x scale)
        far_hand = _transform_landmarks(base_landmarks, scale=0.5, dx=-0.15, dy=-0.1)
        close_hand = _transform_landmarks(base_landmarks, scale=1.6, dx=0.1, dy=0.05)
        tilted_pos = _transform_landmarks(base_landmarks, angle_deg=30.0)
        tilted_neg = _transform_landmarks(base_landmarks, angle_deg=-30.0)
        left_hand = _transform_landmarks(base_landmarks, mirror_x=True)
        left_tilted_far = _transform_landmarks(
            base_landmarks, scale=0.65, dx=0.12, angle_deg=-25.0, mirror_x=True
        )

        assert classify_hand_pose(far_hand) is expected
        assert classify_hand_pose(close_hand) is expected
        assert classify_hand_pose(tilted_pos) is expected
        assert classify_hand_pose(tilted_neg) is expected
        assert classify_hand_pose(left_hand) is expected
        assert classify_hand_pose(left_tilted_far) is expected


def test_pose_debouncer_rejects_short_flickers_below_threshold() -> None:
    debouncer = PoseDebouncer(debounce_s=0.12)

    assert debouncer.stable_pose is HandPose.UNKNOWN
    # Promote to ONE_FINGER after >= 0.12s
    assert debouncer.update(HandPose.ONE_FINGER, 1.00) is HandPose.UNKNOWN
    assert debouncer.update(HandPose.ONE_FINGER, 1.08) is HandPose.UNKNOWN
    assert debouncer.update(HandPose.ONE_FINGER, 1.12) is HandPose.ONE_FINGER

    # 1-2 frame flicker (0.06s < 0.12s) must not switch away from ONE_FINGER
    assert debouncer.update(HandPose.TWO_FINGERS, 1.15) is HandPose.ONE_FINGER
    assert debouncer.update(HandPose.TWO_FINGERS, 1.18) is HandPose.ONE_FINGER
    assert debouncer.update(HandPose.ONE_FINGER, 1.21) is HandPose.ONE_FINGER

    # Candidate timer restarts fresh if TWO_FINGERS reappears (must hold full 0.12s again)
    assert debouncer.update(HandPose.TWO_FINGERS, 1.24) is HandPose.ONE_FINGER
    assert debouncer.update(HandPose.TWO_FINGERS, 1.32) is HandPose.ONE_FINGER


def test_pose_debouncer_switches_after_continuous_hold_and_handles_timestamps() -> None:
    debouncer = PoseDebouncer(debounce_s=0.12)
    debouncer.update(HandPose.ONE_FINGER, 1.00)
    assert debouncer.update(HandPose.ONE_FINGER, 1.12) is HandPose.ONE_FINGER

    assert debouncer.update(HandPose.OPEN_PALM, 2.00) is HandPose.ONE_FINGER
    # Non-increasing timestamp cannot advance candidate duration prematurely
    assert debouncer.update(HandPose.OPEN_PALM, 1.95) is HandPose.ONE_FINGER
    assert debouncer.update(HandPose.OPEN_PALM, 2.11) is HandPose.ONE_FINGER
    assert debouncer.update(HandPose.OPEN_PALM, 2.12) is HandPose.OPEN_PALM
    assert debouncer.stable_pose is HandPose.OPEN_PALM


def test_pose_debouncer_supports_zero_threshold_runtime_update_reset_and_validation() -> None:
    immediate = PoseDebouncer(debounce_s=0.0)
    assert immediate.update(HandPose.TWO_FINGERS, 1.0) is HandPose.TWO_FINGERS
    assert immediate.update(HandPose.ONE_FINGER, 1.01) is HandPose.ONE_FINGER

    immediate.set_debounce_s(0.15)
    assert immediate.debounce_s == pytest.approx(0.15)
    assert immediate.update(HandPose.OPEN_PALM, 2.0) is HandPose.ONE_FINGER

    immediate.reset()
    assert immediate.stable_pose is HandPose.UNKNOWN

    immediate.reset(HandPose.TWO_FINGERS)
    assert immediate.stable_pose is HandPose.TWO_FINGERS

    with pytest.raises(ValueError):
        PoseDebouncer(debounce_s=-0.01)
    with pytest.raises(ValueError):
        immediate.set_debounce_s(-0.1)
