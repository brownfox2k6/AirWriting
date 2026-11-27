"""Unit tests for OneEuroFilter and OneEuroFilter2D."""

from __future__ import annotations

import pytest

from airwriting.config import OneEuroConfig
from airwriting.core.filters import OneEuroFilter, OneEuroFilter2D


def test_one_euro_filter_dampens_static_jitter_without_losing_center() -> None:
    flt = OneEuroFilter(min_cutoff=1.0, beta=0.05, d_cutoff=1.0)
    dt = 1.0 / 30.0
    outputs: list[float] = []

    for idx in range(60):
        jitter = 0.02 if idx % 2 == 0 else -0.02
        outputs.append(flt.filter(0.5 + jitter, idx * dt))

    settled = outputs[20:]
    max_deviation = max(abs(val - 0.5) for val in settled)
    assert max_deviation < 0.008


def test_one_euro_filter_tracks_fast_step_with_low_lag_compared_to_fixed_lowpass() -> None:
    adaptive = OneEuroFilter(min_cutoff=1.0, beta=2.0, d_cutoff=1.0)
    fixed_lowpass = OneEuroFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
    dt = 1.0 / 30.0

    for idx in range(15):
        t = idx * dt
        adaptive.filter(0.1, t)
        fixed_lowpass.filter(0.1, t)

    # Fast ramp from 0.1 -> 0.9 across 5 frames (~0.17s)
    last_adaptive = 0.0
    last_fixed = 0.0
    for step, value in enumerate((0.26, 0.42, 0.58, 0.74, 0.9), start=15):
        t = step * dt
        last_adaptive = adaptive.filter(value, t)
        last_fixed = fixed_lowpass.filter(value, t)

    assert abs(0.9 - last_adaptive) < 0.12
    assert abs(0.9 - last_adaptive) < abs(0.9 - last_fixed)


def test_one_euro_filter_responds_rapidly_to_instantaneous_step_jump() -> None:
    adaptive = OneEuroFilter(min_cutoff=1.0, beta=2.0, d_cutoff=1.0)
    fixed_lowpass = OneEuroFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
    dt = 1.0 / 30.0

    for idx in range(10):
        adaptive.filter(0.1, idx * dt)
        fixed_lowpass.filter(0.1, idx * dt)

    # Instantaneous step jump 0.1 -> 0.9 held for 3 frames (0.10s)
    last_adaptive = 0.0
    last_fixed = 0.0
    for step in range(10, 13):
        t = step * dt
        last_adaptive = adaptive.filter(0.9, t)
        last_fixed = fixed_lowpass.filter(0.9, t)

    assert abs(0.9 - last_adaptive) < 0.05
    assert abs(0.9 - last_adaptive) < abs(0.9 - last_fixed)


def test_one_euro_filter_reset_non_monotonic_timestamp_and_parameter_validation() -> None:
    flt = OneEuroFilter(min_cutoff=1.0, beta=0.05, d_cutoff=1.0)
    assert flt.filter(0.2, 1.0) == 0.2
    # Non-increasing timestamp returns previous filtered value safely without division by zero
    assert flt.filter(0.9, 1.0) == 0.2
    assert flt.filter(0.9, 0.95) == 0.2

    flt.reset()
    # After reset, next sample becomes the exact new initial state
    assert flt.filter(0.85, 0.95) == 0.85

    with pytest.raises(ValueError):
        OneEuroFilter(min_cutoff=0.0, beta=0.05, d_cutoff=1.0)
    with pytest.raises(ValueError):
        OneEuroFilter(min_cutoff=1.0, beta=-0.1, d_cutoff=1.0)
    with pytest.raises(ValueError):
        OneEuroFilter(min_cutoff=1.0, beta=0.05, d_cutoff=0.0)


def test_one_euro_filter_2d_respects_enabled_flag_update_config_and_reset() -> None:
    cfg_enabled = OneEuroConfig(enabled=True, min_cutoff=1.0, beta=0.05, d_cutoff=1.0)
    flt2d = OneEuroFilter2D(cfg_enabled)

    dt = 1.0 / 30.0
    x, y = 0.0, 0.0
    for idx in range(25):
        jitter = 0.03 if idx % 2 == 0 else -0.03
        x, y = flt2d.filter(0.5 + jitter, 0.25 - jitter, idx * dt)

    assert abs(x - 0.5) < 0.01
    assert abs(y - 0.25) < 0.01

    cfg_disabled = OneEuroConfig(enabled=False, min_cutoff=1.0, beta=0.05, d_cutoff=1.0)
    flt2d.update_config(cfg_disabled)
    assert flt2d.filter(0.73, 0.41, 1.0) == (0.73, 0.41)

    flt2d.update_config(cfg_enabled)
    flt2d.reset()
    assert flt2d.filter(0.12, 0.88, 1.1) == (0.12, 0.88)
