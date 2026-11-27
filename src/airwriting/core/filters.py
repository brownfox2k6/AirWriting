"""Real-time One Euro Filter for smoothing 1D and 2D fingertip trajectories."""

from __future__ import annotations

import math

from airwriting.config import OneEuroConfig


def _smoothing_factor(dt: float, cutoff: float) -> float:
    r = 2.0 * math.pi * cutoff * dt
    return r / (r + 1.0)


def _exponential_smoothing(alpha: float, x: float, prev: float) -> float:
    return alpha * x + (1.0 - alpha) * prev


class OneEuroFilter:
    """Adaptive 1D low-pass filter (Casiez et al., 2012) trading jitter for speed."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ) -> None:
        self._validate_params(min_cutoff, beta, d_cutoff)
        self._min_cutoff = float(min_cutoff)
        self._beta = float(beta)
        self._d_cutoff = float(d_cutoff)
        self._prev_x: float | None = None
        self._prev_dx: float = 0.0
        self._prev_t: float | None = None

    @staticmethod
    def _validate_params(min_cutoff: float, beta: float, d_cutoff: float) -> None:
        if min_cutoff <= 0.0 or beta < 0.0 or d_cutoff <= 0.0:
            raise ValueError(
                "OneEuroFilter requires min_cutoff > 0, beta >= 0, d_cutoff > 0; "
                f"got ({min_cutoff}, {beta}, {d_cutoff})"
            )

    def reset(self) -> None:
        """Clear filter history so the next sample starts fresh."""
        self._prev_x = None
        self._prev_dx = 0.0
        self._prev_t = None

    def filter(self, value: float, timestamp: float) -> float:
        """Filter a 1D measurement sampled at `timestamp` seconds."""
        x = float(value)
        t = float(timestamp)
        if self._prev_x is None or self._prev_t is None:
            self._prev_x = x
            self._prev_dx = 0.0
            self._prev_t = t
            return x

        dt = t - self._prev_t
        if dt <= 0.0:
            return self._prev_x

        alpha_d = _smoothing_factor(dt, self._d_cutoff)
        dx = (x - self._prev_x) / dt
        dx_hat = _exponential_smoothing(alpha_d, dx, self._prev_dx)

        cutoff = self._min_cutoff + self._beta * abs(dx_hat)
        alpha = _smoothing_factor(dt, cutoff)
        x_hat = _exponential_smoothing(alpha, x, self._prev_x)

        self._prev_x = x_hat
        self._prev_dx = dx_hat
        self._prev_t = t
        return x_hat


class OneEuroFilter2D:
    """Paired X/Y One Euro Filter driven by `OneEuroConfig`."""

    def __init__(self, config: OneEuroConfig) -> None:
        self._config = config
        self.update_config(config)

    @property
    def config(self) -> OneEuroConfig:
        return self._config

    def reset(self) -> None:
        """Reset both X and Y filters."""
        self._x_filter.reset()
        self._y_filter.reset()

    def update_config(self, config: OneEuroConfig) -> None:
        """Apply new filter settings cleanly and reset filter state."""
        self._config = config
        self._x_filter = OneEuroFilter(
            min_cutoff=config.min_cutoff,
            beta=config.beta,
            d_cutoff=config.d_cutoff,
        )
        self._y_filter = OneEuroFilter(
            min_cutoff=config.min_cutoff,
            beta=config.beta,
            d_cutoff=config.d_cutoff,
        )

    def filter(self, x: float, y: float, timestamp: float) -> tuple[float, float]:
        """Smooth `(x, y)` coordinates, or pass through unchanged when disabled."""
        if not self._config.enabled:
            return (float(x), float(y))
        return (
            self._x_filter.filter(x, timestamp),
            self._y_filter.filter(y, timestamp),
        )
