"""Immutable domain values for character trajectory samples."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from airwriting.config import AppConfig, load_config


class TrajectoryValidationError(ValueError):
    """Raised when a character trajectory is invalid or malformed."""


@dataclass(frozen=True)
class TrajectoryPoint:
    """One normalized ROI coordinate sampled at relative time ``t`` in seconds."""

    x: float
    y: float
    t: float


@dataclass(frozen=True)
class Stroke:
    """A continuous pen-down segment containing ordered trajectory points."""

    points: tuple[TrajectoryPoint, ...]


@dataclass(frozen=True)
class CharacterSample:
    """A labeled character sample containing one or more ordered pen strokes."""

    label: str
    writer_id: str
    mechanism: str
    roi_size: tuple[int, int]
    strokes: tuple[Stroke, ...]
    timestamp_iso: str = ""
    notes: str = ""

    @property
    def num_strokes(self) -> int:
        """Return the number of pen-down segments in this sample."""
        return len(self.strokes)

    @property
    def is_multi_stroke(self) -> bool:
        """Return whether this sample contains at least two strokes."""
        return self.num_strokes >= 2

    @property
    def duration_s(self) -> float:
        """Return elapsed time from the first to the final point in the sample."""
        first_time = self.strokes[0].points[0].t
        last_time = self.strokes[-1].points[-1].t
        return float(last_time - first_time)

    def path_length(self) -> float:
        """Return Euclidean path length, excluding pen-up jumps between strokes."""
        return sum(
            math.hypot(next_point.x - point.x, next_point.y - point.y)
            for stroke in self.strokes
            for point, next_point in zip(stroke.points, stroke.points[1:], strict=False)
        )


def validate_sample(sample: CharacterSample, config: AppConfig | None = None) -> None:
    """Validate the configured label, stroke structure, coordinates, and timestamps."""
    if not isinstance(sample, CharacterSample):
        raise TrajectoryValidationError("Sample must be a CharacterSample")
    selected_config = load_config() if config is None else config
    if (
        not isinstance(sample.label, str)
        or sample.label not in selected_config.characters.all_labels
    ):
        raise TrajectoryValidationError(f"Unknown character label: {sample.label!r}")
    if not isinstance(sample.writer_id, str) or not sample.writer_id.strip():
        raise TrajectoryValidationError("writer_id must be a non-empty string")
    if not isinstance(sample.mechanism, str):
        raise TrajectoryValidationError("mechanism must be a string")
    if not isinstance(sample.timestamp_iso, str) or not isinstance(sample.notes, str):
        raise TrajectoryValidationError("timestamp_iso and notes must be strings")
    if (
        not isinstance(sample.roi_size, tuple)
        or len(sample.roi_size) != 2
        or any(
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
            for dimension in sample.roi_size
        )
    ):
        raise TrajectoryValidationError("roi_size must contain positive integer dimensions")
    if not isinstance(sample.strokes, tuple) or not sample.strokes:
        raise TrajectoryValidationError("A sample must contain at least one stroke")
    previous_time: int | float | None = None
    for index, stroke in enumerate(sample.strokes):
        if not isinstance(stroke, Stroke) or not isinstance(stroke.points, tuple):
            raise TrajectoryValidationError(f"Stroke {index} must contain a tuple of points")
        if len(stroke.points) < 2:
            raise TrajectoryValidationError(f"Stroke {index} must contain at least two points")
        for point_index, point in enumerate(stroke.points):
            if not isinstance(point, TrajectoryPoint):
                raise TrajectoryValidationError(
                    f"Point {point_index} in stroke {index} must be a TrajectoryPoint"
                )
            for coordinate_name, value in (
                ("x", point.x),
                ("y", point.y),
                ("t", point.t),
            ):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TrajectoryValidationError(
                        f"Point {point_index} in stroke {index} has invalid {coordinate_name}"
                    )
                try:
                    is_finite = math.isfinite(value)
                except OverflowError:
                    is_finite = False
                if not is_finite:
                    raise TrajectoryValidationError(
                        f"Point {point_index} in stroke {index} has non-finite {coordinate_name}"
                    )
            if not 0.0 <= point.x <= 1.0 or not 0.0 <= point.y <= 1.0:
                raise TrajectoryValidationError(
                    f"Point {point_index} in stroke {index} is outside normalized ROI bounds"
                )
            if point.t < 0.0:
                raise TrajectoryValidationError("Trajectory timestamps must be non-negative")
            if previous_time is not None and point.t < previous_time:
                raise TrajectoryValidationError(
                    "Trajectory timestamps must be non-decreasing across the sample"
                )
            previous_time = point.t


def sample_to_dict(sample: CharacterSample) -> dict[str, Any]:
    """Convert a sample to the stable JSON-compatible mapping contract."""
    return {
        "label": sample.label,
        "writer_id": sample.writer_id,
        "mechanism": sample.mechanism,
        "roi_size": list(sample.roi_size),
        "strokes": [
            {"points": [{"x": point.x, "y": point.y, "t": point.t} for point in stroke.points]}
            for stroke in sample.strokes
        ],
        "timestamp_iso": sample.timestamp_iso,
        "notes": sample.notes,
    }


def _require_exact_keys(data: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(data.keys())
    missing = sorted(expected - actual)
    unknown = sorted(str(key) for key in actual - expected)
    if missing or unknown:
        raise TrajectoryValidationError(
            f"Invalid fields in {context}: missing={missing}, unknown={unknown}"
        )


def _sample_from_dict(data: Mapping[str, Any], config: AppConfig | None) -> CharacterSample:
    if not isinstance(data, Mapping):
        raise TrajectoryValidationError("Sample data must be a mapping")
    _require_exact_keys(
        data,
        {"label", "writer_id", "mechanism", "roi_size", "strokes", "timestamp_iso", "notes"},
        "sample",
    )

    label = data["label"]
    writer_id = data["writer_id"]
    mechanism = data["mechanism"]
    timestamp_iso = data["timestamp_iso"]
    notes = data["notes"]
    if not all(
        isinstance(value, str) for value in (label, writer_id, mechanism, timestamp_iso, notes)
    ):
        raise TrajectoryValidationError("Sample metadata fields must be strings")

    roi_size = data["roi_size"]
    if not isinstance(roi_size, (list, tuple)) or len(roi_size) != 2:
        raise TrajectoryValidationError("roi_size must be a two-element array")
    if any(isinstance(dimension, bool) or not isinstance(dimension, int) for dimension in roi_size):
        raise TrajectoryValidationError("roi_size dimensions must be integers")

    raw_strokes = data["strokes"]
    if not isinstance(raw_strokes, (list, tuple)):
        raise TrajectoryValidationError("strokes must be an array")
    strokes: list[Stroke] = []
    for stroke_index, raw_stroke in enumerate(raw_strokes):
        if not isinstance(raw_stroke, Mapping):
            raise TrajectoryValidationError(f"Stroke {stroke_index} must be an object")
        _require_exact_keys(raw_stroke, {"points"}, f"stroke {stroke_index}")
        raw_points = raw_stroke["points"]
        if not isinstance(raw_points, (list, tuple)):
            raise TrajectoryValidationError(f"Points in stroke {stroke_index} must be an array")
        points: list[TrajectoryPoint] = []
        for point_index, raw_point in enumerate(raw_points):
            if not isinstance(raw_point, Mapping):
                raise TrajectoryValidationError(
                    f"Point {point_index} in stroke {stroke_index} must be an object"
                )
            _require_exact_keys(raw_point, {"x", "y", "t"}, f"point {point_index}")
            values = (raw_point["x"], raw_point["y"], raw_point["t"])
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float)) for value in values
            ):
                raise TrajectoryValidationError(
                    f"Point {point_index} in stroke {stroke_index} must contain numeric values"
                )
            points.append(TrajectoryPoint(x=values[0], y=values[1], t=values[2]))
        strokes.append(Stroke(points=tuple(points)))

    sample = CharacterSample(
        label=label,
        writer_id=writer_id,
        mechanism=mechanism,
        roi_size=(roi_size[0], roi_size[1]),
        strokes=tuple(strokes),
        timestamp_iso=timestamp_iso,
        notes=notes,
    )
    validate_sample(sample, config=config)
    return sample


def sample_from_dict(data: Mapping[str, Any]) -> CharacterSample:
    """Build and validate a sample from the JSON-compatible mapping contract."""
    return _sample_from_dict(data, config=None)


def save_sample(
    sample: CharacterSample,
    path: Path | str,
    config: AppConfig | None = None,
) -> None:
    """Validate and write a sample as UTF-8 JSON, creating parent directories."""
    validate_sample(sample, config=config)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(sample_to_dict(sample), indent=2, allow_nan=False)
    destination.write_text(payload + "\n", encoding="utf-8")


def load_sample(path: Path | str, config: AppConfig | None = None) -> CharacterSample:
    """Read, parse, and validate a sample from UTF-8 JSON."""
    source = Path(path)
    try:
        raw_data = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TrajectoryValidationError(f"Invalid JSON sample file: {source}") from exc
    if not isinstance(raw_data, Mapping):
        raise TrajectoryValidationError("Sample JSON root must be an object")
    return _sample_from_dict(raw_data, config=config)
