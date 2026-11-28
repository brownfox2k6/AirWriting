"""Tests for immutable character trajectory values and public behavior."""

import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from airwriting.config import load_config
from airwriting.core.trajectory import (
    CharacterSample,
    Stroke,
    TrajectoryPoint,
    TrajectoryValidationError,
    load_sample,
    sample_from_dict,
    sample_to_dict,
    save_sample,
    validate_sample,
)


def _valid_sample(
    *, label: str = "C", strokes: tuple[Stroke, ...] | None = None
) -> CharacterSample:
    selected_strokes = (
        (
            Stroke(
                points=(
                    TrajectoryPoint(0.1, 0.2, 0.0),
                    TrajectoryPoint(0.2, 0.3, 0.1),
                )
            ),
        )
        if strokes is None
        else strokes
    )
    return CharacterSample(
        label=label,
        writer_id="W01",
        mechanism="M1",
        roi_size=(640, 480),
        strokes=selected_strokes,
    )


def test_trajectory_values_are_frozen_and_nested_collections_are_tuples() -> None:
    first = TrajectoryPoint(x=0.1, y=0.2, t=0.0)
    second = TrajectoryPoint(x=0.3, y=0.4, t=0.5)
    stroke = Stroke(points=(first, second))
    sample = CharacterSample(
        label="C",
        writer_id="W01",
        mechanism="M1",
        roi_size=(640, 480),
        strokes=(stroke,),
    )

    assert isinstance(stroke.points, tuple)
    assert isinstance(sample.strokes, tuple)
    with pytest.raises(FrozenInstanceError):
        first.x = 0.5  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        stroke.points = (second,)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        sample.label = "A"  # type: ignore[misc]


def test_sample_measures_count_strokes_duration_and_within_stroke_length() -> None:
    sample = CharacterSample(
        label="A",
        writer_id="W01",
        mechanism="M2",
        roi_size=(640, 480),
        strokes=(
            Stroke(
                points=(
                    TrajectoryPoint(0.0, 0.0, 1.0),
                    TrajectoryPoint(0.3, 0.4, 1.5),
                )
            ),
            Stroke(
                points=(
                    TrajectoryPoint(0.8, 0.8, 3.0),
                    TrajectoryPoint(0.8, 0.9, 4.0),
                )
            ),
        ),
    )

    assert sample.num_strokes == 2
    assert sample.is_multi_stroke is True
    assert sample.duration_s == 3.0
    assert sample.path_length() == pytest.approx(0.6)
    single_stroke = _valid_sample()
    assert single_stroke.num_strokes == 1
    assert single_stroke.is_multi_stroke is False


def test_validation_uses_configured_labels_without_enforcing_expected_stroke_count() -> None:
    sample = CharacterSample(
        label="C",
        writer_id="W01",
        mechanism="M1",
        roi_size=(640, 480),
        strokes=(
            Stroke(points=(TrajectoryPoint(0.1, 0.2, 0.0), TrajectoryPoint(0.2, 0.3, 0.1))),
            Stroke(points=(TrajectoryPoint(0.3, 0.4, 0.2), TrajectoryPoint(0.4, 0.5, 0.3))),
        ),
    )

    assert validate_sample(sample) is None
    with pytest.raises(TrajectoryValidationError):
        validate_sample(
            CharacterSample(
                label="?",
                writer_id="W01",
                mechanism="M1",
                roi_size=(640, 480),
                strokes=sample.strokes,
            )
        )


def test_validation_accepts_labels_added_by_a_custom_config() -> None:
    sample = CharacterSample(
        label="?",
        writer_id="W01",
        mechanism="COLLECTOR",
        roi_size=(640, 480),
        strokes=(Stroke(points=(TrajectoryPoint(0.1, 0.2, 0.0), TrajectoryPoint(0.2, 0.3, 0.1))),),
    )
    config = load_config(overrides={"characters": {"expected_strokes": {"?": 2}}})

    assert validate_sample(sample, config=config) is None


@pytest.mark.parametrize(
    "sample",
    [
        _valid_sample(),
        _valid_sample(
            label="A",
            strokes=(
                Stroke(points=(TrajectoryPoint(0.0, 0.0, 0.0), TrajectoryPoint(0.2, 0.2, 0.1))),
                Stroke(points=(TrajectoryPoint(0.4, 0.4, 0.1), TrajectoryPoint(0.6, 0.6, 0.2))),
                Stroke(points=(TrajectoryPoint(0.8, 0.8, 0.2), TrajectoryPoint(1.0, 1.0, 0.3))),
            ),
        ),
    ],
)
def test_validation_accepts_single_and_three_stroke_samples(sample: CharacterSample) -> None:
    assert validate_sample(sample) is None


def test_validation_allows_equal_timestamps_and_inclusive_coordinate_boundaries() -> None:
    sample = _valid_sample(
        strokes=(
            Stroke(points=(TrajectoryPoint(0.0, 0.0, 0.0), TrajectoryPoint(0.5, 0.5, 0.0))),
            Stroke(points=(TrajectoryPoint(1.0, 1.0, 0.0), TrajectoryPoint(0.8, 0.8, 0.0))),
        )
    )

    assert validate_sample(sample) is None


def test_validation_accepts_integer_values_for_float_fields() -> None:
    sample = _valid_sample(
        strokes=(Stroke(points=(TrajectoryPoint(0, 0, 0), TrajectoryPoint(1, 1, 1))),)
    )

    assert validate_sample(sample) is None
    assert sample.duration_s == 1.0
    assert isinstance(sample.duration_s, float)


@pytest.mark.parametrize(
    "strokes",
    [
        (),
        (Stroke(points=(TrajectoryPoint(0.1, 0.2, 0.0),)),),
    ],
)
def test_validation_rejects_empty_samples_and_strokes_with_too_few_points(
    strokes: tuple[Stroke, ...],
) -> None:
    with pytest.raises(TrajectoryValidationError):
        validate_sample(_valid_sample(strokes=strokes))


@pytest.mark.parametrize(
    "point",
    [
        TrajectoryPoint(-0.01, 0.5, 0.0),
        TrajectoryPoint(1.01, 0.5, 0.0),
        TrajectoryPoint(0.5, -0.01, 0.0),
        TrajectoryPoint(0.5, 1.01, 0.0),
        TrajectoryPoint(0.5, 0.5, -0.01),
        TrajectoryPoint(math.nan, 0.5, 0.0),
        TrajectoryPoint(0.5, math.inf, 0.0),
        TrajectoryPoint(0.5, 0.5, -math.inf),
        TrajectoryPoint(True, 0.5, 0.0),  # type: ignore[arg-type]
        TrajectoryPoint(0.5, False, 0.0),  # type: ignore[arg-type]
        TrajectoryPoint(0.5, 0.5, True),  # type: ignore[arg-type]
    ],
)
def test_validation_rejects_invalid_coordinates_and_timestamps(point: TrajectoryPoint) -> None:
    sample = _valid_sample(strokes=(Stroke(points=(point, TrajectoryPoint(0.5, 0.5, 1.0))),))

    with pytest.raises(TrajectoryValidationError):
        validate_sample(sample)


@pytest.mark.parametrize(
    ("x", "y", "t"),
    [
        (math.nan, 0.5, 0.0),
        (math.inf, 0.5, 0.0),
        (-math.inf, 0.5, 0.0),
        (0.5, math.nan, 0.0),
        (0.5, math.inf, 0.0),
        (0.5, -math.inf, 0.0),
        (0.5, 0.5, math.nan),
        (0.5, 0.5, math.inf),
        (0.5, 0.5, -math.inf),
    ],
)
def test_validation_rejects_every_non_finite_numeric_field(x: float, y: float, t: float) -> None:
    sample = _valid_sample(
        strokes=(Stroke(points=(TrajectoryPoint(x, y, t), TrajectoryPoint(0.5, 0.5, 1.0))),)
    )

    with pytest.raises(TrajectoryValidationError):
        validate_sample(sample)


def test_validation_rejects_time_that_moves_backwards_between_strokes() -> None:
    sample = _valid_sample(
        strokes=(
            Stroke(points=(TrajectoryPoint(0.1, 0.2, 1.0), TrajectoryPoint(0.2, 0.3, 2.0))),
            Stroke(points=(TrajectoryPoint(0.3, 0.4, 1.9), TrajectoryPoint(0.4, 0.5, 3.0))),
        )
    )

    with pytest.raises(TrajectoryValidationError):
        validate_sample(sample)


def test_validation_rejects_time_that_moves_backwards_within_a_stroke() -> None:
    sample = _valid_sample(
        strokes=(Stroke(points=(TrajectoryPoint(0.1, 0.2, 1.0), TrajectoryPoint(0.2, 0.3, 0.9))),)
    )

    with pytest.raises(TrajectoryValidationError):
        validate_sample(sample)


@pytest.mark.parametrize(
    "roi_size",
    [(0, 480), (640, -1), (True, 480), (640.0, 480), ("640", 480)],
)
def test_validation_rejects_invalid_roi_dimensions(roi_size: tuple[int, int]) -> None:
    with pytest.raises(TrajectoryValidationError):
        validate_sample(
            CharacterSample(
                label="C",
                writer_id="W01",
                mechanism="M1",
                roi_size=roi_size,
                strokes=_valid_sample().strokes,
            )
        )


def test_validation_rejects_empty_writer_id() -> None:
    sample = CharacterSample(
        label="C",
        writer_id="",
        mechanism="M1",
        roi_size=(640, 480),
        strokes=_valid_sample().strokes,
    )

    with pytest.raises(TrajectoryValidationError):
        validate_sample(sample)


def test_sample_dictionary_preserves_schema_metadata_and_nested_order() -> None:
    sample = CharacterSample(
        label="A",
        writer_id="W02",
        mechanism="M3",
        roi_size=(800, 600),
        strokes=(
            Stroke(points=(TrajectoryPoint(0.1, 0.2, 0.0), TrajectoryPoint(0.3, 0.4, 0.5))),
            Stroke(points=(TrajectoryPoint(0.8, 0.7, 1.0), TrajectoryPoint(0.6, 0.5, 1.5))),
        ),
        timestamp_iso="2026-11-28T12:00:00Z",
        notes="pilot sample",
    )
    expected = {
        "label": "A",
        "writer_id": "W02",
        "mechanism": "M3",
        "roi_size": [800, 600],
        "strokes": [
            {"points": [{"x": 0.1, "y": 0.2, "t": 0.0}, {"x": 0.3, "y": 0.4, "t": 0.5}]},
            {"points": [{"x": 0.8, "y": 0.7, "t": 1.0}, {"x": 0.6, "y": 0.5, "t": 1.5}]},
        ],
        "timestamp_iso": "2026-11-28T12:00:00Z",
        "notes": "pilot sample",
    }

    assert sample_to_dict(sample) == expected
    assert list(sample_to_dict(sample)) == [
        "label",
        "writer_id",
        "mechanism",
        "roi_size",
        "strokes",
        "timestamp_iso",
        "notes",
    ]
    assert sample_from_dict(expected) == sample


@pytest.mark.parametrize("malformation", ["missing", "extra", "wrong_type", "unknown_label"])
def test_sample_from_dict_rejects_malformed_or_invalid_data(malformation: str) -> None:
    data = sample_to_dict(_valid_sample())
    if malformation == "missing":
        del data["notes"]
    elif malformation == "extra":
        data["unused"] = True
    elif malformation == "wrong_type":
        data["strokes"] = "not an array"
    else:
        data["label"] = "?"

    with pytest.raises(TrajectoryValidationError):
        sample_from_dict(data)


def test_sample_from_dict_rejects_numbers_that_cannot_be_represented_as_floats() -> None:
    data = sample_to_dict(_valid_sample())
    data["strokes"][0]["points"][0]["t"] = 10**1000

    with pytest.raises(TrajectoryValidationError):
        sample_from_dict(data)


def test_file_round_trip_preserves_nested_order_and_utf8_metadata(tmp_path: Path) -> None:
    sample = CharacterSample(
        label="A",
        writer_id="W03",
        mechanism="COLLECTOR",
        roi_size=(800, 600),
        strokes=(
            Stroke(points=(TrajectoryPoint(0.1, 0.2, 0.0), TrajectoryPoint(0.3, 0.4, 0.5))),
            Stroke(points=(TrajectoryPoint(0.8, 0.7, 1.0), TrajectoryPoint(0.6, 0.5, 1.5))),
        ),
        timestamp_iso="2026-11-28T12:00:00Z",
        notes="thử nghiệm",
    )
    destination = tmp_path / "nested" / "sample.json"

    save_sample(sample, destination)

    assert load_sample(destination) == sample
    assert destination.read_bytes().decode("utf-8").endswith("\n")


def test_file_loader_wraps_malformed_json_but_preserves_filesystem_errors(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json", encoding="utf-8")

    with pytest.raises(TrajectoryValidationError) as exc_info:
        load_sample(malformed)
    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)
    with pytest.raises(FileNotFoundError):
        load_sample(tmp_path / "missing.json")


def test_file_loader_wraps_invalid_utf8_as_invalid_sample_data(tmp_path: Path) -> None:
    invalid_encoding = tmp_path / "invalid-encoding.json"
    invalid_encoding.write_bytes(b"\xff")

    with pytest.raises(TrajectoryValidationError):
        load_sample(invalid_encoding)


def test_file_round_trip_uses_a_custom_config_for_labels(tmp_path: Path) -> None:
    sample = _valid_sample(label="?")
    config = load_config(overrides={"characters": {"expected_strokes": {"?": 2}}})
    destination = tmp_path / "custom.json"

    save_sample(sample, destination, config=config)

    assert load_sample(destination, config=config) == sample
