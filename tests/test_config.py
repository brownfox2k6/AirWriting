"""Tests for centralized configuration loading, overrides, validation, and persistence."""

from pathlib import Path

import pytest

from airwriting.config import ConfigValidationError, load_config, save_config


def test_load_default_config_exposes_16_balanced_characters_and_expected_strokes() -> None:
    config = load_config()

    assert config.characters.single_stroke_labels == ("2", "3", "6", "8", "9", "C", "L", "U")
    assert config.characters.multi_stroke_labels == ("4", "5", "A", "E", "H", "K", "T", "X")
    assert len(config.characters.all_labels) == 16
    assert config.characters.char_type("L") == "single"
    assert config.characters.char_type("A") == "multi"
    assert config.characters.expected_strokes("2") == 1
    assert config.characters.expected_strokes("4") == 2
    assert config.characters.expected_strokes("T") == 2
    assert config.characters.expected_strokes("A") == 3
    assert config.characters.expected_strokes("H") == 3
    assert config.characters.expected_strokes("K") == 3
    assert config.characters.expected_strokes("E") == 4


def test_load_default_config_exposes_interaction_filter_and_experiment_defaults() -> None:
    config = load_config()

    assert 0.0 <= config.roi.x_min < config.roi.x_max <= 1.0
    assert 0.0 <= config.roi.y_min < config.roi.y_max <= 1.0
    assert config.one_euro.min_cutoff > 0.0
    assert config.one_euro.beta >= 0.0
    assert config.one_euro.d_cutoff > 0.0
    assert 0.4 <= config.m1_dwell.dwell_start_s <= 0.5
    assert 0.4 <= config.m1_dwell.dwell_end_s <= 0.5
    assert config.m1_dwell.dwell_radius > 0.0
    assert 1.0 <= config.m1_dwell.commit_timeout_s <= 1.2
    assert config.m2_gesture.debounce_s == 0.12
    assert config.m2_gesture.commit_hold_s == 0.25
    assert config.m3_spacebar.commit_timeout_s == 1.0
    assert config.m3_spacebar.use_timeout_commit is False
    assert config.recognition.active_model == "dollar_p"
    assert config.recognition.confidence_threshold == 0.7
    assert config.recognition.top_k == 3
    assert config.experiment.max_attempts_per_char == 3
    assert config.experiment.rest_duration_s == 120.0
    assert config.experiment.false_end_break_s == 0.3


def test_load_config_applies_nested_overrides_without_clobbering_other_fields() -> None:
    config = load_config(
        overrides={
            "m1_dwell": {"dwell_start_s": 0.35},
            "recognition": {"active_model": "mobilenet_v3"},
        }
    )

    assert config.m1_dwell.dwell_start_s == 0.35
    assert config.m1_dwell.dwell_end_s == 0.45
    assert config.m1_dwell.commit_timeout_s == 1.1
    assert config.recognition.active_model == "mobilenet_v3"
    assert config.recognition.confidence_threshold == 0.7
    assert len(config.characters.all_labels) == 16


@pytest.mark.parametrize(
    "invalid_overrides",
    [
        {"m1_dwell": {"dwell_start_s": 0.0}},
        {"m1_dwell": {"commit_timeout_s": -1.0}},
        {"m2_gesture": {"debounce_s": -0.05}},
        {"roi": {"x_min": 0.8, "x_max": 0.2}},
        {"roi": {"y_max": 1.2}},
        {"recognition": {"confidence_threshold": 0.0}},
        {"recognition": {"confidence_threshold": 1.5}},
        {"recognition": {"active_model": "resnet18"}},
        {"experiment": {"max_attempts_per_char": 0}},
        {
            "characters": {
                "single_stroke_labels": ["2", "A"],
                "multi_stroke_labels": ["A", "E"],
            }
        },
        {"unknown_section": {"foo": 1}},
        {"m1_dwell": {"typo_threshold": 0.5}},
    ],
)
def test_load_config_rejects_invalid_or_unknown_parameters(invalid_overrides: dict) -> None:
    with pytest.raises(ConfigValidationError):
        load_config(overrides=invalid_overrides)


def test_save_and_reload_config_round_trips_custom_thresholds(tmp_path: Path) -> None:
    target = tmp_path / "pilot_frozen.json"
    customized = load_config(
        overrides={
            "m1_dwell": {"dwell_start_s": 0.42, "commit_timeout_s": 1.05},
            "m2_gesture": {"debounce_s": 0.14},
            "recognition": {"active_model": "mobilenet_v3"},
        }
    )

    save_config(customized, target)
    reloaded = load_config(path=target)

    assert reloaded == customized
    assert reloaded.m1_dwell.dwell_start_s == 0.42
    assert reloaded.m2_gesture.debounce_s == 0.14
    assert reloaded.recognition.active_model == "mobilenet_v3"
