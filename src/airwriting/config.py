"""Centralized configuration loading and validation for the AirWriting system."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default_config.json"
VALID_RECOGNITION_MODELS = frozenset({"dollar_p", "mobilenet_v3"})
EXPECTED_TOP_LEVEL_KEYS = frozenset(
    {
        "characters",
        "roi",
        "one_euro",
        "m1_dwell",
        "m2_gesture",
        "m3_spacebar",
        "recognition",
        "experiment",
    }
)


class ConfigValidationError(ValueError):
    """Raised when configuration contains invalid values or unknown keys."""


@dataclass(frozen=True)
class CharacterSetConfig:
    expected_strokes_map: dict[str, int]

    @property
    def all_labels(self) -> tuple[str, ...]:
        return tuple(self.expected_strokes_map.keys())

    @property
    def single_stroke_labels(self) -> tuple[str, ...]:
        return tuple(label for label, count in self.expected_strokes_map.items() if count == 1)

    @property
    def multi_stroke_labels(self) -> tuple[str, ...]:
        return tuple(label for label, count in self.expected_strokes_map.items() if count >= 2)

    def char_type(self, label: str) -> Literal["single", "multi"]:
        strokes = self.expected_strokes(label)
        return "single" if strokes == 1 else "multi"

    def expected_strokes(self, label: str) -> int:
        if label not in self.expected_strokes_map:
            raise KeyError(f"Unknown character label: {label!r}")
        return self.expected_strokes_map[label]


@dataclass(frozen=True)
class ROIConfig:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class OneEuroConfig:
    enabled: bool
    min_cutoff: float
    beta: float
    d_cutoff: float


@dataclass(frozen=True)
class M1DwellConfig:
    dwell_start_s: float
    dwell_end_s: float
    dwell_radius: float
    commit_timeout_s: float


@dataclass(frozen=True)
class M2GestureConfig:
    debounce_s: float
    commit_hold_s: float


@dataclass(frozen=True)
class M3SpacebarConfig:
    commit_timeout_s: float
    use_timeout_commit: bool


@dataclass(frozen=True)
class RecognitionConfig:
    active_model: Literal["dollar_p", "mobilenet_v3"]
    confidence_threshold: float
    top_k: int


@dataclass(frozen=True)
class ExperimentConfig:
    max_attempts_per_char: int
    rest_duration_s: float
    false_end_break_s: float


@dataclass(frozen=True)
class AppConfig:
    characters: CharacterSetConfig
    roi: ROIConfig
    one_euro: OneEuroConfig
    m1_dwell: M1DwellConfig
    m2_gesture: M2GestureConfig
    m3_spacebar: M3SpacebarConfig
    recognition: RecognitionConfig
    experiment: ExperimentConfig


def _deep_merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _check_exact_keys(section_name: str, data: Mapping[str, Any], expected: set[str]) -> None:
    actual = set(data.keys())
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise ConfigValidationError(f"Unknown key(s) in {section_name}: {unknown}")
    if missing:
        raise ConfigValidationError(f"Missing key(s) in {section_name}: {missing}")


def _build_section(section_name: str, cls: type[Any], data: Any) -> Any:
    if not isinstance(data, Mapping):
        raise ConfigValidationError(f"Section {section_name!r} must be a mapping")
    expected_keys = {f.name for f in fields(cls)}
    _check_exact_keys(section_name, data, expected_keys)
    return cls(**data)


def _validate_config(config: AppConfig) -> None:
    expected_map = config.characters.expected_strokes_map
    if not expected_map:
        raise ConfigValidationError("Character set must not be empty")
    for label, strokes in expected_map.items():
        if not isinstance(label, str) or not label.strip():
            raise ConfigValidationError(f"Invalid character label: {label!r}")
        if isinstance(strokes, bool) or not isinstance(strokes, int) or strokes < 1:
            raise ConfigValidationError(
                f"Character {label!r} must expect a positive integer stroke count"
            )
    if not config.characters.single_stroke_labels or not config.characters.multi_stroke_labels:
        raise ConfigValidationError(
            "Character set must include both single-stroke and multi-stroke characters"
        )

    roi = config.roi
    if not (0.0 <= roi.x_min < roi.x_max <= 1.0 and 0.0 <= roi.y_min < roi.y_max <= 1.0):
        raise ConfigValidationError(f"Invalid ROI bounds: {roi}")

    oe = config.one_euro
    if (
        not isinstance(oe.enabled, bool)
        or oe.min_cutoff <= 0.0
        or oe.beta < 0.0
        or oe.d_cutoff <= 0.0
    ):
        raise ConfigValidationError(f"Invalid One Euro parameters: {oe}")

    m1 = config.m1_dwell
    if (
        m1.dwell_start_s <= 0.0
        or m1.dwell_end_s <= 0.0
        or m1.dwell_radius <= 0.0
        or m1.commit_timeout_s <= 0.0
    ):
        raise ConfigValidationError(f"Invalid M1 dwell thresholds: {m1}")

    m2 = config.m2_gesture
    if m2.debounce_s < 0.0 or m2.commit_hold_s <= 0.0:
        raise ConfigValidationError(f"Invalid M2 gesture thresholds: {m2}")

    m3 = config.m3_spacebar
    if m3.commit_timeout_s <= 0.0 or not isinstance(m3.use_timeout_commit, bool):
        raise ConfigValidationError(f"Invalid M3 spacebar thresholds: {m3}")

    rec = config.recognition
    if rec.active_model not in VALID_RECOGNITION_MODELS:
        raise ConfigValidationError(f"Unknown recognition model: {rec.active_model!r}")
    if not (0.0 < rec.confidence_threshold <= 1.0) or rec.top_k < 1:
        raise ConfigValidationError(f"Invalid recognition parameters: {rec}")

    exp = config.experiment
    if exp.max_attempts_per_char < 1 or exp.rest_duration_s < 0.0 or exp.false_end_break_s <= 0.0:
        raise ConfigValidationError(f"Invalid experiment parameters: {exp}")


def load_config(
    path: Path | str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    raw: dict[str, Any] = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    if path is not None:
        custom_raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(custom_raw, Mapping):
            raise ConfigValidationError("Config file root must be a JSON object")
        raw = _deep_merge(raw, custom_raw)
    if overrides is not None:
        raw = _deep_merge(raw, overrides)

    _check_exact_keys("config", raw, set(EXPECTED_TOP_LEVEL_KEYS))

    chars_raw = raw["characters"]
    if not isinstance(chars_raw, Mapping):
        raise ConfigValidationError("Section 'characters' must be a mapping")
    _check_exact_keys("characters", chars_raw, {"expected_strokes"})
    expected_strokes = chars_raw["expected_strokes"]
    if not isinstance(expected_strokes, Mapping):
        raise ConfigValidationError("'characters.expected_strokes' must be a mapping")
    characters = CharacterSetConfig(expected_strokes_map=dict(expected_strokes))

    config = AppConfig(
        characters=characters,
        roi=_build_section("roi", ROIConfig, raw["roi"]),
        one_euro=_build_section("one_euro", OneEuroConfig, raw["one_euro"]),
        m1_dwell=_build_section("m1_dwell", M1DwellConfig, raw["m1_dwell"]),
        m2_gesture=_build_section("m2_gesture", M2GestureConfig, raw["m2_gesture"]),
        m3_spacebar=_build_section("m3_spacebar", M3SpacebarConfig, raw["m3_spacebar"]),
        recognition=_build_section("recognition", RecognitionConfig, raw["recognition"]),
        experiment=_build_section("experiment", ExperimentConfig, raw["experiment"]),
    )
    _validate_config(config)
    return config


def save_config(config: AppConfig, path: Path | str) -> None:
    _validate_config(config)
    payload = {
        "characters": {
            "expected_strokes": dict(config.characters.expected_strokes_map),
        },
        "roi": asdict(config.roi),
        "one_euro": asdict(config.one_euro),
        "m1_dwell": asdict(config.m1_dwell),
        "m2_gesture": asdict(config.m2_gesture),
        "m3_spacebar": asdict(config.m3_spacebar),
        "recognition": asdict(config.recognition),
        "experiment": asdict(config.experiment),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
