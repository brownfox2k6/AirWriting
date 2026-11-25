"""Smoke tests for verifying core CPU runtime dependencies."""

import importlib
from types import ModuleType

from airwriting.environment import check_runtime_environment


def test_check_runtime_environment_imports_all_core_libraries_on_cpu() -> None:
    report = check_runtime_environment()

    assert report.ok is True
    assert report.missing == ()
    assert set(report.versions.keys()) == {
        "cv2",
        "mediapipe",
        "PyQt5",
        "torch",
        "torchvision",
        "scipy",
        "numpy",
    }
    assert all(bool(version) for version in report.versions.values())
    assert report.cpu_tensor_check_ok is True


def test_check_runtime_environment_reports_missing_dependency() -> None:
    def failing_importer(name: str) -> ModuleType:
        if name == "mediapipe":
            raise ModuleNotFoundError("No module named 'mediapipe'")
        return importlib.import_module(name)

    report = check_runtime_environment(importer=failing_importer)

    assert report.ok is False
    assert report.missing == ("mediapipe",)
