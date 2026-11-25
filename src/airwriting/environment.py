"""Runtime environment smoke check for CPU core dependencies."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

CORE_DEPENDENCIES: tuple[str, ...] = (
    "cv2",
    "mediapipe",
    "PyQt5",
    "torch",
    "torchvision",
    "scipy",
    "numpy",
)


@dataclass(frozen=True)
class EnvironmentReport:
    python_version: str
    versions: dict[str, str]
    missing: tuple[str, ...]
    cpu_tensor_check_ok: bool

    @property
    def ok(self) -> bool:
        return len(self.missing) == 0 and self.cpu_tensor_check_ok


def _read_module_version(
    name: str, module: ModuleType, importer: Callable[[str], ModuleType]
) -> str:
    version = getattr(module, "__version__", None)
    if isinstance(version, str) and version:
        return version
    if name == "PyQt5":
        qt_core = importer("PyQt5.QtCore")
        qt_ver = getattr(qt_core, "PYQT_VERSION_STR", None)
        if isinstance(qt_ver, str) and qt_ver:
            return qt_ver
    return "unknown"


def check_runtime_environment(
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> EnvironmentReport:
    versions: dict[str, str] = {}
    missing: list[str] = []
    torch_module: ModuleType | None = None

    for dep in CORE_DEPENDENCIES:
        try:
            module = importer(dep)
        except Exception:
            missing.append(dep)
            continue
        versions[dep] = _read_module_version(dep, module, importer)
        if dep == "torch":
            torch_module = module

    cpu_ok = False
    if torch_module is not None:
        tensor = torch_module.zeros((2, 2), device="cpu")
        cpu_ok = float((tensor + 1.0).sum().item()) == 4.0

    return EnvironmentReport(
        python_version=sys.version.split()[0],
        versions=versions,
        missing=tuple(missing),
        cpu_tensor_check_ok=cpu_ok,
    )
