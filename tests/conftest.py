"""Shared pytest fixtures and Qt event-loop helpers for headless tests."""

from __future__ import annotations

import os
import time
from collections.abc import Callable

import pytest
from PyQt6.QtCore import QCoreApplication


@pytest.fixture(scope="session")
def qt_app() -> QCoreApplication:
    """Provide a single session-wide Qt application instance (QApplication when available)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    existing = QCoreApplication.instance()
    if existing is not None:
        return existing
    try:
        from PyQt6.QtWidgets import QApplication

        return QApplication([])
    except ImportError:
        return QCoreApplication([])


def pump_until(
    app: QCoreApplication,
    predicate: Callable[[], bool],
    *,
    timeout_s: float = 3.0,
) -> bool:
    """Process Qt events until `predicate()` returns True or `timeout_s` elapses."""
    deadline = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    return predicate()
