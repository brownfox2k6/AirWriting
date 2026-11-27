"""CLI entry point for launching the AirWriting PyQt6 application."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from PyQt6.QtWidgets import QApplication

from airwriting.config import load_config
from airwriting.ui.main_window import MainWindow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="airwriting",
        description=(
            "AirWriting PyQt6 application with mirrored webcam, Active ROI, and handedness."
        ),
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera device index (default: 0)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a custom JSON configuration file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(path=args.config)

    app = QApplication(sys.argv if argv is None else ["airwriting", *argv])
    window = MainWindow(config, camera_index=args.camera_index, auto_start=True)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
