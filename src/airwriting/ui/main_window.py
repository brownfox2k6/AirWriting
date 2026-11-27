"""Main PyQt6 application window with mirrored webcam, Active ROI, FPS, and handedness toggle."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QCloseEvent, QImage
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from airwriting.config import AppConfig
from airwriting.core.camera import CameraWorker, FrameSource, Handedness, OpenCVFrameSource
from airwriting.ui.camera_view import CameraViewWidget


class MainWindow(QMainWindow):
    """Primary AirWriting PyQt6 window hosting the webcam viewport and live controls."""

    handedness_changed = pyqtSignal(object, arguments=["handedness"])

    def __init__(
        self,
        config: AppConfig,
        *,
        frame_source_factory: Callable[[], FrameSource] | None = None,
        camera_index: int = 0,
        auto_start: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._camera_index = camera_index
        self._frame_source_factory: Callable[[], FrameSource] = (
            frame_source_factory
            if frame_source_factory is not None
            else lambda: OpenCVFrameSource(camera_index=self._camera_index)
        )
        self._handedness = Handedness.RIGHT
        self._fps = 0.0
        self._worker: CameraWorker | None = None

        self.setWindowTitle("AirWriting — Webcam, Active ROI & Tay thuận")
        self.resize(960, 680)

        self._camera_view = CameraViewWidget(config.roi, parent=self)
        self._fps_label = QLabel("FPS: 0.0", parent=self)
        self._handedness_label = QLabel(
            f"Tay thuận: {self._handedness.display_label}",
            parent=self,
        )

        self.right_hand_button = QPushButton("Phải", parent=self)
        self.right_hand_button.setCheckable(True)
        self.right_hand_button.setChecked(True)

        self.left_hand_button = QPushButton("Trái", parent=self)
        self.left_hand_button.setCheckable(True)

        self._hand_group = QButtonGroup(self)
        self._hand_group.setExclusive(True)
        self._hand_group.addButton(self.right_hand_button)
        self._hand_group.addButton(self.left_hand_button)

        self.right_hand_button.clicked.connect(lambda: self.set_handedness(Handedness.RIGHT))
        self.left_hand_button.clicked.connect(lambda: self.set_handedness(Handedness.LEFT))

        self.retry_button = QPushButton("Thử lại", parent=self)
        self.retry_button.clicked.connect(self.start_camera)

        self._build_layout()

        if auto_start:
            self.start_camera()

    def _build_layout(self) -> None:
        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        controls.addWidget(self._handedness_label)
        controls.addWidget(self.right_hand_button)
        controls.addWidget(self.left_hand_button)
        controls.addStretch(1)
        controls.addWidget(self._fps_label)
        controls.addWidget(self.retry_button)

        root_layout.addLayout(controls)
        root_layout.addWidget(self._camera_view, stretch=1)
        self.setCentralWidget(central)

    @property
    def handedness(self) -> Handedness:
        return self._handedness

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def camera_view(self) -> CameraViewWidget:
        return self._camera_view

    @property
    def status_message(self) -> str:
        return self._camera_view.status_message

    def set_handedness(self, handedness: Handedness) -> None:
        """Update dominant writing hand immediately and emit `handedness_changed`."""
        self.right_hand_button.setChecked(handedness is Handedness.RIGHT)
        self.left_hand_button.setChecked(handedness is Handedness.LEFT)
        if handedness is self._handedness:
            return
        self._handedness = handedness
        self._handedness_label.setText(f"Tay thuận: {handedness.display_label}")
        self.handedness_changed.emit(handedness)

    def start_camera(self) -> None:
        """Start (or restart) the background webcam capture worker."""
        self.stop_camera()
        self._fps = 0.0
        self._fps_label.setText("FPS: 0.0")
        self._camera_view.set_status_message("Đang kết nối webcam...")

        worker = CameraWorker(self._frame_source_factory(), mirror=True, parent=self)
        worker.frame_ready.connect(self._on_frame_ready)
        worker.error_occurred.connect(self._on_camera_error)
        self._worker = worker
        worker.start()

    def stop_camera(self) -> None:
        """Stop the background capture worker and release the camera cleanly."""
        if self._worker is not None:
            worker = self._worker
            self._worker = None
            try:
                worker.frame_ready.disconnect(self._on_frame_ready)
            except TypeError:
                pass
            try:
                worker.error_occurred.disconnect(self._on_camera_error)
            except TypeError:
                pass
            worker.stop()
            worker.deleteLater()

    def _on_frame_ready(self, image: QImage, fps: float) -> None:
        self._fps = fps
        self._fps_label.setText(f"FPS: {fps:.1f}")
        self._camera_view.set_frame(image)

    def _on_camera_error(self, message: str) -> None:
        self._fps = 0.0
        self._fps_label.setText("FPS: 0.0")
        self._camera_view.set_status_message(message)

    def closeEvent(self, event: QCloseEvent | None) -> None:
        self.stop_camera()
        super().closeEvent(event)
