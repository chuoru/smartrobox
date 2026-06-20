##
# @file qt_app.py
#
# @brief Qt application for displaying a MuJoCo simulation scene and live camera feeds.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/20.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import os
import sys
import time
from collections import deque
from enum import Enum

# External library
import cv2
import mujoco
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QImage, QPixmap, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMenu,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# Internal library
from app.config import Config
from app.controller import Controller
from app.logger import Logger

_ENV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "projects", "anlab", "envs", "anlab_mujoco.xml"
)

_DEVICE_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "projects", "anlab", "device.yaml"
)

_format = QSurfaceFormat()
_format.setDepthBufferSize(24)
_format.setStencilBufferSize(8)
_format.setSamples(4)
_format.setSwapInterval(1)
_format.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
_format.setVersion(2, 0)
_format.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
QSurfaceFormat.setDefaultFormat(_format)


class Mode(Enum):
    """! Application layout mode."""

    NORMAL = "normal"
    MASSAGE = "massage"


class Viewport(QOpenGLWidget):
    """! OpenGL viewport that renders a MuJoCo scene at 60 Hz.

    Supports mouse-driven camera control (rotate, pan, zoom) and emits
    per-frame average render time via updateRuntime.
    """

    updateRuntime = Signal(float)

    def __init__(self, model, data, cam, opt, scn) -> None:
        """! Initialise the viewport with MuJoCo simulation objects.

        @param model<mujoco.MjModel>: Compiled MuJoCo model.
        @param data<mujoco.MjData>: Simulation state.
        @param cam<mujoco.MjvCamera>: Camera descriptor.
        @param opt<mujoco.MjvOption>: Visualisation options.
        @param scn<mujoco.MjvScene>: Scene graph.
        """
        super().__init__()
        self._model = model
        self._data = data
        self._cam = cam
        self._opt = opt
        self._scn = scn
        self._last_pos = None
        self._runtime = deque(maxlen=1000)
        self._timer = QTimer()
        self._timer.setInterval(int(1 / 60 * 1000))
        self._timer.timeout.connect(self.update)
        self._timer.start()

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def mousePressEvent(self, event) -> None:
        self._last_pos = event.position()

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.RightButton:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif event.buttons() & Qt.MouseButton.LeftButton:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif event.buttons() & Qt.MouseButton.MiddleButton:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        else:
            return
        pos = event.position()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()
        mujoco.mjv_moveCamera(
            self._model, action, dx / self.height(), dy / self.height(), self._scn, self._cam
        )
        self._last_pos = pos

    def wheelEvent(self, event) -> None:
        mujoco.mjv_moveCamera(
            self._model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0,
            -0.0005 * event.angleDelta().y(),
            self._scn,
            self._cam,
        )

    def initializeGL(self) -> None:
        self._con = mujoco.MjrContext(self._model, mujoco.mjtFontScale.mjFONTSCALE_100)

    def resizeGL(self, w: int, h: int) -> None:
        pass

    def paintGL(self) -> None:
        t = time.time()
        mujoco.mjv_updateScene(
            self._model,
            self._data,
            self._opt,
            None,
            self._cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self._scn,
        )
        ratio = self.devicePixelRatio()
        viewport = mujoco.MjrRect(0, 0, int(self.width() * ratio), int(self.height() * ratio))
        mujoco.mjr_render(viewport, self._scn, self._con)
        self._runtime.append(time.time() - t)
        self.updateRuntime.emit(float(np.average(self._runtime)))


class _SimThread(QThread):
    """! Background thread that steps the MuJoCo simulation at full speed."""

    def __init__(self, model, data, parent=None) -> None:
        """! Initialise with shared model and data.

        @param model<mujoco.MjModel>: Compiled MuJoCo model.
        @param data<mujoco.MjData>: Mutable simulation state.
        @param parent<QObject|None>: Optional Qt parent.
        """
        super().__init__(parent)
        self._model = model
        self._data = data
        self._running = True

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def run(self) -> None:
        while self._running:
            mujoco.mj_step(self._model, self._data)

    def stop(self) -> None:
        """! Signal the thread to stop and block until it exits."""
        self._running = False
        self.wait()


class _PoseEstimationThread(QThread):
    """! Background QThread that runs YOLO11 pose estimation on a camera feed.

    Loads the YOLO model inside run() so the main thread is never blocked at
    construction time.  Emits frameReady with an annotated BGR ndarray on each
    successful inference, or None when no camera frame is available.
    """

    frameReady = Signal(object)

    def __init__(
        self,
        controller: Controller,
        camera_name: str,
        model_name: str = "yolo11n-pose.pt",
        parent=None,
    ) -> None:
        """! Store parameters; the YOLO model is deferred to run().

        @param controller<Controller>: Device controller used for frame retrieval.
        @param camera_name<str>: Registered name of the head Orbbec camera.
        @param model_name<str>: YOLO11 pose model weight name or path.
        @param parent<QObject|None>: Optional Qt parent.
        """
        super().__init__(parent)
        self._controller = controller
        self._camera_name = camera_name
        self._model_name = model_name
        self._running = False

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def run(self) -> None:
        from ultralytics import YOLO

        model = YOLO(self._model_name)
        self._running = True
        while self._running:
            try:
                frame = self._controller.execute(self._camera_name, "get_color_frame")
            except Exception:
                frame = None
            if frame is None:
                self.frameReady.emit(None)
            else:
                results = model(frame, verbose=False)
                self.frameReady.emit(results[0].plot())

    def stop(self) -> None:
        """! Signal the thread to stop and block until it exits."""
        self._running = False
        self.quit()
        self.wait()


class _HeadCameraPanel(QWidget):
    """! Left panel for massage mode: head camera feed with live YOLO11 pose overlay.

    Runs _PoseEstimationThread in the background and updates _CameraWidget via a
    Qt signal-slot QueuedConnection on each annotated frame, keeping UI updates
    on the main thread.
    """

    def __init__(
        self,
        controller: Controller,
        camera_name: str,
        model_name: str = "yolo11n-pose.pt",
        parent=None,
    ) -> None:
        """! Build the widget and start the pose estimation thread.

        @param controller<Controller>: Device controller for frame retrieval.
        @param camera_name<str>: Registered name of the head Orbbec camera.
        @param model_name<str>: YOLO11 pose model weight name or path.
        @param parent<QWidget|None>: Optional Qt parent.
        """
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._camera_widget = _CameraWidget()
        layout.addWidget(self._camera_widget, stretch=1)
        self._thread = _PoseEstimationThread(controller, camera_name, model_name, parent=self)
        self._thread.frameReady.connect(self._on_frame)
        self._thread.start()

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def stop(self) -> None:
        """! Stop the pose estimation thread and block until it exits."""
        self._thread.stop()

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    @Slot(object)
    def _on_frame(self, frame: np.ndarray | None) -> None:
        """! Receive an annotated frame from the pose thread and update the display.

        @param frame<np.ndarray|None>: BGR annotated frame, or None for no signal.
        """
        self._camera_widget.set_frame(frame, "No signal")


class _CameraWidget(QLabel):
    """! QLabel-based widget that renders a single Orbbec camera's BGR color frame.

    Converts incoming BGR numpy arrays to a scaled QPixmap.  When no frame is
    available, a centered placeholder string is displayed instead.
    """

    def __init__(self, parent=None) -> None:
        """! Initialise with blank placeholder state.

        @param parent<QWidget|None>: Optional Qt parent.
        """
        super().__init__(parent)
        self._last_pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setText("No signal")
        self.setStyleSheet("color: gray; font-size: 14px;")

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def set_frame(self, frame: np.ndarray | None, placeholder: str = "No signal") -> None:
        """! Update the displayed image from a BGR numpy array.

        @param frame<np.ndarray|None>: H×W×3 uint8 BGR array, or None.
        @param placeholder<str>: Text shown when frame is None.
        """
        if frame is None:
            self._last_pixmap = None
            self.setPixmap(QPixmap())
            self.setText(placeholder)
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        q_img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self._last_pixmap = pixmap
        self.setText("")
        self.setPixmap(
            pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        """! Re-scale the cached pixmap when the widget is resized.

        @param event<QResizeEvent>: Resize event from Qt.
        """
        super().resizeEvent(event)
        if self._last_pixmap is not None:
            self.setPixmap(
                self._last_pixmap.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )


class _CameraPanel(QWidget):
    """! Side panel that displays all orbbec camera feeds simultaneously.

    One _CameraWidget per camera is stacked vertically.  A 30 Hz QTimer
    refreshes every feed on each tick.  Camera lifecycle (open/close) is
    handled by the caller; this panel only reads frames.
    """

    _REFRESH_INTERVAL_MS: int = 33

    def __init__(self, controller: Controller, camera_names: list[str], parent=None) -> None:
        """! Build one camera widget per name and start the refresh timer.

        @param controller<Controller>: Device controller used for frame retrieval.
        @param camera_names<list[str]>: Ordered list of orbbec device names to display.
        @param parent<QWidget|None>: Optional Qt parent.
        """
        super().__init__(parent)
        self._controller = controller
        self._camera_names = camera_names
        self._logger = Logger("CameraPanel", Logger.MAGENTA)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._widgets: dict[str, _CameraWidget] = {}
        for name in camera_names:
            widget = _CameraWidget()
            self._widgets[name] = widget
            layout.addWidget(widget, stretch=1)

        self._timer = QTimer()
        self._timer.setInterval(self._REFRESH_INTERVAL_MS)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    @Slot()
    def _refresh(self) -> None:
        """! Fetch the current color frame for every camera and update its widget."""
        for name, widget in self._widgets.items():
            try:
                frame = self._controller.execute(name, "get_color_frame")
            except Exception as exc:
                self._logger.error(f"Frame fetch failed for '{name}': {exc}")
                frame = None
            widget.set_frame(frame)


class _MassageRightPanel(QWidget):
    """! Right panel for massage mode: MuJoCo viewport above the remaining camera feeds.

    Stacks the viewport (stretch 2) on top of a _CameraPanel for the non-primary
    cameras (stretch 1).
    """

    def __init__(
        self,
        viewport: Viewport,
        controller: Controller,
        camera_names: list[str],
        parent=None,
    ) -> None:
        """! Build the composite panel.

        @param viewport<Viewport>: Shared MuJoCo viewport widget.
        @param controller<Controller>: Device controller for camera frame retrieval.
        @param camera_names<list[str]>: Camera device names to show below the viewport.
        @param parent<QWidget|None>: Optional Qt parent.
        """
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(viewport, stretch=2)
        layout.addWidget(_CameraPanel(controller, camera_names), stretch=1)


class MuJoCoWindow(QMainWindow):
    """! Main window that hosts the MuJoCo viewport and an optional camera panel.

    Supports two layout modes toggled via the View menu or the M key:
    - Normal: MuJoCo viewer on the left, all camera feeds on the right.
    - Massage: Head camera (device index 2) on the left; MuJoCo viewer with
      remaining cameras stacked on the right.
    """

    def __init__(
        self,
        xml_path: str,
        controller: Controller | None = None,
        mode: Mode = Mode.NORMAL,
    ) -> None:
        """! Load the scene from an XML file and build the UI.

        @param xml_path<str>: Absolute path to the MuJoCo XML scene file.
        @param controller<Controller|None>: Device controller; when None the
            camera panel is omitted and only the MuJoCo viewport is shown.
        @param mode<Mode>: Initial layout mode.
        """
        super().__init__()
        self._controller = controller
        self._mode = mode
        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data = mujoco.MjData(self._model)
        self._cam = self._create_free_camera()
        self._opt = mujoco.MjvOption()
        self._scn = mujoco.MjvScene(self._model, maxgeom=10000)
        self._viewport = Viewport(self._model, self._data, self._cam, self._opt, self._scn)
        self._viewport.updateRuntime.connect(self._show_runtime)

        self._orbbec_names: list[str] = []
        if controller is not None:
            self._orbbec_names = [
                name
                for name in controller.list_devices()
                if controller.status(name)["type"] == "orbbec"
            ]

        self._build_layout()
        self._build_menu()
        self.showMaximized()
        self._sim_thread = _SimThread(self._model, self._data, self)
        self._sim_thread.start()
        self._head_panel: _HeadCameraPanel | None = None

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def closeEvent(self, event) -> None:
        """! Stop all threads and close all camera devices before closing."""
        if self._head_panel is not None:
            self._head_panel.stop()
            self._head_panel = None
        self._sim_thread.stop()
        if self._controller is not None:
            self._controller.close_all()
        super().closeEvent(event)

    def keyPressEvent(self, event) -> None:
        """! Toggle layout mode when M is pressed.

        @param event<QKeyEvent>: Key event from Qt.
        """
        if event.key() == Qt.Key.Key_M and self._controller is not None:
            self._toggle_mode()
        super().keyPressEvent(event)

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _build_menu(self) -> None:
        view_menu = QMenu("View", self)
        toggle_action = QAction("Toggle Mode [M]", self)
        toggle_action.triggered.connect(self._toggle_mode)
        view_menu.addAction(toggle_action)
        self.menuBar().addMenu(view_menu)

    def _toggle_mode(self) -> None:
        if self._head_panel is not None:
            self._head_panel.stop()
            self._head_panel = None
        self._viewport.setParent(None)
        self._mode = Mode.MASSAGE if self._mode == Mode.NORMAL else Mode.NORMAL
        self._build_layout()

    def _build_layout(self) -> None:
        if self._controller is None or not self._orbbec_names:
            self.setCentralWidget(self._viewport)
            self.setWindowTitle("SMARTROBOX")
            return
        if self._mode == Mode.MASSAGE:
            self._build_massage_layout()
        else:
            self._build_normal_layout()

    def _build_normal_layout(self) -> None:
        panel = _CameraPanel(self._controller, self._orbbec_names, self)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._viewport)
        splitter.addWidget(panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)
        self.setWindowTitle("SMARTROBOX — Normal")

    def _build_massage_layout(self) -> None:
        if len(self._orbbec_names) < 3:
            self._build_normal_layout()
            return
        main_cam = self._orbbec_names[2]
        side_cams = self._orbbec_names[:2]
        self._head_panel = _HeadCameraPanel(self._controller, main_cam, parent=self)
        right = _MassageRightPanel(self._viewport, self._controller, side_cams, self)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._head_panel)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.setWindowTitle("SMARTROBOX — Massage")

    @Slot(float)
    def _show_runtime(self, avg_time: float) -> None:
        self.statusBar().showMessage(
            f"Render: {avg_time:.2e}s  |  Sim time: {self._data.time:.1f}s"
        )

    def _create_free_camera(self) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1
        for i in range(3):
            cam.lookat[i] = np.median(self._data.geom_xpos[:, i])
        cam.distance = self._model.stat.extent
        cam.azimuth = 135
        cam.elevation = -25
        return cam


def main() -> None:
    """! Entry point — launch the Qt application and show the MuJoCo viewer."""
    xml_path = os.path.abspath(_ENV_PATH)
    config = Config(os.path.abspath(_DEVICE_CONFIG_PATH))
    controller = Controller(config)
    for name in controller.list_devices():
        if controller.status(name)["type"] == "orbbec":
            controller.open(name)
    app = QApplication(sys.argv)
    window = MuJoCoWindow(xml_path, controller)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
