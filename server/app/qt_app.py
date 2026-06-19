##
# @file qt_app.py
#
# @brief Qt application for displaying a MuJoCo simulation scene.
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

# External library
import mujoco
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
)

_ENV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "projects", "anlab", "envs", "anlab_mujoco.xml"
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


class MuJoCoWindow(QMainWindow):
    """! Main window that hosts the MuJoCo viewport."""

    def __init__(self, xml_path: str) -> None:
        """! Load the scene from an XML file and build the UI.

        @param xml_path<str>: Absolute path to the MuJoCo XML scene file.
        """
        super().__init__()
        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data = mujoco.MjData(self._model)
        self._cam = self._create_free_camera()
        self._opt = mujoco.MjvOption()
        self._scn = mujoco.MjvScene(self._model, maxgeom=10000)
        self._viewport = Viewport(self._model, self._data, self._cam, self._opt, self._scn)
        self._viewport.updateRuntime.connect(self._show_runtime)
        self.setCentralWidget(self._viewport)
        self.setWindowTitle("SmartRoBox — MuJoCo Viewer")
        self.showMaximized()
        self._sim_thread = _SimThread(self._model, self._data, self)
        self._sim_thread.start()

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def closeEvent(self, event) -> None:
        """! Stop the simulation thread before closing."""
        self._sim_thread.stop()
        super().closeEvent(event)

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

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
    app = QApplication(sys.argv)
    window = MuJoCoWindow(xml_path)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
