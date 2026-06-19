#!/usr/bin/env python3
##
# @file orbbec_interface.py
#
# @brief Threaded Orbbec depth camera interface with color+depth capture
#        and pixel-to-world coordinate projection.
#
# Copyright (c) 2025 HACHIX
##

# ========================
# Standard library
# ========================
import threading
import time

# ========================
# External library
# ========================
import cv2
import numpy as np
from pyorbbecsdk import (
    Context,
    Pipeline,
    Config,
    OBSensorType,
    OBAlignMode,
    OBFormat,
    ColorFrame,
    DepthFrame,
)


class OrbbecInterface:
    """
    One Orbbec depth camera = one capture thread.
    Captures aligned color + depth frames continuously.
    Provides pixel-to-world projection using camera intrinsics.
    """

    _MAX_FAIL = 10
    _WAIT_TIMEOUT_MS = 100

    def __init__(self, device_index: int = 0):
        self.device_index = device_index

        self._context: Context | None = None
        self._pipeline: Pipeline | None = None

        self._color_frame: np.ndarray | None = None  # BGR uint8 H×W×3
        self._depth_frame: np.ndarray | None = None  # uint16 H×W

        self._fx: float | None = None
        self._fy: float | None = None
        self._cx: float | None = None
        self._cy: float | None = None
        self._depth_scale: float | None = None  # raw uint16 → metres

        self.running = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._fail_count = 0

    # ===========================================================
    # CONNECTION
    # ===========================================================
    def start(self):
        if self.running:
            return

        self._context = Context()
        device_list = self._context.query_devices()

        if device_list.get_count() == 0:
            self._context = None
            raise RuntimeError("No Orbbec devices found")

        if self.device_index >= device_list.get_count():
            self._context = None
            raise RuntimeError(
                f"Orbbec device index {self.device_index} out of range "
                f"(found {device_list.get_count()} device(s))"
            )

        device = device_list.get_device_by_index(self.device_index)
        self._pipeline = Pipeline(device)

        config = Config()
        color_profiles = self._pipeline.get_stream_profile_list(OBSensorType.COLOR)
        depth_profiles = self._pipeline.get_stream_profile_list(OBSensorType.DEPTH)
        config.enable_stream(color_profiles.get_default_video_stream_profile())
        config.enable_stream(depth_profiles.get_default_video_stream_profile())
        config.set_align_mode(OBAlignMode.SW_MODE)

        self._pipeline.start(config)
        self._pipeline.enable_frame_sync()

        cam_param = self._pipeline.get_camera_param()
        intr = cam_param.rgb_intrinsic
        self._fx = intr.fx
        self._fy = intr.fy
        self._cx = intr.cx
        self._cy = intr.cy

        self.running = True
        self._fail_count = 0

        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"Orbbec-{self.device_index}",
        )
        self._thread.start()

    def stop(self):
        self.running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

        self._context = None

        with self._lock:
            self._color_frame = None
            self._depth_frame = None

    def is_alive(self) -> bool:
        if not self.running:
            return False
        if self._pipeline is None:
            return False
        if self._fail_count >= self._MAX_FAIL:
            return False
        return True

    # ===========================================================
    # FRAME ACCESS
    # ===========================================================
    def get_color_frame(self) -> np.ndarray | None:
        """Return a copy of the latest BGR color frame, or None."""
        if not self.is_alive():
            return None
        with self._lock:
            if self._color_frame is None:
                return None
            return self._color_frame.copy()

    def get_depth_frame(self) -> np.ndarray | None:
        """Return a copy of the latest raw uint16 depth frame, or None."""
        if not self.is_alive():
            return None
        with self._lock:
            if self._depth_frame is None:
                return None
            return self._depth_frame.copy()

    # ===========================================================
    # COORDINATE PROJECTION
    # ===========================================================
    def pixel_to_world(
        self, u: int, v: int
    ) -> tuple[float, float, float] | None:
        """
        Convert pixel (u, v) to a 3D point in the camera frame (metres).

        Depth is software-aligned to the color stream, so RGB intrinsics
        (cached at start) apply directly. Returns None when depth is
        unavailable, zero, or the pixel is out of bounds.
        """
        if self._fx is None or self._depth_scale is None:
            return None

        with self._lock:
            if self._depth_frame is None:
                return None
            depth_copy = self._depth_frame.copy()

        h, w = depth_copy.shape
        if not (0 <= v < h and 0 <= u < w):
            return None

        raw = int(depth_copy[v, u])
        if raw == 0:
            return None

        Z = raw * self._depth_scale
        X = (u - self._cx) * Z / self._fx
        Y = (v - self._cy) * Z / self._fy
        return (X, Y, Z)

    # ===========================================================
    # INTERNAL
    # ===========================================================
    def _capture_loop(self):
        while self.running:
            if self._pipeline is None:
                self._fail_count += 1
                time.sleep(0.2)
                continue

            try:
                frameset = self._pipeline.wait_for_frames(self._WAIT_TIMEOUT_MS)
            except Exception:
                self._fail_count += 1
                time.sleep(0.05)
                continue

            if frameset is None:
                time.sleep(0.01)
                continue

            color_frame: ColorFrame | None = frameset.get_color_frame()
            depth_frame: DepthFrame | None = frameset.get_depth_frame()

            if color_frame is None or depth_frame is None:
                time.sleep(0.01)
                continue

            try:
                color_bgr = self._decode_color(color_frame)
                depth_arr = self._decode_depth(depth_frame)

                if self._depth_scale is None:
                    self._depth_scale = depth_frame.get_depth_scale()

                self._fail_count = 0
                with self._lock:
                    self._color_frame = color_bgr
                    self._depth_frame = depth_arr

            except Exception:
                self._fail_count += 1

        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

    @staticmethod
    def _decode_color(frame: ColorFrame) -> np.ndarray:
        data = np.frombuffer(frame.get_data(), dtype=np.uint8)
        h, w = frame.get_height(), frame.get_width()
        fmt = frame.get_format()

        if fmt == OBFormat.MJPG:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imdecode failed for MJPEG frame")
            return img

        if fmt == OBFormat.RGB:
            return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)

        if fmt == OBFormat.BGR:
            return data.reshape(h, w, 3).copy()

        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _decode_depth(frame: DepthFrame) -> np.ndarray:
        data = np.frombuffer(frame.get_data(), dtype=np.uint16)
        return data.reshape(frame.get_height(), frame.get_width()).copy()
