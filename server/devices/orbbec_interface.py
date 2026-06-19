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
import collections
import threading

# ========================
# External library
# ========================
import cv2
import numpy as np
from pyorbbecsdk import (
    Config,
    Context,
    FrameSet,
    OBError,
    OBSensorType,
    Pipeline,
    ColorFrame,
    DepthFrame,
)

_QUEUE_MAXSIZE = 5


class _FrameQueue:
    """Bounded queue that silently drops the oldest item when full."""

    def __init__(self, maxsize: int = _QUEUE_MAXSIZE):
        self._queue: collections.deque = collections.deque(maxlen=maxsize)
        self._lock = threading.Lock()

    def put(self, item: np.ndarray) -> None:
        with self._lock:
            self._queue.append(item)

    def get(self) -> np.ndarray | None:
        with self._lock:
            return self._queue.popleft() if self._queue else None

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()


class OrbbecInterface:
    """
    Orbbec depth camera interface using the SDK callback pipeline.

    The SDK delivers FrameSet objects via its own internal thread.
    Raw frames are queued and decoded lazily on the caller's thread.
    Provides pixel-to-world projection using camera intrinsics.
    """

    _MAX_FAIL = 10

    def __init__(self, device_index: int = 0):
        self.device_index = device_index

        self._pipeline: Pipeline | None = None

        self._color_q: _FrameQueue = _FrameQueue()  # holds decoded BGR arrays
        self._depth_q: _FrameQueue = _FrameQueue()  # holds decoded uint16 arrays

        # Latest decoded frames — updated by get_color_frame / get_depth_frame.
        # Callers always receive the most recent frame even when the queue is empty.
        self._latest_color: np.ndarray | None = None
        self._latest_depth: np.ndarray | None = None
        self._lock = threading.Lock()

        self._fx: float | None = None
        self._fy: float | None = None
        self._cx: float | None = None
        self._cy: float | None = None
        self._depth_scale: float | None = None  # raw uint16 → metres

        self.running = False
        self._fail_count = 0
        self._frame_callback: object = None

    # ===========================================================
    # CONNECTION
    # ===========================================================
    def start(self):
        if self.running:
            return

        ctx = Context()
        device_list = ctx.query_devices()
        count = device_list.get_count()

        if count == 0:
            del ctx
            raise RuntimeError("No Orbbec devices found")

        if self.device_index >= count:
            del ctx
            raise RuntimeError(
                f"Orbbec device index {self.device_index} out of range "
                f"(found {count} device(s))"
            )

        device = device_list.get_device_by_index(self.device_index)
        del ctx

        self._pipeline = Pipeline(device)
        config = Config()

        for sensor_type in [OBSensorType.COLOR_SENSOR, OBSensorType.DEPTH_SENSOR]:
            try:
                profiles = self._pipeline.get_stream_profile_list(sensor_type)
                config.enable_stream(profiles.get_default_video_stream_profile())
            except OBError as exc:
                raise RuntimeError(f"Failed to configure sensor {sensor_type}: {exc}")

        self._fail_count = 0
        self.running = True
        # Store the bound method so it is not garbage-collected before the
        # SDK's internal thread invokes it (pyorbbecsdk2 may not Py_INCREF it).
        self._frame_callback = self._on_frame
        try:
            self._pipeline.start(config, self._frame_callback)
        except Exception:
            self.running = False
            raise

        cam_param = self._pipeline.get_camera_param()
        intr = cam_param.rgb_intrinsic
        self._fx = intr.fx
        self._fy = intr.fy
        self._cx = intr.cx
        self._cy = intr.cy

    def stop(self):
        self.running = False

        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        self._frame_callback = None

        self._color_q.clear()
        self._depth_q.clear()
        with self._lock:
            self._latest_color = None
            self._latest_depth = None

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
        """Return the latest BGR color frame, or None if none has arrived yet."""
        if not self.is_alive():
            return None
        while True:
            arr: np.ndarray | None = self._color_q.get()
            if arr is None:
                break
            with self._lock:
                self._latest_color = arr
        with self._lock:
            return self._latest_color.copy() if self._latest_color is not None else None

    def get_depth_frame(self) -> np.ndarray | None:
        """Return the latest uint16 depth frame, or None if none has arrived yet."""
        if not self.is_alive():
            return None
        while True:
            arr: np.ndarray | None = self._depth_q.get()
            if arr is None:
                break
            with self._lock:
                self._latest_depth = arr
        with self._lock:
            return self._latest_depth.copy() if self._latest_depth is not None else None

    # ===========================================================
    # COORDINATE PROJECTION
    # ===========================================================
    def pixel_to_world(
        self, u: int, v: int
    ) -> tuple[float, float, float] | None:
        """
        Convert pixel (u, v) to a 3D point in the camera frame (metres).

        Uses RGB intrinsics cached at start. Returns None when depth is
        unavailable, zero, or the pixel is out of bounds.
        """
        if self._fx is None or self._depth_scale is None:
            return None

        # Refresh cache from any pending queue frames, then read cache.
        depth = self.get_depth_frame()
        if depth is None:
            return None

        h, w = depth.shape
        if not (0 <= v < h and 0 <= u < w):
            return None

        raw = int(depth[v, u])
        if raw == 0:
            return None

        Z = raw * self._depth_scale
        X = (u - self._cx) * Z / self._fx
        Y = (v - self._cy) * Z / self._fy
        return (X, Y, Z)

    # ===========================================================
    # INTERNAL
    # ===========================================================
    def _on_frame(self, frame_set: FrameSet):
        """SDK callback — decode immediately to copy data out of the SDK frame pool."""
        if frame_set is None or not self.running:
            return

        color_frame: ColorFrame | None = frame_set.get_color_frame()
        depth_frame: DepthFrame | None = frame_set.get_depth_frame()

        try:
            if color_frame is not None:
                self._color_q.put(self._decode_color(color_frame))

            if depth_frame is not None:
                if self._depth_scale is None:
                    self._depth_scale = depth_frame.get_depth_scale()
                self._depth_q.put(self._decode_depth(depth_frame))

            self._fail_count = 0
        except Exception:
            self._fail_count += 1

    @staticmethod
    def _decode_color(frame: ColorFrame) -> np.ndarray:
        data = np.frombuffer(frame.get_data(), dtype=np.uint8)
        h, w = frame.get_height(), frame.get_width()

        if len(data) == h * w * 3:
            # Uncompressed RGB (3 bytes/pixel)
            return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)

        if len(data) == h * w * 2:
            # YUYV / YUV422 (2 bytes/pixel) — Gemini 305 default color format
            return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)

        # Compressed — decode as JPEG/MJPEG
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode failed for color frame")
        return img

    @staticmethod
    def _decode_depth(frame: DepthFrame) -> np.ndarray:
        data = np.frombuffer(frame.get_data(), dtype=np.uint16)
        return data.reshape(frame.get_height(), frame.get_width()).copy()
