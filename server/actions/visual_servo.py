#!/usr/bin/env python3
##
# @file visual_servo.py
#
# @brief Action that visually servos both Fairino arms toward the detected
#        shoulder keypoints of a person using YOLO11 pose estimation.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/20.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import math
import time

# External library
import numpy as np
from ultralytics import YOLO

# Internal library
from actions.base import BaseAction
from app.controller import Controller


class VisualServoAction(BaseAction):
    """! Drives left and right Fairino arms toward the detected left and right
    shoulders of a person using YOLO11 pose estimation as the feedback signal.

    Each servo tick:
      1. Captures a frame and runs YOLO11 pose estimation once.
      2. Lifts the detected left and right shoulder keypoints to 3D positions
         in their respective arm base frames via pixel_to_world + extrinsic.
      3. Computes the error vector from each arm's current TCP to the shoulder
         (both in mm).
      4. Applies a proportional correction:
         new_tcp[:3] = tcp[:3] - servo_gain * (tcp[:3] - shoulder_mm)
      5. Sends servo_c commands to both arms.

    A tick where a shoulder is undetected or depth is unavailable is neutral —
    the stable counter for that arm neither advances nor resets.  The action
    exits successfully when both arms have been within error_threshold for
    stable_ticks consecutive measured ticks, or raises RuntimeError on timeout.

    Result format::

        {
            "converged":          bool,
            "left_stable_ticks":  int,
            "right_stable_ticks": int,
            "left_final_error":   float,  # mm at last measured tick, inf if never
            "right_final_error":  float,
        }
    """

    _KP_LEFT_SHOULDER  = 5
    _KP_RIGHT_SHOULDER = 6

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def __init__(
        self,
        controller: Controller,
        left_robot_device: str,
        right_robot_device: str,
        camera_device: str,
        left_arm_extrinsic: list[list[float]],
        right_arm_extrinsic: list[list[float]],
        error_threshold: float,
        stable_ticks: int,
        servo_gain: float = 0.5,
        cmd_period: float = 0.016,
        timeout: float = 30.0,
        model_name: str = "yolo11n-pose.pt",
        warmup_timeout: float = 3.0,
        keypoint_conf_min: float = 0.5,
    ) -> None:
        """! Load the YOLO model and store servo parameters.

        @param controller<Controller>: Controller used to dispatch device calls.
        @param left_robot_device<str>: Registered name of the left Fairino arm.
        @param right_robot_device<str>: Registered name of the right Fairino arm.
        @param camera_device<str>: Registered name of the Orbbec head camera.
        @param left_arm_extrinsic<list[list[float]]>: 4×4 T_cam_to_left_base
            homogeneous transform (metres).
        @param right_arm_extrinsic<list[list[float]]>: 4×4 T_cam_to_right_base
            homogeneous transform (metres).
        @param error_threshold<float>: 3D distance in mm below which an arm is
            considered converged for a tick.
        @param stable_ticks<int>: Number of consecutive below-threshold ticks
            required from both arms to declare convergence.
        @param servo_gain<float>: Proportional gain in (0, 1].  Controls how
            aggressively the TCP chases the shoulder each tick.  Default 0.5.
        @param cmd_period<float>: Servo command interval in seconds.  Default 0.016.
        @param timeout<float>: Maximum run time in seconds.  Default 30.0.
        @param model_name<str>: YOLO11 pose model name or path.
        @param warmup_timeout<float>: Seconds to poll for the first color and
            depth frame before raising RuntimeError.
        @param keypoint_conf_min<float>: Minimum YOLO keypoint confidence to
            accept a shoulder detection.
        @raises ValueError: If servo_gain is not in (0, 1].
        """
        super().__init__(controller)
        if not (0.0 < servo_gain <= 1.0):
            raise ValueError(f"servo_gain must be in (0, 1], got {servo_gain}")
        self._left_robot_device  = left_robot_device
        self._right_robot_device = right_robot_device
        self._camera_device      = camera_device
        self._left_arm_extrinsic  = np.array(left_arm_extrinsic,  dtype=np.float64)
        self._right_arm_extrinsic = np.array(right_arm_extrinsic, dtype=np.float64)
        self._error_threshold  = error_threshold
        self._stable_ticks     = stable_ticks
        self._servo_gain       = servo_gain
        self._cmd_period       = cmd_period
        self._timeout          = timeout
        self._warmup_timeout   = warmup_timeout
        self._keypoint_conf_min = keypoint_conf_min
        self._model_name = model_name
        self._model = YOLO(model_name)

    def parameters(self) -> dict:
        """! Return the action's configuration parameters.

        @return<dict>: Configuration snapshot.
        """
        return {
            "left_robot_device":  self._left_robot_device,
            "right_robot_device": self._right_robot_device,
            "camera_device":      self._camera_device,
            "error_threshold":    self._error_threshold,
            "stable_ticks":       self._stable_ticks,
            "servo_gain":         self._servo_gain,
            "cmd_period":         self._cmd_period,
            "timeout":            self._timeout,
            "model_name":         self._model_name,
        }

    # =========================================================================
    # PROTECTED METHODS
    # =========================================================================

    def _run(self) -> dict:
        """! Run the dual-arm visual servo loop until convergence or timeout.

        @return<dict>: {"converged", "left_stable_ticks", "right_stable_ticks",
            "left_final_error", "right_final_error"}.
        @raises RuntimeError: If no camera frame or depth frame arrives within
            warmup_timeout, if tpos() fails on either arm, or if timeout elapses
            before both arms converge.
        """
        frame = self._poll_for_frame()
        if frame is None:
            raise RuntimeError(
                f"No color frame received from camera '{self._camera_device}'"
            )

        depth_deadline = time.monotonic() + self._warmup_timeout
        while True:
            if self._call(self._camera_device, "get_depth_frame") is not None:
                break
            if time.monotonic() >= depth_deadline:
                raise RuntimeError(
                    f"No depth frame received from camera '{self._camera_device}'"
                )
            if not self._checkpoint():
                return {
                    "converged": False,
                    "left_stable_ticks": 0,
                    "right_stable_ticks": 0,
                    "left_final_error": math.inf,
                    "right_final_error": math.inf,
                }
            time.sleep(0.05)

        try:
            self._call(self._left_robot_device,  "servo_start")
            self._call(self._right_robot_device, "servo_start")

            left_ret,  left_tcp  = self._call(self._left_robot_device,  "tpos")
            right_ret, right_tcp = self._call(self._right_robot_device, "tpos")

            if left_ret != 0:
                raise RuntimeError(
                    f"tpos() failed on '{self._left_robot_device}': code={left_ret}"
                )
            if right_ret != 0:
                raise RuntimeError(
                    f"tpos() failed on '{self._right_robot_device}': code={right_ret}"
                )

            left_tcp  = list(left_tcp)
            right_tcp = list(right_tcp)

            left_stable  = 0
            right_stable = 0
            left_error   = math.inf
            right_error  = math.inf
            converged    = False
            deadline     = time.monotonic() + self._timeout

            while True:
                frame = self._call(self._camera_device, "get_color_frame")
                if frame is None:
                    if not self._checkpoint():
                        break
                    time.sleep(self._cmd_period)
                    continue

                results = self._model(frame, verbose=False)
                result  = results[0]

                left_kp_xy  = None
                right_kp_xy = None

                if result.boxes is not None and result.boxes.conf.tolist():
                    kp_xy   = result.keypoints.xy.tolist()[0]
                    kp_conf = result.keypoints.conf.tolist()[0]

                    if (len(kp_xy) > self._KP_LEFT_SHOULDER
                            and kp_conf[self._KP_LEFT_SHOULDER] >= self._keypoint_conf_min):
                        left_kp_xy = kp_xy[self._KP_LEFT_SHOULDER]

                    if (len(kp_xy) > self._KP_RIGHT_SHOULDER
                            and kp_conf[self._KP_RIGHT_SHOULDER] >= self._keypoint_conf_min):
                        right_kp_xy = kp_xy[self._KP_RIGHT_SHOULDER]

                if left_kp_xy is not None:
                    left_shoulder_mm = self._lift_shoulder(
                        left_kp_xy, self._left_arm_extrinsic
                    )
                    if left_shoulder_mm is not None:
                        ex = left_tcp[0] - left_shoulder_mm[0]
                        ey = left_tcp[1] - left_shoulder_mm[1]
                        ez = left_tcp[2] - left_shoulder_mm[2]
                        left_error = math.sqrt(ex * ex + ey * ey + ez * ez)
                        if left_error >= self._error_threshold:
                            left_tcp[0] -= self._servo_gain * ex
                            left_tcp[1] -= self._servo_gain * ey
                            left_tcp[2] -= self._servo_gain * ez
                            self._call(
                                self._left_robot_device, "servo_c",
                                left_tcp, self._cmd_period
                            )
                            left_stable = 0
                        else:
                            left_stable += 1

                if right_kp_xy is not None:
                    right_shoulder_mm = self._lift_shoulder(
                        right_kp_xy, self._right_arm_extrinsic
                    )
                    if right_shoulder_mm is not None:
                        ex = right_tcp[0] - right_shoulder_mm[0]
                        ey = right_tcp[1] - right_shoulder_mm[1]
                        ez = right_tcp[2] - right_shoulder_mm[2]
                        right_error = math.sqrt(ex * ex + ey * ey + ez * ez)
                        if right_error >= self._error_threshold:
                            right_tcp[0] -= self._servo_gain * ex
                            right_tcp[1] -= self._servo_gain * ey
                            right_tcp[2] -= self._servo_gain * ez
                            self._call(
                                self._right_robot_device, "servo_c",
                                right_tcp, self._cmd_period
                            )
                            right_stable = 0
                        else:
                            right_stable += 1

                if left_stable >= self._stable_ticks and right_stable >= self._stable_ticks:
                    converged = True
                    break

                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"Visual servo timed out after {self._timeout}s "
                        f"(left_error={left_error:.1f}mm, right_error={right_error:.1f}mm)"
                    )

                if not self._checkpoint():
                    break

                time.sleep(self._cmd_period)

        finally:
            self._call(self._left_robot_device,  "servo_end")
            self._call(self._right_robot_device, "servo_end")

        return {
            "converged":          converged,
            "left_stable_ticks":  left_stable,
            "right_stable_ticks": right_stable,
            "left_final_error":   left_error,
            "right_final_error":  right_error,
        }

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _poll_for_frame(self) -> np.ndarray | None:
        """! Poll get_color_frame until a frame arrives or warmup_timeout elapses.

        @return<np.ndarray|None>: BGR frame, or None on timeout or cancellation.
        """
        deadline = time.monotonic() + self._warmup_timeout
        while True:
            frame = self._call(self._camera_device, "get_color_frame")
            if frame is not None:
                return frame
            if time.monotonic() >= deadline:
                return None
            if not self._checkpoint():
                return None
            time.sleep(0.05)

    def _lift_shoulder(
        self,
        kp_xy: list[float],
        extrinsic: np.ndarray,
    ) -> tuple[float, float, float] | None:
        """! Lift a keypoint pixel to a 3D position in the arm base frame (mm).

        @param kp_xy<list[float]>: [u, v] pixel coordinate of the shoulder.
        @param extrinsic<np.ndarray>: 4×4 T_cam_to_arm_base (metres).
        @return<tuple[float,float,float]|None>: (X, Y, Z) in mm in the arm base
            frame, or None if depth is unavailable at that pixel.
        """
        u = int(round(kp_xy[0]))
        v = int(round(kp_xy[1]))
        cam_pt = self._call(self._camera_device, "pixel_to_world", u, v)
        if cam_pt is None:
            return None
        p_h    = np.array([cam_pt[0], cam_pt[1], cam_pt[2], 1.0])
        p_base = extrinsic @ p_h
        return (float(p_base[0] * 1000.0),
                float(p_base[1] * 1000.0),
                float(p_base[2] * 1000.0))
