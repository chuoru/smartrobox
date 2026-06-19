#!/usr/bin/env python3
##
# @file visual_servo.py
#
# @brief Action that visually servos a Fairino arm toward a target keypoint
#        configuration using YOLO11 pose estimation as the feedback signal.
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
    """! Visually servos a Fairino arm until pixel error drops below a threshold.

    Each servo tick:
      1. Captures a camera frame and runs YOLO11 pose estimation.
      2. Computes the mean pixel offset between detected and target keypoints
         (only keypoints whose confidence exceeds keypoint_conf_min contribute).
      3. Sends a ServoJ command with joint corrections proportional to the
         offset via gain_matrix.
      4. Increments a stable counter when error < error_threshold; resets it
         otherwise.

    The action exits successfully when stable_counter reaches stable_ticks, or
    raises RuntimeError if timeout elapses first.

    Result format::

        {
            "converged":    bool,
            "stable_ticks": int,   # consecutive ticks below threshold at exit
            "final_error":  float, # pixel error on the last tick (inf if no detection)
        }
    """

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def __init__(
        self,
        controller: Controller,
        robot_device: str,
        camera_device: str,
        target_keypoints: list[list[float]],
        error_threshold: float,
        stable_ticks: int,
        gain_matrix: list[list[float]] | None = None,
        cmd_period: float = 0.016,
        timeout: float = 30.0,
        model_name: str = "yolo11n-pose.pt",
        warmup_timeout: float = 3.0,
        keypoint_conf_min: float = 0.5,
    ) -> None:
        """! Load the YOLO model and store servo parameters.

        @param controller<Controller>: Controller used to dispatch device calls.
        @param robot_device<str>: Registered name of the Fairino robot device.
        @param camera_device<str>: Registered name of the Orbbec camera device.
        @param target_keypoints<list[list[float]]>: COCO keypoint pixel targets
            [[x, y], ...].  Length must match the model's output (17 for YOLO11).
        @param error_threshold<float>: Pixel-distance threshold for convergence.
        @param stable_ticks<int>: Consecutive ticks below threshold required to
            declare convergence.
        @param gain_matrix<list[list[float]]|None>: 6×2 matrix where
            gain_matrix[i] = [gx, gy] maps (mean_ex, mean_ey) to Δji.
            Defaults to all-zero (monitoring only, no motion).
        @param cmd_period<float>: Servo command interval in seconds.  Default 0.016.
        @param timeout<float>: Maximum run time in seconds.  Default 30.0.
        @param model_name<str>: YOLO11 pose model name or path.
        @param warmup_timeout<float>: Seconds to poll for the first camera frame.
        @param keypoint_conf_min<float>: Minimum YOLO keypoint confidence to
            include a keypoint in the error computation.
        """
        super().__init__(controller)
        self._robot_device = robot_device
        self._camera_device = camera_device
        self._target_keypoints = target_keypoints
        self._error_threshold = error_threshold
        self._stable_ticks = stable_ticks
        self._gain_matrix = gain_matrix if gain_matrix is not None else [[0.0, 0.0]] * 6
        self._cmd_period = cmd_period
        self._timeout = timeout
        self._warmup_timeout = warmup_timeout
        self._keypoint_conf_min = keypoint_conf_min
        self._model = YOLO(model_name)
        self._model_name = model_name

    def parameters(self) -> dict:
        """! Return the action's configuration parameters.

        @return<dict>: Configuration snapshot.
        """
        return {
            "robot_device": self._robot_device,
            "camera_device": self._camera_device,
            "error_threshold": self._error_threshold,
            "stable_ticks": self._stable_ticks,
            "cmd_period": self._cmd_period,
            "timeout": self._timeout,
            "model_name": self._model_name,
        }

    # =========================================================================
    # PROTECTED METHODS
    # =========================================================================

    def _run(self) -> dict:
        """! Run the visual servo loop until convergence or timeout.

        @return<dict>: {"converged", "stable_ticks", "final_error"}.
        @raises RuntimeError: If no camera frame arrives within warmup_timeout,
            or if timeout elapses before convergence.
        """
        frame = self._poll_for_frame()
        if frame is None:
            raise RuntimeError(
                f"No frame received from camera '{self._camera_device}'"
            )

        self._call(self._robot_device, "servo_start")

        ret, joint_pos = self._call(self._robot_device, "get_joint_pos")
        if ret != 0:
            joint_pos = [0.0] * 6

        stable_counter = 0
        error = math.inf
        deadline = time.monotonic() + self._timeout
        converged = False

        while True:
            frame = self._call(self._camera_device, "get_color_frame")
            if frame is not None:
                error, mean_ex, mean_ey = self._compute_error(frame)

                if error < self._error_threshold:
                    stable_counter += 1
                else:
                    stable_counter = 0

                if stable_counter >= self._stable_ticks:
                    converged = True
                    break

                if time.monotonic() > deadline:
                    self._call(self._robot_device, "servo_end")
                    raise RuntimeError(
                        f"Visual servo timed out after {self._timeout}s "
                        f"(final error={error:.2f}px)"
                    )

                if not math.isinf(error):
                    delta = [
                        self._gain_matrix[i][0] * mean_ex
                        + self._gain_matrix[i][1] * mean_ey
                        for i in range(6)
                    ]
                    joint_pos = [jp + dj for jp, dj in zip(joint_pos, delta)]
                    self._call(self._robot_device, "servo_j", joint_pos, self._cmd_period)

            if not self._checkpoint():
                break

            time.sleep(self._cmd_period)

        self._call(self._robot_device, "servo_end")
        return {
            "converged": converged,
            "stable_ticks": stable_counter,
            "final_error": error,
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

    def _compute_error(
        self, frame: np.ndarray
    ) -> tuple[float, float, float]:
        """! Run YOLO on a frame and compute the mean pixel error vs. target.

        Only the first detected person is used.  Keypoints below
        keypoint_conf_min are excluded.

        @param frame<np.ndarray>: BGR camera frame.
        @return<tuple[float, float, float]>: (error, mean_ex, mean_ey).
            error is inf when no valid keypoints are available.
        """
        results = self._model(frame, verbose=False)
        result = results[0]

        if result.boxes is None or not result.boxes.conf.tolist():
            return math.inf, 0.0, 0.0

        kp_xy = result.keypoints.xy.tolist()[0]
        kp_conf = result.keypoints.conf.tolist()[0]

        pairs = [
            (kp_xy[i], self._target_keypoints[i])
            for i in range(min(len(kp_xy), len(self._target_keypoints)))
            if kp_conf[i] >= self._keypoint_conf_min
        ]

        if not pairs:
            return math.inf, 0.0, 0.0

        mean_ex = sum(cur[0] - tgt[0] for cur, tgt in pairs) / len(pairs)
        mean_ey = sum(cur[1] - tgt[1] for cur, tgt in pairs) / len(pairs)
        error = math.sqrt(mean_ex ** 2 + mean_ey ** 2)
        return error, mean_ex, mean_ey
