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
        servo_space: str = "joint",
        camera_config: str = "eye_in_hand",
        camera_extrinsic: list[list[float]] | None = None,
        target_keypoints_3d: list[list[float]] | None = None,
        servo_gain_3d: float = 0.5,
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
            gain_matrix[i] = [gx, gy] maps (mean_ex, mean_ey) to Δi.
            In "joint" space Δi is a joint increment (deg); in "cart" space Δi
            is a TCP increment (mm for i<3, deg for i>=3).
            Defaults to all-zero (monitoring only, no motion).
        @param cmd_period<float>: Servo command interval in seconds.  Default 0.016.
        @param timeout<float>: Maximum run time in seconds.  Default 30.0.
        @param model_name<str>: YOLO11 pose model name or path.
        @param warmup_timeout<float>: Seconds to poll for the first camera frame.
        @param keypoint_conf_min<float>: Minimum YOLO keypoint confidence to
            include a keypoint in the error computation.
        @param servo_space<str>: "joint" (default) or "cart".  Selects whether
            corrections are applied via servo_j or servo_c.
        @param camera_config<str>: "eye_in_hand" (default) — existing IBVS
            gain_matrix approach.  "eye_to_hand" — PBVS: pixel_to_world +
            camera_extrinsic gives base-frame 3D error applied as a TCP
            correction via servo_c.
        @param camera_extrinsic<list[list[float]]|None>: 4×4 T_cam_to_base
            homogeneous transform; required when camera_config == "eye_to_hand".
        @param target_keypoints_3d<list[list[float]]|None>: Desired 3D keypoint
            positions [[X, Y, Z], ...] in base frame (metres); required for
            "eye_to_hand".
        @param servo_gain_3d<float>: Scalar gain in [0, 1] applied to the mean
            3D error vector.  Default 0.5.
        @raises ValueError: If camera_config == "eye_to_hand" and
            camera_extrinsic or target_keypoints_3d is None.
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
        self._servo_space = servo_space
        if camera_config == "eye_to_hand":
            if camera_extrinsic is None or target_keypoints_3d is None:
                raise ValueError(
                    "eye_to_hand requires camera_extrinsic and target_keypoints_3d"
                )
        self._camera_config = camera_config
        self._camera_extrinsic = (
            np.array(camera_extrinsic, dtype=np.float64)
            if camera_extrinsic is not None
            else None
        )
        self._target_keypoints_3d = target_keypoints_3d
        self._servo_gain_3d = servo_gain_3d
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
            "servo_space": self._servo_space,
            "camera_config": self._camera_config,
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

        if self._camera_config == "eye_to_hand":
            self._servo_space = "cart"
            depth_deadline = time.monotonic() + self._warmup_timeout
            while True:
                if self._call(self._camera_device, "get_depth_frame") is not None:
                    break
                if time.monotonic() >= depth_deadline:
                    raise RuntimeError(
                        f"No depth frame received from camera '{self._camera_device}'"
                    )
                if not self._checkpoint():
                    return {"converged": False, "stable_ticks": 0, "final_error": math.inf}
                time.sleep(0.05)

        self._call(self._robot_device, "servo_start")

        if self._servo_space == "cart":
            ret, pose = self._call(self._robot_device, "tpos")
            if ret != 0:
                pose = [0.0] * 6
        else:
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
                if self._camera_config == "eye_to_hand":
                    yolo_results = self._model(frame, verbose=False)
                    error, mean_ex, mean_ey = self._compute_error(frame, yolo_results)
                else:
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
                    if self._camera_config == "eye_to_hand":
                        dx, dy, dz = self._compute_3d_correction(frame, yolo_results)
                        if dx != 0.0 or dy != 0.0 or dz != 0.0:
                            pose[0] += dx * 1000.0 * self._servo_gain_3d
                            pose[1] += dy * 1000.0 * self._servo_gain_3d
                            pose[2] += dz * 1000.0 * self._servo_gain_3d
                            self._call(self._robot_device, "servo_c", pose, self._cmd_period)
                    else:
                        delta = [
                            self._gain_matrix[i][0] * mean_ex
                            + self._gain_matrix[i][1] * mean_ey
                            for i in range(6)
                        ]
                        if self._servo_space == "cart":
                            pose = [p + d for p, d in zip(pose, delta)]
                            self._call(self._robot_device, "servo_c", pose, self._cmd_period)
                        else:
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
        self, frame: np.ndarray, yolo_results=None
    ) -> tuple[float, float, float]:
        """! Run YOLO on a frame and compute the mean pixel error vs. target.

        Only the first detected person is used.  Keypoints below
        keypoint_conf_min are excluded.

        @param frame<np.ndarray>: BGR camera frame.
        @param yolo_results: Pre-computed YOLO output; runs inference if None.
        @return<tuple[float, float, float]>: (error, mean_ex, mean_ey).
            error is inf when no valid keypoints are available.
        """
        results = yolo_results if yolo_results is not None else self._model(frame, verbose=False)
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

    def _compute_3d_correction(
        self, frame: np.ndarray, yolo_results=None
    ) -> tuple[float, float, float]:
        """! Compute mean 3D correction vector from detected keypoints to target.

        Lifts each valid YOLO keypoint to the camera frame via pixel_to_world,
        transforms to the base frame via camera_extrinsic, and returns the mean
        offset to target_keypoints_3d.  Keypoints with no depth are skipped.

        @param frame<np.ndarray>: BGR camera frame.
        @param yolo_results: Pre-computed YOLO output; runs inference if None.
        @return<tuple[float, float, float]>: Mean (dx, dy, dz) in base frame
            metres (detected − target).  Returns (0.0, 0.0, 0.0) when no valid
            keypoints are available.
        """
        results = yolo_results if yolo_results is not None else self._model(frame, verbose=False)
        result = results[0]

        if result.boxes is None or not result.boxes.conf.tolist():
            return 0.0, 0.0, 0.0

        kp_xy = result.keypoints.xy.tolist()[0]
        kp_conf = result.keypoints.conf.tolist()[0]

        deltas = []
        n_kp = min(len(kp_xy), len(self._target_keypoints_3d))
        for i in range(n_kp):
            if kp_conf[i] < self._keypoint_conf_min:
                continue
            u = int(round(kp_xy[i][0]))
            v = int(round(kp_xy[i][1]))
            cam_pt = self._call(self._camera_device, "pixel_to_world", u, v)
            if cam_pt is None:
                continue
            p_cam = np.array([cam_pt[0], cam_pt[1], cam_pt[2], 1.0])
            p_base = self._camera_extrinsic @ p_cam
            tgt = self._target_keypoints_3d[i]
            deltas.append((p_base[0] - tgt[0], p_base[1] - tgt[1], p_base[2] - tgt[2]))

        if not deltas:
            return 0.0, 0.0, 0.0

        n = len(deltas)
        return (
            sum(d[0] for d in deltas) / n,
            sum(d[1] for d in deltas) / n,
            sum(d[2] for d in deltas) / n,
        )
