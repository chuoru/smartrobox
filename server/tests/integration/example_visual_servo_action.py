#!/usr/bin/env python3
##
# @file example_visual_servo_action.py
#
# @brief Integration example: VisualServoAction with live camera feed overlay.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/20.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# External library
import cv2
import yaml

# Internal library
from actions.base import ActionState
from actions.visual_servo import VisualServoAction
from app.config import Config
from app.controller import Controller


_ROBOT_DEVICE = "robot"
_CAMERA_DEVICE = "camera"
_MODEL_NAME = "yolo11n-pose.pt"
_KP_CONF_THRESHOLD = 0.5
_ACTION_TIMEOUT = 30.0

# Placeholder target keypoints (17 COCO keypoints at image centre).
# Replace with real target positions captured from the camera.
_TARGET_KEYPOINTS = [[320.0, 240.0]] * 17


def _draw_target_keypoints(frame, target_kps: list[list[float]]) -> None:
    """! Draw target keypoints as green circles on the frame.

    @param frame: BGR numpy array; modified in-place.
    @param target_kps<list[list[float]]>: Target pixel positions [[x, y], ...].
    """
    for x, y in target_kps:
        cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 0), -1)


def _draw_detected_keypoints(
    frame, kps: list[list[float]], confs: list[float], conf_min: float
) -> None:
    """! Draw detected keypoints as red circles on the frame.

    @param frame: BGR numpy array; modified in-place.
    @param kps<list[list[float]]>: Detected keypoints [[x, y], ...].
    @param confs<list[float]>: Per-keypoint confidences.
    @param conf_min<float>: Minimum confidence to draw a keypoint.
    """
    for (x, y), c in zip(kps, confs):
        if c >= conf_min:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)


def _draw_error(frame, error: float) -> None:
    """! Overlay the current pixel error on the frame.

    @param frame: BGR numpy array; modified in-place.
    @param error<float>: Current pixel error value.
    """
    label = f"error: {error:.1f}px" if error != float("inf") else "error: --"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


def main() -> None:
    """! Open robot (debug) + camera, run VisualServoAction, and display live feed."""
    device_cfg = {
        "devices": {
            _ROBOT_DEVICE: {
                "type": "fairino",
                "params": {"ip": "192.168.57.2"},
            },
            _CAMERA_DEVICE: {
                "type": "orbbec",
                "params": {"device_index": 2},
            },
        }
    }

    fd, cfg_path = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(device_cfg, f)

        ctrl = Controller(Config(cfg_path))

        if not ctrl.open(_CAMERA_DEVICE):
            print(f"[example] Failed to open camera '{_CAMERA_DEVICE}'")
            return

        if not ctrl.open(_ROBOT_DEVICE):
            print(f"[example] Failed to open robot '{_ROBOT_DEVICE}'")
            ctrl.close(_CAMERA_DEVICE)
            return

        print("[example] Starting VisualServoAction. Press 'q' to quit.")
        print(f"[example] Target: {_TARGET_KEYPOINTS[0]} (first keypoint shown)")

        try:
            action = VisualServoAction(
                ctrl,
                robot_device=_ROBOT_DEVICE,
                camera_device=_CAMERA_DEVICE,
                target_keypoints=_TARGET_KEYPOINTS,
                error_threshold=20.0,
                stable_ticks=10,
                gain_matrix=[[0.0, 0.0]] * 6,
                cmd_period=0.016,
                timeout=_ACTION_TIMEOUT,
                model_name=_MODEL_NAME,
            )
            action.start()

            while not action.wait(timeout=0.05):
                frame = ctrl.execute(_CAMERA_DEVICE, "get_color_frame")
                if frame is None:
                    continue

                _draw_target_keypoints(frame, _TARGET_KEYPOINTS)
                _draw_error(frame, float("inf"))
                cv2.imshow("VisualServoAction", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    action.cancel()
                    action.wait(timeout=2.0)
                    break

            state = action.state()
            result = action.result()
            if state == ActionState.DONE and result:
                print(
                    f"[example] Converged={result['converged']}  "
                    f"stable_ticks={result['stable_ticks']}  "
                    f"final_error={result['final_error']:.2f}px"
                )
            elif state == ActionState.FAILED:
                print(f"[example] Action failed: {action.error()}")
            else:
                print(f"[example] Action ended with state={state.value}")

            frame = ctrl.execute(_CAMERA_DEVICE, "get_color_frame")
            if frame is not None and result:
                _draw_error(frame, result.get("final_error", float("inf")))
                _draw_target_keypoints(frame, _TARGET_KEYPOINTS)
                cv2.imshow("VisualServoAction — final", frame)
                cv2.waitKey(2000)

        finally:
            ctrl.close(_CAMERA_DEVICE)
            ctrl.close(_ROBOT_DEVICE)
            cv2.destroyAllWindows()
    finally:
        os.unlink(cfg_path)


if __name__ == "__main__":
    main()
