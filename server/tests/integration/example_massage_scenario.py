#!/usr/bin/env python3
##
# @file example_massage_scenario.py
#
# @brief Integration example: full bilateral massage scenario.
#
#        Steps:
#          1. Move both arms to massage_left / massage_right teaching points.
#          2. Close thumb adduction (yaw) on both hands.
#          3. Auto-capture first confident person from head_camera.
#          4. Visual servo both arms in parallel (eye_to_hand, PBVS) until
#             each converges or times out.
#          5. Run MassageAction on both hands simultaneously.
#          6. Return both arms to home_left / home_right; open both hands.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/20.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# External library
import cv2
import numpy as np
import yaml

# Internal library
from actions.base import ActionState
from actions.estimate_pose import EstimatePoseAction
from actions.massage import MassageAction
from actions.visual_servo import VisualServoAction
from app.config import Config
from app.controller import Controller


_LEFT_ARM = "left_arm"
_RIGHT_ARM = "right_arm"
_LEFT_HAND = "left_hand"
_RIGHT_HAND = "right_hand"
_HEAD_CAMERA = "head_camera"

_LEFT_MASSAGE_KEY = "massage_left"
_RIGHT_MASSAGE_KEY = "massage_right"
_LEFT_HOME_KEY = "home_left"
_RIGHT_HOME_KEY = "home_right"

_MODEL_NAME = "yolo11n-pose.pt"
_KP_CONF_THRESHOLD = 0.4
_MIN_VALID_KPS = 5
_AUTO_CAPTURE_RETRIES = 10
_ESTIMATE_TIMEOUT = 3.0

_MOVE_VEL = 20.0
_TORQUE_LIMIT = 180
_OPEN_POSE = [255] * 6
# O6: thumb_cmc_pitch=open, thumb_cmc_yaw=closed, four fingers=open
_THUMB_ADDUCT = [255, 0, 255, 255, 255, 255]

_ERROR_THRESHOLD = 30.0
_STABLE_TICKS = 10
_SERVO_GAIN_3D = 0.5
_SERVO_TIMEOUT = 30.0

_CYCLES = 5
_HALF_CLOSE_DURATION = 0.4
_OPEN_DURATION = 0.4
_MASSAGE_TIMEOUT = 30.0

_DEVICE_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "projects", "anlab", "device.yaml")
)
_TEACHING_POINT_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "projects", "anlab", "teaching_point.yaml")
)

# 4×4 T_cam_to_base for head_camera → left_arm base (metres).
# Replace with calibrated values from example_hand_eye_calibrate.py.
_LEFT_EXTRINSIC = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 1.5],
    [0.0, 0.0, 0.0, 1.0],
]

# 4×4 T_cam_to_base for head_camera → right_arm base (metres).
# Replace with calibrated values from example_hand_eye_calibrate.py.
_RIGHT_EXTRINSIC = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 1.5],
    [0.0, 0.0, 0.0, 1.0],
]

_TERMINAL_STATES = {ActionState.DONE, ActionState.FAILED, ActionState.CANCELLED}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_joints(key: str) -> list[float] | None:
    """! Load joint angles for a named teaching point from teaching_point.yaml.

    @param key<str>: Top-level key (e.g. ``"massage_left"``).
    @return<list[float]|None>: [j1..j6] in degrees, or None if not found.
    """
    if not os.path.exists(_TEACHING_POINT_FILE):
        print(f"[scenario] Teaching point file not found: {_TEACHING_POINT_FILE}")
        return None
    with open(_TEACHING_POINT_FILE, "r") as fh:
        data = yaml.safe_load(fh) or {}
    if key not in data:
        print(f"[scenario] Teaching point '{key}' not found in teaching_point.yaml")
        return None
    block = data[key]["joint"]
    return [float(block[k]) for k in ("j1", "j2", "j3", "j4", "j5", "j6")]


def _poll_depth_warmup(ctrl: Controller, timeout: float = 3.0) -> bool:
    """! Block until the head camera depth stream delivers its first frame.

    @param ctrl<Controller>: Active controller.
    @param timeout<float>: Maximum wait in seconds.
    @return<bool>: True if depth is ready, False on timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ctrl.execute(_HEAD_CAMERA, "get_depth_frame") is not None:
            return True
        time.sleep(0.05)
    return False


def _lift_to_base(
    ctrl: Controller,
    kps_2d: list[list[float]],
    kp_confs: list[float],
    extrinsic: np.ndarray,
) -> list[list[float]]:
    """! Lift 2D keypoints to a robot base frame via depth + extrinsic transform.

    @param ctrl<Controller>: Active controller.
    @param kps_2d<list[list[float]]>: 17 detected keypoint pixel positions.
    @param kp_confs<list[float]>: 17 keypoint confidence values.
    @param extrinsic<np.ndarray>: 4×4 T_cam_to_base (float64).
    @return<list[list[float]]>: 17 [X, Y, Z] positions in base frame (metres).
    """
    pts_base = []
    for kp, conf in zip(kps_2d, kp_confs):
        if conf < _KP_CONF_THRESHOLD:
            pts_base.append(None)
            continue
        cam_pt = ctrl.execute(_HEAD_CAMERA, "pixel_to_world", int(kp[0]), int(kp[1]))
        if cam_pt is None:
            pts_base.append(None)
            continue
        p_cam = np.array([cam_pt[0], cam_pt[1], cam_pt[2], 1.0])
        pts_base.append((extrinsic @ p_cam)[:3].tolist())

    valid = [p for p in pts_base if p is not None]
    centroid = (
        [sum(p[i] for p in valid) / len(valid) for i in range(3)] if valid else [0.0, 0.0, 0.0]
    )
    return [p if p is not None else centroid for p in pts_base]


def _draw_target_2d(frame: np.ndarray, target_kps: list[list[float]]) -> None:
    """! Draw target keypoints as green cross markers.

    @param frame: BGR numpy array; modified in-place.
    @param target_kps<list[list[float]]>: Target pixel positions [[x, y], ...].
    """
    for x, y in target_kps:
        cv2.drawMarker(frame, (int(x), int(y)), (0, 255, 0), cv2.MARKER_CROSS, 10, 1)


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------

def _auto_capture(
    ctrl: Controller,
    extrinsic_left: np.ndarray,
    extrinsic_right: np.ndarray,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]] | None:
    """! Run EstimatePoseAction until a confident person is detected.

    @param ctrl<Controller>: Active controller.
    @param extrinsic_left<np.ndarray>: 4×4 T_cam_to_base for left arm (float64).
    @param extrinsic_right<np.ndarray>: 4×4 T_cam_to_base for right arm (float64).
    @return<tuple|None>: (kps_2d, kps_3d_left, kps_3d_right) or None if no
        confident detection found within _AUTO_CAPTURE_RETRIES attempts.
    """
    print("[scenario] Auto-capturing target pose from head camera ...")
    action = EstimatePoseAction(ctrl, _HEAD_CAMERA, _MODEL_NAME)
    for attempt in range(1, _AUTO_CAPTURE_RETRIES + 1):
        action.reset()
        action.start()
        action.wait(timeout=_ESTIMATE_TIMEOUT + 1.0)

        poses = action.result() or []
        if not poses:
            print(f"[scenario]   attempt {attempt}/{_AUTO_CAPTURE_RETRIES}: no person detected")
            continue

        person = poses[0]
        kps_2d = person["keypoints"]
        kp_confs = person["keypoint_conf"]
        n_valid = sum(1 for c in kp_confs if c >= _KP_CONF_THRESHOLD)

        if n_valid < _MIN_VALID_KPS:
            print(
                f"[scenario]   attempt {attempt}/{_AUTO_CAPTURE_RETRIES}: "
                f"only {n_valid}/{len(kps_2d)} confident keypoints"
            )
            continue

        kps_3d_left = _lift_to_base(ctrl, kps_2d, kp_confs, extrinsic_left)
        kps_3d_right = _lift_to_base(ctrl, kps_2d, kp_confs, extrinsic_right)
        print(
            f"[scenario] Target captured ({n_valid}/17 confident keypoints) — "
            f"wrist_l={kps_3d_left[9]}, wrist_r={kps_3d_right[10]}"
        )
        return kps_2d, kps_3d_left, kps_3d_right

    print("[scenario] Auto-capture failed: no confident detection after all retries.")
    return None


def _phase_servo_both(
    ctrl: Controller,
    target_kps_2d: list[list[float]],
    kps_3d_left: list[list[float]],
    kps_3d_right: list[list[float]],
) -> None:
    """! Start visual servo on both arms simultaneously and poll until both converge.

    @param ctrl<Controller>: Active controller.
    @param target_kps_2d<list[list[float]]>: 17-keypoint pixel targets.
    @param kps_3d_left<list[list[float]]>: 17 3D targets in left arm base frame.
    @param kps_3d_right<list[list[float]]>: 17 3D targets in right arm base frame.
    """
    print("[scenario] Phase servo — both arms in parallel (Q to cancel)")

    left_servo = VisualServoAction(
        ctrl,
        robot_device=_LEFT_ARM,
        camera_device=_HEAD_CAMERA,
        target_keypoints=target_kps_2d,
        error_threshold=_ERROR_THRESHOLD,
        stable_ticks=_STABLE_TICKS,
        cmd_period=0.016,
        timeout=_SERVO_TIMEOUT,
        model_name=_MODEL_NAME,
        keypoint_conf_min=_KP_CONF_THRESHOLD,
        camera_config="eye_to_hand",
        camera_extrinsic=_LEFT_EXTRINSIC,
        target_keypoints_3d=kps_3d_left,
        servo_gain_3d=_SERVO_GAIN_3D,
    )
    right_servo = VisualServoAction(
        ctrl,
        robot_device=_RIGHT_ARM,
        camera_device=_HEAD_CAMERA,
        target_keypoints=target_kps_2d,
        error_threshold=_ERROR_THRESHOLD,
        stable_ticks=_STABLE_TICKS,
        cmd_period=0.016,
        timeout=_SERVO_TIMEOUT,
        model_name=_MODEL_NAME,
        keypoint_conf_min=_KP_CONF_THRESHOLD,
        camera_config="eye_to_hand",
        camera_extrinsic=_RIGHT_EXTRINSIC,
        target_keypoints_3d=kps_3d_right,
        servo_gain_3d=_SERVO_GAIN_3D,
    )

    left_servo.start()
    right_servo.start()

    while True:
        frame = ctrl.execute(_HEAD_CAMERA, "get_color_frame")
        if frame is not None:
            _draw_target_2d(frame, target_kps_2d)
            cv2.imshow("Massage servo — head camera", frame)

        if cv2.waitKey(33) & 0xFF == ord("q"):
            print("[scenario] Servo cancelled by operator.")
            left_servo.cancel()
            right_servo.cancel()
            break

        if left_servo.state() in _TERMINAL_STATES and right_servo.state() in _TERMINAL_STATES:
            break

    left_servo.wait(timeout=2.0)
    right_servo.wait(timeout=2.0)
    cv2.destroyAllWindows()

    for label, action in (("left", left_servo), ("right", right_servo)):
        result = action.result()
        if action.state() == ActionState.DONE and result:
            print(
                f"[scenario] {label} servo: converged={result['converged']}  "
                f"stable_ticks={result['stable_ticks']}  "
                f"final_error={result['final_error']:.2f}px"
            )
        elif action.state() == ActionState.FAILED:
            print(f"[scenario] {label} servo failed: {action.error()}")
        else:
            print(f"[scenario] {label} servo ended: state={action.state().value}")


def _phase_massage_both(ctrl: Controller) -> None:
    """! Run MassageAction on both hands simultaneously.

    @param ctrl<Controller>: Active controller.
    """
    print(f"[scenario] Phase massage — both hands in parallel ({_CYCLES} cycles)")

    left_massage = MassageAction(
        ctrl,
        device_name=_LEFT_HAND,
        cycles=_CYCLES,
        half_close_duration=_HALF_CLOSE_DURATION,
        open_duration=_OPEN_DURATION,
        torque_limit=_TORQUE_LIMIT,
    )
    right_massage = MassageAction(
        ctrl,
        device_name=_RIGHT_HAND,
        cycles=_CYCLES,
        half_close_duration=_HALF_CLOSE_DURATION,
        open_duration=_OPEN_DURATION,
        torque_limit=_TORQUE_LIMIT,
    )

    left_massage.start()
    right_massage.start()
    left_massage.wait(timeout=_MASSAGE_TIMEOUT)
    right_massage.wait(timeout=_MASSAGE_TIMEOUT)

    for label, action in (("left", left_massage), ("right", right_massage)):
        if action.state() == ActionState.DONE:
            print(f"[scenario] {label} massage complete.")
        elif action.state() == ActionState.FAILED:
            print(f"[scenario] {label} massage failed: {action.error()}")
        else:
            print(f"[scenario] {label} massage ended: state={action.state().value}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """! Run the full bilateral massage scenario end to end."""
    ctrl = Controller(Config(_DEVICE_FILE))

    opened: list[str] = []
    for device in (_LEFT_ARM, _RIGHT_ARM, _LEFT_HAND, _RIGHT_HAND, _HEAD_CAMERA):
        if not ctrl.open(device):
            print(f"[scenario] Failed to open '{device}'")
            for d in opened:
                ctrl.close(d)
            return
        opened.append(device)

    try:
        # -- Enable both arms --------------------------------------------------
        for device in (_LEFT_ARM, _RIGHT_ARM):
            if not ctrl.execute(device, "enable"):
                print(f"[scenario] Failed to enable '{device}'")
                return

        # -- Move to massage positions -----------------------------------------
        for device, key in ((_LEFT_ARM, _LEFT_MASSAGE_KEY), (_RIGHT_ARM, _RIGHT_MASSAGE_KEY)):
            joints = _load_joints(key)
            if joints is None:
                return
            print(f"[scenario] Moving '{device}' to {key} ...")
            if not ctrl.execute(device, "movej", *joints, vel=_MOVE_VEL):
                print(f"[scenario] movej failed for '{device}'")
                return
            print(f"[scenario] '{device}' reached {key}.")

        # -- Close thumb adduction on both hands --------------------------------
        for device in (_LEFT_HAND, _RIGHT_HAND):
            ctrl.execute(device, "set_torque", [_TORQUE_LIMIT] * 6)
            if not ctrl.execute(device, "move", _THUMB_ADDUCT):
                print(f"[scenario] Thumb adduct failed for '{device}'")
                return
            print(f"[scenario] '{device}' thumb adducted.")

        # -- Depth warmup ------------------------------------------------------
        print("[scenario] Waiting for depth stream ...")
        if not _poll_depth_warmup(ctrl):
            print("[scenario] Depth stream did not start — check head_camera connection.")
            return

        # -- Auto-capture target pose -----------------------------------------
        extrinsic_left = np.array(_LEFT_EXTRINSIC, dtype=np.float64)
        extrinsic_right = np.array(_RIGHT_EXTRINSIC, dtype=np.float64)

        captured = _auto_capture(ctrl, extrinsic_left, extrinsic_right)
        if captured is None:
            return
        target_kps_2d, kps_3d_left, kps_3d_right = captured

        # -- Visual servo both arms -------------------------------------------
        _phase_servo_both(ctrl, target_kps_2d, kps_3d_left, kps_3d_right)

        # -- Massage both hands -----------------------------------------------
        _phase_massage_both(ctrl)

        # -- Return to home ---------------------------------------------------
        for device, key in ((_LEFT_ARM, _LEFT_HOME_KEY), (_RIGHT_ARM, _RIGHT_HOME_KEY)):
            joints = _load_joints(key)
            if joints is None:
                return
            print(f"[scenario] Moving '{device}' to {key} ...")
            if not ctrl.execute(device, "movej", *joints, vel=_MOVE_VEL):
                print(f"[scenario] movej failed for '{device}'")
                return
            print(f"[scenario] '{device}' reached {key}.")

        for device in (_LEFT_HAND, _RIGHT_HAND):
            ctrl.execute(device, "move", _OPEN_POSE)
            print(f"[scenario] '{device}' fingers opened.")

        print("[scenario] Massage scenario complete.")

    finally:
        for device in opened:
            ctrl.close(device)


if __name__ == "__main__":
    main()
