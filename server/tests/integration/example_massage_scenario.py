#!/usr/bin/env python3
##
# @file example_massage_scenario.py
#
# @brief Integration example: full bilateral massage scenario.
#
#        Steps:
#          1. Move both arms to massage_left / massage_right teaching points.
#          2. Close thumb adduction (yaw) on both hands.
#          3. Wait for depth stream to become available.
#          4. Visual servo both arms in parallel toward live shoulder positions
#             until each converges or times out.
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
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Disable X11 MIT-SHM so cv2.imshow renders correctly over X11 forwarding.
os.environ.setdefault("QT_X11_NO_MITSHM", "1")

# External library
import cv2
import numpy as np
import yaml

# Internal library
from actions.base import ActionState
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

_MOVE_VEL = 20.0
_TORQUE_LIMIT = 180
_OPEN_POSE = [255] * 6
# O6: thumb_cmc_pitch=open, thumb_cmc_yaw=closed, four fingers=open
_THUMB_ADDUCT = [255, 0, 255, 255, 255, 255]

_ERROR_THRESHOLD = 30.0   # mm — 3D distance between TCP and shoulder
_STABLE_TICKS = 10
_SERVO_GAIN = 0.5
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
# Axes: left_arm_z+ ~ cam_x-, left_arm_y+ ~ cam_z-, left_arm_x+ ~ cam_y+.
# Left arm origin in camera frame: x=-0.1 m, y=+0.2 m, z=0.
_LEFT_EXTRINSIC = [
    [ 0.0,  1.0,  0.0, -0.2],
    [ 0.0,  0.0, -1.0,  0.0],
    [-1.0,  0.0,  0.0, -0.1],
    [ 0.0,  0.0,  0.0,  1.0],
]

# 4×4 T_cam_to_base for head_camera → right_arm base (metres).
# Axes: right_arm_z+ ~ cam_x+, right_arm_y+ ~ cam_y+, right_arm_x+ ~ cam_z-.
# Right arm origin in camera frame: x=+0.1 m, y=+0.2 m, z=0.
_RIGHT_EXTRINSIC = [
    [ 0.0,  0.0, -1.0,  0.0],
    [ 0.0,  1.0,  0.0, -0.2],
    [ 1.0,  0.0,  0.0, -0.1],
    [ 0.0,  0.0,  0.0,  1.0],
]

_TERMINAL_STATES = {ActionState.DONE, ActionState.FAILED, ActionState.CANCELLED}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _move_dual_arm(
    ctrl: Controller,
    left_key: str,
    right_key: str,
) -> bool:
    """! Move left and right arms to named teaching points simultaneously.

    @param ctrl<Controller>: Active controller.
    @param left_key<str>: Teaching point key for the left arm.
    @param right_key<str>: Teaching point key for the right arm.
    @return<bool>: True if both arms reached their targets, False on any failure.
    """
    joints_left = _load_joints(left_key)
    joints_right = _load_joints(right_key)
    if joints_left is None or joints_right is None:
        return False

    results: dict[str, bool] = {}

    def _move(device: str, joints: list[float]) -> None:
        print(f"[scenario] Moving '{device}' to {left_key if device == _LEFT_ARM else right_key} ...")
        ok = ctrl.execute(device, "movej", *joints, vel=_MOVE_VEL)
        results[device] = bool(ok)
        if ok:
            print(f"[scenario] '{device}' reached target.")
        else:
            print(f"[scenario] movej failed for '{device}'")

    left_thread = threading.Thread(target=_move, args=(_LEFT_ARM, joints_left))
    right_thread = threading.Thread(target=_move, args=(_RIGHT_ARM, joints_right))
    left_thread.start()
    right_thread.start()
    left_thread.join()
    right_thread.join()

    return results.get(_LEFT_ARM, False) and results.get(_RIGHT_ARM, False)


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


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------

def _phase_servo_both(ctrl: Controller) -> None:
    """! Start the dual-arm visual servo action and poll until convergence.

    The action detects shoulder positions live each tick via YOLO11 and
    drives each arm's TCP toward its respective shoulder until the 3D
    error drops below _ERROR_THRESHOLD mm for _STABLE_TICKS ticks.

    @param ctrl<Controller>: Active controller.
    """
    print("[scenario] Phase servo — dual-arm live shoulder tracking (Q to cancel)")

    servo = VisualServoAction(
        ctrl,
        left_robot_device=_LEFT_ARM,
        right_robot_device=_RIGHT_ARM,
        camera_device=_HEAD_CAMERA,
        left_arm_extrinsic=_LEFT_EXTRINSIC,
        right_arm_extrinsic=_RIGHT_EXTRINSIC,
        error_threshold=_ERROR_THRESHOLD,
        stable_ticks=_STABLE_TICKS,
        servo_gain=_SERVO_GAIN,
        cmd_period=0.016,
        timeout=_SERVO_TIMEOUT,
        model_name=_MODEL_NAME,
        keypoint_conf_min=_KP_CONF_THRESHOLD,
    )
    servo.start()

    while True:
        frame = ctrl.execute(_HEAD_CAMERA, "get_color_frame")
        if frame is not None:
            cv2.imshow("Massage servo — head camera", frame)

        if cv2.waitKey(33) & 0xFF == ord("q"):
            print("[scenario] Servo cancelled by operator.")
            servo.cancel()
            break

        if servo.state() in _TERMINAL_STATES:
            break

    servo.wait(timeout=2.0)
    cv2.destroyAllWindows()

    result = servo.result()
    if servo.state() == ActionState.DONE and result:
        print(
            f"[scenario] servo: converged={result['converged']}  "
            f"L stable={result['left_stable_ticks']}  "
            f"R stable={result['right_stable_ticks']}  "
            f"L err={result['left_final_error']:.1f}mm  "
            f"R err={result['right_final_error']:.1f}mm"
        )
    elif servo.state() == ActionState.FAILED:
        print(f"[scenario] servo failed: {servo.error()}")
    else:
        print(f"[scenario] servo ended: state={servo.state().value}")


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
        if not _move_dual_arm(ctrl, _LEFT_MASSAGE_KEY, _RIGHT_MASSAGE_KEY):
            return

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

        # -- Visual servo both arms toward live shoulders ----------------------
        _phase_servo_both(ctrl)

        # -- Massage both hands -----------------------------------------------
        _phase_massage_both(ctrl)

        # -- Return to home ---------------------------------------------------
        if not _move_dual_arm(ctrl, _LEFT_HOME_KEY, _RIGHT_HOME_KEY):
            return

        for device in (_LEFT_HAND, _RIGHT_HAND):
            ctrl.execute(device, "move", _OPEN_POSE)
            print(f"[scenario] '{device}' fingers opened.")

        print("[scenario] Massage scenario complete.")

    finally:
        for device in opened:
            ctrl.close(device)


if __name__ == "__main__":
    main()
