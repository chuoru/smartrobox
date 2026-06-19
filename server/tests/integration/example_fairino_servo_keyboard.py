#!/usr/bin/env python3
##
# @file example_fairino_servo_keyboard.py
#
# @brief Integration example: jog a Fairino robot joint-by-joint or in Cartesian
#        space using keyboard keys via high-rate ServoJ / ServoC control (~60 Hz).
#
# @section author Author(s)
# - Created by chuoru on 2026/06/20.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import os
import sys
import time
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# External library
import cv2
import numpy as np
import yaml

# Internal library
from app.config import Config
from app.controller import Controller


_ROBOT_DEVICE = "robot"
_CMD_PERIOD = 0.016
_JOINT_DELTA = 0.5       # degrees per keypress
_CART_DELTA_TRANS = 1.0  # mm per keypress
_CART_DELTA_ROT = 0.5    # degrees per keypress
_HUD_WIDTH = 640
_HUD_HEIGHT = 400
_WINDOW_TITLE = "Fairino Servo Keyboard"
_ESC = 27
_TAB = 9
_MODE_JOINT = "JOINT"
_MODE_CART = "CART"

# Number row increases DOF; QWERTY row decreases. Indices 0-5 map to J1-J6 (joint)
# or X/Y/Z/Rx/Ry/Rz (cartesian) depending on the active mode.
_KEY_BINDINGS: dict[int, tuple[int, int]] = {
    ord("1"): (0, +1),  ord("q"): (0, -1),
    ord("2"): (1, +1),  ord("w"): (1, -1),
    ord("3"): (2, +1),  ord("e"): (2, -1),
    ord("4"): (3, +1),  ord("r"): (3, -1),
    ord("5"): (4, +1),  ord("t"): (4, -1),
    ord("6"): (5, +1),  ord("y"): (5, -1),
}

_JOINT_LABELS = ["J1", "J2", "J3", "J4", "J5", "J6"]
_CART_LABELS  = ["X",  "Y",  "Z",  "Rx", "Ry", "Rz"]
_CART_UNITS   = ["mm", "mm", "mm", "deg", "deg", "deg"]


def _build_key_hints() -> dict[int, tuple[str, str]]:
    """! Build per-DOF mapping to increase/decrease key characters.

    @return<dict[int, tuple[str, str]]>: {dof_idx: (inc_char, dec_char)}.
    """
    hints: dict[int, list] = {}
    for key_code, (dof_idx, direction) in _KEY_BINDINGS.items():
        hints.setdefault(dof_idx, ["", ""])
        if direction > 0:
            hints[dof_idx][0] = chr(key_code)
        else:
            hints[dof_idx][1] = chr(key_code)
    return {idx: (v[0], v[1]) for idx, v in hints.items()}


_KEY_HINTS = _build_key_hints()


def _draw_hud(
    canvas: np.ndarray,
    mode: str,
    joints: list[float],
    cart: list[float],
) -> None:
    """! Render current state and key-binding help onto the HUD canvas in-place.

    @param canvas<np.ndarray>: BGR image array of shape (_HUD_HEIGHT, _HUD_WIDTH, 3).
    @param mode<str>: Active control mode (_MODE_JOINT or _MODE_CART).
    @param joints<list[float]>: Current joint angles [j1..j6] in degrees.
    @param cart<list[float]>: Current TCP pose [x, y, z, rx, ry, rz] in mm/deg.
    """
    canvas[:] = 0

    mode_color = (0, 255, 128) if mode == _MODE_JOINT else (0, 200, 255)
    cv2.putText(
        canvas, f"Fairino Servo Keyboard  [{mode} MODE]",
        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2,
    )
    cv2.line(canvas, (10, 50), (_HUD_WIDTH - 10, 50), (80, 80, 80), 1)

    if mode == _MODE_JOINT:
        values = joints
        labels = _JOINT_LABELS
        units  = ["deg"] * 6
    else:
        values = cart
        labels = _CART_LABELS
        units  = _CART_UNITS

    for i, (label, value, unit) in enumerate(zip(labels, values, units)):
        inc_key, dec_key = _KEY_HINTS[i]
        y = 90 + i * 45
        cv2.putText(
            canvas, f"{label}: {value:+9.3f} {unit}",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1,
        )
        cv2.putText(
            canvas, f"[+] {inc_key}   [-] {dec_key}",
            (330, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 1,
        )

    cv2.line(canvas, (10, _HUD_HEIGHT - 60), (_HUD_WIDTH - 10, _HUD_HEIGHT - 60), (80, 80, 80), 1)
    cv2.putText(
        canvas, "Tab = toggle mode   ESC = quit",
        (10, _HUD_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1,
    )


def main() -> None:
    """! Open robot in debug mode, enter servo mode, and jog via keyboard."""
    device_cfg = {
        "devices": {
            _ROBOT_DEVICE: {
                "type": "fairino",
                "params": {"ip": "192.168.57.2"},
            }
        }
    }

    fd, cfg_path = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(device_cfg, f)

        ctrl = Controller(Config(cfg_path))

        if not ctrl.open(_ROBOT_DEVICE):
            print(f"[example] Failed to open robot '{_ROBOT_DEVICE}'")
            return

        print("[example] Robot opened.")
        print("[example] Keys: 1-6 increase DOF; Q/W/E/R/T/Y decrease DOF.")
        print("[example] Tab toggles Joint / Cartesian mode. ESC to quit.")

        try:
            err, joints = ctrl.execute(_ROBOT_DEVICE, "get_joint_pos")
            if err != 0:
                print(f"[example] get_joint_pos failed (err={err})")
                return
            joints = list(joints)

            err, cart = ctrl.execute(_ROBOT_DEVICE, "tpos")
            if err != 0:
                print(f"[example] tpos failed (err={err})")
                return
            cart = list(cart)

            if not ctrl.execute(_ROBOT_DEVICE, "servo_start"):
                print("[example] servo_start failed")
                return

            mode = _MODE_JOINT
            canvas = np.zeros((_HUD_HEIGHT, _HUD_WIDTH, 3), dtype=np.uint8)
            cv2.namedWindow(_WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

            try:
                while True:
                    t_start = time.perf_counter()

                    key = cv2.waitKey(1) & 0xFF
                    if key == _ESC:
                        print("[example] ESC pressed — quitting.")
                        break

                    if key == _TAB:
                        mode = _MODE_CART if mode == _MODE_JOINT else _MODE_JOINT
                        print(f"[example] Switched to {mode} mode.")

                    if key in _KEY_BINDINGS:
                        dof_idx, direction = _KEY_BINDINGS[key]
                        if mode == _MODE_JOINT:
                            joints[dof_idx] += direction * _JOINT_DELTA
                        else:
                            delta = _CART_DELTA_TRANS if dof_idx < 3 else _CART_DELTA_ROT
                            cart[dof_idx] += direction * delta

                    if mode == _MODE_JOINT:
                        if not ctrl.execute(_ROBOT_DEVICE, "servo_j", joints, _CMD_PERIOD):
                            print("[example] servo_j failed — stopping.")
                            break
                    else:
                        if not ctrl.execute(_ROBOT_DEVICE, "servo_c", cart, _CMD_PERIOD):
                            print("[example] servo_c failed — stopping.")
                            break

                    _draw_hud(canvas, mode, joints, cart)
                    cv2.imshow(_WINDOW_TITLE, canvas)

                    elapsed = time.perf_counter() - t_start
                    remaining = _CMD_PERIOD - elapsed
                    if remaining > 0:
                        time.sleep(remaining)

            finally:
                ctrl.execute(_ROBOT_DEVICE, "servo_end")
                cv2.destroyAllWindows()

        finally:
            ctrl.close(_ROBOT_DEVICE)

    finally:
        os.unlink(cfg_path)


if __name__ == "__main__":
    main()
