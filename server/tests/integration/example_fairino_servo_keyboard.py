#!/usr/bin/env python3
##
# @file example_fairino_servo_keyboard.py
#
# @brief Integration example: jog a Fairino robot joint-by-joint using keyboard
#        keys via high-rate ServoJ control (~60 Hz).
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
_JOINT_DELTA = 0.5
_HUD_WIDTH = 640
_HUD_HEIGHT = 400
_WINDOW_TITLE = "Fairino Servo Keyboard"
_ESC = 27

# Number row increases joint angle; QWERTY row decreases (number key sits above its letter).
_KEY_BINDINGS: dict[int, tuple[int, int]] = {
    ord("1"): (0, +1),  ord("q"): (0, -1),
    ord("2"): (1, +1),  ord("w"): (1, -1),
    ord("3"): (2, +1),  ord("e"): (2, -1),
    ord("4"): (3, +1),  ord("r"): (3, -1),
    ord("5"): (4, +1),  ord("t"): (4, -1),
    ord("6"): (5, +1),  ord("y"): (5, -1),
}


def _build_joint_hints() -> dict[int, tuple[str, str]]:
    """! Build a per-joint mapping to the increase/decrease key characters.

    @return<dict[int, tuple[str, str]]>: {joint_idx: (inc_char, dec_char)}.
    """
    hints: dict[int, list] = {}
    for key_code, (joint_idx, direction) in _KEY_BINDINGS.items():
        hints.setdefault(joint_idx, ["", ""])
        if direction > 0:
            hints[joint_idx][0] = chr(key_code)
        else:
            hints[joint_idx][1] = chr(key_code)
    return {idx: (v[0], v[1]) for idx, v in hints.items()}


_JOINT_HINTS = _build_joint_hints()


def _draw_hud(canvas: np.ndarray, joints: list[float]) -> None:
    """! Render joint angles and key-binding help onto the HUD canvas in-place.

    @param canvas<np.ndarray>: BGR image array of shape (_HUD_HEIGHT, _HUD_WIDTH, 3).
    @param joints<list[float]>: Current joint angles [j1..j6] in degrees.
    """
    canvas[:] = 0
    cv2.putText(
        canvas, "Fairino Servo Keyboard Jog",
        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
    )
    cv2.line(canvas, (10, 50), (_HUD_WIDTH - 10, 50), (80, 80, 80), 1)

    for i, angle in enumerate(joints):
        inc_key, dec_key = _JOINT_HINTS[i]
        y = 90 + i * 45
        cv2.putText(
            canvas, f"J{i + 1}: {angle:+8.3f} deg",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1,
        )
        cv2.putText(
            canvas, f"[+] {inc_key}   [-] {dec_key}",
            (330, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 1,
        )

    cv2.line(canvas, (10, _HUD_HEIGHT - 45), (_HUD_WIDTH - 10, _HUD_HEIGHT - 45), (80, 80, 80), 1)
    cv2.putText(
        canvas, "ESC = quit",
        (10, _HUD_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1,
    )


def main() -> None:
    """! Open robot in debug mode, enter servo mode, and jog joints via keyboard."""
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

        print("[example] Robot opened. Press keys to jog joints. ESC to quit.")
        print("[example] Keys: 1-6 increase J1-J6; Q/W/E/R/T/Y decrease J1-J6.")

        try:
            err, joints = ctrl.execute(_ROBOT_DEVICE, "get_joint_pos")
            if err != 0:
                print(f"[example] get_joint_pos failed (err={err})")
                return

            joints = list(joints)

            if not ctrl.execute(_ROBOT_DEVICE, "servo_start"):
                print("[example] servo_start failed")
                return

            canvas = np.zeros((_HUD_HEIGHT, _HUD_WIDTH, 3), dtype=np.uint8)
            cv2.namedWindow(_WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

            try:
                while True:
                    t_start = time.perf_counter()

                    key = cv2.waitKey(1) & 0xFF
                    if key == _ESC:
                        print("[example] ESC pressed — quitting.")
                        break

                    if key in _KEY_BINDINGS:
                        joint_idx, direction = _KEY_BINDINGS[key]
                        joints[joint_idx] += direction * _JOINT_DELTA

                    if not ctrl.execute(_ROBOT_DEVICE, "servo_j", joints, _CMD_PERIOD):
                        print("[example] servo_j failed — stopping.")
                        break

                    _draw_hud(canvas, joints)
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
