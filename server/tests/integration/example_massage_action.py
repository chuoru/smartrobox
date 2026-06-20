#!/usr/bin/env python3
##
# @file example_massage_action.py
#
# @brief Integration example: run MassageAction on the left LinkerBot O6 hand.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/20.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Internal library
from actions.base import ActionState
from actions.massage import MassageAction
from app.config import Config
from app.controller import Controller


_DEVICE_NAME = "left_hand"
_CYCLES = 5
_HALF_CLOSE_DURATION = 0.4
_OPEN_DURATION = 0.4
_TORQUE_LIMIT = 180
_ACTION_TIMEOUT = 30.0
_OPEN_POSE = [255] * 6
_DEVICE_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "projects", "anlab", "device.yaml")
)


def main() -> None:
    """! Open the left hand and run the massage action for 5 rhythm cycles."""
    ctrl = Controller(Config(_DEVICE_FILE))

    if not ctrl.open(_DEVICE_NAME):
        print(f"[example] Failed to open '{_DEVICE_NAME}' — check connection and device.yaml")
        return

    try:
        action = MassageAction(
            ctrl,
            device_name=_DEVICE_NAME,
            cycles=_CYCLES,
            half_close_duration=_HALF_CLOSE_DURATION,
            open_duration=_OPEN_DURATION,
            torque_limit=_TORQUE_LIMIT,
        )

        print(f"[example] Starting massage on '{_DEVICE_NAME}' ({_CYCLES} cycles) ...")
        action.start()
        finished = action.wait(timeout=_ACTION_TIMEOUT)

        if not finished:
            print(f"[example] Action timed out after {_ACTION_TIMEOUT}s")
        elif action.state() == ActionState.DONE:
            print(f"[example] Massage complete.")
        elif action.state() == ActionState.CANCELLED:
            print(f"[example] Massage cancelled.")
        else:
            print(f"[example] Massage failed: {action.error()}")

        ctrl.execute(_DEVICE_NAME, "move", _OPEN_POSE)

    finally:
        ctrl.close(_DEVICE_NAME)


if __name__ == "__main__":
    main()
