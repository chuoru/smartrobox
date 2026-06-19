#!/usr/bin/env python3
##
# @file example_move_to_home.py
#
# @brief Integration example: move left and right Fairino arms to their saved
#        home positions read from teaching_point.yaml.
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
import yaml

# Internal library
from app.config import Config
from app.controller import Controller


_LEFT_ARM = "left_arm"
_RIGHT_ARM = "right_arm"
_LEFT_HOME_KEY = "home_left"
_RIGHT_HOME_KEY = "home_right"
_MOVE_VEL = 20.0
_TEACHING_POINT_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "projects", "anlab", "teaching_point.yaml")
)


def _load_home_joints(key: str) -> list[float] | None:
    """! Load joint home angles for a named teaching point from teaching_point.yaml.

    @param key<str>: Top-level key to look up (e.g. ``"home_left"``).
    @return<list[float] | None>: [j1..j6] in degrees, or None if not found.
    """
    if not os.path.exists(_TEACHING_POINT_FILE):
        print(f"[example] Teaching point file not found: {_TEACHING_POINT_FILE}")
        print("[example] Run example_register_home.py first.")
        return None

    with open(_TEACHING_POINT_FILE, "r") as fh:
        data = yaml.safe_load(fh) or {}

    if key not in data:
        print(f"[example] Home position '{key}' not found in {_TEACHING_POINT_FILE}")
        print("[example] Run example_register_home.py first.")
        return None

    block = data[key]["joint"]
    return [float(block[k]) for k in ("j1", "j2", "j3", "j4", "j5", "j6")]


def main() -> None:
    """! Open both arms and move each to its saved home joint position."""
    device_cfg = {
        "devices": {
            _LEFT_ARM: {
                "type": "fairino",
                "params": {"ip": "192.168.58.2", "debug": True},
            },
            _RIGHT_ARM: {
                "type": "fairino",
                "params": {"ip": "192.168.58.2", "debug": True},
            },
        }
    }

    fd, cfg_path = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(device_cfg, f)

        ctrl = Controller(Config(cfg_path))

        opened: list[str] = []
        for device in (_LEFT_ARM, _RIGHT_ARM):
            if not ctrl.open(device):
                print(f"[example] Failed to open '{device}'")
                for d in opened:
                    ctrl.close(d)
                return
            opened.append(device)

        try:
            for device, key in ((_LEFT_ARM, _LEFT_HOME_KEY), (_RIGHT_ARM, _RIGHT_HOME_KEY)):
                joints = _load_home_joints(key)
                if joints is None:
                    return

                print(
                    f"[example] Moving '{device}' to home "
                    f"[{', '.join(f'{j:+.3f}' for j in joints)}] ..."
                )
                ok = ctrl.execute(device, "movej", *joints, vel=_MOVE_VEL)
                if not ok:
                    print(f"[example] movej failed for '{device}'")
                    return

                print(f"[example] '{device}' reached home.")

        finally:
            for device in opened:
                ctrl.close(device)

    finally:
        os.unlink(cfg_path)


if __name__ == "__main__":
    main()
