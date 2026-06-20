#!/usr/bin/env python3
##
# @file example_register_teaching_point.py
#
# @brief Integration example: capture current joint and Cartesian poses of the
#        left and right Fairino arms and write them as a named teaching point
#        to teaching_point.yaml.  The name is supplied via a CLI argument.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/20.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# External library
import yaml

# Internal library
from app.config import Config
from app.controller import Controller


_LEFT_ARM = "left_arm"
_RIGHT_ARM = "right_arm"
_DEVICE_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "projects", "anlab", "device.yaml")
)
_TEACHING_POINT_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "projects", "anlab", "teaching_point.yaml")
)


def _read_arm_state(
    ctrl: Controller, device: str
) -> tuple[list[float], list[float]] | None:
    """! Read current joint and Cartesian positions from an arm.

    @param ctrl<Controller>: Active controller instance.
    @param device<str>: Registered device name.
    @return<tuple[list[float], list[float]] | None>: (joints, cartesian) or None on error.
    """
    err, joints = ctrl.execute(device, "get_joint_pos")
    if err != 0 or joints is None:
        print(f"[example] {device}: get_joint_pos failed (err={err})")
        return None

    err, pose = ctrl.execute(device, "tpos")
    if err != 0 or pose is None:
        print(f"[example] {device}: tpos failed (err={err})")
        return None

    return list(joints), list(pose)


def _save_teaching_point(key: str, joints: list[float], pose: list[float]) -> None:
    """! Upsert a named teaching point entry in teaching_point.yaml.

    @param key<str>: Top-level key to write (e.g. ``"pick_left"``).
    @param joints<list[float]>: Joint angles [j1..j6] in degrees.
    @param pose<list[float]>: Cartesian pose [x, y, z, rx, ry, rz] in mm / deg.
    """
    if os.path.exists(_TEACHING_POINT_FILE):
        with open(_TEACHING_POINT_FILE, "r") as fh:
            data = yaml.safe_load(fh) or {}
    else:
        data = {}

    data[key] = {
        "joint": {
            "j1": joints[0], "j2": joints[1], "j3": joints[2],
            "j4": joints[3], "j5": joints[4], "j6": joints[5],
        },
        "cartesian": {
            "x": pose[0], "y": pose[1], "z": pose[2],
            "rx": pose[3], "ry": pose[4], "rz": pose[5],
        },
    }

    with open(_TEACHING_POINT_FILE, "w") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)

    print(f"[example] '{key}' saved to {_TEACHING_POINT_FILE}")


def main() -> None:
    """! Open both arms, read current poses, and register them as a named teaching point."""
    parser = argparse.ArgumentParser(
        description="Register a named teaching point for both arms."
    )
    parser.add_argument("name", help="Base name for the teaching point (e.g. 'pick').")
    args = parser.parse_args()

    left_key = f"{args.name}_left"
    right_key = f"{args.name}_right"

    ctrl = Controller(Config(_DEVICE_FILE))

    opened: list[str] = []
    for device in (_LEFT_ARM, _RIGHT_ARM):
        if not ctrl.open(device):
            print(f"[example] Failed to open '{device}'")
            for d in opened:
                ctrl.close(d)
            return
        opened.append(device)

    try:
        for device, key in ((_LEFT_ARM, left_key), (_RIGHT_ARM, right_key)):
            state = _read_arm_state(ctrl, device)
            if state is None:
                return
            joints, pose = state
            print(f"[example] {device} joints    : {[f'{j:+.3f}' for j in joints]}")
            print(f"[example] {device} cartesian : {[f'{v:+.3f}' for v in pose]}")
            _save_teaching_point(key, joints, pose)

    finally:
        for device in opened:
            ctrl.close(device)


if __name__ == "__main__":
    main()
