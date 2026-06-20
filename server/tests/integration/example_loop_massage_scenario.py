#!/usr/bin/env python3
##
# @file example_loop_massage_scenario.py
#
# @brief Integration example: continuous bilateral massage loop scenario.
#
#        OUTER LOOP (infinite — press Ctrl+C to stop):
#          INNER LOOP (10 times):
#            Step 1: massage both hands + move both arms to home          [4 in parallel]
#            Step 2: massage both hands + move both arms to massage       [4 in parallel]
#            Step 3: massage both hands + move both arms to open position [4 in parallel]
#          END INNER LOOP
#          Step 4: open both hands + move both arms to handshake         [4 in parallel]
#          Step 5: wait 5 seconds
#          Step 6: close right hand only
#        END OUTER LOOP
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

# External library
import yaml

# Internal library
from actions.base import ActionState
from actions.massage import MassageAction
from app.config import Config
from app.controller import Controller


_LEFT_ARM  = "left_arm"
_RIGHT_ARM = "right_arm"
_LEFT_HAND  = "left_hand"
_RIGHT_HAND = "right_hand"

_HOME_LEFT_KEY       = "home_left"
_HOME_RIGHT_KEY      = "home_right"
_MASSAGE_LEFT_KEY    = "massage_left"
_MASSAGE_RIGHT_KEY   = "massage_right"
_OPEN_LEFT_KEY       = "open_left"
_OPEN_RIGHT_KEY      = "open_right"
_HANDSHAKE_LEFT_KEY  = "handshake_left"
_HANDSHAKE_RIGHT_KEY = "handshake_right"

_MOVE_VEL    = 20.0
_TORQUE_LIMIT = 180
_OPEN_POSE   = [255] * 6
_CLOSED_POSE = [0] * 6

# 3 cycles × (0.4 s half-close + 0.4 s open) ≈ 2.4 s, matching typical arm move time
_CYCLES              = 3
_HALF_CLOSE_DURATION = 0.4
_OPEN_DURATION       = 0.4
_MASSAGE_TIMEOUT     = 30.0

_INNER_LOOP_COUNT = 10
_HANDSHAKE_WAIT   = 5.0

_DEVICE_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "projects", "anlab", "device.yaml")
)
_TEACHING_POINT_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "projects", "anlab", "teaching_point.yaml")
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_joints(key: str) -> list[float] | None:
    """! Load joint angles for a named teaching point from teaching_point.yaml.

    @param key<str>: Top-level key (e.g. ``"home_left"``).
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


def _run_parallel_step(
    ctrl: Controller,
    left_arm_key: str,
    right_arm_key: str,
    label: str,
) -> bool:
    """! Massage both hands and move both arms to the given positions, all in parallel.

    Fires two MassageActions (each uses its own daemon thread via BaseAction.start())
    and two arm-move threads simultaneously, then waits for all four to finish.

    @param ctrl<Controller>: Active controller.
    @param left_arm_key<str>: Teaching point key for the left arm destination.
    @param right_arm_key<str>: Teaching point key for the right arm destination.
    @param label<str>: Human-readable name for logging (e.g. "home").
    @return<bool>: True if all four tasks completed successfully, False otherwise.
    """
    joints_left  = _load_joints(left_arm_key)
    joints_right = _load_joints(right_arm_key)
    if joints_left is None or joints_right is None:
        return False

    print(f"[scenario] Step '{label}' — massage (both hands) + move (both arms) in parallel ...")

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

    arm_results: dict[str, bool] = {}

    def _move(device: str, joints: list[float], key: str) -> None:
        print(f"[scenario]   Moving '{device}' to '{key}' ...")
        ok = bool(ctrl.execute(device, "movej", *joints, vel=_MOVE_VEL))
        arm_results[device] = ok
        if ok:
            print(f"[scenario]   '{device}' reached '{key}'.")
        else:
            print(f"[scenario]   movej failed for '{device}'.")

    left_arm_t  = threading.Thread(target=_move, args=(_LEFT_ARM,  joints_left,  left_arm_key))
    right_arm_t = threading.Thread(target=_move, args=(_RIGHT_ARM, joints_right, right_arm_key))

    left_massage.start()
    right_massage.start()
    left_arm_t.start()
    right_arm_t.start()

    left_massage.wait(timeout=_MASSAGE_TIMEOUT)
    right_massage.wait(timeout=_MASSAGE_TIMEOUT)
    left_arm_t.join()
    right_arm_t.join()

    all_ok = True
    for hand_label, action in (("left_hand", left_massage), ("right_hand", right_massage)):
        if action.state() == ActionState.DONE:
            print(f"[scenario]   {hand_label} massage complete.")
        elif action.state() == ActionState.FAILED:
            print(f"[scenario]   {hand_label} massage FAILED: {action.error()}")
            all_ok = False
        else:
            print(f"[scenario]   {hand_label} massage ended: state={action.state().value}")
            all_ok = False

    if not arm_results.get(_LEFT_ARM, False):
        all_ok = False
    if not arm_results.get(_RIGHT_ARM, False):
        all_ok = False

    return all_ok


def _step_handshake(ctrl: Controller) -> bool:
    """! Open both hands and move both arms to handshake positions, all in parallel.

    @param ctrl<Controller>: Active controller.
    @return<bool>: True if all four tasks completed successfully, False otherwise.
    """
    joints_left  = _load_joints(_HANDSHAKE_LEFT_KEY)
    joints_right = _load_joints(_HANDSHAKE_RIGHT_KEY)
    if joints_left is None or joints_right is None:
        return False

    print("[scenario] Step 'handshake' — open both hands + move both arms in parallel ...")

    results: dict[str, bool] = {}

    def _open_hand(device: str) -> None:
        ok = bool(ctrl.execute(device, "move", _OPEN_POSE))
        results[device] = ok
        if ok:
            print(f"[scenario]   '{device}' opened.")
        else:
            print(f"[scenario]   open failed for '{device}'.")

    def _move_arm(device: str, joints: list[float], key: str) -> None:
        ok = bool(ctrl.execute(device, "movej", *joints, vel=_MOVE_VEL))
        results[device] = ok
        if ok:
            print(f"[scenario]   '{device}' reached '{key}'.")
        else:
            print(f"[scenario]   movej failed for '{device}'.")

    lh_t = threading.Thread(target=_open_hand, args=(_LEFT_HAND,))
    rh_t = threading.Thread(target=_open_hand, args=(_RIGHT_HAND,))
    la_t = threading.Thread(target=_move_arm,  args=(_LEFT_ARM,  joints_left,  _HANDSHAKE_LEFT_KEY))
    ra_t = threading.Thread(target=_move_arm,  args=(_RIGHT_ARM, joints_right, _HANDSHAKE_RIGHT_KEY))

    lh_t.start()
    rh_t.start()
    la_t.start()
    ra_t.start()

    lh_t.join()
    rh_t.join()
    la_t.join()
    ra_t.join()

    return all(results.get(d, False) for d in (_LEFT_HAND, _RIGHT_HAND, _LEFT_ARM, _RIGHT_ARM))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """! Run the infinite bilateral massage loop scenario.

    Press Ctrl+C to stop cleanly after the current step.
    """
    ctrl = Controller(Config(_DEVICE_FILE))

    opened: list[str] = []
    for device in (_LEFT_ARM, _RIGHT_ARM, _LEFT_HAND, _RIGHT_HAND):
        if not ctrl.open(device):
            print(f"[scenario] Failed to open '{device}'")
            for d in opened:
                ctrl.close(d)
            return
        opened.append(device)

    try:
        for device in (_LEFT_ARM, _RIGHT_ARM):
            if not ctrl.execute(device, "enable"):
                print(f"[scenario] Failed to enable '{device}'")
                return

        outer_count = 0
        while True:
            outer_count += 1
            print(f"\n[scenario] ===== Outer loop iteration {outer_count} =====")

            for inner_idx in range(1, _INNER_LOOP_COUNT + 1):
                print(f"[scenario] --- Inner {inner_idx}/{_INNER_LOOP_COUNT} ---")

                if not _run_parallel_step(ctrl, _HOME_LEFT_KEY, _HOME_RIGHT_KEY, "home"):
                    print("[scenario] Step 1 (home) failed — aborting.")
                    return

                if not _run_parallel_step(ctrl, _MASSAGE_LEFT_KEY, _MASSAGE_RIGHT_KEY, "massage"):
                    print("[scenario] Step 2 (massage) failed — aborting.")
                    return

                if not _run_parallel_step(ctrl, _OPEN_LEFT_KEY, _OPEN_RIGHT_KEY, "open"):
                    print("[scenario] Step 3 (open) failed — aborting.")
                    return

            if not _step_handshake(ctrl):
                print("[scenario] Step 4 (handshake) failed — aborting.")
                return

            print(f"[scenario] Step 5 — holding handshake for {_HANDSHAKE_WAIT:.0f} s ...")
            time.sleep(_HANDSHAKE_WAIT)

            print("[scenario] Step 6 — closing right hand ...")
            if not bool(ctrl.execute(_RIGHT_HAND, "move", _CLOSED_POSE)):
                print("[scenario] Step 6: right hand close failed — aborting.")
                return
            print("[scenario] Step 6: right hand closed.")

    except KeyboardInterrupt:
        print("\n[scenario] Interrupted by operator — stopping.")

    finally:
        for device in opened:
            ctrl.close(device)


if __name__ == "__main__":
    main()
