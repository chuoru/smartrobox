#!/usr/bin/env python3
##
# @file example_estimate_hand_action.py
#
# @brief Integration example: EstimateHandAction streaming with cv2 overlay,
#        showing distance from camera to each detected hand via depth data.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/20.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import math
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# External library
import cv2
import numpy as np

# Internal library
from actions.base import ActionState
from actions.estimate_hand import EstimateHandAction
from app.config import Config
from app.controller import Controller


_DEVICE_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "projects", "anlab", "device.yaml")
)
_DEVICE_NAME = "left_camera"
_KP_CONF_THRESHOLD = 0.5
_ACTION_TIMEOUT = 5.0

_KP_WRIST = 0  # MediaPipe landmark index — hand root used as distance reference

# 4×4 T_cam_to_left_arm_base: left camera frame → left arm base frame (metres).
# Replace with the result of a hand-eye calibration routine.
_LEFT_ARM_EXTRINSIC = [
    [ 1.0,  0.0,  0.0,  0.0],
    [ 0.0,  1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0,  0.3],
    [ 0.0,  0.0,  0.0,  1.0],
]

# 4×4 T_cam_to_right_arm_base: left camera frame → right arm base frame (metres).
# Replace with the result of a hand-eye calibration routine.
_RIGHT_ARM_EXTRINSIC = [
    [ 1.0,  0.0,  0.0,  0.0],
    [ 0.0,  1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0,  0.3],
    [ 0.0,  0.0,  0.0,  1.0],
]

# MediaPipe 21-landmark hand skeleton connections
_HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def _draw_hands(frame, hands: list[dict]) -> None:
    """! Overlay bounding boxes, keypoints, and skeleton on a BGR frame.

    @param frame: BGR numpy array; modified in-place.
    @param hands<list[dict]>: Hand dicts from EstimateHandAction.result().
    """
    for hand in hands:
        x1, y1, x2, y2 = (int(v) for v in hand["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{hand['conf']:.2f}",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        kps = hand["keypoints"]
        kp_confs = hand["keypoint_conf"]

        for i, ((x, y), c) in enumerate(zip(kps, kp_confs)):
            if c > _KP_CONF_THRESHOLD:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

        for i, j in _HAND_SKELETON:
            if kp_confs[i] > _KP_CONF_THRESHOLD and kp_confs[j] > _KP_CONF_THRESHOLD:
                pt1 = (int(kps[i][0]), int(kps[i][1]))
                pt2 = (int(kps[j][0]), int(kps[j][1]))
                cv2.line(frame, pt1, pt2, (255, 0, 0), 2)


def _poll_depth_warmup(ctrl: Controller, timeout: float = 3.0) -> bool:
    """! Block until the camera depth stream delivers its first frame.

    @param ctrl<Controller>: Active controller.
    @param timeout<float>: Maximum wait in seconds.
    @return<bool>: True if depth is ready, False on timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ctrl.execute(_DEVICE_NAME, "get_depth_frame") is not None:
            return True
        time.sleep(0.05)
    return False


def _get_hand_3d(
    ctrl: Controller,
    wrist_xy: list[float],
    left_ext: np.ndarray,
    right_ext: np.ndarray,
) -> tuple:
    """! Get wrist position in camera frame, both arm base frames, and distance.

    Calls pixel_to_world once and applies both extrinsics to avoid a second
    depth lookup.

    @param ctrl<Controller>: Active controller.
    @param wrist_xy<list[float]>: [u, v] pixel coordinate of the wrist.
    @param left_ext<np.ndarray>: 4×4 T_cam_to_left_arm (float64).
    @param right_ext<np.ndarray>: 4×4 T_cam_to_right_arm (float64).
    @return<tuple>: (cam_xyz, dist_m, left_xyz, right_xyz) — cam_xyz is
        (X, Y, Z) in metres, dist_m is Euclidean distance, left_xyz and
        right_xyz are ndarray(3,) in each arm base frame; all None on no depth.
    """
    cam_pt = ctrl.execute(_DEVICE_NAME, "pixel_to_world", int(wrist_xy[0]), int(wrist_xy[1]))
    if cam_pt is None:
        return (None, None, None, None)
    dist_m = math.sqrt(cam_pt[0] ** 2 + cam_pt[1] ** 2 + cam_pt[2] ** 2)
    p_h = np.array([cam_pt[0], cam_pt[1], cam_pt[2], 1.0])
    left_xyz  = (left_ext  @ p_h)[:3]
    right_xyz = (right_ext @ p_h)[:3]
    return (cam_pt, dist_m, left_xyz, right_xyz)


def _draw_hand_overlay(
    frame,
    hand_idx: int,
    cam_xyz: tuple | None,
    dist_m: float | None,
    left_xyz: np.ndarray | None,
    right_xyz: np.ndarray | None,
    wrist_px: tuple[int, int],
) -> None:
    """! Overlay wrist position in all frames and distance near the wrist keypoint.

    @param frame: BGR numpy array; modified in-place.
    @param hand_idx<int>: Zero-based hand index for labelling.
    @param cam_xyz<tuple|None>: Camera-frame (X, Y, Z) in metres, or None.
    @param dist_m<float|None>: Euclidean distance in metres, or None.
    @param left_xyz<np.ndarray|None>: Left arm base-frame position, or None.
    @param right_xyz<np.ndarray|None>: Right arm base-frame position, or None.
    @param wrist_px<tuple[int, int]>: Pixel (u, v) of the wrist keypoint.
    """
    cam_str   = f"{cam_xyz[0]:.3f} {cam_xyz[1]:.3f} {cam_xyz[2]:.3f}" if cam_xyz is not None else "no depth"
    dist_str  = f"{dist_m:.3f} m" if dist_m is not None else "--"
    left_str  = f"{left_xyz[0]:.3f} {left_xyz[1]:.3f} {left_xyz[2]:.3f}" if left_xyz is not None else "--"
    right_str = f"{right_xyz[0]:.3f} {right_xyz[1]:.3f} {right_xyz[2]:.3f}" if right_xyz is not None else "--"

    lines = [
        f"hand {hand_idx}",
        f"cam:  {cam_str}",
        f"dist: {dist_str}",
        f"larm: {left_str}",
        f"rarm: {right_str}",
    ]
    x = wrist_px[0] + 8
    y = wrist_px[1] + 4
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


def _print_hand_3d(
    hand_idx: int,
    cam_xyz: tuple | None,
    dist_m: float | None,
    left_xyz: np.ndarray | None,
    right_xyz: np.ndarray | None,
) -> None:
    """! Print wrist position in all frames and distance to stdout.

    @param hand_idx<int>: Zero-based hand index.
    @param cam_xyz<tuple|None>: Camera-frame (X, Y, Z) metres, or None.
    @param dist_m<float|None>: Euclidean distance in metres, or None.
    @param left_xyz<np.ndarray|None>: Left arm base-frame position, or None.
    @param right_xyz<np.ndarray|None>: Right arm base-frame position, or None.
    """
    cam_str   = f"({cam_xyz[0]:.3f}, {cam_xyz[1]:.3f}, {cam_xyz[2]:.3f})" if cam_xyz is not None else "no depth"
    dist_str  = f"{dist_m:.3f} m" if dist_m is not None else "--"
    left_str  = f"({left_xyz[0]:.3f}, {left_xyz[1]:.3f}, {left_xyz[2]:.3f})" if left_xyz is not None else "--"
    right_str = f"({right_xyz[0]:.3f}, {right_xyz[1]:.3f}, {right_xyz[2]:.3f})" if right_xyz is not None else "--"
    print(
        f"[example] hand {hand_idx}: cam={cam_str}  dist={dist_str}"
        f"  left_arm={left_str}  right_arm={right_str}"
    )


def main() -> None:
    """! Open an Orbbec camera, run EstimateHandAction in a loop, and display results
    including camera-frame position and distance to each detected hand wrist.
    """
    ctrl = Controller(Config(_DEVICE_FILE))

    if not ctrl.open(_DEVICE_NAME):
        print(f"[example] Failed to open device '{_DEVICE_NAME}'")
        return

    print("[example] Waiting for depth stream …")
    if not _poll_depth_warmup(ctrl):
        print("[example] Depth stream did not start — check camera connection.")
        ctrl.close(_DEVICE_NAME)
        return

    left_ext  = np.array(_LEFT_ARM_EXTRINSIC,  dtype=np.float64)
    right_ext = np.array(_RIGHT_ARM_EXTRINSIC, dtype=np.float64)

    print("[example] Press 'q' to quit.")
    action = EstimateHandAction(ctrl, _DEVICE_NAME)
    try:
        while True:
            action.reset()
            action.start()
            finished = action.wait(timeout=_ACTION_TIMEOUT)

            if not finished or action.state() != ActionState.DONE:
                print(f"[example] Action did not complete — state={action.state()}, error={action.error()}")
                break

            frame = ctrl.execute(_DEVICE_NAME, "get_color_frame")
            if frame is None:
                continue

            hands = action.result() or []
            _draw_hands(frame, hands)

            for idx, hand in enumerate(hands):
                kps   = hand["keypoints"]
                confs = hand["keypoint_conf"]
                if len(kps) > _KP_WRIST and confs[_KP_WRIST] >= _KP_CONF_THRESHOLD:
                    cam_xyz, dist_m, left_xyz, right_xyz = _get_hand_3d(
                        ctrl, kps[_KP_WRIST], left_ext, right_ext
                    )
                    wrist_px = (int(kps[_KP_WRIST][0]), int(kps[_KP_WRIST][1]))
                    _draw_hand_overlay(frame, idx, cam_xyz, dist_m, left_xyz, right_xyz, wrist_px)
                    _print_hand_3d(idx, cam_xyz, dist_m, left_xyz, right_xyz)

            cv2.imshow("EstimateHandAction", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        ctrl.close(_DEVICE_NAME)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
