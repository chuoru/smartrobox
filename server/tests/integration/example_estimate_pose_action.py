#!/usr/bin/env python3
##
# @file example_estimate_pose_action.py
#
# @brief Integration example: EstimatePoseAction streaming with cv2 overlay,
#        showing shoulder 3D positions in camera frame and both arm base frames.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/20.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# External library
import cv2
import numpy as np
import yaml

# Internal library
from actions.base import ActionState
from actions.estimate_pose import EstimatePoseAction
from app.config import Config
from app.controller import Controller


_DEVICE_NAME = "head_camera"
_MODEL_NAME = "yolo11n-pose.pt"
_KP_CONF_THRESHOLD = 0.3
_ACTION_TIMEOUT = 5.0

_KP_LEFT_SHOULDER  = 5
_KP_RIGHT_SHOULDER = 6

# COCO 17-keypoint skeleton pairs (0-indexed)
_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# 4×4 T_cam_to_left_arm_base: head camera frame → left arm base frame (metres).
# Replace with the result of a hand-eye calibration routine.
_LEFT_ARM_EXTRINSIC = [
    [ 1.0,  0.0,  0.0,  0.0],
    [ 0.0,  1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0,  1.5],
    [ 0.0,  0.0,  0.0,  1.0],
]

# 4×4 T_cam_to_right_arm_base: head camera frame → right arm base frame (metres).
# Replace with the result of a hand-eye calibration routine.
_RIGHT_ARM_EXTRINSIC = [
    [ 1.0,  0.0,  0.0,  0.0],
    [ 0.0,  1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0,  1.5],
    [ 0.0,  0.0,  0.0,  1.0],
]


def _draw_poses(frame, poses: list[dict]) -> None:
    """! Overlay bounding boxes, keypoints, and skeleton on a BGR frame.

    @param frame: BGR numpy array; modified in-place.
    @param poses<list[dict]>: Pose dicts from EstimatePoseAction.result().
    """
    for person in poses:
        x1, y1, x2, y2 = (int(v) for v in person["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{person['conf']:.2f}",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        kps = person["keypoints"]
        kp_confs = person["keypoint_conf"]

        for i, ((x, y), c) in enumerate(zip(kps, kp_confs)):
            if c > _KP_CONF_THRESHOLD:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

        for i, j in _SKELETON:
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


def _get_shoulder_3d(
    ctrl: Controller,
    kp_xy: list[float],
    left_ext: np.ndarray,
    right_ext: np.ndarray,
) -> tuple:
    """! Convert a shoulder keypoint pixel to 3D in camera and both arm base frames.

    Calls pixel_to_world once and applies both extrinsics to avoid a second
    depth lookup.

    @param ctrl<Controller>: Active controller.
    @param kp_xy<list[float]>: [u, v] pixel coordinate of the keypoint.
    @param left_ext<np.ndarray>: 4×4 T_cam_to_left_arm (float64).
    @param right_ext<np.ndarray>: 4×4 T_cam_to_right_arm (float64).
    @return<tuple>: (cam_xyz, left_xyz, right_xyz) — each is a tuple/ndarray
        of (X, Y, Z) in metres, or None if depth is unavailable.
    """
    cam_pt = ctrl.execute(_DEVICE_NAME, "pixel_to_world", int(kp_xy[0]), int(kp_xy[1]))
    if cam_pt is None:
        return (None, None, None)
    p_h = np.array([cam_pt[0], cam_pt[1], cam_pt[2], 1.0])
    left_xyz  = (left_ext  @ p_h)[:3]
    right_xyz = (right_ext @ p_h)[:3]
    return (cam_pt, left_xyz, right_xyz)


def _draw_shoulder_overlay(
    frame,
    label: str,
    cam_xyz: tuple | None,
    left_xyz: np.ndarray | None,
    right_xyz: np.ndarray | None,
    origin_px: tuple[int, int],
) -> None:
    """! Overlay 3D shoulder position text near a keypoint on the frame.

    @param frame: BGR numpy array; modified in-place.
    @param label<str>: Keypoint label, e.g. "L_shoulder".
    @param cam_xyz<tuple|None>: Camera-frame (X, Y, Z) in metres, or None.
    @param left_xyz<np.ndarray|None>: Left arm base-frame position, or None.
    @param right_xyz<np.ndarray|None>: Right arm base-frame position, or None.
    @param origin_px<tuple[int, int]>: Pixel (u, v) to anchor the text block.
    """
    cam_str   = f"{cam_xyz[0]:.3f} {cam_xyz[1]:.3f} {cam_xyz[2]:.3f}" if cam_xyz is not None else "no depth"
    left_str  = f"{left_xyz[0]:.3f} {left_xyz[1]:.3f} {left_xyz[2]:.3f}" if left_xyz is not None else "--"
    right_str = f"{right_xyz[0]:.3f} {right_xyz[1]:.3f} {right_xyz[2]:.3f}" if right_xyz is not None else "--"

    lines = [label, f"cam:  {cam_str}", f"larm: {left_str}", f"rarm: {right_str}"]
    x = origin_px[0] + 8
    y = origin_px[1] + 4
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


def _print_shoulder_3d(
    label: str,
    cam_xyz: tuple | None,
    left_xyz: np.ndarray | None,
    right_xyz: np.ndarray | None,
) -> None:
    """! Print shoulder 3D positions in all three frames to stdout.

    @param label<str>: Keypoint label, e.g. "L_shoulder".
    @param cam_xyz<tuple|None>: Camera-frame (X, Y, Z) metres, or None.
    @param left_xyz<np.ndarray|None>: Left arm base-frame position, or None.
    @param right_xyz<np.ndarray|None>: Right arm base-frame position, or None.
    """
    cam_str   = f"({cam_xyz[0]:.3f}, {cam_xyz[1]:.3f}, {cam_xyz[2]:.3f})" if cam_xyz is not None else "no depth"
    left_str  = f"({left_xyz[0]:.3f}, {left_xyz[1]:.3f}, {left_xyz[2]:.3f})" if left_xyz is not None else "--"
    right_str = f"({right_xyz[0]:.3f}, {right_xyz[1]:.3f}, {right_xyz[2]:.3f})" if right_xyz is not None else "--"
    print(f"[example] {label}: cam={cam_str}  left_arm={left_str}  right_arm={right_str}")


def main() -> None:
    """! Open an Orbbec camera, run EstimatePoseAction in a loop, and display results
    including shoulder 3D positions in camera and arm base frames.
    """
    device_cfg = {
        "devices": {
            _DEVICE_NAME: {
                "type": "orbbec",
                "params": {"device_index": 2},
            }
        }
    }
    fd, cfg_path = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(device_cfg, f)

        ctrl = Controller(Config(cfg_path))

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
        try:
            while True:
                action = EstimatePoseAction(ctrl, _DEVICE_NAME, _MODEL_NAME)
                action.start()
                finished = action.wait(timeout=_ACTION_TIMEOUT)

                if not finished or action.state() != ActionState.DONE:
                    print(f"[example] Action did not complete — state={action.state()}, error={action.error()}")
                    break

                frame = ctrl.execute(_DEVICE_NAME, "get_color_frame")
                if frame is None:
                    continue

                poses = action.result() or []
                _draw_poses(frame, poses)

                if poses:
                    person = poses[0]
                    kps    = person["keypoints"]
                    confs  = person["keypoint_conf"]
                    if len(kps) > _KP_RIGHT_SHOULDER:
                        for idx, lbl in [(_KP_LEFT_SHOULDER, "L_shoulder"),
                                         (_KP_RIGHT_SHOULDER, "R_shoulder")]:
                            if confs[idx] >= _KP_CONF_THRESHOLD:
                                cam_xyz, left_xyz, right_xyz = _get_shoulder_3d(
                                    ctrl, kps[idx], left_ext, right_ext
                                )
                                _draw_shoulder_overlay(
                                    frame, lbl, cam_xyz, left_xyz, right_xyz,
                                    (int(kps[idx][0]), int(kps[idx][1])),
                                )
                                _print_shoulder_3d(lbl, cam_xyz, left_xyz, right_xyz)

                cv2.imshow("EstimatePoseAction", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            ctrl.close(_DEVICE_NAME)
            cv2.destroyAllWindows()
    finally:
        os.unlink(cfg_path)


if __name__ == "__main__":
    main()
