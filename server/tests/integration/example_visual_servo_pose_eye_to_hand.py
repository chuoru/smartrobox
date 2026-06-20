#!/usr/bin/env python3
##
# @file example_visual_servo_pose_eye_to_hand.py
#
# @brief Integration example: eye-to-hand visual servo that positions the left
#        Fairino arm relative to a detected human pose using the fixed head
#        Orbbec camera.
#
#        Phase 1 — Capture target
#            EstimatePoseAction streams in a loop with COCO skeleton overlay.
#            Press SPACE on a frame with a detected person to lock that pose
#            as the servo target.  pixel_to_world lifts each keypoint to the
#            left arm base frame for the 3D PBVS correction law.
#
#        Phase 2 — Servo
#            VisualServoAction (eye_to_hand, cart space) drives the left arm
#            TCP until the person's 3D keypoints return to the captured base-
#            frame positions or timeout elapses.  Press Q to cancel.
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

# Disable X11 MIT-SHM so cv2.imshow renders correctly over X11 forwarding.
os.environ.setdefault("QT_X11_NO_MITSHM", "1")

# External library
import cv2
import numpy as np

# Internal library
from actions.base import ActionState
from actions.estimate_pose import EstimatePoseAction
from actions.visual_servo import VisualServoAction
from app.config import Config
from app.controller import Controller


_LEFT_ARM    = "left_arm"
_RIGHT_ARM   = "right_arm"
_HEAD_CAMERA = "head_camera"
_MODEL_NAME        = "yolo11n-pose.pt"
_KP_CONF_THRESHOLD = 0.4
_ERROR_THRESHOLD   = 30.0
_STABLE_TICKS      = 10
_ACTION_TIMEOUT    = 30.0
_SERVO_GAIN_3D     = 0.5

_DEVICE_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "projects", "anlab", "device.yaml")
)

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

# COCO 17-keypoint skeleton pairs (0-indexed).
_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def _draw_poses(frame, poses: list[dict]) -> None:
    """! Overlay bounding boxes, keypoints, and skeleton for each detected person.

    @param frame: BGR numpy array; modified in-place.
    @param poses<list[dict]>: Pose dicts from EstimatePoseAction.result().
    """
    for person in poses:
        x1, y1, x2, y2 = (int(v) for v in person["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(frame, f"{person['conf']:.2f}", (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        kps = person["keypoints"]
        confs = person["keypoint_conf"]

        for i, j in _SKELETON:
            if (len(kps) > max(i, j)
                    and confs[i] >= _KP_CONF_THRESHOLD
                    and confs[j] >= _KP_CONF_THRESHOLD):
                cv2.line(frame,
                         (int(kps[i][0]), int(kps[i][1])),
                         (int(kps[j][0]), int(kps[j][1])),
                         (255, 0, 0), 1)

        for (x, y), c in zip(kps, confs):
            if c >= _KP_CONF_THRESHOLD:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)


def _draw_target_2d(frame, target_kps: list[list[float]]) -> None:
    """! Draw target keypoints as green cross markers.

    @param frame: BGR numpy array; modified in-place.
    @param target_kps<list[list[float]]>: Target pixel positions [[x, y], ...].
    """
    for x, y in target_kps:
        cv2.drawMarker(frame, (int(x), int(y)), (0, 255, 0),
                       cv2.MARKER_CROSS, 10, 1)


def _draw_status(frame, text: str) -> None:
    """! Overlay a status line at the bottom of the frame.

    @param frame: BGR numpy array; modified in-place.
    @param text<str>: Status message.
    """
    h = frame.shape[0]
    cv2.putText(frame, text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def _lift_to_base(
    ctrl: Controller,
    kps_2d: list[list[float]],
    kp_confs: list[float],
    extrinsic: np.ndarray,
) -> list[list[float]]:
    """! Lift 2D keypoints to the robot base frame via depth + extrinsic.

    For each keypoint whose confidence meets the threshold, pixel_to_world
    is called to get the camera-frame 3D position and then transformed by
    extrinsic.  Keypoints below threshold or without depth fall back to the
    centroid of valid points so that index alignment with kps_2d is preserved.

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
        cam_pt = ctrl.execute(
            _HEAD_CAMERA, "pixel_to_world", int(kp[0]), int(kp[1])
        )
        if cam_pt is None:
            pts_base.append(None)
            continue
        p_cam = np.array([cam_pt[0], cam_pt[1], cam_pt[2], 1.0])
        p_base = (extrinsic @ p_cam)[:3]
        pts_base.append(p_base.tolist())

    valid = [p for p in pts_base if p is not None]
    if not valid:
        centroid = [0.0, 0.0, 0.0]
    else:
        centroid = [sum(p[i] for p in valid) / len(valid) for i in range(3)]

    return [p if p is not None else centroid for p in pts_base]


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


def _poll_color_warmup(ctrl: Controller, timeout: float = 5.0) -> bool:
    """! Block until the head camera delivers a non-black color frame.

    Orbbec sensors may output all-zero frames for several seconds after
    startup even though depth is already streaming; this gate ensures the
    color path is live before pose estimation begins.

    @param ctrl<Controller>: Active controller.
    @param timeout<float>: Maximum wait in seconds.
    @return<bool>: True if a valid color frame arrived, False on timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = ctrl.execute(_HEAD_CAMERA, "get_color_frame")
        if frame is not None and np.any(frame):
            return True
        time.sleep(0.05)
    return False


def _phase_capture(
    ctrl: Controller, extrinsic: np.ndarray
) -> tuple[list[list[float]], list[list[float]]] | None:
    """! Stream pose estimation until the user captures a target pose.

    @param ctrl<Controller>: Active controller.
    @param extrinsic<np.ndarray>: 4×4 T_cam_to_left_arm_base (float64).
    @return<tuple|None>: (target_kps_2d, target_kps_3d) or None if cancelled.
    """
    print("[example] Phase 1 — Pose capture (head camera)")
    print("[example]   SPACE: capture person as target | Q: quit")

    action = EstimatePoseAction(ctrl, _HEAD_CAMERA, _MODEL_NAME)
    while True:
        action.reset()
        action.start()
        finished = action.wait(timeout=_ACTION_TIMEOUT)

        if not finished or action.state() != ActionState.DONE:
            print(f"[example] Pose action did not finish — "
                  f"state={action.state()}, error={action.error()}")
            continue

        frame = ctrl.execute(_HEAD_CAMERA, "get_color_frame")
        if frame is None:
            continue

        poses = action.result() or []
        _draw_poses(frame, poses)
        hint = "SPACE: capture" if poses else "no person detected"
        _draw_status(frame, hint)
        cv2.imshow("Capture target — head camera", frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            return None
        if key == ord(" ") and poses:
            person = poses[0]
            kps_2d = person["keypoints"]
            kp_confs = person["keypoint_conf"]
            kps_3d = _lift_to_base(ctrl, kps_2d, kp_confs, extrinsic)
            n_valid = sum(1 for c in kp_confs if c >= _KP_CONF_THRESHOLD)
            print(
                f"[example] Target captured: {n_valid}/17 keypoints with depth — "
                f"wrist_l={kps_3d[9]}, wrist_r={kps_3d[10]}"
            )
            return kps_2d, kps_3d


def _phase_servo(
    ctrl: Controller,
    target_kps_2d: list[list[float]],
    target_kps_3d: list[list[float]],
    extrinsic_raw: list[list[float]],
) -> None:
    """! Run eye-to-hand VisualServoAction toward the captured pose target.

    @param ctrl<Controller>: Active controller.
    @param target_kps_2d<list[list[float]]>: 17-keypoint pixel targets.
    @param target_kps_3d<list[list[float]]>: 17-keypoint 3D targets in left arm base frame.
    @param extrinsic_raw<list[list[float]]>: 4×4 T_cam_to_left_arm_base as plain list.
    """
    print("[example] Phase 2 — Visual servo (eye_to_hand, cart space)")
    print("[example]   Q: cancel")

    action = VisualServoAction(
        ctrl,
        robot_device=_LEFT_ARM,
        camera_device=_HEAD_CAMERA,
        target_keypoints=target_kps_2d,
        error_threshold=_ERROR_THRESHOLD,
        stable_ticks=_STABLE_TICKS,
        cmd_period=0.016,
        timeout=_ACTION_TIMEOUT,
        model_name=_MODEL_NAME,
        keypoint_conf_min=_KP_CONF_THRESHOLD,
        camera_config="eye_to_hand",
        camera_extrinsic=extrinsic_raw,
        target_keypoints_3d=target_kps_3d,
        servo_gain_3d=_SERVO_GAIN_3D,
    )
    action.start()

    est_action = EstimatePoseAction(ctrl, _HEAD_CAMERA, _MODEL_NAME)
    est_action.reset()
    est_action.start()
    latest_poses: list[dict] = []

    while not action.wait(timeout=0.05):
        if est_action.wait(timeout=0):
            if est_action.state() == ActionState.DONE:
                latest_poses = est_action.result() or []
            est_action.reset()
            est_action.start()

        frame = ctrl.execute(_HEAD_CAMERA, "get_color_frame")
        if frame is None:
            continue

        _draw_poses(frame, latest_poses)
        _draw_target_2d(frame, target_kps_2d)
        _draw_status(frame, "servoing — Q to cancel")
        cv2.imshow("Visual servo — head camera", frame)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            action.cancel()
            action.wait(timeout=2.0)
            break

    if est_action.state() in (ActionState.RUNNING, ActionState.PAUSED):
        est_action.cancel()
        est_action.wait(timeout=2.0)

    state = action.state()
    result = action.result()

    if state == ActionState.DONE and result:
        print(
            f"[example] Converged={result['converged']}  "
            f"stable_ticks={result['stable_ticks']}  "
            f"final_error={result['final_error']:.2f}px"
        )
        frame = ctrl.execute(_HEAD_CAMERA, "get_color_frame")
        if frame is not None:
            _draw_target_2d(frame, target_kps_2d)
            _draw_status(frame, f"done — error={result['final_error']:.1f}px")
            cv2.imshow("Visual servo — final", frame)
            cv2.waitKey(2000)
    elif state == ActionState.FAILED:
        print(f"[example] Action failed: {action.error()}")
    else:
        print(f"[example] Action ended with state={state.value}")


def main() -> None:
    """! Open head camera and both arms, capture a human pose target, then servo."""
    ctrl = Controller(Config(_DEVICE_FILE))

    if not ctrl.open(_HEAD_CAMERA):
        print(f"[example] Failed to open '{_HEAD_CAMERA}'")
        return

    if not ctrl.open(_LEFT_ARM):
        print(f"[example] Failed to open '{_LEFT_ARM}'")
        ctrl.close(_HEAD_CAMERA)
        return

    if not ctrl.open(_RIGHT_ARM):
        print(f"[example] Failed to open '{_RIGHT_ARM}'")
        ctrl.close(_LEFT_ARM)
        ctrl.close(_HEAD_CAMERA)
        return

    try:
        print("[example] Waiting for depth stream …")
        if not _poll_depth_warmup(ctrl):
            print("[example] Depth stream did not start — check camera connection.")
            return

        print("[example] Waiting for color stream …")
        if not _poll_color_warmup(ctrl):
            print("[example] Color stream did not start — check camera connection.")
            return

        left_ext = np.array(_LEFT_ARM_EXTRINSIC, dtype=np.float64)

        captured = _phase_capture(ctrl, left_ext)
        cv2.destroyAllWindows()

        if captured is None:
            print("[example] Capture cancelled — exiting.")
            return

        target_kps_2d, target_kps_3d = captured
        _phase_servo(ctrl, target_kps_2d, target_kps_3d, _LEFT_ARM_EXTRINSIC)

    finally:
        ctrl.close(_RIGHT_ARM)
        ctrl.close(_LEFT_ARM)
        ctrl.close(_HEAD_CAMERA)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
