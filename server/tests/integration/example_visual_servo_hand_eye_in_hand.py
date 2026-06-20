#!/usr/bin/env python3
##
# @file example_visual_servo_hand_eye_in_hand.py
#
# @brief Integration example: eye-in-hand visual servo that tracks a human
#        hand using the left Fairino arm and the left Orbbec camera.
#
#        Phase 1 — Capture target
#            EstimateHandAction streams in a loop; detected hand landmarks are
#            drawn on screen.  Press SPACE on a frame with a detected hand to
#            lock those pixel positions as the servo target.
#
#        Phase 2 — Servo
#            VisualServoAction (eye_in_hand, joint space) runs until the hand
#            returns to the captured target positions or timeout elapses.
#            Press Q to cancel.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/20.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# External library
import cv2

# Internal library
from actions.base import ActionState
from actions.estimate_hand import EstimateHandAction
from actions.visual_servo import VisualServoAction
from app.config import Config
from app.controller import Controller


_LEFT_ARM = "left_arm"
_LEFT_CAMERA = "left_camera"
_SERVO_MODEL_NAME = "yolo11n-hand-pose.pt"
_KP_CONF_THRESHOLD = 0.5
_ERROR_THRESHOLD = 25.0
_STABLE_TICKS = 10
_ACTION_TIMEOUT = 30.0
_ESTIMATE_TIMEOUT = 3.0

_DEVICE_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "projects", "anlab", "device.yaml")
)

# MediaPipe 21-landmark hand skeleton connections.
_HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Gain matrix: gain_matrix[j] = [gx, gy] maps (mean_ex, mean_ey) -> joint j
# increment in degrees.  All-zero → monitoring only.  Calibrate per setup.
_GAIN_MATRIX = [[0.0, 0.0]] * 6


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_hand(
    frame, hand: dict, conf_min: float, color: tuple, label: str = ""
) -> None:
    """! Draw landmarks, skeleton, and bounding box for one detected hand.

    @param frame: BGR numpy array; modified in-place.
    @param hand<dict>: Hand dict from EstimateHandAction.result().
    @param conf_min<float>: Minimum keypoint confidence to draw.
    @param color<tuple>: BGR draw color.
    @param label<str>: Optional text drawn above bounding box.
    """
    kps = hand["keypoints"]
    confs = hand["keypoint_conf"]

    for i, j in _HAND_SKELETON:
        if (
            len(kps) > max(i, j)
            and confs[i] >= conf_min
            and confs[j] >= conf_min
        ):
            pt1 = (int(kps[i][0]), int(kps[i][1]))
            pt2 = (int(kps[j][0]), int(kps[j][1]))
            cv2.line(frame, pt1, pt2, color, 1)

    for (x, y), c in zip(kps, confs):
        if c >= conf_min:
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)

    x1, y1, x2, y2 = (int(v) for v in hand["bbox"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    text = label or f"{hand['conf']:.2f}"
    cv2.putText(frame, text, (x1, max(y1 - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def _draw_target(frame, target_kps: list[list[float]]) -> None:
    """! Draw target keypoints as green cross markers.

    @param frame: BGR numpy array; modified in-place.
    @param target_kps<list[list[float]]>: Target pixel positions [[x, y], ...].
    """
    for x, y in target_kps:
        cv2.drawMarker(
            frame, (int(x), int(y)), (0, 255, 0), cv2.MARKER_CROSS, 10, 1
        )


def _draw_status(frame, text: str) -> None:
    """! Overlay a status line at the bottom of the frame.

    @param frame: BGR numpy array; modified in-place.
    @param text<str>: Status message.
    """
    h = frame.shape[0]
    cv2.putText(frame, text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------

def _phase_capture(ctrl: Controller) -> list[list[float]] | None:
    """! Stream hand detection until the user captures a target.

    Runs EstimateHandAction in a loop.  Overlays detected landmarks on each
    frame.  Returns the 21-keypoint pixel positions of the first detected hand
    when the user presses SPACE, or None if the user presses Q or closes the
    window.

    @param ctrl<Controller>: Active controller.
    @return<list[list[float]]|None>: Captured target keypoints or None.
    """
    print("[example] Phase 1 — Hand capture")
    print("[example]   SPACE: capture hand as target | Q: quit")

    while True:
        action = EstimateHandAction(
            ctrl, _LEFT_CAMERA, warmup_timeout=_ESTIMATE_TIMEOUT
        )
        action.start()
        action.wait(timeout=_ESTIMATE_TIMEOUT + 1.0)

        frame = ctrl.execute(_LEFT_CAMERA, "get_color_frame")
        if frame is None:
            continue

        hands = action.result() or []
        for hand in hands:
            _draw_hand(frame, hand, _KP_CONF_THRESHOLD, (0, 200, 255))

        hint = "SPACE: capture" if hands else "no hand detected"
        _draw_status(frame, hint)
        cv2.imshow("Capture target — left camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return None
        if key == ord(" ") and hands:
            target = hands[0]["keypoints"]
            print(f"[example] Target captured: wrist={target[0]}, index_tip={target[8]}")
            return target


def _phase_servo(ctrl: Controller, target_kps: list[list[float]]) -> None:
    """! Run eye-in-hand VisualServoAction toward the captured hand target.

    @param ctrl<Controller>: Active controller.
    @param target_kps<list[list[float]]>: 21-keypoint pixel targets.
    """
    print("[example] Phase 2 — Visual servo (eye_in_hand)")
    print("[example]   Q: cancel")

    action = VisualServoAction(
        ctrl,
        robot_device=_LEFT_ARM,
        camera_device=_LEFT_CAMERA,
        target_keypoints=target_kps,
        error_threshold=_ERROR_THRESHOLD,
        stable_ticks=_STABLE_TICKS,
        gain_matrix=_GAIN_MATRIX,
        cmd_period=0.016,
        timeout=_ACTION_TIMEOUT,
        model_name=_SERVO_MODEL_NAME,
        keypoint_conf_min=_KP_CONF_THRESHOLD,
        servo_space="joint",
        camera_config="eye_in_hand",
    )
    action.start()

    while not action.wait(timeout=0.05):
        frame = ctrl.execute(_LEFT_CAMERA, "get_color_frame")
        if frame is None:
            continue

        _draw_target(frame, target_kps)
        _draw_status(frame, "servoing — Q to cancel")
        cv2.imshow("Visual servo — left camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            action.cancel()
            action.wait(timeout=2.0)
            break

    state = action.state()
    result = action.result()

    if state == ActionState.DONE and result:
        print(
            f"[example] Converged={result['converged']}  "
            f"stable_ticks={result['stable_ticks']}  "
            f"final_error={result['final_error']:.2f}px"
        )
        frame = ctrl.execute(_LEFT_CAMERA, "get_color_frame")
        if frame is not None:
            _draw_target(frame, target_kps)
            _draw_status(frame, f"done — error={result['final_error']:.1f}px")
            cv2.imshow("Visual servo — final", frame)
            cv2.waitKey(2000)
    elif state == ActionState.FAILED:
        print(f"[example] Action failed: {action.error()}")
    else:
        print(f"[example] Action ended with state={state.value}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """! Open left arm and left camera, capture a hand target, then servo."""
    ctrl = Controller(Config(_DEVICE_FILE))

    if not ctrl.open(_LEFT_CAMERA):
        print(f"[example] Failed to open '{_LEFT_CAMERA}'")
        return

    if not ctrl.open(_LEFT_ARM):
        print(f"[example] Failed to open '{_LEFT_ARM}'")
        ctrl.close(_LEFT_CAMERA)
        return

    try:
        target_kps = _phase_capture(ctrl)
        cv2.destroyAllWindows()

        if target_kps is None:
            print("[example] Capture cancelled — exiting.")
            return

        _phase_servo(ctrl, target_kps)

    finally:
        ctrl.close(_LEFT_ARM)
        ctrl.close(_LEFT_CAMERA)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
