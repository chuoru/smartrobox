#!/usr/bin/env python3
##
# @file example_estimate_hand_action.py
#
# @brief Integration example: EstimateHandAction streaming with cv2 overlay.
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
import cv2
import yaml

# Internal library
from actions.base import ActionState
from actions.estimate_hand import EstimateHandAction
from app.config import Config
from app.controller import Controller


_DEVICE_NAME = "camera"
_KP_CONF_THRESHOLD = 0.5
_ACTION_TIMEOUT = 5.0

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


def main() -> None:
    """! Open an Orbbec camera, run EstimateHandAction in a loop, and display results."""
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

        print("[example] Press 'q' to quit.")
        try:
            while True:
                action = EstimateHandAction(ctrl, _DEVICE_NAME)
                action.start()
                finished = action.wait(timeout=_ACTION_TIMEOUT)

                if not finished or action.state() != ActionState.DONE:
                    print(f"[example] Action did not complete — state={action.state()}, error={action.error()}")
                    break

                frame = ctrl.execute(_DEVICE_NAME, "get_color_frame")
                if frame is None:
                    continue

                _draw_hands(frame, action.result() or [])
                cv2.imshow("EstimateHandAction", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            ctrl.close(_DEVICE_NAME)
            cv2.destroyAllWindows()
    finally:
        os.unlink(cfg_path)


if __name__ == "__main__":
    main()
