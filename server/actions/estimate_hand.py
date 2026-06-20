#!/usr/bin/env python3
##
# @file estimate_hand.py
#
# @brief Action that runs MediaPipe hand pose estimation on a frame from an
#        Orbbec camera and returns per-hand landmark detections.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/20.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import time

# External library
import cv2
import mediapipe as mp
import numpy as np

# Internal library
from actions.base import BaseAction
from app.controller import Controller


class EstimateHandAction(BaseAction):
    """! Captures one color frame from an Orbbec camera and runs MediaPipe hand
    pose estimation to detect hand landmarks.

    The MediaPipe Hands solution is initialised once at construction time.
    Each call to start() grabs a fresh frame, runs inference, and returns
    detected hands ordered by descending detection confidence.

    Keypoint layout (21 MediaPipe hand landmarks):
        0  WRIST
        1  THUMB_CMC          2  THUMB_MCP       3  THUMB_IP         4  THUMB_TIP
        5  INDEX_MCP          6  INDEX_PIP        7  INDEX_DIP        8  INDEX_TIP
        9  MIDDLE_MCP        10  MIDDLE_PIP      11  MIDDLE_DIP      12  MIDDLE_TIP
        13 RING_MCP          14  RING_PIP        15  RING_DIP        16  RING_TIP
        17 PINKY_MCP         18  PINKY_PIP       19  PINKY_DIP       20  PINKY_TIP

    Result format — list of dicts, one per detected hand::

        [
            {
                "keypoints":     [[x, y], ...],  # 21 landmarks, pixel coords
                "keypoint_conf": [v, ...],        # 21 visibility scores (0–1)
                "bbox":          [x1, y1, x2, y2],
                "conf":          float,           # hand detection confidence
            },
            ...
        ]
    """

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def __init__(
        self,
        controller: Controller,
        device_name: str,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        warmup_timeout: float = 3.0,
    ) -> None:
        """! Initialise the MediaPipe Hands solution and store action parameters.

        @param controller<Controller>: Controller used to dispatch device calls.
        @param device_name<str>: Registered name of the Orbbec camera device.
        @param max_num_hands<int>: Maximum number of hands to detect.
        @param min_detection_confidence<float>: Minimum confidence threshold for
            hand detection (0.0–1.0).
        @param warmup_timeout<float>: Seconds to poll for the first frame before
            giving up; accommodates camera hardware start-up delay.
        """
        super().__init__(controller)
        self._device_name = device_name
        self._max_num_hands = max_num_hands
        self._min_detection_confidence = min_detection_confidence
        self._warmup_timeout = warmup_timeout
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
        )

    def parameters(self) -> dict:
        """! Return the action's configuration parameters.

        @return<dict>: {"device_name": ..., "max_num_hands": ...,
            "min_detection_confidence": ..., "warmup_timeout": ...}
        """
        return {
            "device_name": self._device_name,
            "max_num_hands": self._max_num_hands,
            "min_detection_confidence": self._min_detection_confidence,
            "warmup_timeout": self._warmup_timeout,
        }

    # =========================================================================
    # PROTECTED METHODS
    # =========================================================================

    def _run(self) -> list[dict] | None:
        """! Grab a frame, run hand pose inference, and return detected hands.

        @return<list[dict]|None>: List of hand dicts on success, None if cancelled.
        @raises RuntimeError: If no frame arrives within warmup_timeout seconds.
        """
        frame = self._poll_for_frame()
        if frame is None:
            raise RuntimeError(
                f"No frame received from camera '{self._device_name}'"
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        if not self._checkpoint():
            return None
        h, w = frame.shape[:2]
        return self._extract_hands(results, w, h)

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _poll_for_frame(self) -> np.ndarray | None:
        """! Poll get_color_frame until a frame arrives or warmup_timeout elapses.

        @return<np.ndarray|None>: BGR frame, or None on timeout or cancellation.
        """
        deadline = time.monotonic() + self._warmup_timeout
        while True:
            frame = self._call(self._device_name, "get_color_frame")
            if frame is not None:
                return frame
            if time.monotonic() >= deadline:
                return None
            if not self._checkpoint():
                return None
            time.sleep(0.05)

    def _extract_hands(self, results, width: int, height: int) -> list[dict]:
        """! Convert MediaPipe Hands results into a list of hand landmark dicts.

        @param results: MediaPipe Hands solution output object.
        @param width<int>: Frame width in pixels, used to denormalize x coords.
        @param height<int>: Frame height in pixels, used to denormalize y coords.
        @return<list[dict]>: One dict per detected hand; empty list if none found.
        """
        hands = []
        if not results.multi_hand_landmarks:
            return hands
        handedness_list = results.multi_handedness or []
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            keypoints = [
                [lm.x * width, lm.y * height] for lm in hand_landmarks.landmark
            ]
            keypoint_conf = [lm.visibility for lm in hand_landmarks.landmark]
            xs = [pt[0] for pt in keypoints]
            ys = [pt[1] for pt in keypoints]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            conf = (
                handedness_list[i].classification[0].score
                if i < len(handedness_list)
                else 1.0
            )
            hands.append(
                {
                    "keypoints": keypoints,
                    "keypoint_conf": keypoint_conf,
                    "bbox": bbox,
                    "conf": conf,
                }
            )
        hands.sort(key=lambda h: h["conf"], reverse=True)
        return hands
