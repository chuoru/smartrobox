#!/usr/bin/env python3
##
# @file estimate_pose.py
#
# @brief Action that runs YOLO11 pose estimation on a frame from an Orbbec camera.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/20.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import time

# External library
import numpy as np
from ultralytics import YOLO

# Internal library
from actions.base import BaseAction
from app.controller import Controller


class EstimatePoseAction(BaseAction):
    """! Captures one color frame from an Orbbec camera and runs YOLO11 pose estimation.

    The YOLO model is loaded once at construction time.  Each call to start()
    grabs a fresh frame, runs inference, and stores the list of detected persons
    as the action result accessible via result().

    Result format — list of dicts, one per detected person::

        [
            {
                "keypoints":      [[x, y], ...],  # 17 COCO keypoints, pixel coords
                "keypoint_conf":  [c, ...],        # 17 confidence floats
                "bbox":           [x1, y1, x2, y2],
                "conf":           float,
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
        model_name: str = "yolo11n-pose.pt",
        warmup_timeout: float = 3.0,
    ) -> None:
        """! Load the YOLO model and store action parameters.

        @param controller<Controller>: Controller used to dispatch device calls.
        @param device_name<str>: Registered name of the Orbbec camera device.
        @param model_name<str>: YOLO11 pose model weight name or path.
            Ultralytics auto-downloads named weights on first use.
        @param warmup_timeout<float>: Seconds to poll for the first frame before
            giving up; accommodates camera hardware start-up delay.
        """
        super().__init__(controller)
        self._device_name = device_name
        self._model_name = model_name
        self._warmup_timeout = warmup_timeout
        self._model = YOLO(model_name)

    def parameters(self) -> dict:
        """! Return the action's configuration parameters.

        @return<dict>: {"device_name": ..., "model_name": ...}
        """
        return {
            "device_name": self._device_name,
            "model_name": self._model_name,
            "warmup_timeout": self._warmup_timeout,
        }

    # =========================================================================
    # PROTECTED METHODS
    # =========================================================================

    def _run(self) -> list[dict] | None:
        """! Grab a frame, run pose inference, and return detected persons.

        @return<list[dict]|None>: List of pose dicts on success, None if cancelled.
        @raises RuntimeError: If no frame arrives within warmup_timeout seconds.
        """
        frame = self._poll_for_frame()
        if frame is None:
            raise RuntimeError(
                f"No frame received from camera '{self._device_name}'"
            )
        results = self._model(frame, verbose=False)
        if not self._checkpoint():
            return None
        return self._extract_poses(results[0])

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

    def _extract_poses(self, result) -> list[dict]:
        """! Convert a YOLO Results object into a list of pose dicts.

        @param result: ultralytics Results object from a single image.
        @return<list[dict]>: One dict per detected person; empty list if none found.
        """
        poses = []
        if result.keypoints is None or result.boxes is None:
            return poses
        keypoints_xy = result.keypoints.xy.tolist()
        keypoints_conf = result.keypoints.conf.tolist()
        boxes_xyxy = result.boxes.xyxy.tolist()
        boxes_conf = result.boxes.conf.tolist()
        for kp_xy, kp_conf, bbox, conf in zip(
            keypoints_xy, keypoints_conf, boxes_xyxy, boxes_conf
        ):
            poses.append(
                {
                    "keypoints": kp_xy,
                    "keypoint_conf": kp_conf,
                    "bbox": bbox,
                    "conf": conf,
                }
            )
        return poses
