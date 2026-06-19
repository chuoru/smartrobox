#!/usr/bin/env python3
##
# @file grasp.py
#
# @brief Action that closes a LinkerBot hand in a structured two-phase grasp.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/19.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Internal library
from actions.base import BaseAction
from app.controller import Controller

_HALF = 128  # half-closed position value (0–255)
_FULL = 255  # fully-closed position value (0–255)


class GraspAction(BaseAction):
    """! Action that drives a LinkerBot hand through a two-phase grasp sequence.

    Phase 1 (prepare): thumb abduction in, selected fingers half-closed.
    Phase 2 (grasp):   thumb pitch in, selected fingers fully closed.

    Grasp levels determine which fingers participate:
      1 — index only
      2 — index + middle
      3 — index + middle + ring
      4 — all fingers (no half-in prepare for fingers; thumb abduction in only)

    Torque limit is applied to all joints before motion begins.
    _checkpoint() is called after the prepare move to support cooperative
    pause/resume/cancel between phases.

    L10 joint layout (pose index → joint):
      0  thumb_cmc_pitch   4  ring_mcp_pitch    8  pinky_mcp_roll
      1  thumb_cmc_roll    5  pinky_mcp_pitch   9  thumb_cmc_yaw
      2  index_mcp_pitch   6  index_mcp_roll
      3  middle_mcp_pitch  7  ring_mcp_roll
    """

    _FINGER_INDICES_BY_LEVEL: dict[int, list[int]] = {
        1: [2],
        2: [2, 3],
        3: [2, 3, 4],
        4: [2, 3, 4, 5],
    }

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def __init__(
        self,
        controller: Controller,
        device_name: str,
        grasp_level: int,
        torque_limit: int,
    ) -> None:
        """! Initialise the grasp action.

        @param controller<Controller>: Controller used to dispatch device calls.
        @param device_name<str>: Registered name of the LinkerBot device.
        @param grasp_level<int>: Grasp intensity level (1–4).
        @param torque_limit<int>: Maximum joint torque applied before motion (0–255).
        @raises ValueError: If grasp_level not in [1..4] or torque_limit not in [0..255].
        """
        super().__init__(controller)
        if grasp_level not in self._FINGER_INDICES_BY_LEVEL:
            raise ValueError(f"grasp_level must be 1–4, got {grasp_level!r}")
        if not (0 <= torque_limit <= 255):
            raise ValueError(f"torque_limit must be 0–255, got {torque_limit!r}")
        self._device_name = device_name
        self._grasp_level = grasp_level
        self._torque_limit = torque_limit

    def parameters(self) -> dict:
        """! Return the grasp action's configuration parameters.

        @return<dict>: {"device_name": ..., "grasp_level": ..., "torque_limit": ...}
        """
        return {
            "device_name": self._device_name,
            "grasp_level": self._grasp_level,
            "torque_limit": self._torque_limit,
        }

    # =========================================================================
    # PROTECTED METHODS
    # =========================================================================

    def _run(self) -> bool:
        """! Apply torque limit then execute the two-phase grasp.

        @return<bool>: True on successful completion, False if cancelled.
        @raises RuntimeError: If any device call returns False.
        """
        torque = [self._torque_limit] * 10
        if not self._call(self._device_name, "set_torque", torque):
            raise RuntimeError(f"set_torque failed on device '{self._device_name}'")
        prepare, grasp = self._build_poses()
        if not self._call(self._device_name, "move", prepare):
            raise RuntimeError(f"prepare move failed on device '{self._device_name}'")
        if not self._checkpoint():
            return False
        if not self._call(self._device_name, "move", grasp):
            raise RuntimeError(f"grasp move failed on device '{self._device_name}'")
        if not self._checkpoint():
            return False
        return True

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _build_poses(self) -> tuple[list[int], list[int]]:
        """! Build the prepare and grasp pose arrays for the configured level.

        @return<tuple[list[int], list[int]]>: (prepare_pose, grasp_pose), each length 10.
        """
        prepare = [0] * 10
        grasp = [0] * 10

        # Thumb abduction in for both phases; thumb pitch closes during grasp
        prepare[9] = _FULL  # thumb_cmc_yaw
        grasp[0] = _FULL    # thumb_cmc_pitch
        grasp[9] = _FULL    # thumb_cmc_yaw

        fingers = self._FINGER_INDICES_BY_LEVEL[self._grasp_level]
        for idx in fingers:
            if self._grasp_level < 4:
                prepare[idx] = _HALF
            grasp[idx] = _FULL

        return prepare, grasp
