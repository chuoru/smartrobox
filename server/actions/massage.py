#!/usr/bin/env python3
##
# @file massage.py
#
# @brief Action that drives a LinkerBot hand through a rhythmic massage sequence.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/20.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import time

# Internal library
from actions.base import BaseAction
from app.controller import Controller

_HALF = 128    # half-closed position value (O6: 128 = midpoint)
_CLOSED = 0    # fully-closed position value (O6: 0 = bent/closed)
_OPEN = 255    # fully-open position value  (O6: 255 = extended/open)

# Thumb adducted (yaw closed), all four fingers extended
_THUMB_ADDUCT = [_OPEN, _CLOSED, _OPEN, _OPEN, _OPEN, _OPEN]

# Thumb adducted, all four fingers half-closed
_FINGERS_HALF = [_OPEN, _CLOSED, _HALF, _HALF, _HALF, _HALF]


class MassageAction(BaseAction):
    """! Action that drives a LinkerBot hand through a rhythmic massage sequence.

    Phase 0 (setup): thumb adduction in (yaw closes), all fingers open.
    Phase 1 (rhythm, repeated): all four fingers half-close, then re-open.
                                Thumb stays adducted throughout.

    O6 joint layout (pose index → joint):
      0  thumb_cmc_pitch   3  middle_mcp_pitch
      1  thumb_cmc_yaw     4  ring_mcp_pitch
      2  index_mcp_pitch   5  pinky_mcp_pitch
    """

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def __init__(
        self,
        controller: Controller,
        device_name: str,
        cycles: int = 5,
        half_close_duration: float = 0.4,
        open_duration: float = 0.4,
        torque_limit: int = 180,
    ) -> None:
        """! Initialise the massage action.

        @param controller<Controller>: Controller used to dispatch device calls.
        @param device_name<str>: Registered name of the LinkerBot device.
        @param cycles<int>: Number of half-close/open rhythm repetitions (≥ 1).
        @param half_close_duration<float>: Seconds to hold half-close per cycle (> 0).
        @param open_duration<float>: Seconds to hold open per cycle (> 0).
        @param torque_limit<int>: Maximum joint torque applied before motion (0–255).
        @raises ValueError: If any parameter is out of its valid range.
        """
        super().__init__(controller)
        if cycles < 1:
            raise ValueError(f"cycles must be ≥ 1, got {cycles!r}")
        if half_close_duration <= 0:
            raise ValueError(f"half_close_duration must be > 0, got {half_close_duration!r}")
        if open_duration <= 0:
            raise ValueError(f"open_duration must be > 0, got {open_duration!r}")
        if not (0 <= torque_limit <= 255):
            raise ValueError(f"torque_limit must be 0–255, got {torque_limit!r}")
        self._device_name = device_name
        self._cycles = cycles
        self._half_close_duration = half_close_duration
        self._open_duration = open_duration
        self._torque_limit = torque_limit

    def parameters(self) -> dict:
        """! Return the massage action's configuration parameters.

        @return<dict>: {"device_name", "cycles", "half_close_duration",
                        "open_duration", "torque_limit"}
        """
        return {
            "device_name": self._device_name,
            "cycles": self._cycles,
            "half_close_duration": self._half_close_duration,
            "open_duration": self._open_duration,
            "torque_limit": self._torque_limit,
        }

    # =========================================================================
    # PROTECTED METHODS
    # =========================================================================

    def _run(self) -> bool:
        """! Apply torque limit, adduct thumb, then execute rhythmic finger cycles.

        @return<bool>: True on successful completion, False if cancelled.
        @raises RuntimeError: If any device call returns False.
        """
        torque = [self._torque_limit] * 6
        if not self._call(self._device_name, "set_torque", torque):
            raise RuntimeError(f"set_torque failed on device '{self._device_name}'")

        if not self._call(self._device_name, "move", _THUMB_ADDUCT):
            raise RuntimeError(f"thumb adduct move failed on device '{self._device_name}'")
        if not self._checkpoint():
            return False

        for _ in range(self._cycles):
            if not self._call(self._device_name, "move", _FINGERS_HALF):
                raise RuntimeError(f"half-close move failed on device '{self._device_name}'")
            if not self._checkpoint():
                return False
            time.sleep(self._half_close_duration)

            if not self._call(self._device_name, "move", _THUMB_ADDUCT):
                raise RuntimeError(f"open move failed on device '{self._device_name}'")
            if not self._checkpoint():
                return False
            time.sleep(self._open_duration)

        return True
