#!/usr/bin/env python3
##
# @file robot_program.py
#
# @brief Action that executes an ordered list of robot move steps,
#        with text serialization and deserialization support.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/19.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
from dataclasses import dataclass

# Internal library
from actions.base import BaseAction
from app.controller import Controller


@dataclass
class MoveJStep:
    """! Joint-space move step for a robot program."""

    j1: float
    j2: float
    j3: float
    j4: float
    j5: float
    j6: float
    vel: float = 20.0
    tool_offset: list[float] | None = None
    base_offset: list[float] | None = None


@dataclass
class MoveLStep:
    """! Cartesian linear move step for a robot program."""

    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float
    vel: float = 20.0
    tool_offset: list[float] | None = None
    base_offset: list[float] | None = None


def _offset_fields(step) -> str:
    """! Serialize the offset portion of a step as ``mode ox oy oz orx ory orz``."""
    if step.tool_offset is not None:
        o = step.tool_offset
        return f"2 {o[0]} {o[1]} {o[2]} {o[3]} {o[4]} {o[5]}"
    if step.base_offset is not None:
        o = step.base_offset
        return f"1 {o[0]} {o[1]} {o[2]} {o[3]} {o[4]} {o[5]}"
    return "0 0.0 0.0 0.0 0.0 0.0 0.0"


def serialize(steps: list) -> str:
    """! Serialize a list of MoveJStep / MoveLStep to a newline-separated string.

    Each step becomes one line:
    ``movej j1 j2 j3 j4 j5 j6 vel offset_mode ox oy oz orx ory orz`` or
    ``movel x y z rx ry rz vel offset_mode ox oy oz orx ory orz``.
    offset_mode: 0=none, 1=base frame, 2=tool frame.

    @param steps<list>: Ordered list of MoveJStep / MoveLStep instances.
    @return<str>: Newline-separated text representation.
    @raises TypeError: If an unrecognised step type is encountered.
    """
    lines = []
    for step in steps:
        if isinstance(step, MoveJStep):
            lines.append(
                f"movej {step.j1} {step.j2} {step.j3}"
                f" {step.j4} {step.j5} {step.j6} {step.vel} {_offset_fields(step)}"
            )
        elif isinstance(step, MoveLStep):
            lines.append(
                f"movel {step.x} {step.y} {step.z}"
                f" {step.rx} {step.ry} {step.rz} {step.vel} {_offset_fields(step)}"
            )
        else:
            raise TypeError(f"Unknown step type: {type(step)}")
    return "\n".join(lines)


def deserialize(text: str) -> list:
    """! Deserialize a newline-separated string to a list of move steps.

    Blank lines are skipped.  Each non-blank line must start with ``movej``
    or ``movel`` followed by exactly 13 space-separated fields:
    6 motion values, vel, offset_mode (0/1/2), and 6 offset floats.

    @param text<str>: Text produced by serialize() or compatible format.
    @return<list>: Ordered list of MoveJStep / MoveLStep instances.
    @raises ValueError: If a line has an unknown kind, wrong field count,
        non-numeric field, or unknown offset_mode.
    """
    steps = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        kind = parts[0]
        if kind not in ("movej", "movel"):
            raise ValueError(f"Line {lineno}: unknown step kind {kind!r}")
        if len(parts) != 15:
            raise ValueError(
                f"Line {lineno}: {kind} expects 13 fields,"
                f" got {len(parts) - 1}"
            )
        try:
            f = [float(p) for p in parts[1:]]
        except ValueError:
            raise ValueError(
                f"Line {lineno}: non-numeric field in {line!r}"
            )
        offset_mode = int(f[7])
        offset_vals = list(f[8:14])
        if offset_mode == 0:
            tool_offset = base_offset = None
        elif offset_mode == 1:
            tool_offset, base_offset = None, offset_vals
        elif offset_mode == 2:
            tool_offset, base_offset = offset_vals, None
        else:
            raise ValueError(
                f"Line {lineno}: unknown offset_mode {offset_mode!r}"
            )
        if kind == "movej":
            steps.append(
                MoveJStep(f[0], f[1], f[2], f[3], f[4], f[5], f[6],
                          tool_offset=tool_offset, base_offset=base_offset)
            )
        else:
            steps.append(
                MoveLStep(f[0], f[1], f[2], f[3], f[4], f[5], f[6],
                          tool_offset=tool_offset, base_offset=base_offset)
            )
    return steps


class RobotProgramAction(BaseAction):
    """! Action that executes an ordered list of MoveJStep / MoveLStep commands.

    Each step is dispatched through the Controller to the named Fairino device.
    _checkpoint() is called after each step to support cooperative pause/cancel.
    A move that returns False raises RuntimeError, ending the action in FAILED.
    On success or cancellation, result() returns the count of completed steps.
    """

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def __init__(
        self, controller: Controller, device_name: str, steps: list
    ) -> None:
        """! Initialise the robot program action.

        @param controller<Controller>: Controller used to dispatch device calls.
        @param device_name<str>: Registered name of the Fairino device.
        @param steps<list>: Ordered list of MoveJStep / MoveLStep instances.
        """
        super().__init__(controller)
        self._device_name = device_name
        self._steps = list(steps)

    # =========================================================================
    # PROTECTED METHODS
    # =========================================================================

    def _run(self) -> int:
        """! Execute all steps in order.

        @return<int>: Number of steps successfully completed before DONE or CANCELLED.
        @raises RuntimeError: If a move step returns False (device-level failure).
        @raises TypeError: If an unrecognised step type is encountered.
        """
        completed = 0
        for step in self._steps:
            if isinstance(step, MoveJStep):
                ok = self._call(
                    self._device_name,
                    "movej",
                    step.j1,
                    step.j2,
                    step.j3,
                    step.j4,
                    step.j5,
                    step.j6,
                    vel=step.vel,
                    tool_offset=step.tool_offset,
                    base_offset=step.base_offset,
                )
            elif isinstance(step, MoveLStep):
                ok = self._call(
                    self._device_name,
                    "movel",
                    step.x,
                    step.y,
                    step.z,
                    step.rx,
                    step.ry,
                    step.rz,
                    vel=step.vel,
                    tool_offset=step.tool_offset,
                    base_offset=step.base_offset,
                )
            else:
                raise TypeError(f"Unknown step type: {type(step)}")
            if not ok:
                raise RuntimeError(
                    f"Step {completed} ({type(step).__name__}) failed"
                    f" on device '{self._device_name}'"
                )
            completed += 1
            if not self._checkpoint():
                return completed
        return completed
