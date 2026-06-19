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
import os
from dataclasses import dataclass

# External library
import yaml

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


@dataclass
class PTPStep:
    """! Joint-space move step resolved from a named teaching point.

    Joint angles are loaded at execution time from
    ``{data_folder}/teaching_point.yaml`` under the key ``teaching_point_name``.
    The teaching point name must not contain spaces.
    """

    teaching_point_name: str
    vel: float = 20.0
    tool_offset: list[float] | None = None
    base_offset: list[float] | None = None


@dataclass
class LinearStep:
    """! Cartesian linear move step resolved from a named teaching point.

    Cartesian pose is loaded at execution time from
    ``{data_folder}/teaching_point.yaml`` under the key ``teaching_point_name``.
    The teaching point name must not contain spaces.
    """

    teaching_point_name: str
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
    """! Serialize a list of move steps to a newline-separated string.

    Each step becomes one line:
    ``movej j1 j2 j3 j4 j5 j6 vel offset_mode ox oy oz orx ory orz`` or
    ``movel x y z rx ry rz vel offset_mode ox oy oz orx ory orz`` or
    ``ptp name vel offset_mode ox oy oz orx ory orz`` or
    ``linear name vel offset_mode ox oy oz orx ory orz``.
    offset_mode: 0=none, 1=base frame, 2=tool frame.

    @param steps<list>: Ordered list of MoveJStep / MoveLStep / PTPStep / LinearStep instances.
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
        elif isinstance(step, PTPStep):
            lines.append(
                f"ptp {step.teaching_point_name} {step.vel} {_offset_fields(step)}"
            )
        elif isinstance(step, LinearStep):
            lines.append(
                f"linear {step.teaching_point_name} {step.vel} {_offset_fields(step)}"
            )
        else:
            raise TypeError(f"Unknown step type: {type(step)}")
    return "\n".join(lines)


def deserialize(text: str) -> list:
    """! Deserialize a newline-separated string to a list of move steps.

    Blank lines are skipped.  Each non-blank line must start with one of
    ``movej``, ``movel``, ``ptp``, or ``linear``.

    ``movej`` / ``movel`` lines have exactly 13 fields after the kind:
    6 motion values, vel, offset_mode (0/1/2), and 6 offset floats (15 parts total).

    ``ptp`` / ``linear`` lines have exactly 8 fields after the kind:
    teaching_point_name, vel, offset_mode (0/1/2), and 6 offset floats (10 parts total).

    @param text<str>: Text produced by serialize() or compatible format.
    @return<list>: Ordered list of MoveJStep / MoveLStep / PTPStep / LinearStep instances.
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
        if kind not in ("movej", "movel", "ptp", "linear"):
            raise ValueError(f"Line {lineno}: unknown step kind {kind!r}")
        if kind in ("movej", "movel"):
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
        else:
            if len(parts) != 10:
                raise ValueError(
                    f"Line {lineno}: {kind} expects 8 fields,"
                    f" got {len(parts) - 1}"
                )
            name = parts[1]
            try:
                f = [float(p) for p in parts[2:]]
            except ValueError:
                raise ValueError(
                    f"Line {lineno}: non-numeric field in {line!r}"
                )
            vel = f[0]
            offset_mode = int(f[1])
            offset_vals = list(f[2:8])
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
            if kind == "ptp":
                steps.append(
                    PTPStep(name, vel, tool_offset=tool_offset, base_offset=base_offset)
                )
            else:
                steps.append(
                    LinearStep(name, vel, tool_offset=tool_offset, base_offset=base_offset)
                )
    return steps


class RobotProgramAction(BaseAction):
    """! Action that loads and executes a named robot program file.

    The program file is read from ``{data_folder}/robot_program/{program_name}``
    at the start of each _run() call, deserialized to move steps, then dispatched
    through the Controller to the named Fairino device.
    _checkpoint() is called after each step to support cooperative pause/cancel.
    A move that returns False raises RuntimeError, ending the action in FAILED.
    On success or cancellation, result() returns the count of completed steps.
    """

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def __init__(
        self,
        controller: Controller,
        program_name: str,
        device_name: str,
        data_folder: str,
    ) -> None:
        """! Initialise the robot program action.

        @param controller<Controller>: Controller used to dispatch device calls.
        @param program_name<str>: Filename inside {data_folder}/robot_program/ to execute.
        @param device_name<str>: Registered name of the Fairino device.
        @param data_folder<str>: Root data directory (value of config["data_folder"]).
        """
        super().__init__(controller)
        self._program_name = program_name
        self._device_name = device_name
        self._data_folder = data_folder

    def parameters(self) -> dict:
        """! Return the robot program's configuration parameters.

        @return<dict>: {"program_name": ..., "device_name": ...}
        """
        return {"program_name": self._program_name, "device_name": self._device_name}

    # =========================================================================
    # PROTECTED METHODS
    # =========================================================================

    def _run(self) -> int:
        """! Load program from file and execute all steps in order.

        @return<int>: Number of steps successfully completed before DONE or CANCELLED.
        @raises FileNotFoundError: If the program file or a teaching point file does not exist.
        @raises ValueError: If the program file contains invalid syntax.
        @raises KeyError: If a teaching point YAML is missing a required section or key.
        @raises RuntimeError: If a move step returns False (device-level failure).
        @raises TypeError: If an unrecognised step type is encountered.
        """
        path = os.path.join(self._data_folder, "robot_program", self._program_name)
        with open(path, "r") as f:
            steps = deserialize(f.read())
        completed = 0
        for step in steps:
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
            elif isinstance(step, PTPStep):
                joints = self._load_teaching_point(step.teaching_point_name, "joint")
                ok = self._call(
                    self._device_name,
                    "movej",
                    joints[0], joints[1], joints[2],
                    joints[3], joints[4], joints[5],
                    vel=step.vel,
                    tool_offset=step.tool_offset,
                    base_offset=step.base_offset,
                )
            elif isinstance(step, LinearStep):
                pose = self._load_teaching_point(step.teaching_point_name, "cartesian")
                ok = self._call(
                    self._device_name,
                    "movel",
                    pose[0], pose[1], pose[2],
                    pose[3], pose[4], pose[5],
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

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _load_teaching_point(self, name: str, section: str) -> list[float]:
        """! Load the requested section for a named teaching point.

        @param name<str>: Teaching point name; looked up as a top-level key in
            ``{data_folder}/teaching_point.yaml``.
        @param section<str>: ``"joint"`` returns ``[j1, j2, j3, j4, j5, j6]``;
            ``"cartesian"`` returns ``[x, y, z, rx, ry, rz]``.
        @return<list[float]>: Six coordinate values in call-argument order.
        @raises FileNotFoundError: If teaching_point.yaml does not exist.
        @raises KeyError: If the point name, section, or any sub-key is absent.
        """
        path = os.path.join(self._data_folder, "teaching_point.yaml")
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict) or name not in data:
            raise KeyError(
                f"Teaching point {name!r} not found in {path!r}"
            )
        point = data[name]
        if not isinstance(point, dict) or section not in point:
            raise KeyError(
                f"Teaching point {name!r}: missing section {section!r}"
            )
        block = point[section]
        keys = ("j1", "j2", "j3", "j4", "j5", "j6") if section == "joint" \
               else ("x", "y", "z", "rx", "ry", "rz")
        missing = [k for k in keys if k not in block]
        if missing:
            raise KeyError(
                f"Teaching point {name!r}: missing key(s) {missing!r} in section {section!r}"
            )
        return [float(block[k]) for k in keys]
