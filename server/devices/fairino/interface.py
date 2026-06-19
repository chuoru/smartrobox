#!/usr/bin/env python3
##
# @file fairino_interface.py
#
# @brief Fairino device interface module
#
# @section author
# - Converted by Tran Viet Thanh (2025/12/10)
# - Modifed by Le Quang Minh (2025/12/10)
#
# Copyright (c) 2025 HACHIX. All rights reserved.

# Standard library
import time
import random
import sys

# Internal library
from devices.fairino import Robot


class FairinoInterface:
    """! Fairino device interface class."""

    # =====================================================================
    # PUBLIC METHODS
    # =====================================================================
    def __init__(self, ip="192.168.58.2", debug: bool = False):
        """! Initialize the TISInterface class.
        @param ip<str>: IP address of the Fairino robot.
        @param debug<bool>: Enable debug mode if True.
        """
        self.ip = ip
        self._is_opened = False
        self._debug = debug
        self.robot = None

    def open(self):
        """! Open Fairino connection."""
        if self._is_opened:
            print("[FairinoInterface] Open requested but already opened.")
            return
        if not self._debug:
            self.robot = Robot.RPC(self.ip)
            self.robot.RobotEnable(1)
        self._is_opened = True
        print(f"[FairinoInterface] Connection opened (debug={self._debug}).")

    def close(self):
        """! Close the fairino device connection."""
        if not self._is_opened:
            print("[FairinoInterface] Close requested but already closed.")
            return

        if self._debug:
            self._is_opened = False
            print("[FairinoInterface] Connection closed in debug mode.")
            return

        try:
            if self.robot is not None:
                self.robot.CloseRPC()
        except Exception as exception:
            print(f"[FairinoInterface] CloseRPC failed: {exception}")
        finally:
            self.robot = None
        self._is_opened = False
        print("[FairinoInterface] Connection closed.")

    def tpos(self):
        """! Get the current tcp position of the robot.
        @return<list>: List of current tcp position [x, y, z, rx, ry, rz].
        @return<int>: Error code.
        """
        if self._debug:
            self.pose = [
                random.uniform(-500, 500),
                random.uniform(-500, 500),
                random.uniform(-500, 500),
                random.uniform(-180, 180),
                random.uniform(-180, 180),
                random.uniform(-180, 180),
            ]
            self.error = 0
            return self.error, self.pose
        try:
            self.error, self.pose = self.robot.GetActualToolFlangePose()
        except Exception as e:
            print(f"[tpos] Failed to get TCP pose: {e}")
            self.error = -1
            self.pose = None
        return self.error, self.pose

    def move(
        self, x: float, y: float, z: float, rx: float, ry: float, rz: float
    ) -> bool:
        """! Move the robot to the specified position.
        @param x<float>: X coordinate.
        @param y<float>: Y coordinate.
        @param z<float>: Z coordinate.
        @param rx<float>: Rotation around X axis.
        @param ry<float>: Rotation around Y axis.
        @param rz<float>: Rotation around Z axis.
        @return<bool>: True if move is successful, False otherwise.
        """
        if self._debug:
            time.sleep(5)
            return False
        x, y, z, rx, ry, rz = (
            float(x),
            float(y),
            float(z),
            float(rx),
            float(ry),
            float(rz),
        )
        try:
            ret, (j1, j2, j3, j4, j5, j6) = self.robot.GetInverseKin(
                0, [x, y, z, rx, ry, rz], -1
            )
            if ret != 0:
                raise Exception(f"Inverse kinematics error: {ret}")
            self.robot.MoveJ(
                [j1, j2, j3, j4, j5, j6],
                0,
                0,
                [x, y, z, rx, ry, rz]
            )
        except Exception as exception:
            print(f"[move] Failed to move robot: {exception}")
            return False
        print(f"[move] Moved to position: {x}, {y}, {z}, {rx}, {ry}, {rz}")
        return True

    def get_inverse_kinematics(
        self, x: float, y: float, z: float, rx: float, ry: float, rz: float
    ) -> tuple[float, float, float, float, float, float]:
        """! Get the inverse kinematics for the specified position.
        @param x<float>: X coordinate.
        @param y<float>: Y coordinate.
        @param z<float>: Z coordinate.
        @param rx<float>: Rotation around X axis.
        @param ry<float>: Rotation around Y axis.
        @param rz<float>: Rotation around Z axis.
        @return<tuple>: Joint angles (j1, j2, j3, j4, j5, j6).
        """
        if self._debug:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        try:
            ret, (j1, j2, j3, j4, j5, j6) = self.robot.GetInverseKin(
                0, [x, y, z, rx, ry, rz], -1
            )
            if ret != 0:
                raise Exception(f"Inverse kinematics error: {ret}")
            return (j1, j2, j3, j4, j5, j6)
        except Exception as exception:
            print(f"[get_inverse_kinematics] Failed to get inverse kinematics: {exception}")
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def movej(
        self,
        j1: float, j2: float, j3: float, j4: float, j5: float, j6: float,
        vel: float = 20.0,
        tool_offset: list[float] | None = None,
        base_offset: list[float] | None = None,
    ) -> bool:
        """! Move the robot to the specified joint angles.
        @param j1<float>: Joint 1 angle in degrees.
        @param j2<float>: Joint 2 angle in degrees.
        @param j3<float>: Joint 3 angle in degrees.
        @param j4<float>: Joint 4 angle in degrees.
        @param j5<float>: Joint 5 angle in degrees.
        @param j6<float>: Joint 6 angle in degrees.
        @param vel<float>: Velocity percentage [0-100]. Default 20.0.
        @param tool_offset<list[float]|None>: Pose offset [x,y,z,rx,ry,rz] in tool frame. Mutually exclusive with base_offset.
        @param base_offset<list[float]|None>: Pose offset [x,y,z,rx,ry,rz] in base frame. Mutually exclusive with tool_offset.
        @return<bool>: True if move is successful, False otherwise.
        """
        if self._debug:
            time.sleep(1)
            return True
        if tool_offset is not None and base_offset is not None:
            raise ValueError("tool_offset and base_offset are mutually exclusive")
        j1, j2, j3, j4, j5, j6, vel = (
            float(j1),
            float(j2),
            float(j3),
            float(j4),
            float(j5),
            float(j6),
            float(vel),
        )
        if tool_offset is not None:
            offset_flag, offset_pos = 2, [float(v) for v in tool_offset]
        elif base_offset is not None:
            offset_flag, offset_pos = 1, [float(v) for v in base_offset]
        else:
            offset_flag, offset_pos = 0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        try:
            ret = self.robot.MoveJ(
                [j1, j2, j3, j4, j5, j6], 0, 0,
                vel=vel, offset_flag=offset_flag, offset_pos=offset_pos,
            )
            if ret != 0:
                raise Exception(f"MoveJ error: {ret}")
        except Exception as exception:
            print(f"[movej] Failed to move robot: {exception}")
            return False
        print(f"[movej] Moved to joints: {j1}, {j2}, {j3}, {j4}, {j5}, {j6}")
        return True

    def movel(
        self,
        x: float, y: float, z: float, rx: float, ry: float, rz: float,
        vel: float = 20.0,
        tool_offset: list[float] | None = None,
        base_offset: list[float] | None = None,
    ) -> bool:
        """! Move the robot linearly to the specified Cartesian pose.
        @param x<float>: X coordinate in mm.
        @param y<float>: Y coordinate in mm.
        @param z<float>: Z coordinate in mm.
        @param rx<float>: Rotation around X axis in degrees.
        @param ry<float>: Rotation around Y axis in degrees.
        @param rz<float>: Rotation around Z axis in degrees.
        @param vel<float>: Velocity percentage [0-100]. Default 20.0.
        @param tool_offset<list[float]|None>: Pose offset [x,y,z,rx,ry,rz] in tool frame. Mutually exclusive with base_offset.
        @param base_offset<list[float]|None>: Pose offset [x,y,z,rx,ry,rz] in base frame. Mutually exclusive with tool_offset.
        @return<bool>: True if move is successful, False otherwise.
        """
        if self._debug:
            time.sleep(1)
            return True
        if tool_offset is not None and base_offset is not None:
            raise ValueError("tool_offset and base_offset are mutually exclusive")
        x, y, z, rx, ry, rz, vel = (
            float(x),
            float(y),
            float(z),
            float(rx),
            float(ry),
            float(rz),
            float(vel),
        )
        if tool_offset is not None:
            offset_flag, offset_pos = 2, [float(v) for v in tool_offset]
        elif base_offset is not None:
            offset_flag, offset_pos = 1, [float(v) for v in base_offset]
        else:
            offset_flag, offset_pos = 0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        try:
            ret = self.robot.MoveL(
                [x, y, z, rx, ry, rz], 0, 0,
                vel=vel, offset_flag=offset_flag, offset_pos=offset_pos,
            )
            if ret != 0:
                raise Exception(f"MoveL error: {ret}")
        except Exception as exception:
            print(f"[movel] Failed to move robot: {exception}")
            return False
        print(f"[movel] Moved to position: {x}, {y}, {z}, {rx}, {ry}, {rz}")
        return True

    def is_opened(self):
        """! Check if the fairino interface is opened.
        @return<bool>: True if opened, False otherwise.
        """
        return self._is_opened