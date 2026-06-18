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

    def is_opened(self):
        """! Check if the fairino interface is opened.
        @return<bool>: True if opened, False otherwise.
        """
        return self._is_opened