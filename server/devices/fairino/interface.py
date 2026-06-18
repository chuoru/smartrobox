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

    def jog_joints_with_keyboard(
        self,
        keymap: dict[str, tuple[int, float]],
        vel: float = 20,
        acc: float = 20,
        quit_key: str = "q",
    ) -> int:
        """! Jog multiple joints in one keyboard session.
        @param keymap<dict>: Mapping of key -> (joint_index, delta_deg).
            Example: {"k": (0, +2.0), "j": (0, -2.0), "a": (2, +1.0), "s": (2, -1.0)}
        @param vel<float>: Joint move velocity.
        @param acc<float>: Joint move acceleration.
        @param quit_key<str>: Key to exit jogging mode.
        @return<int>: 0 if normal exit, -1 on validation/communication error.
        """
        if not self._is_opened:
            print("[jog_joints_with_keyboard] Interface is not opened.")
            return -1

        if self._debug:
            print("[jog_joints_with_keyboard] Debug mode: keyboard jogging is disabled.")
            return -1

        if len(quit_key) != 1:
            print("[jog_joints_with_keyboard] quit_key must be a single character.")
            return -1

        if not keymap:
            print("[jog_joints_with_keyboard] keymap cannot be empty.")
            return -1

        for key, action in keymap.items():
            if len(key) != 1:
                print(f"[jog_joints_with_keyboard] Invalid key '{key}': must be single character.")
                return -1
            if key == quit_key:
                print("[jog_joints_with_keyboard] quit_key cannot be used in keymap.")
                return -1
            if not isinstance(action, tuple) or len(action) != 2:
                print(f"[jog_joints_with_keyboard] Invalid mapping for key '{key}': expected (joint_index, delta_deg).")
                return -1

            joint_index, delta_deg = action
            if joint_index < 0 or joint_index > 5:
                print(f"[jog_joints_with_keyboard] Invalid joint index for key '{key}': {joint_index}. Must be 0..5.")
                return -1
            if delta_deg == 0:
                print(f"[jog_joints_with_keyboard] Delta cannot be 0 for key '{key}'.")
                return -1

        joint_state = self.robot.GetActualJointPosDegree(0)
        if not isinstance(joint_state, tuple) or len(joint_state) != 2:
            print(
                "[jog_joints_with_keyboard] GetActualJointPosDegree returned invalid data: "
                f"{joint_state}"
            )
            return -1

        ret, current_joints = joint_state
        if ret != 0:
            print(f"[jog_joints_with_keyboard] GetActualJointPosDegree errcode: {ret}")
            return -1

        exaxis_pos = [0, 0, 0, 0]
        offset_pos = [0, 0, 0, 0, 0, 0]

        print(f"[jog_joints_with_keyboard] Ready. Press '{quit_key}' to quit.")
        for key, (joint_index, delta_deg) in keymap.items():
            sign = "+" if delta_deg > 0 else ""
            joint_label = joint_index + 1
            print(f"  key '{key}' => J{joint_label} (index {joint_index}) {sign}{delta_deg} deg")

        while True:
            key = self._read_single_key()

            if key == quit_key:
                print("[jog_joints_with_keyboard] Quit requested.")
                return 0

            if key not in keymap:
                continue

            joint_index, delta_deg = keymap[key]
            current_joints[joint_index] += float(delta_deg)

            try:
                move_ret = self.robot.MoveJ(
                    joint_pos=current_joints,
                    tool=0,
                    user=0,
                    vel=vel,
                    acc=acc,
                    ovl=100,
                    exaxis_pos=exaxis_pos,
                    blendT=-1,
                    offset_flag=0,
                    offset_pos=offset_pos,
                )
                if move_ret != 0:
                    print(f"[jog_joints_with_keyboard] MoveJ errcode: {move_ret}")
                    continue

                joint_state = self.robot.GetActualJointPosDegree(0)
                if isinstance(joint_state, tuple) and len(joint_state) == 2:
                    ret, actual_joints = joint_state
                    if ret == 0:
                        current_joints = actual_joints

                joint_label = joint_index + 1
                print(
                    f"[jog_joints_with_keyboard] J{joint_label} (index {joint_index}) = "
                    f"{current_joints[joint_index]:.3f} deg"
                )
            except Exception as exception:
                print(f"[jog_joints_with_keyboard] Move failed: {exception}")
                return -1

    def jog_cartesian_with_keyboard(
        self,
        keymap: dict[str, tuple[str, float]],
        vel: float = 20,
        acc: float = 20,
        quit_key: str = "q",
    ) -> int:
        """! Jog in Cartesian coordinates (x,y,z,rx,ry,rz) using keyboard.
        @param keymap<dict>: Mapping of key -> (axis, delta_mm_or_deg).
            Axes: 'x', 'y', 'z' for translation (mm), 'rx', 'ry', 'rz' for rotation (deg)
            Example: {"w": ("z", +10), "s": ("z", -10), "a": ("x", -5), "d": ("x", +5)}
        @param vel<float>: Movement velocity.
        @param acc<float>: Movement acceleration.
        @param quit_key<str>: Key to exit jogging mode.
        @return<int>: 0 if normal exit, -1 on error.
        """
        if not self._is_opened:
            print("[jog_cartesian_with_keyboard] Interface is not opened.")
            return -1

        if self._debug:
            print("[jog_cartesian_with_keyboard] Debug mode: keyboard jogging is disabled.")
            return -1

        if len(quit_key) != 1:
            print("[jog_cartesian_with_keyboard] quit_key must be a single character.")
            return -1

        if not keymap:
            print("[jog_cartesian_with_keyboard] keymap cannot be empty.")
            return -1

        # Validate keymap
        valid_axes = {'x', 'y', 'z', 'rx', 'ry', 'rz'}
        for key, action in keymap.items():
            if len(key) != 1:
                print(f"[jog_cartesian_with_keyboard] Invalid key '{key}': must be single character.")
                return -1
            if key == quit_key:
                print("[jog_cartesian_with_keyboard] quit_key cannot be used in keymap.")
                return -1
            if not isinstance(action, tuple) or len(action) != 2:
                print(f"[jog_cartesian_with_keyboard] Invalid mapping for key '{key}': expected (axis, delta).")
                return -1
            axis, delta = action
            if axis not in valid_axes:
                print(f"[jog_cartesian_with_keyboard] Invalid axis '{axis}': must be one of {valid_axes}")
                return -1
            if delta == 0:
                print(f"[jog_cartesian_with_keyboard] Delta cannot be 0 for key '{key}'")
                return -1

        # Get current TCP position
        error, current_pose = self.tpos()
        if error != 0 or current_pose is None:
            print("[jog_cartesian_with_keyboard] Failed to get current TCP position")
            return -1

        x, y, z, rx, ry, rz = current_pose
        print(f"[jog_cartesian_with_keyboard] Starting position: X={x:.1f}, Y={y:.1f}, Z={z:.1f}, "
              f"RX={rx:.1f}, RY={ry:.1f}, RZ={rz:.1f}")
        print(f"[jog_cartesian_with_keyboard] Ready. Press '{quit_key}' to quit.")

        for key, (axis, delta) in keymap.items():
            sign = "+" if delta > 0 else ""
            unit = "mm" if axis in ['x', 'y', 'z'] else "deg"
            print(f"  key '{key}' => {axis.upper()} {sign}{delta} {unit}")

        while True:
            key = self._read_single_key()

            if key == quit_key:
                print("[jog_cartesian_with_keyboard] Quit requested.")
                return 0

            if key not in keymap:
                continue

            axis, delta = keymap[key]

            # Update the appropriate coordinate
            if axis == 'x':
                x += delta
            elif axis == 'y':
                y += delta
            elif axis == 'z':
                z += delta
            elif axis == 'rx':
                rx += delta
            elif axis == 'ry':
                ry += delta
            elif axis == 'rz':
                rz += delta

            # Convert Cartesian to joint angles and move
            try:
                joints = self.get_inverse_kinematics(x, y, z, rx, ry, rz)
                if joints == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0):
                    print(f"[jog_cartesian_with_keyboard] Inverse kinematics failed for position: "
                          f"({x:.1f}, {y:.1f}, {z:.1f}, {rx:.1f}, {ry:.1f}, {rz:.1f})")
                    continue

                ret = self.robot.MoveJ(
                    joint_pos=list(joints),
                    tool=0,
                    user=0,
                    vel=vel,
                    acc=acc,
                    offset_pos=[0, 0, 0, 0, 0, 0]
                )
                if ret != 0:
                    print(f"[jog_cartesian_with_keyboard] MoveJ failed with error code: {ret}")
                    continue

                print(f"  -> ({x:.1f}, {y:.1f}, {z:.1f}, {rx:.1f}, {ry:.1f}, {rz:.1f})")
            except Exception as e:
                print(f"[jog_cartesian_with_keyboard] Movement error: {e}")
                continue

    def is_opened(self):
        """! Check if the fairino interface is opened.
        @return<bool>: True if opened, False otherwise.
        """
        return self._is_opened

    # =====================================================================
    # PRIVATE METHODS
    # =====================================================================
    def _read_single_key(self) -> str:
        """Read one key press from terminal without requiring Enter."""
        
        try:
            import msvcrt

            return msvcrt.getch().decode("utf-8", errors="ignore")
        except ImportError:
            import tty
            import termios

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                return sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
