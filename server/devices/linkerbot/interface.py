#!/usr/bin/env python3
##
# @file interface.py
#
# @brief LinkerBot hand device interface.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/18.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Internal library
from devices.linkerbot.LinkerHand.linker_hand_api import LinkerHandApi


class LinkerbotInterface:
    """! LinkerBot robotic hand device interface."""

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def __init__(
        self,
        hand_type: str = "left",
        hand_joint: str = "L10",
        modbus: str = "COM3",
        debug: bool = False,
    ) -> None:
        """! Initialize the LinkerbotInterface.
        @param hand_type<str>: Hand side, "left" or "right".
        @param hand_joint<str>: Joint model, e.g. "L10", "L25".
        @param modbus<str>: RS485 serial port, e.g. "COM3" or "/dev/ttyUSB0".
        @param debug<bool>: Enable debug mode if True.
        """
        self._hand_type = hand_type
        self._hand_joint = hand_joint
        self._modbus = modbus
        self._debug = debug
        self._is_opened = False
        self._hand = None

    def open(self) -> None:
        """! Open the LinkerBot hand connection."""
        if self._is_opened:
            print("[LinkerbotInterface] Open requested but already opened.")
            return
        if not self._debug:
            self._hand = LinkerHandApi(
                hand_type=self._hand_type,
                hand_joint=self._hand_joint,
                modbus=self._modbus,
            )
        self._is_opened = True
        print(f"[LinkerbotInterface] Connection opened (debug={self._debug}).")

    def close(self) -> None:
        """! Close the LinkerBot hand connection."""
        if not self._is_opened:
            print("[LinkerbotInterface] Close requested but already closed.")
            return
        if self._debug:
            self._is_opened = False
            print("[LinkerbotInterface] Connection closed in debug mode.")
            return
        try:
            self._hand = None
        finally:
            self._is_opened = False
        print("[LinkerbotInterface] Connection closed.")

    def is_opened(self) -> bool:
        """! Check if the interface is opened.
        @return<bool>: True if opened, False otherwise.
        """
        return self._is_opened

    def move(self, pose: list[int]) -> bool:
        """! Move the hand joints to the specified pose.
        @param pose<list[int]>: Joint position values (0–255 each).
            Length must match the hand joint model (L10=10, L25=25, etc.).
        @return<bool>: True if the command was sent successfully, False otherwise.
        """
        if not self._is_opened:
            print("[LinkerbotInterface] move() called but interface is not opened.")
            return False
        if self._debug:
            print(f"[LinkerbotInterface] Debug move: {pose}")
            return True
        try:
            self._hand.finger_move(pose)
        except Exception as exception:
            print(f"[LinkerbotInterface] move() failed: {exception}")
            return False
        return True

    def set_speed(self, speed: list[int]) -> bool:
        """! Set the hand joint speeds.
        @param speed<list[int]>: Speed values (0–255 each), minimum 5 elements.
        @return<bool>: True if the command was sent successfully, False otherwise.
        """
        if not self._is_opened:
            print("[LinkerbotInterface] set_speed() called but interface is not opened.")
            return False
        if self._debug:
            print(f"[LinkerbotInterface] Debug set_speed: {speed}")
            return True
        try:
            self._hand.set_speed(speed)
        except Exception as exception:
            print(f"[LinkerbotInterface] set_speed() failed: {exception}")
            return False
        return True

    def set_torque(self, torque: list[int]) -> bool:
        """! Set the hand joint torques.
        @param torque<list[int]>: Torque values (0–255 each), minimum 5 elements.
        @return<bool>: True if the command was sent successfully, False otherwise.
        """
        if not self._is_opened:
            print("[LinkerbotInterface] set_torque() called but interface is not opened.")
            return False
        if self._debug:
            print(f"[LinkerbotInterface] Debug set_torque: {torque}")
            return True
        try:
            self._hand.set_torque(torque)
        except Exception as exception:
            print(f"[LinkerbotInterface] set_torque() failed: {exception}")
            return False
        return True

    def get_state(self) -> list | None:
        """! Get the current joint state.
        @return<list|None>: List of joint position values, or None on failure.
        """
        if not self._is_opened:
            print("[LinkerbotInterface] get_state() called but interface is not opened.")
            return None
        if self._debug:
            return [0] * 10
        try:
            return self._hand.get_state()
        except Exception as exception:
            print(f"[LinkerbotInterface] get_state() failed: {exception}")
            return None

    def get_speed(self) -> list | None:
        """! Get the current joint speeds.
        @return<list|None>: List of speed values, or None on failure.
        """
        if not self._is_opened:
            print("[LinkerbotInterface] get_speed() called but interface is not opened.")
            return None
        if self._debug:
            return [100] * 10
        try:
            return self._hand.get_speed()
        except Exception as exception:
            print(f"[LinkerbotInterface] get_speed() failed: {exception}")
            return None

    def get_torque(self) -> list | None:
        """! Get the current joint torques.
        @return<list|None>: List of torque values, or None on failure.
        """
        if not self._is_opened:
            print("[LinkerbotInterface] get_torque() called but interface is not opened.")
            return None
        if self._debug:
            return [180] * 10
        try:
            return self._hand.get_torque()
        except Exception as exception:
            print(f"[LinkerbotInterface] get_torque() failed: {exception}")
            return None

    def get_touch(self) -> list | None:
        """! Get touch sensor data.
        @return<list|None>: Touch sensor readings, or None on failure.
        """
        if not self._is_opened:
            print("[LinkerbotInterface] get_touch() called but interface is not opened.")
            return None
        if self._debug:
            return [0] * 6
        try:
            return self._hand.get_touch()
        except Exception as exception:
            print(f"[LinkerbotInterface] get_touch() failed: {exception}")
            return None

    def get_force(self) -> list | None:
        """! Get force sensor data (normal, tangential, direction, approach).
        @return<list|None>: Nested list of force readings, or None on failure.
        """
        if not self._is_opened:
            print("[LinkerbotInterface] get_force() called but interface is not opened.")
            return None
        if self._debug:
            return [[0] * 5, [0] * 5, [0] * 5, [0] * 5]
        try:
            return self._hand.get_force()
        except Exception as exception:
            print(f"[LinkerbotInterface] get_force() failed: {exception}")
            return None

    def get_version(self) -> str | None:
        """! Get the embedded firmware version.
        @return<str|None>: Version string, or None on failure.
        """
        if not self._is_opened:
            print("[LinkerbotInterface] get_version() called but interface is not opened.")
            return None
        if self._debug:
            return "debug-version"
        try:
            return self._hand.get_embedded_version()
        except Exception as exception:
            print(f"[LinkerbotInterface] get_version() failed: {exception}")
            return None

    def get_serial_number(self) -> str | None:
        """! Get the device serial number.
        @return<str|None>: Serial number string, or None on failure.
        """
        if not self._is_opened:
            print("[LinkerbotInterface] get_serial_number() called but interface is not opened.")
            return None
        if self._debug:
            return "debug-serial"
        try:
            return self._hand.get_serial_number()
        except Exception as exception:
            print(f"[LinkerbotInterface] get_serial_number() failed: {exception}")
            return None

    def get_temperature(self) -> list | None:
        """! Get the motor temperatures.
        @return<list|None>: List of temperature values, or None on failure.
        """
        if not self._is_opened:
            print("[LinkerbotInterface] get_temperature() called but interface is not opened.")
            return None
        if self._debug:
            return [25] * 10
        try:
            return self._hand.get_temperature()
        except Exception as exception:
            print(f"[LinkerbotInterface] get_temperature() failed: {exception}")
            return None

    def get_fault(self) -> list | None:
        """! Get motor fault codes.
        @return<list|None>: List of fault codes, or None on failure.
        """
        if not self._is_opened:
            print("[LinkerbotInterface] get_fault() called but interface is not opened.")
            return None
        if self._debug:
            return [0] * 10
        try:
            return self._hand.get_fault()
        except Exception as exception:
            print(f"[LinkerbotInterface] get_fault() failed: {exception}")
            return None

    def clear_faults(self) -> list:
        """! Clear all motor fault codes.
        @return<list>: List of result codes, or empty list if not opened.
        """
        if not self._is_opened:
            print("[LinkerbotInterface] clear_faults() called but interface is not opened.")
            return []
        if self._debug:
            return [0] * 5
        try:
            return self._hand.clear_faults()
        except Exception as exception:
            print(f"[LinkerbotInterface] clear_faults() failed: {exception}")
            return []
