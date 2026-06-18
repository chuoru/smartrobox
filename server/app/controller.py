#!/usr/bin/env python3
##
# @file controller.py
#
# @brief Manage device instances, lifecycle, status, and method dispatch.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/19.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import threading
from dataclasses import dataclass

# Internal library
from app.config import Config
from app.logger import Logger
from devices.fairino.interface import FairinoInterface
from devices.linkerbot.interface import LinkerbotInterface
from devices.orbbec_interface import OrbbecInterface


_DEVICE_FACTORIES: dict[str, type] = {
    "fairino": FairinoInterface,
    "linkerbot": LinkerbotInterface,
    "orbbec": OrbbecInterface,
}


@dataclass
class _DeviceEntry:
    name: str
    type: str
    instance: object


class Controller:
    """! Manages device lifecycle and method dispatch for all configured devices."""

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def __init__(self, config: Config) -> None:
        """! Initialize the controller and build the device registry from config.
        @param config<Config>: Config instance backed by device.yaml.
        """
        self._lock = threading.Lock()
        self._logger = Logger("Controller", Logger.CYAN)
        self._registry = self._build_registry(config.get("devices") or {})
        self._logger.info(f"Registered {len(self._registry)} device(s): {list(self._registry.keys())}")

    def open(self, device_name: str) -> bool:
        """! Open (or start) a single device by name.
        @param device_name<str>: Registered device name.
        @return<bool>: True if opened successfully, False otherwise.
        """
        with self._lock:
            entry = self._registry.get(device_name)
        if entry is None:
            self._logger.error(f"Device '{device_name}' not found.")
            return False
        return self._lifecycle_open(entry)

    def close(self, device_name: str) -> bool:
        """! Close (or stop) a single device by name.
        @param device_name<str>: Registered device name.
        @return<bool>: True if closed successfully, False otherwise.
        """
        with self._lock:
            entry = self._registry.get(device_name)
        if entry is None:
            self._logger.error(f"Device '{device_name}' not found.")
            return False
        return self._lifecycle_close(entry)

    def open_all(self) -> dict[str, bool]:
        """! Open all registered devices.
        @return<dict[str, bool]>: Map of device name to open success.
        """
        return {name: self.open(name) for name in self.list_devices()}

    def close_all(self) -> dict[str, bool]:
        """! Close all registered devices.
        @return<dict[str, bool]>: Map of device name to close success.
        """
        return {name: self.close(name) for name in self.list_devices()}

    def status(self, device_name: str) -> dict | None:
        """! Return the status of a single device.
        @param device_name<str>: Registered device name.
        @return<dict|None>: {name, type, is_opened} or None if device not found.
        """
        with self._lock:
            entry = self._registry.get(device_name)
        if entry is None:
            return None
        return {
            "name": entry.name,
            "type": entry.type,
            "is_opened": self._probe_status(entry),
        }

    def list_devices(self) -> list[str]:
        """! Return a snapshot list of all registered device names.
        @return<list[str]>: Registered device names.
        """
        with self._lock:
            return list(self._registry.keys())

    def execute(self, device_name: str, method: str, *args, **kwargs) -> object:
        """! Call a public method on a registered device.
        @param device_name<str>: Registered device name.
        @param method<str>: Public method name to invoke.
        @return<object>: Return value of the method.
        @raises KeyError: If device_name is not registered.
        @raises AttributeError: If method is private, missing, or not callable.
        """
        with self._lock:
            entry = self._registry.get(device_name)
        if entry is None:
            raise KeyError(f"Device '{device_name}' not found.")
        if method.startswith("_"):
            raise AttributeError(f"Method '{method}' is private and cannot be executed.")
        fn = getattr(entry.instance, method, None)
        if fn is None or not callable(fn):
            raise AttributeError(f"Device '{device_name}' has no public method '{method}'.")
        try:
            return fn(*args, **kwargs)
        except Exception as exception:
            self._logger.error(f"[{device_name}.{method}] {exception}")
            raise

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _build_registry(self, devices_cfg: dict) -> dict[str, "_DeviceEntry"]:
        """! Instantiate device interfaces from config and return the registry.
        @param devices_cfg<dict>: Devices section from device.yaml.
        @return<dict[str, _DeviceEntry]>: Registry keyed by device name.
        """
        registry = {}
        for name, cfg in devices_cfg.items():
            device_type = cfg.get("type")
            if device_type is None:
                self._logger.warning(f"Device '{name}' has no 'type' — skipping.")
                continue
            factory = _DEVICE_FACTORIES.get(device_type)
            if factory is None:
                self._logger.warning(f"Unknown device type '{device_type}' for '{name}' — skipping.")
                continue
            params = cfg.get("params", {})
            try:
                instance = factory(**params)
            except Exception as exception:
                self._logger.error(f"Failed to instantiate '{name}' ({device_type}): {exception}")
                continue
            registry[name] = _DeviceEntry(name=name, type=device_type, instance=instance)
            self._logger.info(f"Registered device '{name}' (type={device_type}).")
        return registry

    def _lifecycle_open(self, entry: "_DeviceEntry") -> bool:
        """! Call start() for orbbec or open() for all other device types.
        @param entry<_DeviceEntry>: Device registry entry.
        @return<bool>: True on success, False on exception.
        """
        try:
            if entry.type == "orbbec":
                entry.instance.start()
            else:
                entry.instance.open()
            self._logger.info(f"Device '{entry.name}' opened.")
            return True
        except Exception as exception:
            self._logger.error(f"Failed to open '{entry.name}': {exception}")
            return False

    def _lifecycle_close(self, entry: "_DeviceEntry") -> bool:
        """! Call stop() for orbbec or close() for all other device types.
        @param entry<_DeviceEntry>: Device registry entry.
        @return<bool>: True on success, False on exception.
        """
        try:
            if entry.type == "orbbec":
                entry.instance.stop()
            else:
                entry.instance.close()
            self._logger.info(f"Device '{entry.name}' closed.")
            return True
        except Exception as exception:
            self._logger.error(f"Failed to close '{entry.name}': {exception}")
            return False

    def _probe_status(self, entry: "_DeviceEntry") -> bool:
        """! Return True if the device is open/alive.
        @param entry<_DeviceEntry>: Device registry entry.
        @return<bool>: Connection/alive state.
        """
        try:
            if entry.type == "orbbec":
                return entry.instance.is_alive()
            return entry.instance.is_opened()
        except Exception:
            return False
