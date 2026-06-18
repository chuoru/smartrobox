#!/usr/bin/env python3
##
# @file config.py
#
# @brief Provide the configuration management for the measurement system.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2025/09/07.

# Standard libraries
import yaml


class Config:
    """! Configuration management for the measurement system."""

    def __init__(self, config_path: str):
        """! Initialize the configuration by reading from a YAML file.
        @param config_path: Path to the YAML configuration file.
        """
        self._config_path = config_path
        self._config = self._read_yaml_config()

    def get(self, key: str = None):
        """! Get a configuration parameter by key.
        @param key: The configuration key.
        @return: The configuration value or default.
        """
        if key is None:
            return self._config
        
        return self._config.get(key)

    def set(self, key: str, value):
        """! Set a configuration parameter and write to the YAML file.
        @param key: The configuration key.
        @param value: The configuration value.
        """
        self._config[key] = value
        self._write_yaml_config()

    def get_all(self) -> dict:
        """! Get all configuration parameters.
        @return: A dictionary of all configuration parameters.
        """
        return self._config
    
    def set_all(self, config: dict):
        """! Set all configuration parameters and write to the YAML file.
        @param config: A dictionary of configuration parameters.
        """
        self._config = config
        self._write_yaml_config()

    def _write_yaml_config(self):
        """! Write the current configuration parameters to a YAML file.
        """
        with open(self._config_path, "w") as file:
            yaml.safe_dump(self._config, file)

    def _read_yaml_config(self) -> dict:
        """! Read configuration parameters from a YAML file."""
        with open(self._config_path, "r") as file:
            config = yaml.safe_load(file)
            return config
