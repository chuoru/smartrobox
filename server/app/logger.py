#!/usr/bin/env python3
##
# @file logger.py
#
# @brief Provide logger for the HATS SDK.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2025/02/26.
#
# Copyright (c) 2025 HACHIX.  All rights reserved.

# Standard library
import os
import sys
import logging
from logging.handlers import RotatingFileHandler


class Logger:
    _loggers = {}
    _log_level = logging.INFO
    BLACK = "\033[90m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    CLEAR = "\033[0m"
    PURPLE = "\033[38;5;129m"
    TURQUOISE = "\033[38;5;44m"
    LAVENDER = "\033[38;5;105m"
    BEIGE = "\033[38;5;180m"

    # ==============================================================================
    # PUBLIC METHODS
    # ==============================================================================
    def __init__(self, tag="", color="\033[38;5;180m"):
        """! Logger constructor.
        @param tag<str> Logger tag.
        @param color<str> Logger color.
        """
        self._tag = tag
        self._color = color
        self._logger = logging.getLogger(tag)
        self._logger.setLevel(Logger._log_level)
        self._loggers[tag] = self._logger

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s.%(msecs)d000] [%(thread)d] [%(levelname)s]"
                " %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self._logger.addHandler(handler)

    def info(self, message):
        """! Log info message.
        @param message<str> Log message.
        """
        self._log(logging.INFO, message)

    def debug(self, message):
        """! Log debug message.
        @param message<str> Log message.
        """
        self._log(logging.DEBUG, message)

    def warning(self, message):
        """! Log warning message.
        @param message<str> Log message.
        """
        self._log(logging.WARNING, f"{Logger.YELLOW}{message}{Logger.CLEAR}")

    def error(self, message):
        """! Log error message.
        @param message<str> Log message.
        """
        self._log(logging.ERROR, f"{Logger.RED}{message}{Logger.CLEAR}")

    def fatal(self, message):
        """! Log fatal message.
        @param message<str> Log message.
        """
        self._log(logging.CRITICAL, f"{Logger.RED}{message}{Logger.CLEAR}")

    def trace(self, message):
        """! Log trace message.
        @param message<str> Log message.
        """
        self._log(logging.DEBUG, message)

    @staticmethod
    def set_level(level: str):
        """! Set logging level.
        @param level<str> Logging level.
        """
        levels = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "fatal": logging.CRITICAL,
            "trace": logging.DEBUG,
        }
        if level in levels:
            Logger.set_logging_level(levels[level])
            Logger._log_level = levels[level]
        else:
            raise ValueError(f"Invalid log level: {level}")

    @staticmethod
    def set_logging_level(level: int):
        """! Set logging level.
        @param level<int> Logging level.
        """
        for logger in Logger._loggers.values():
            logger.setLevel(level)

    def set_log_directory(
        self,
        log_directory: str,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ):
        """! Set log directory with rotation.
        @param log_directory<str> Log directory.
        @param max_bytes<int> Maximum size of a log file in bytes before 
        rotation.
        @param backup_count<int> Number of backup files to retain.
        """
        os.makedirs(log_directory, exist_ok=True)
        log_basename = os.path.join(log_directory, f"{self._tag}.log")
        handler = RotatingFileHandler(
            log_basename, maxBytes=max_bytes, backupCount=backup_count
        )
        handler.setFormatter(
            logging.Formatter(
                '{"timestamp": "%(asctime)s.%(msecs)d000", "thread":'
                ' %(thread)d, "level": "%(levelname)s", "message":'
                ' "%(message)s"}',
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self._logger.addHandler(handler)

    # ==============================================================================
    # PRIVATE METHODS
    # ==============================================================================
    def _log(self, level, message):
        """! Log message with color.
        @param level<int> Log level.
        @param message<str> Log message.
        """
        colored_message = f"{self._color}[{self._tag}]{Logger.CLEAR} {message}"
        self._logger.log(level, colored_message)
