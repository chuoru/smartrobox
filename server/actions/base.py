#!/usr/bin/env python3
##
# @file base.py
#
# @brief Base class for long-running actions that orchestrate multiple devices.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/19.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import threading
from enum import Enum

# Internal library
from app.controller import Controller
from app.logger import Logger


class ActionState(Enum):
    """! Lifecycle states for a BaseAction."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BaseAction:
    """! Abstract base class for long-running, multi-device actions.

    Subclasses implement _run() to define the action body and call
    _checkpoint() between logical steps to support cooperative
    pause/resume/cancel.  Actions are composable via start() + wait():

        # Consecutive
        a1.start(); a1.wait(); a2.start(); a2.wait()

        # Parallel
        a1.start(); a2.start(); a1.wait(); a2.wait()
    """

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def __init__(self, controller: Controller) -> None:
        """! Initialise the action with a device controller.

        @param controller<Controller>: Controller used to dispatch device calls.
        """
        self._controller = controller
        self._state = ActionState.IDLE
        self._lock = threading.Lock()
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._done_event = threading.Event()
        self._cancelled: bool = False
        self._thread: threading.Thread | None = None
        self._result = None
        self._error: Exception | None = None
        self._logger = Logger(type(self).__name__, Logger.GREEN)

    def start(self) -> bool:
        """! Start the action in a background daemon thread.

        @return<bool>: True if started, False if already started or completed.
        """
        with self._lock:
            if self._state != ActionState.IDLE:
                self._logger.warning(
                    f"start() ignored — state is {self._state.value}"
                )
                return False
            self._state = ActionState.RUNNING
        self._done_event.clear()
        self._thread = threading.Thread(
            target=self._thread_entry,
            daemon=True,
            name=type(self).__name__,
        )
        self._thread.start()
        return True

    def pause(self) -> bool:
        """! Request the action to pause at the next checkpoint.

        @return<bool>: True if pause was requested, False if not running.
        """
        with self._lock:
            if self._state != ActionState.RUNNING:
                return False
            self._state = ActionState.PAUSED
        self._pause_event.clear()
        return True

    def resume(self) -> bool:
        """! Resume a paused action.

        @return<bool>: True if resumed, False if not paused.
        """
        with self._lock:
            if self._state != ActionState.PAUSED:
                return False
            self._state = ActionState.RUNNING
        self._pause_event.set()
        return True

    def cancel(self) -> bool:
        """! Request cancellation; unblocks any active checkpoint.

        Cancellation is cooperative: _run() must check _checkpoint() and
        return when it returns False.  In-progress device calls complete
        before the action exits.

        @return<bool>: True if cancellation was requested, False if already
            in a terminal state.
        """
        with self._lock:
            if self._state in (
                ActionState.DONE,
                ActionState.FAILED,
                ActionState.CANCELLED,
            ):
                return False
            self._cancelled = True
        self._pause_event.set()
        return True

    def wait(self, timeout: float | None = None) -> bool:
        """! Block until the action reaches a terminal state or timeout expires.

        @param timeout<float|None>: Seconds to wait; None waits indefinitely.
        @return<bool>: True if the action finished with DONE, False otherwise.
        """
        self._done_event.wait(timeout=timeout)
        return self.state() == ActionState.DONE

    def state(self) -> ActionState:
        """! Return the current action state.

        @return<ActionState>: Current state.
        """
        with self._lock:
            return self._state

    def result(self):
        """! Return the value produced by _run() on success.

        Safe to call after wait() returns True.

        @return: Action result, or None if not complete or failed.
        """
        return self._result

    def error(self) -> Exception | None:
        """! Return the exception that caused a FAILED state.

        Safe to call after wait() returns.

        @return<Exception|None>: Captured exception, or None.
        """
        return self._error

    def reset(self) -> bool:
        """! Reset the action to IDLE so it can be started again.

        Safe to call on an already-IDLE action (no-op).  Must not be called
        while the action is RUNNING or PAUSED; wait() first.

        @return<bool>: True if the action is now IDLE, False if it is still
            running or paused.
        """
        with self._lock:
            if self._state in (ActionState.RUNNING, ActionState.PAUSED):
                return False
            if self._state == ActionState.IDLE:
                return True
            self._state = ActionState.IDLE
            self._result = None
            self._error = None
            self._cancelled = False
            self._thread = None
        self._pause_event.set()
        self._done_event.clear()
        return True

    def parameters(self) -> dict:
        """! Return the action's configuration parameters.

        Override in subclasses to expose the parameters the action was
        created with (e.g. device name, program name).

        @return<dict>: Configuration parameters dict (empty by default).
        """
        return {}

    # =========================================================================
    # PROTECTED METHODS
    # =========================================================================

    def _run(self):
        """! Override in subclass to implement the action body.

        Call _checkpoint() after each logical step.  Return immediately
        when _checkpoint() returns False (cancelled).
        """
        raise NotImplementedError

    def _checkpoint(self) -> bool:
        """! Yield a pause/cancel opportunity between action steps.

        Blocks while the action is paused.  Returns False immediately
        when the action has been cancelled.

        @return<bool>: True to continue, False if the action is cancelled.
        """
        self._pause_event.wait()
        return not self._cancelled

    def _call(self, device_name: str, method: str, *args, **kwargs):
        """! Dispatch a device method through the controller.

        @param device_name<str>: Registered device name.
        @param method<str>: Public method name on the device interface.
        @return: Whatever the device method returns.
        """
        return self._controller.execute(device_name, method, *args, **kwargs)

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _thread_entry(self) -> None:
        """! Thread target: run _run(), then set the terminal state."""
        try:
            self._result = self._run()
            with self._lock:
                self._state = (
                    ActionState.CANCELLED if self._cancelled else ActionState.DONE
                )
        except Exception as exception:
            self._error = exception
            self._logger.error(f"action failed: {exception}")
            with self._lock:
                self._state = ActionState.FAILED
        finally:
            self._done_event.set()
