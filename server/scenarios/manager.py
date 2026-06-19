#!/usr/bin/env python3
##
# @file manager.py
#
# @brief ScenarioManager: load, run, pause, resume, and cancel named scenarios.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/19.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import os
import threading

# Internal library
from actions.base import ActionState
from app.controller import Controller
from scenarios.parser import parse_scenario
from scenarios.runner import ScenarioRunner
from scenarios.step import Scenario


class ScenarioManager:
    """! Manages scenario lifecycle: load from YAML, run, pause, resume, cancel.

    Scenarios are loaded from ``<data_folder>/scenarios/`` and stored by name.
    Running a scenario creates a ScenarioRunner that is tracked so that
    pause/resume/cancel can be forwarded to it.

    Multiple scenarios may be loaded and run simultaneously (on different devices).
    Starting a second run of the same scenario replaces the tracked runner.
    """

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def __init__(self, controller: Controller, data_folder: str) -> None:
        """! Initialise the manager.

        @param controller<Controller>: Controller used to dispatch device calls.
        @param data_folder<str>: Root data directory; scenario files live under
            ``<data_folder>/scenarios/``.
        """
        self._controller = controller
        self._data_folder = data_folder
        self._scenarios: dict[str, Scenario] = {}
        self._runners: dict[str, ScenarioRunner] = {}
        self._lock = threading.Lock()

    def load(self, name: str) -> None:
        """! Load a scenario from ``<data_folder>/scenarios/<name>.yaml``.

        @param name<str>: Scenario name (YAML filename without extension).
        @raises FileNotFoundError: If the YAML file does not exist.
        @raises ValueError: If the YAML content is malformed.
        """
        path = os.path.join(self._data_folder, "scenarios", f"{name}.yaml")
        scenario = parse_scenario(path)
        with self._lock:
            self._scenarios[scenario.name] = scenario

    def load_all(self) -> list[str]:
        """! Load all ``.yaml`` files from ``<data_folder>/scenarios/``.

        @return<list[str]>: Names of successfully loaded scenarios, sorted.
        """
        folder = os.path.join(self._data_folder, "scenarios")
        if not os.path.isdir(folder):
            return []
        loaded = []
        for filename in sorted(os.listdir(folder)):
            if filename.endswith(".yaml"):
                self.load(filename[:-5])
                loaded.append(filename[:-5])
        return loaded

    def list_scenarios(self) -> list[str]:
        """! Return all loaded scenario names.

        @return<list[str]>: Sorted list of loaded scenario names.
        """
        with self._lock:
            return sorted(self._scenarios.keys())

    def run(self, name: str) -> ScenarioRunner:
        """! Start a loaded scenario and return its runner.

        If a runner for this name already exists it is replaced by the new one.

        @param name<str>: Name of a previously loaded scenario.
        @return<ScenarioRunner>: Running scenario runner.
        @raises KeyError: If the scenario has not been loaded.
        """
        with self._lock:
            if name not in self._scenarios:
                raise KeyError(f"Scenario '{name}' not loaded")
            scenario = self._scenarios[name]
        runner = ScenarioRunner(self._controller, scenario, self._data_folder)
        with self._lock:
            self._runners[name] = runner
        runner.start()
        return runner

    def pause(self, name: str) -> bool:
        """! Pause the active runner for the named scenario.

        @param name<str>: Scenario name.
        @return<bool>: True if paused, False if no active runner or not running.
        """
        runner = self._get_runner(name)
        return runner.pause() if runner is not None else False

    def resume(self, name: str) -> bool:
        """! Resume the active runner for the named scenario.

        @param name<str>: Scenario name.
        @return<bool>: True if resumed, False if no active runner or not paused.
        """
        runner = self._get_runner(name)
        return runner.resume() if runner is not None else False

    def cancel(self, name: str) -> bool:
        """! Cancel the active runner for the named scenario.

        @param name<str>: Scenario name.
        @return<bool>: True if cancellation was requested, False if no active runner.
        """
        runner = self._get_runner(name)
        return runner.cancel() if runner is not None else False

    def state(self, name: str) -> ActionState | None:
        """! Return the state of the active runner for the named scenario.

        @param name<str>: Scenario name.
        @return<ActionState|None>: Current runner state, or None if no runner exists.
        """
        runner = self._get_runner(name)
        return runner.state() if runner is not None else None

    def get_runner(self, name: str) -> ScenarioRunner | None:
        """! Return the active runner for the named scenario.

        @param name<str>: Scenario name.
        @return<ScenarioRunner|None>: Runner instance, or None if none exists.
        """
        return self._get_runner(name)

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _get_runner(self, name: str) -> ScenarioRunner | None:
        """! Thread-safely retrieve the runner for a scenario.

        @param name<str>: Scenario name.
        @return<ScenarioRunner|None>: Runner instance, or None.
        """
        with self._lock:
            return self._runners.get(name)
