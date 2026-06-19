#!/usr/bin/env python3
##
# @file parser.py
#
# @brief Parse a YAML scenario file into a Scenario data structure.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/19.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import os

# External library
import yaml

# Internal library
from scenarios.step import ActionStep, ParallelStep, Scenario, SequenceStep


def parse_scenario(path: str) -> Scenario:
    """! Load and parse a YAML scenario file.

    Top-level steps are executed sequentially.  A step whose ``type`` is
    ``"parallel"`` groups sub-steps that run concurrently.

    @param path<str>: Absolute or relative path to the scenario YAML file.
    @return<Scenario>: Parsed scenario ready for ScenarioRunner.
    @raises FileNotFoundError: If the file does not exist.
    @raises ValueError: If the YAML content is malformed or missing required keys.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Scenario file not found: {path!r}")
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Scenario file must be a YAML mapping: {path!r}")
    name = data.get("name")
    if not name:
        raise ValueError(f"Scenario file missing 'name' key: {path!r}")
    raw_steps = data.get("steps", [])
    if not isinstance(raw_steps, list):
        raise ValueError(f"'steps' must be a list in: {path!r}")
    return Scenario(name=str(name), steps=[_parse_step(s) for s in raw_steps])


# =========================================================================
# PRIVATE HELPERS
# =========================================================================


def _parse_step(data: dict) -> ActionStep | ParallelStep:
    """! Parse a single step dict from YAML.

    @param data<dict>: Raw step dictionary from YAML.
    @return<ActionStep|ParallelStep>: Parsed step.
    @raises ValueError: If the step dict is malformed or missing 'type'.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Each step must be a YAML mapping, got: {type(data).__name__!r}")
    step_type = data.get("type")
    if not step_type:
        raise ValueError(f"Step missing 'type' key: {data!r}")
    if step_type == "parallel":
        raw_threads = data.get("threads")
        if not raw_threads:
            raise ValueError(
                f"'parallel' step must have a non-empty 'threads' list: {data!r}"
            )
        if not isinstance(raw_threads, list):
            raise ValueError(f"'threads' must be a list in parallel step: {data!r}")
        threads = []
        for raw_thread in raw_threads:
            if not isinstance(raw_thread, dict):
                raise ValueError(
                    f"Each thread must be a mapping with a 'steps' key: {raw_thread!r}"
                )
            raw_steps = raw_thread.get("steps")
            if not raw_steps:
                raise ValueError(
                    f"Each thread must have a non-empty 'steps' list: {raw_thread!r}"
                )
            if not isinstance(raw_steps, list):
                raise ValueError(f"Thread 'steps' must be a list: {raw_thread!r}")
            threads.append(SequenceStep(steps=[_parse_step(s) for s in raw_steps]))
        return ParallelStep(threads=threads)
    params = {k: v for k, v in data.items() if k != "type"}
    return ActionStep(action_type=str(step_type), params=params)
