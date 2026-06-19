#!/usr/bin/env python3
##
# @file step.py
#
# @brief Data structures for scenario steps and the Scenario container.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/19.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
from dataclasses import dataclass, field


@dataclass
class ActionStep:
    """! A single named action with its parameters."""

    action_type: str
    params: dict = field(default_factory=dict)


@dataclass
class SequenceStep:
    """! An ordered list of steps forming one thread in a parallel group.

    Each SequenceStep in a ParallelStep runs in its own thread.  Steps
    inside a SequenceStep execute sequentially.  SequenceStep can itself
    contain ParallelStep entries for nested parallelism.
    """

    steps: list  # list[ActionStep | ParallelStep]


@dataclass
class ParallelStep:
    """! Multiple SequenceStep threads executing concurrently.

    All threads start at the same time.  The parent waits for every
    thread to reach a terminal state before continuing to the next
    scenario step.
    """

    threads: list  # list[SequenceStep]


@dataclass
class Scenario:
    """! A named, top-level sequential list of steps (some may be parallel groups)."""

    name: str
    steps: list  # list[ActionStep | ParallelStep]
