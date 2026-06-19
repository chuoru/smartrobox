#!/usr/bin/env python3
##
# @file test_scenario_parser.py
#
# @brief Unit tests for the scenario YAML parser.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/19.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import os
import tempfile
import unittest

# Internal library
from scenarios.parser import parse_scenario
from scenarios.step import ActionStep, ParallelStep, Scenario, SequenceStep


def _write_yaml(directory: str, filename: str, content: str) -> str:
    path = os.path.join(directory, filename)
    with open(path, "w") as fh:
        fh.write(content)
    return path


class TestParseScenarioFileErrors(unittest.TestCase):
    """File-level error conditions."""

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            parse_scenario("/nonexistent/path/scenario.yaml")

    def test_non_mapping_yaml_raises_value_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_yaml(d, "bad.yaml", "- item1\n- item2\n")
            with self.assertRaises(ValueError):
                parse_scenario(path)

    def test_missing_name_key_raises_value_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_yaml(d, "no_name.yaml", "steps: []\n")
            with self.assertRaises(ValueError):
                parse_scenario(path)

    def test_steps_not_list_raises_value_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_yaml(d, "bad_steps.yaml", "name: x\nsteps: not_a_list\n")
            with self.assertRaises(ValueError):
                parse_scenario(path)


class TestParseScenarioBasic(unittest.TestCase):
    """Happy-path sequential step parsing."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def _parse(self, content: str) -> Scenario:
        path = _write_yaml(self._tmpdir.name, "s.yaml", content)
        return parse_scenario(path)

    def test_name_is_parsed(self):
        s = self._parse("name: my_scenario\nsteps: []\n")
        self.assertEqual(s.name, "my_scenario")

    def test_empty_steps_list(self):
        s = self._parse("name: empty\nsteps: []\n")
        self.assertEqual(s.steps, [])

    def test_single_action_step_is_action_step_instance(self):
        content = (
            "name: s\nsteps:\n"
            "  - type: robot_program\n    device: fairino\n    program: x.txt\n"
        )
        s = self._parse(content)
        self.assertIsInstance(s.steps[0], ActionStep)

    def test_single_action_step_action_type(self):
        content = (
            "name: s\nsteps:\n"
            "  - type: grasp\n    device: left_hand\n    grasp_level: 2\n    torque_limit: 100\n"
        )
        self.assertEqual(self._parse(content).steps[0].action_type, "grasp")

    def test_params_exclude_type_key(self):
        content = (
            "name: s\nsteps:\n"
            "  - type: grasp\n    device: left_hand\n    grasp_level: 2\n    torque_limit: 100\n"
        )
        params = self._parse(content).steps[0].params
        self.assertNotIn("type", params)

    def test_params_values_are_correct(self):
        content = (
            "name: s\nsteps:\n"
            "  - type: grasp\n    device: left_hand\n    grasp_level: 3\n    torque_limit: 128\n"
        )
        params = self._parse(content).steps[0].params
        self.assertEqual(params["device"], "left_hand")
        self.assertEqual(params["grasp_level"], 3)
        self.assertEqual(params["torque_limit"], 128)

    def test_multiple_sequential_steps(self):
        content = (
            "name: s\nsteps:\n"
            "  - type: robot_program\n    device: fairino\n    program: a.txt\n"
            "  - type: grasp\n    device: left_hand\n    grasp_level: 1\n    torque_limit: 50\n"
        )
        s = self._parse(content)
        self.assertEqual(len(s.steps), 2)
        self.assertEqual(s.steps[0].action_type, "robot_program")
        self.assertEqual(s.steps[1].action_type, "grasp")

    def test_step_missing_type_raises_value_error(self):
        content = "name: s\nsteps:\n  - device: fairino\n"
        path = _write_yaml(self._tmpdir.name, "no_type.yaml", content)
        with self.assertRaises(ValueError):
            parse_scenario(path)


class TestParseScenarioParallel(unittest.TestCase):
    """Parallel step parsing with SequenceStep threads."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def _parse(self, content: str) -> Scenario:
        path = _write_yaml(self._tmpdir.name, "s.yaml", content)
        return parse_scenario(path)

    _TWO_THREAD_YAML = (
        "name: s\n"
        "steps:\n"
        "  - type: parallel\n"
        "    threads:\n"
        "      - steps:\n"
        "          - type: grasp\n"
        "            device: left_hand\n"
        "            grasp_level: 2\n"
        "            torque_limit: 100\n"
        "      - steps:\n"
        "          - type: robot_program\n"
        "            device: fairino\n"
        "            program: hold.txt\n"
    )

    def test_parallel_is_parallel_step_instance(self):
        self.assertIsInstance(self._parse(self._TWO_THREAD_YAML).steps[0], ParallelStep)

    def test_parallel_has_two_threads(self):
        step = self._parse(self._TWO_THREAD_YAML).steps[0]
        self.assertEqual(len(step.threads), 2)

    def test_threads_are_sequence_step_instances(self):
        step = self._parse(self._TWO_THREAD_YAML).steps[0]
        for thread in step.threads:
            self.assertIsInstance(thread, SequenceStep)

    def test_thread_steps_are_action_steps(self):
        step = self._parse(self._TWO_THREAD_YAML).steps[0]
        for thread in step.threads:
            for sub in thread.steps:
                self.assertIsInstance(sub, ActionStep)

    def test_thread_action_types(self):
        step = self._parse(self._TWO_THREAD_YAML).steps[0]
        types = [thread.steps[0].action_type for thread in step.threads]
        self.assertIn("grasp", types)
        self.assertIn("robot_program", types)

    def test_thread_with_multiple_steps(self):
        content = (
            "name: s\nsteps:\n"
            "  - type: parallel\n"
            "    threads:\n"
            "      - steps:\n"
            "          - type: robot_program\n"
            "            device: fairino\n"
            "            program: a.txt\n"
            "          - type: robot_program\n"
            "            device: fairino\n"
            "            program: b.txt\n"
            "      - steps:\n"
            "          - type: grasp\n"
            "            device: left_hand\n"
            "            grasp_level: 1\n"
            "            torque_limit: 80\n"
        )
        step = self._parse(content).steps[0]
        self.assertEqual(len(step.threads[0].steps), 2)
        self.assertEqual(len(step.threads[1].steps), 1)

    def test_parallel_missing_threads_raises_value_error(self):
        content = "name: s\nsteps:\n  - type: parallel\n"
        path = _write_yaml(self._tmpdir.name, "no_threads.yaml", content)
        with self.assertRaises(ValueError):
            parse_scenario(path)

    def test_parallel_empty_threads_raises_value_error(self):
        content = "name: s\nsteps:\n  - type: parallel\n    threads: []\n"
        path = _write_yaml(self._tmpdir.name, "empty_threads.yaml", content)
        with self.assertRaises(ValueError):
            parse_scenario(path)

    def test_thread_missing_steps_raises_value_error(self):
        content = (
            "name: s\nsteps:\n"
            "  - type: parallel\n"
            "    threads:\n"
            "      - device: fairino\n"
        )
        path = _write_yaml(self._tmpdir.name, "no_steps.yaml", content)
        with self.assertRaises(ValueError):
            parse_scenario(path)

    def test_thread_empty_steps_raises_value_error(self):
        content = (
            "name: s\nsteps:\n"
            "  - type: parallel\n"
            "    threads:\n"
            "      - steps: []\n"
        )
        path = _write_yaml(self._tmpdir.name, "empty_steps.yaml", content)
        with self.assertRaises(ValueError):
            parse_scenario(path)

    def test_missing_type_in_thread_step_raises_value_error(self):
        content = (
            "name: s\nsteps:\n"
            "  - type: parallel\n"
            "    threads:\n"
            "      - steps:\n"
            "          - device: left_hand\n"
        )
        path = _write_yaml(self._tmpdir.name, "bad_sub.yaml", content)
        with self.assertRaises(ValueError):
            parse_scenario(path)

    def test_mixed_sequential_and_parallel(self):
        content = (
            "name: s\nsteps:\n"
            "  - type: robot_program\n    device: fairino\n    program: a.txt\n"
            "  - type: parallel\n"
            "    threads:\n"
            "      - steps:\n"
            "          - type: grasp\n            device: left_hand\n"
            "            grasp_level: 1\n            torque_limit: 80\n"
            "      - steps:\n"
            "          - type: robot_program\n            device: fairino\n"
            "            program: hold.txt\n"
            "  - type: robot_program\n    device: fairino\n    program: b.txt\n"
        )
        s = self._parse(content)
        self.assertEqual(len(s.steps), 3)
        self.assertIsInstance(s.steps[0], ActionStep)
        self.assertIsInstance(s.steps[1], ParallelStep)
        self.assertIsInstance(s.steps[2], ActionStep)


if __name__ == "__main__":
    unittest.main()
