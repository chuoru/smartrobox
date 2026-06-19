#!/usr/bin/env python3
##
# @file test_robot_program_action.py
#
# @brief Unit tests for RobotProgramAction, MoveJStep, MoveLStep,
#        serialize, and deserialize.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/19.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import threading
import unittest
from unittest.mock import MagicMock, call, patch

# Internal library
from actions.base import ActionState
from actions.robot_program import (
    MoveLStep,
    MoveJStep,
    RobotProgramAction,
    deserialize,
    serialize,
)


# ---------------------------------------------------------------------------
# Shared setUp / tearDown mixin
# ---------------------------------------------------------------------------

class _ActionTestMixin:
    def setUp(self):
        self._logger_patcher = patch("actions.base.Logger")
        self._logger_patcher.start()
        self._controller = MagicMock()
        self._controller.execute.return_value = True

    def tearDown(self):
        self._logger_patcher.stop()

    def _make(self, steps=None, device_name="left_arm"):
        return RobotProgramAction(
            self._controller, device_name, steps if steps is not None else []
        )


# ---------------------------------------------------------------------------
# Synchronisation helper for checkpoint tests
# ---------------------------------------------------------------------------

def _blocking_side_effect(n):
    """Return (side_effect_fn, at_step_events, can_proceed_events) for n steps."""
    at_step = [threading.Event() for _ in range(n)]
    can_proceed = [threading.Event() for _ in range(n)]
    idx = [0]

    def side_effect(*args, **kwargs):
        i = idx[0]
        idx[0] += 1
        at_step[i].set()
        can_proceed[i].wait()
        return True

    return side_effect, at_step, can_proceed


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestStepDataclasses(unittest.TestCase):
    """Tests for MoveJStep and MoveLStep dataclasses."""

    def test_movej_step_stores_all_fields(self):
        s = MoveJStep(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        self.assertEqual((s.j1, s.j2, s.j3, s.j4, s.j5, s.j6), (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

    def test_movej_step_default_vel_is_20(self):
        self.assertEqual(MoveJStep(0, 0, 0, 0, 0, 0).vel, 20.0)

    def test_movej_step_custom_vel(self):
        self.assertEqual(MoveJStep(0, 0, 0, 0, 0, 0, vel=50.0).vel, 50.0)

    def test_movej_step_default_tool_offset_is_none(self):
        self.assertIsNone(MoveJStep(0, 0, 0, 0, 0, 0).tool_offset)

    def test_movej_step_default_base_offset_is_none(self):
        self.assertIsNone(MoveJStep(0, 0, 0, 0, 0, 0).base_offset)

    def test_movej_step_custom_tool_offset(self):
        self.assertEqual(MoveJStep(0, 0, 0, 0, 0, 0, tool_offset=[1, 2, 3, 4, 5, 6]).tool_offset, [1, 2, 3, 4, 5, 6])

    def test_movej_step_custom_base_offset(self):
        self.assertEqual(MoveJStep(0, 0, 0, 0, 0, 0, base_offset=[7, 8, 9, 0, 0, 0]).base_offset, [7, 8, 9, 0, 0, 0])

    def test_movel_step_stores_all_fields(self):
        s = MoveLStep(10.0, 20.0, 30.0, 1.0, 2.0, 3.0)
        self.assertEqual((s.x, s.y, s.z, s.rx, s.ry, s.rz), (10.0, 20.0, 30.0, 1.0, 2.0, 3.0))

    def test_movel_step_default_vel_is_20(self):
        self.assertEqual(MoveLStep(0, 0, 0, 0, 0, 0).vel, 20.0)

    def test_movel_step_custom_vel(self):
        self.assertEqual(MoveLStep(0, 0, 0, 0, 0, 0, vel=80.0).vel, 80.0)

    def test_movel_step_default_tool_offset_is_none(self):
        self.assertIsNone(MoveLStep(0, 0, 0, 0, 0, 0).tool_offset)

    def test_movel_step_default_base_offset_is_none(self):
        self.assertIsNone(MoveLStep(0, 0, 0, 0, 0, 0).base_offset)

    def test_movel_step_custom_tool_offset(self):
        self.assertEqual(MoveLStep(0, 0, 0, 0, 0, 0, tool_offset=[1, 2, 3, 4, 5, 6]).tool_offset, [1, 2, 3, 4, 5, 6])

    def test_movel_step_custom_base_offset(self):
        self.assertEqual(MoveLStep(0, 0, 0, 0, 0, 0, base_offset=[7, 8, 9, 0, 0, 0]).base_offset, [7, 8, 9, 0, 0, 0])


class TestRobotProgramActionInit(_ActionTestMixin, unittest.TestCase):
    """Tests for RobotProgramAction construction."""

    def test_initial_state_is_idle(self):
        self.assertEqual(self._make().state(), ActionState.IDLE)

    def test_empty_steps_list_accepted(self):
        action = self._make(steps=[])
        action.start()
        self.assertTrue(action.wait(timeout=2.0))

    def test_steps_list_is_copied_defensively(self):
        steps = [MoveJStep(0, 0, 0, 0, 0, 0)]
        action = self._make(steps=steps)
        steps.append(MoveLStep(1, 2, 3, 4, 5, 6))
        action.start()
        action.wait(timeout=2.0)
        self.assertEqual(self._controller.execute.call_count, 1)


class TestRobotProgramActionRun(_ActionTestMixin, unittest.TestCase):
    """Tests for step dispatch, ordering, result, and failure handling."""

    def test_empty_program_returns_zero(self):
        action = self._make(steps=[])
        action.start()
        action.wait(timeout=2.0)
        self.assertEqual(action.state(), ActionState.DONE)
        self.assertEqual(action.result(), 0)

    def test_single_movej_step_calls_correct_method(self):
        step = MoveJStep(10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
        action = self._make(steps=[step])
        action.start()
        action.wait(timeout=2.0)
        self._controller.execute.assert_called_once_with(
            "left_arm", "movej", 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
            vel=20.0, tool_offset=None, base_offset=None,
        )

    def test_single_movel_step_calls_correct_method(self):
        step = MoveLStep(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        action = self._make(steps=[step])
        action.start()
        action.wait(timeout=2.0)
        self._controller.execute.assert_called_once_with(
            "left_arm", "movel", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            vel=20.0, tool_offset=None, base_offset=None,
        )

    def test_movej_step_passes_custom_vel_as_keyword(self):
        step = MoveJStep(0, 0, 0, 0, 0, 0, vel=75.0)
        action = self._make(steps=[step])
        action.start()
        action.wait(timeout=2.0)
        _, kwargs = self._controller.execute.call_args
        self.assertEqual(kwargs.get("vel"), 75.0)

    def test_movel_step_passes_custom_vel_as_keyword(self):
        step = MoveLStep(0, 0, 0, 0, 0, 0, vel=40.0)
        action = self._make(steps=[step])
        action.start()
        action.wait(timeout=2.0)
        _, kwargs = self._controller.execute.call_args
        self.assertEqual(kwargs.get("vel"), 40.0)

    def test_movej_step_passes_none_offsets_by_default(self):
        step = MoveJStep(0, 0, 0, 0, 0, 0)
        action = self._make(steps=[step])
        action.start()
        action.wait(timeout=2.0)
        _, kwargs = self._controller.execute.call_args
        self.assertIsNone(kwargs.get("tool_offset"))
        self.assertIsNone(kwargs.get("base_offset"))

    def test_movel_step_passes_none_offsets_by_default(self):
        step = MoveLStep(0, 0, 0, 0, 0, 0)
        action = self._make(steps=[step])
        action.start()
        action.wait(timeout=2.0)
        _, kwargs = self._controller.execute.call_args
        self.assertIsNone(kwargs.get("tool_offset"))
        self.assertIsNone(kwargs.get("base_offset"))

    def test_movej_step_passes_tool_offset_as_keyword(self):
        offset = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        step = MoveJStep(0, 0, 0, 0, 0, 0, tool_offset=offset)
        action = self._make(steps=[step])
        action.start()
        action.wait(timeout=2.0)
        _, kwargs = self._controller.execute.call_args
        self.assertEqual(kwargs.get("tool_offset"), offset)
        self.assertIsNone(kwargs.get("base_offset"))

    def test_movej_step_passes_base_offset_as_keyword(self):
        offset = [7.0, 8.0, 9.0, 0.0, 0.0, 0.0]
        step = MoveJStep(0, 0, 0, 0, 0, 0, base_offset=offset)
        action = self._make(steps=[step])
        action.start()
        action.wait(timeout=2.0)
        _, kwargs = self._controller.execute.call_args
        self.assertIsNone(kwargs.get("tool_offset"))
        self.assertEqual(kwargs.get("base_offset"), offset)

    def test_movel_step_passes_tool_offset_as_keyword(self):
        offset = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        step = MoveLStep(0, 0, 0, 0, 0, 0, tool_offset=offset)
        action = self._make(steps=[step])
        action.start()
        action.wait(timeout=2.0)
        _, kwargs = self._controller.execute.call_args
        self.assertEqual(kwargs.get("tool_offset"), offset)

    def test_movel_step_passes_base_offset_as_keyword(self):
        offset = [0.0, 0.0, 5.0, 0.0, 0.0, 0.0]
        step = MoveLStep(0, 0, 0, 0, 0, 0, base_offset=offset)
        action = self._make(steps=[step])
        action.start()
        action.wait(timeout=2.0)
        _, kwargs = self._controller.execute.call_args
        self.assertEqual(kwargs.get("base_offset"), offset)

    def test_result_equals_completed_step_count(self):
        steps = [
            MoveJStep(0, 0, 0, 0, 0, 0),
            MoveLStep(1, 2, 3, 4, 5, 6),
            MoveJStep(7, 8, 9, 10, 11, 12),
        ]
        action = self._make(steps=steps)
        action.start()
        action.wait(timeout=2.0)
        self.assertEqual(action.result(), 3)

    def test_multiple_steps_dispatched_in_order(self):
        steps = [
            MoveJStep(1, 2, 3, 4, 5, 6),
            MoveLStep(10, 20, 30, 1, 2, 3),
        ]
        action = self._make(steps=steps)
        action.start()
        action.wait(timeout=2.0)
        self._controller.execute.assert_has_calls([
            call("left_arm", "movej", 1, 2, 3, 4, 5, 6, vel=20.0, tool_offset=None, base_offset=None),
            call("left_arm", "movel", 10, 20, 30, 1, 2, 3, vel=20.0, tool_offset=None, base_offset=None),
        ])

    def test_movej_failure_sets_failed_state(self):
        self._controller.execute.return_value = False
        action = self._make(steps=[MoveJStep(0, 0, 0, 0, 0, 0)])
        action.start()
        action.wait(timeout=2.0)
        self.assertEqual(action.state(), ActionState.FAILED)

    def test_movel_failure_sets_failed_state(self):
        self._controller.execute.return_value = False
        action = self._make(steps=[MoveLStep(0, 0, 0, 0, 0, 0)])
        action.start()
        action.wait(timeout=2.0)
        self.assertEqual(action.state(), ActionState.FAILED)

    def test_error_is_runtime_error_on_move_failure(self):
        self._controller.execute.return_value = False
        action = self._make(steps=[MoveJStep(0, 0, 0, 0, 0, 0)])
        action.start()
        action.wait(timeout=2.0)
        self.assertIsInstance(action.error(), RuntimeError)

    def test_failure_on_second_step_executes_exactly_two_calls(self):
        self._controller.execute.side_effect = [True, False]
        steps = [MoveJStep(0, 0, 0, 0, 0, 0), MoveLStep(1, 2, 3, 4, 5, 6)]
        action = self._make(steps=steps)
        action.start()
        action.wait(timeout=2.0)
        self.assertEqual(self._controller.execute.call_count, 2)
        self.assertEqual(action.state(), ActionState.FAILED)

    def test_unknown_step_type_sets_failed_state(self):
        action = self._make(steps=[{"type": "invalid"}])
        action.start()
        action.wait(timeout=2.0)
        self.assertEqual(action.state(), ActionState.FAILED)

    def test_unknown_step_type_error_is_type_error(self):
        action = self._make(steps=[{"type": "invalid"}])
        action.start()
        action.wait(timeout=2.0)
        self.assertIsInstance(action.error(), TypeError)


class TestRobotProgramActionCheckpoint(_ActionTestMixin, unittest.TestCase):
    """Tests for pause/cancel interaction via _checkpoint()."""

    def test_cancel_between_steps_stops_further_execution(self):
        side_effect, at_step, can_proceed = _blocking_side_effect(2)
        self._controller.execute.side_effect = side_effect
        steps = [MoveJStep(0, 0, 0, 0, 0, 0), MoveLStep(1, 2, 3, 4, 5, 6)]
        action = self._make(steps=steps)
        action.start()
        at_step[0].wait()
        action.cancel()
        can_proceed[0].set()
        action.wait(timeout=2.0)
        self.assertEqual(action.state(), ActionState.CANCELLED)
        self.assertEqual(self._controller.execute.call_count, 1)

    def test_cancel_result_is_completed_count_at_cancellation(self):
        side_effect, at_step, can_proceed = _blocking_side_effect(3)
        self._controller.execute.side_effect = side_effect
        steps = [
            MoveJStep(0, 0, 0, 0, 0, 0),
            MoveLStep(1, 2, 3, 4, 5, 6),
            MoveJStep(7, 8, 9, 10, 11, 12),
        ]
        action = self._make(steps=steps)
        action.start()
        at_step[0].wait()
        can_proceed[0].set()
        at_step[1].wait()
        action.cancel()
        can_proceed[1].set()
        action.wait(timeout=2.0)
        self.assertEqual(action.result(), 2)

    def test_pause_blocks_second_step_until_resume(self):
        side_effect, at_step, can_proceed = _blocking_side_effect(2)
        self._controller.execute.side_effect = side_effect
        steps = [MoveJStep(0, 0, 0, 0, 0, 0), MoveLStep(1, 2, 3, 4, 5, 6)]
        action = self._make(steps=steps)
        action.start()
        at_step[0].wait()
        can_proceed[0].set()
        action.pause()
        self.assertFalse(at_step[1].wait(timeout=0.1))
        action.resume()
        at_step[1].wait()
        can_proceed[1].set()
        self.assertTrue(action.wait(timeout=2.0))

    def test_pause_resume_all_steps_complete(self):
        side_effect, at_step, can_proceed = _blocking_side_effect(3)
        self._controller.execute.side_effect = side_effect
        steps = [
            MoveJStep(0, 0, 0, 0, 0, 0),
            MoveLStep(1, 2, 3, 4, 5, 6),
            MoveJStep(7, 8, 9, 10, 11, 12),
        ]
        action = self._make(steps=steps)
        action.start()
        at_step[0].wait()
        can_proceed[0].set()
        action.pause()
        action.resume()
        at_step[1].wait()
        can_proceed[1].set()
        at_step[2].wait()
        can_proceed[2].set()
        action.wait(timeout=2.0)
        self.assertEqual(action.state(), ActionState.DONE)
        self.assertEqual(action.result(), 3)

    def test_cancel_while_paused_stops_action(self):
        side_effect, at_step, can_proceed = _blocking_side_effect(2)
        self._controller.execute.side_effect = side_effect
        steps = [MoveJStep(0, 0, 0, 0, 0, 0), MoveLStep(1, 2, 3, 4, 5, 6)]
        action = self._make(steps=steps)
        action.start()
        at_step[0].wait()
        can_proceed[0].set()
        action.pause()
        action.cancel()
        action.wait(timeout=2.0)
        self.assertEqual(action.state(), ActionState.CANCELLED)
        self.assertEqual(self._controller.execute.call_count, 1)

    def test_cancel_immediately_after_start_produces_cancelled_or_done(self):
        action = self._make(steps=[MoveJStep(0, 0, 0, 0, 0, 0)])
        action.start()
        action.cancel()
        action.wait(timeout=2.0)
        self.assertIn(
            action.state(), (ActionState.CANCELLED, ActionState.DONE)
        )

    def test_wait_returns_false_on_cancelled(self):
        side_effect, at_step, can_proceed = _blocking_side_effect(2)
        self._controller.execute.side_effect = side_effect
        steps = [MoveJStep(0, 0, 0, 0, 0, 0), MoveLStep(1, 2, 3, 4, 5, 6)]
        action = self._make(steps=steps)
        action.start()
        at_step[0].wait()
        action.cancel()
        can_proceed[0].set()
        self.assertFalse(action.wait(timeout=2.0))


class TestRobotProgramActionDeviceName(_ActionTestMixin, unittest.TestCase):
    """Tests that device_name is forwarded correctly."""

    def test_device_name_is_first_arg_to_controller_execute(self):
        action = self._make(steps=[MoveJStep(0, 0, 0, 0, 0, 0)], device_name="my_robot")
        action.start()
        action.wait(timeout=2.0)
        args, _ = self._controller.execute.call_args
        self.assertEqual(args[0], "my_robot")

    def test_two_actions_with_different_device_names_dispatch_independently(self):
        ctrl_a = MagicMock()
        ctrl_a.execute.return_value = True
        ctrl_b = MagicMock()
        ctrl_b.execute.return_value = True
        a = RobotProgramAction(ctrl_a, "robot_a", [MoveJStep(0, 0, 0, 0, 0, 0)])
        b = RobotProgramAction(ctrl_b, "robot_b", [MoveLStep(1, 2, 3, 4, 5, 6)])
        a.start(); b.start()
        a.wait(timeout=2.0); b.wait(timeout=2.0)
        ctrl_a.execute.assert_called_once()
        ctrl_b.execute.assert_called_once()
        self.assertEqual(ctrl_a.execute.call_args[0][0], "robot_a")
        self.assertEqual(ctrl_b.execute.call_args[0][0], "robot_b")


class TestSerialize(unittest.TestCase):
    """Tests for the serialize() function."""

    def test_empty_list_returns_empty_string(self):
        self.assertEqual(serialize([]), "")

    def test_single_movej_format(self):
        step = MoveJStep(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        self.assertEqual(serialize([step]), "movej 1.0 2.0 3.0 4.0 5.0 6.0 20.0 0 0.0 0.0 0.0 0.0 0.0 0.0")

    def test_single_movel_format(self):
        step = MoveLStep(10.0, 20.0, 30.0, 1.0, 2.0, 3.0)
        self.assertEqual(serialize([step]), "movel 10.0 20.0 30.0 1.0 2.0 3.0 20.0 0 0.0 0.0 0.0 0.0 0.0 0.0")

    def test_custom_vel_included_in_output(self):
        step = MoveJStep(0, 0, 0, 0, 0, 0, vel=75.0)
        self.assertIn("75.0", serialize([step]))

    def test_tool_offset_serialized_with_mode_2(self):
        step = MoveJStep(0, 0, 0, 0, 0, 0, tool_offset=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        parts = serialize([step]).split()
        self.assertEqual(parts[8], "2")
        self.assertEqual(parts[9:15], ["1.0", "2.0", "3.0", "4.0", "5.0", "6.0"])

    def test_base_offset_serialized_with_mode_1(self):
        step = MoveLStep(0, 0, 0, 0, 0, 0, base_offset=[7.0, 8.0, 9.0, 0.0, 0.0, 0.0])
        parts = serialize([step]).split()
        self.assertEqual(parts[8], "1")
        self.assertEqual(parts[9:15], ["7.0", "8.0", "9.0", "0.0", "0.0", "0.0"])

    def test_no_offset_serialized_with_mode_0(self):
        step = MoveJStep(0, 0, 0, 0, 0, 0)
        parts = serialize([step]).split()
        self.assertEqual(parts[8], "0")

    def test_offset_values_appear_in_output(self):
        step = MoveJStep(0, 0, 0, 0, 0, 0, tool_offset=[99.5, 0, 0, 0, 0, 0])
        self.assertIn("99.5", serialize([step]))

    def test_multiple_steps_are_newline_separated(self):
        steps = [MoveJStep(1, 2, 3, 4, 5, 6), MoveLStep(7, 8, 9, 0, 0, 0)]
        lines = serialize(steps).splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("movej"))
        self.assertTrue(lines[1].startswith("movel"))

    def test_step_order_preserved_in_output(self):
        steps = [
            MoveLStep(1, 2, 3, 4, 5, 6),
            MoveJStep(7, 8, 9, 10, 11, 12),
            MoveLStep(0, 0, 0, 0, 0, 0),
        ]
        kinds = [line.split()[0] for line in serialize(steps).splitlines()]
        self.assertEqual(kinds, ["movel", "movej", "movel"])

    def test_unknown_step_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            serialize([{"type": "bad"}])

    def test_negative_values_serialized_correctly(self):
        step = MoveJStep(-45.0, -90.0, 0.0, 0.0, 0.0, 0.0)
        text = serialize([step])
        self.assertIn("-45.0", text)
        self.assertIn("-90.0", text)


class TestDeserialize(unittest.TestCase):
    """Tests for the deserialize() function."""

    def test_empty_string_returns_empty_list(self):
        self.assertEqual(deserialize(""), [])

    def test_blank_lines_are_skipped(self):
        text = "\nmovej 0.0 0.0 0.0 0.0 0.0 0.0 20.0 0 0.0 0.0 0.0 0.0 0.0 0.0\n\n"
        self.assertEqual(len(deserialize(text)), 1)

    def test_movej_line_produces_movej_step(self):
        step = deserialize("movej 1.0 2.0 3.0 4.0 5.0 6.0 20.0 0 0.0 0.0 0.0 0.0 0.0 0.0")[0]
        self.assertIsInstance(step, MoveJStep)
        self.assertEqual((step.j1, step.j2, step.j3, step.j4, step.j5, step.j6, step.vel),
                         (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 20.0))
        self.assertIsNone(step.tool_offset)
        self.assertIsNone(step.base_offset)

    def test_movel_line_produces_movel_step(self):
        step = deserialize("movel 10.0 20.0 30.0 1.0 2.0 3.0 50.0 0 0.0 0.0 0.0 0.0 0.0 0.0")[0]
        self.assertIsInstance(step, MoveLStep)
        self.assertEqual((step.x, step.y, step.z, step.rx, step.ry, step.rz, step.vel),
                         (10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 50.0))
        self.assertIsNone(step.tool_offset)
        self.assertIsNone(step.base_offset)

    def test_multiple_lines_produce_correct_count(self):
        text = (
            "movej 0 0 0 0 0 0 20 0 0 0 0 0 0 0\n"
            "movel 1 2 3 4 5 6 20 0 0 0 0 0 0 0\n"
            "movej 7 8 9 10 11 12 20 0 0 0 0 0 0 0"
        )
        self.assertEqual(len(deserialize(text)), 3)

    def test_unknown_kind_raises_value_error(self):
        with self.assertRaises(ValueError):
            deserialize("rotate 1 2 3 4 5 6 20 0 0 0 0 0 0 0")

    def test_too_few_fields_raises_value_error(self):
        with self.assertRaises(ValueError):
            deserialize("movej 1 2 3 4 5 6")

    def test_too_many_fields_raises_value_error(self):
        with self.assertRaises(ValueError):
            deserialize("movej 1 2 3 4 5 6 20 0 0 0 0 0 0 0 99")

    def test_non_float_field_raises_value_error(self):
        with self.assertRaises(ValueError):
            deserialize("movej 1.0 2.0 3.0 4.0 5.0 abc 20.0 0 0.0 0.0 0.0 0.0 0.0 0.0")

    def test_negative_values_parsed_correctly(self):
        step = deserialize("movej -45.0 -90.0 0.0 0.0 0.0 0.0 20.0 0 0.0 0.0 0.0 0.0 0.0 0.0")[0]
        self.assertEqual(step.j1, -45.0)
        self.assertEqual(step.j2, -90.0)

    def test_mode_2_produces_tool_offset(self):
        step = deserialize("movej 0 0 0 0 0 0 20 2 1.0 2.0 3.0 4.0 5.0 6.0")[0]
        self.assertEqual(step.tool_offset, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertIsNone(step.base_offset)

    def test_mode_1_produces_base_offset(self):
        step = deserialize("movel 0 0 0 0 0 0 20 1 7.0 8.0 9.0 0.0 0.0 0.0")[0]
        self.assertIsNone(step.tool_offset)
        self.assertEqual(step.base_offset, [7.0, 8.0, 9.0, 0.0, 0.0, 0.0])

    def test_mode_0_produces_none_offsets(self):
        step = deserialize("movej 0 0 0 0 0 0 20 0 0 0 0 0 0 0")[0]
        self.assertIsNone(step.tool_offset)
        self.assertIsNone(step.base_offset)

    def test_unknown_offset_mode_raises_value_error(self):
        with self.assertRaises(ValueError):
            deserialize("movej 0 0 0 0 0 0 20 9 0 0 0 0 0 0")

    def test_old_8part_format_raises_value_error(self):
        with self.assertRaises(ValueError):
            deserialize("movej 1 2 3 4 5 6 20")


class TestSerializeDeserializeRoundtrip(unittest.TestCase):
    """Roundtrip tests: serialize then deserialize must reproduce the original steps."""

    def test_roundtrip_single_movej(self):
        original = [MoveJStep(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, vel=35.0)]
        self.assertEqual(deserialize(serialize(original)), original)

    def test_roundtrip_single_movel(self):
        original = [MoveLStep(100.0, 200.0, 300.0, 1.5, 2.5, 3.5, vel=60.0)]
        self.assertEqual(deserialize(serialize(original)), original)

    def test_roundtrip_mixed_steps_preserves_order(self):
        original = [
            MoveJStep(0, 0, 0, 0, 0, 0),
            MoveLStep(1, 2, 3, 4, 5, 6),
            MoveJStep(7, 8, 9, 10, 11, 12, vel=50.0),
        ]
        self.assertEqual(deserialize(serialize(original)), original)

    def test_roundtrip_empty_list(self):
        self.assertEqual(deserialize(serialize([])), [])

    def test_roundtrip_preserves_custom_vel(self):
        original = [MoveJStep(0, 0, 0, 0, 0, 0, vel=99.0)]
        result = deserialize(serialize(original))
        self.assertEqual(result[0].vel, 99.0)

    def test_roundtrip_preserves_tool_offset(self):
        original = [MoveJStep(0, 0, 0, 0, 0, 0, tool_offset=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]
        result = deserialize(serialize(original))
        self.assertEqual(result[0].tool_offset, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertIsNone(result[0].base_offset)

    def test_roundtrip_preserves_base_offset(self):
        original = [MoveLStep(1, 2, 3, 4, 5, 6, base_offset=[7.0, 8.0, 9.0, 0.0, 0.0, 0.0])]
        result = deserialize(serialize(original))
        self.assertIsNone(result[0].tool_offset)
        self.assertEqual(result[0].base_offset, [7.0, 8.0, 9.0, 0.0, 0.0, 0.0])

    def test_roundtrip_preserves_no_offset(self):
        original = [MoveJStep(1, 2, 3, 4, 5, 6)]
        result = deserialize(serialize(original))
        self.assertIsNone(result[0].tool_offset)
        self.assertIsNone(result[0].base_offset)


if __name__ == "__main__":
    unittest.main()
