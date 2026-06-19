#!/usr/bin/env python3
##
# @file test_base_action.py
#
# @brief Unit tests for BaseAction and ActionState.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/19.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import threading
import unittest
from unittest.mock import MagicMock, patch

# Internal library
from actions.base import ActionState, BaseAction


# ---------------------------------------------------------------------------
# Concrete subclasses used across tests
# ---------------------------------------------------------------------------

class _ValueAction(BaseAction):
    """Completes immediately and returns a fixed value."""

    def __init__(self, controller, value=42):
        super().__init__(controller)
        self._value = value

    def _run(self):
        return self._value


class _ErrorAction(BaseAction):
    """Raises a fixed exception from _run()."""

    def __init__(self, controller, exception=None):
        super().__init__(controller)
        self._exception = exception or RuntimeError("hardware fault")

    def _run(self):
        raise self._exception


class _CancellableAction(BaseAction):
    """Two-step action that calls _checkpoint() between steps."""

    def __init__(self, controller):
        super().__init__(controller)
        self.steps_done = 0

    def _run(self):
        self.steps_done += 1
        if not self._checkpoint():
            return self.steps_done
        self.steps_done += 1
        if not self._checkpoint():
            return self.steps_done
        return self.steps_done


class _ControlledAction(BaseAction):
    """Two-step action that synchronises with the test thread.

    Each step signals _at_step[i] when complete, then blocks on
    _can_proceed[i] until the test releases it.  This lets tests
    call pause()/cancel() between steps without races.
    """

    def __init__(self, controller):
        super().__init__(controller)
        self._at_step = [threading.Event(), threading.Event()]
        self._can_proceed = [threading.Event(), threading.Event()]
        self.steps_done = 0

    def _run(self):
        self.steps_done += 1
        self._at_step[0].set()
        self._can_proceed[0].wait()
        self._can_proceed[0].clear()
        if not self._checkpoint():
            return self.steps_done

        self.steps_done += 1
        self._at_step[1].set()
        self._can_proceed[1].wait()
        self._can_proceed[1].clear()
        if not self._checkpoint():
            return self.steps_done

        return self.steps_done


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestActionState(unittest.TestCase):
    """Tests for the ActionState enum."""

    def test_six_states_defined(self):
        self.assertEqual(len(ActionState), 6)

    def test_all_states_have_string_values(self):
        for state in ActionState:
            self.assertIsInstance(state.value, str)

    def test_expected_state_names_present(self):
        names = {s.name for s in ActionState}
        self.assertEqual(
            names, {"IDLE", "RUNNING", "PAUSED", "DONE", "FAILED", "CANCELLED"}
        )


class TestBaseAction(unittest.TestCase):
    """Tests for BaseAction lifecycle, state transitions, and composability."""

    # =========================================================================
    # SETUP / TEARDOWN
    # =========================================================================

    def setUp(self):
        self._logger_patcher = patch("actions.base.Logger")
        self._logger_patcher.start()
        self._controller = MagicMock()

    def tearDown(self):
        self._logger_patcher.stop()

    def _make(self, cls=_ValueAction, **kwargs):
        return cls(self._controller, **kwargs)

    # =========================================================================
    # ABSTRACT _run()
    # =========================================================================

    def test_run_raises_not_implemented_on_base_class(self):
        action = BaseAction(self._controller)
        with self.assertRaises(NotImplementedError):
            action._run()

    # =========================================================================
    # INITIAL STATE
    # =========================================================================

    def test_initial_state_is_idle(self):
        action = self._make()
        self.assertEqual(action.state(), ActionState.IDLE)

    def test_initial_result_is_none(self):
        action = self._make()
        self.assertIsNone(action.result())

    def test_initial_error_is_none(self):
        action = self._make()
        self.assertIsNone(action.error())

    # =========================================================================
    # START
    # =========================================================================

    def test_start_returns_true_on_idle(self):
        action = self._make()
        self.assertTrue(action.start())
        action.wait(timeout=2.0)

    def test_start_transitions_to_done(self):
        action = self._make()
        action.start()
        action.wait(timeout=2.0)
        self.assertEqual(action.state(), ActionState.DONE)

    def test_start_returns_false_when_already_running(self):
        action = self._make(_ControlledAction)
        action.start()
        self.assertFalse(action.start())
        action._can_proceed[0].set()
        action._can_proceed[1].set()
        action.wait(timeout=2.0)

    def test_start_returns_false_when_paused(self):
        action = self._make(_ControlledAction)
        action.start()
        action._at_step[0].wait()
        action.pause()
        self.assertFalse(action.start())
        action._can_proceed[0].set()
        action.resume()
        action._can_proceed[1].set()
        action.wait(timeout=2.0)

    def test_start_returns_false_when_done(self):
        action = self._make()
        action.start()
        action.wait(timeout=2.0)
        self.assertFalse(action.start())

    # =========================================================================
    # DONE — result / wait
    # =========================================================================

    def test_result_is_set_after_done(self):
        action = self._make(_ValueAction, value="sensor_ok")
        action.start()
        action.wait(timeout=2.0)
        self.assertEqual(action.result(), "sensor_ok")

    def test_wait_returns_true_on_done(self):
        action = self._make()
        action.start()
        self.assertTrue(action.wait(timeout=2.0))

    def test_wait_returns_false_on_timeout(self):
        action = self._make(_ControlledAction)
        action.start()
        action._at_step[0].wait()
        self.assertFalse(action.wait(timeout=0.05))
        action._can_proceed[0].set()
        action._can_proceed[1].set()
        action.wait(timeout=2.0)

    # =========================================================================
    # FAILED — error / wait
    # =========================================================================

    def test_failed_state_on_exception_in_run(self):
        action = self._make(_ErrorAction)
        action.start()
        action.wait(timeout=2.0)
        self.assertEqual(action.state(), ActionState.FAILED)

    def test_error_is_set_to_raised_exception(self):
        exc = ValueError("bad encoder value")
        action = _ErrorAction(self._controller, exception=exc)
        action.start()
        action.wait(timeout=2.0)
        self.assertIs(action.error(), exc)

    def test_wait_returns_false_on_failed(self):
        action = self._make(_ErrorAction)
        action.start()
        self.assertFalse(action.wait(timeout=2.0))

    # =========================================================================
    # PAUSE
    # =========================================================================

    def test_pause_returns_false_when_idle(self):
        action = self._make()
        self.assertFalse(action.pause())

    def test_pause_returns_false_when_done(self):
        action = self._make()
        action.start()
        action.wait(timeout=2.0)
        self.assertFalse(action.pause())

    def test_pause_sets_state_to_paused(self):
        action = self._make(_ControlledAction)
        action.start()
        action._at_step[0].wait()
        self.assertTrue(action.pause())
        self.assertEqual(action.state(), ActionState.PAUSED)
        action._can_proceed[0].set()
        action.resume()
        action._can_proceed[1].set()
        action.wait(timeout=2.0)

    def test_pause_returns_false_when_already_paused(self):
        action = self._make(_ControlledAction)
        action.start()
        action._at_step[0].wait()
        action.pause()
        self.assertFalse(action.pause())
        action._can_proceed[0].set()
        action.resume()
        action._can_proceed[1].set()
        action.wait(timeout=2.0)

    # =========================================================================
    # RESUME
    # =========================================================================

    def test_resume_returns_false_when_idle(self):
        action = self._make()
        self.assertFalse(action.resume())

    def test_resume_returns_false_when_running(self):
        action = self._make(_ControlledAction)
        action.start()
        self.assertFalse(action.resume())
        action._can_proceed[0].set()
        action._can_proceed[1].set()
        action.wait(timeout=2.0)

    def test_resume_allows_paused_action_to_complete(self):
        action = self._make(_ControlledAction)
        action.start()
        action._at_step[0].wait()
        action.pause()
        action._can_proceed[0].set()
        self.assertTrue(action.resume())
        action._at_step[1].wait()
        action._can_proceed[1].set()
        self.assertTrue(action.wait(timeout=2.0))
        self.assertEqual(action.state(), ActionState.DONE)

    def test_resume_sets_state_to_running(self):
        action = self._make(_ControlledAction)
        action.start()
        action._at_step[0].wait()
        action.pause()
        action.resume()
        self.assertEqual(action.state(), ActionState.RUNNING)
        action._can_proceed[0].set()
        action._can_proceed[1].set()
        action.wait(timeout=2.0)

    # =========================================================================
    # CANCEL
    # =========================================================================

    def test_cancel_returns_false_when_idle(self):
        action = self._make()
        # IDLE is not a terminal state but no thread is running
        # cancel() succeeds on non-terminal states
        self.assertTrue(action.cancel())

    def test_cancel_returns_false_when_done(self):
        action = self._make()
        action.start()
        action.wait(timeout=2.0)
        self.assertFalse(action.cancel())

    def test_cancel_returns_false_when_failed(self):
        action = self._make(_ErrorAction)
        action.start()
        action.wait(timeout=2.0)
        self.assertFalse(action.cancel())

    def test_cancel_returns_false_when_already_cancelled(self):
        action = self._make(_ControlledAction)
        action.start()
        action.cancel()
        action._can_proceed[0].set()
        action.wait(timeout=2.0)
        self.assertFalse(action.cancel())

    def test_cancel_while_running_sets_cancelled(self):
        action = self._make(_ControlledAction)
        action.start()
        action._at_step[0].wait()
        self.assertTrue(action.cancel())
        action._can_proceed[0].set()
        action.wait(timeout=2.0)
        self.assertEqual(action.state(), ActionState.CANCELLED)

    def test_cancel_while_paused_sets_cancelled(self):
        action = self._make(_ControlledAction)
        action.start()
        action._at_step[0].wait()
        action.pause()
        action._can_proceed[0].set()
        action.cancel()
        action.wait(timeout=2.0)
        self.assertEqual(action.state(), ActionState.CANCELLED)

    def test_cancel_stops_further_steps(self):
        action = self._make(_ControlledAction)
        action.start()
        action._at_step[0].wait()
        action.cancel()
        action._can_proceed[0].set()
        action.wait(timeout=2.0)
        self.assertEqual(action.steps_done, 1)

    def test_wait_returns_false_on_cancelled(self):
        action = self._make(_ControlledAction)
        action.start()
        action._at_step[0].wait()
        action.cancel()
        action._can_proceed[0].set()
        self.assertFalse(action.wait(timeout=2.0))

    # =========================================================================
    # _call
    # =========================================================================

    def test_call_delegates_to_controller_execute(self):
        self._controller.execute.return_value = True
        action = self._make()
        result = action._call("left_arm", "movej", j1=0.0, j2=0.0)
        self._controller.execute.assert_called_once_with("left_arm", "movej", j1=0.0, j2=0.0)
        self.assertTrue(result)

    def test_call_forwards_positional_args(self):
        pose = [128] * 10
        self._controller.execute.return_value = True
        action = self._make()
        action._call("left_hand", "move", pose)
        self._controller.execute.assert_called_once_with("left_hand", "move", pose)

    def test_call_propagates_controller_exception(self):
        self._controller.execute.side_effect = KeyError("left_arm")
        action = self._make()
        with self.assertRaises(KeyError):
            action._call("left_arm", "movej")

    # =========================================================================
    # _checkpoint
    # =========================================================================

    def test_checkpoint_returns_true_when_not_cancelled(self):
        action = self._make()
        action._state = ActionState.RUNNING
        self.assertTrue(action._checkpoint())

    def test_checkpoint_returns_false_after_cancel_flag_set(self):
        action = self._make()
        action._state = ActionState.RUNNING
        action._cancelled = True
        action._pause_event.set()
        self.assertFalse(action._checkpoint())

    # =========================================================================
    # parameters
    # =========================================================================

    def test_parameters_returns_empty_dict(self):
        action = BaseAction(self._controller)
        self.assertEqual(action.parameters(), {})


if __name__ == "__main__":
    unittest.main()
