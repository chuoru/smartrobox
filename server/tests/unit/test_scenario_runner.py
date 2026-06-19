#!/usr/bin/env python3
##
# @file test_scenario_runner.py
#
# @brief Unit tests for ScenarioRunner.
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
from scenarios.runner import ScenarioRunner
from scenarios.step import ActionStep, ParallelStep, Scenario, SequenceStep


# =========================================================================
# Mock action helpers
# =========================================================================


class _InstantAction(BaseAction):
    """Completes immediately."""

    def _run(self) -> bool:
        return self._checkpoint()


class _BlockingAction(BaseAction):
    """Blocks at a proceed event, then yields a checkpoint."""

    def __init__(self, controller):
        super().__init__(controller)
        self.started = threading.Event()
        self.proceed = threading.Event()

    def _run(self) -> bool:
        self.started.set()
        self.proceed.wait()
        return self._checkpoint()


class _FailingAction(BaseAction):
    """Raises RuntimeError immediately."""

    def _run(self):
        raise RuntimeError("device error")


# =========================================================================
# YAML-structure helpers
# =========================================================================


def _scenario(steps: list) -> Scenario:
    return Scenario(name="test", steps=steps)


def _action(type_name: str = "noop") -> ActionStep:
    return ActionStep(action_type=type_name, params={})


def _parallel(*threads: list) -> ParallelStep:
    """Build a ParallelStep whose threads are the given step lists."""
    return ParallelStep(threads=[SequenceStep(steps=s) for s in threads])


# =========================================================================
# Base test mixin
# =========================================================================


class _Mixin:
    def setUp(self):
        self._logger_patcher = patch("actions.base.Logger")
        self._logger_patcher.start()
        self._controller = MagicMock()

    def tearDown(self):
        self._logger_patcher.stop()

    def _runner(self, steps: list) -> ScenarioRunner:
        return ScenarioRunner(self._controller, _scenario(steps), "/data")

    def _run(self, steps: list, factory) -> ScenarioRunner:
        """Run a scenario with a class-level factory patch, wait for completion."""
        runner = self._runner(steps)
        with patch.object(
            ScenarioRunner, "_build_action", lambda self, step: factory(step)
        ):
            runner.start()
            runner.wait(timeout=2.0)
        return runner


# =========================================================================
# Init & parameters
# =========================================================================


class TestScenarioRunnerInit(_Mixin, unittest.TestCase):
    def test_initial_state_is_idle(self):
        self.assertEqual(self._runner([]).state(), ActionState.IDLE)

    def test_parameters_returns_scenario_name(self):
        self.assertEqual(self._runner([]).parameters()["scenario"], "test")


# =========================================================================
# Sequential execution
# =========================================================================


class TestScenarioRunnerSequential(_Mixin, unittest.TestCase):
    def test_empty_scenario_is_done(self):
        runner = self._runner([])
        runner.start()
        runner.wait(timeout=2.0)
        self.assertEqual(runner.state(), ActionState.DONE)

    def test_single_step_completes_done(self):
        runner = self._run(
            [_action()],
            lambda step: _InstantAction(self._controller),
        )
        self.assertEqual(runner.state(), ActionState.DONE)

    def test_single_step_result_true(self):
        runner = self._run(
            [_action()],
            lambda step: _InstantAction(self._controller),
        )
        self.assertTrue(runner.result())

    def test_two_steps_run_in_order(self):
        order = []

        class _Record(BaseAction):
            def __init__(self, ctrl, label):
                super().__init__(ctrl)
                self._label = label

            def _run(self):
                order.append(self._label)
                return True

        labels = iter(["first", "second"])
        runner = self._run(
            [_action("a"), _action("b")],
            lambda step: _Record(self._controller, next(labels)),
        )
        self.assertEqual(order, ["first", "second"])

    def test_failing_step_sets_runner_failed(self):
        runner = self._run(
            [_action()],
            lambda step: _FailingAction(self._controller),
        )
        self.assertEqual(runner.state(), ActionState.FAILED)

    def test_failing_step_error_contains_action_type(self):
        runner = self._run(
            [_action("my_action")],
            lambda step: _FailingAction(self._controller),
        )
        self.assertIn("my_action", str(runner.error()))

    def test_second_step_skipped_after_first_fails(self):
        executed = []

        def factory(step):
            if not executed:
                executed.append("first")
                return _FailingAction(self._controller)
            executed.append("second")
            return _InstantAction(self._controller)

        self._run([_action(), _action()], factory)
        self.assertNotIn("second", executed)


# =========================================================================
# Parallel execution — threads with single steps
# =========================================================================


class TestScenarioRunnerParallelSingle(_Mixin, unittest.TestCase):
    """Parallel step where each thread has exactly one action."""

    def test_two_threads_runner_is_done(self):
        runner = self._run(
            [_parallel([_action()], [_action()])],
            lambda step: _InstantAction(self._controller),
        )
        self.assertEqual(runner.state(), ActionState.DONE)

    def test_three_threads_all_start(self):
        started = []
        lock = threading.Lock()
        all_started = threading.Event()
        proceed = threading.Event()

        class _Track(BaseAction):
            def _run(self):
                with lock:
                    started.append(1)
                    if len(started) == 3:
                        all_started.set()
                proceed.wait()
                return True

        runner = self._runner([_parallel([_action()], [_action()], [_action()])])
        with patch.object(
            ScenarioRunner, "_build_action", lambda self, step: _Track(self._controller)
        ):
            runner.start()
            all_started.wait(timeout=2.0)
            proceed.set()
            runner.wait(timeout=2.0)

        self.assertTrue(all_started.is_set())
        self.assertEqual(len(started), 3)

    def test_one_failing_thread_sets_runner_failed(self):
        calls = [0]

        def factory(step):
            calls[0] += 1
            return _FailingAction(self._controller) if calls[0] == 1 else _InstantAction(self._controller)

        runner = self._run(
            [_parallel([_action()], [_action()])],
            factory,
        )
        self.assertEqual(runner.state(), ActionState.FAILED)


# =========================================================================
# Parallel execution — threads with multiple steps
# =========================================================================


class TestScenarioRunnerParallelMultiStep(_Mixin, unittest.TestCase):
    """Parallel step where at least one thread has multiple sequential steps."""

    def test_steps_within_thread_run_sequentially(self):
        """Actions in the same thread execute in order."""
        order = []
        lock = threading.Lock()

        class _Record(BaseAction):
            def __init__(self, ctrl, label):
                super().__init__(ctrl)
                self._label = label

            def _run(self):
                with lock:
                    order.append(self._label)
                return True

        runner = self._runner(
            [_parallel([_action("first"), _action("second")], [_action("third")])]
        )
        with patch.object(
            ScenarioRunner,
            "_build_action",
            lambda self, step: _Record(self._controller, step.action_type),
        ):
            runner.start()
            runner.wait(timeout=2.0)

        self.assertEqual(runner.state(), ActionState.DONE)
        # "first" and "second" are in thread 1 — "first" must precede "second"
        self.assertLess(order.index("first"), order.index("second"))

    def test_threads_run_concurrently(self):
        """Both threads start before the slower one unblocks."""
        started = []
        lock = threading.Lock()
        both_started = threading.Event()
        proceed = threading.Event()

        class _Sync(BaseAction):
            def _run(self):
                with lock:
                    started.append(1)
                    if len(started) == 2:
                        both_started.set()
                proceed.wait()
                return True

        # Thread 1: one blocking step; Thread 2: one instant step followed by one blocking step
        runner = self._runner(
            [_parallel([_action("t1")], [_action("t2a"), _action("t2b")])]
        )

        call_count = [0]

        def factory(step):
            call_count[0] += 1
            # First call per thread is the blocking action that signals started
            if step.action_type in ("t1", "t2a"):
                return _Sync(self._controller)
            return _InstantAction(self._controller)

        with patch.object(ScenarioRunner, "_build_action", lambda self, step: factory(step)):
            runner.start()
            both_started.wait(timeout=2.0)
            proceed.set()
            runner.wait(timeout=2.0)

        self.assertTrue(both_started.is_set())

    def test_multi_step_thread_failure_sets_runner_failed(self):
        calls = [0]

        def factory(step):
            calls[0] += 1
            # Second call (second step in thread 1) fails
            return _FailingAction(self._controller) if calls[0] == 2 else _InstantAction(self._controller)

        runner = self._run(
            [_parallel([_action("a"), _action("b")], [_action("c")])],
            factory,
        )
        self.assertEqual(runner.state(), ActionState.FAILED)

    def test_runner_done_after_multi_step_threads(self):
        runner = self._run(
            [_parallel([_action("a"), _action("b")], [_action("c"), _action("d")])],
            lambda step: _InstantAction(self._controller),
        )
        self.assertEqual(runner.state(), ActionState.DONE)


# =========================================================================
# Cancel
# =========================================================================


class TestScenarioRunnerCancel(_Mixin, unittest.TestCase):
    def test_cancel_before_start_is_noop(self):
        runner = self._runner([])
        runner.cancel()
        self.assertEqual(runner.state(), ActionState.IDLE)

    def test_cancel_during_step_sets_cancelled(self):
        blocker = _BlockingAction(self._controller)
        runner = self._runner([_action()])
        with patch.object(ScenarioRunner, "_build_action", lambda self, step: blocker):
            runner.start()
            blocker.started.wait(timeout=2.0)
            runner.cancel()
            blocker.proceed.set()
            runner.wait(timeout=2.0)
        self.assertEqual(runner.state(), ActionState.CANCELLED)

    def test_cancel_propagates_to_active_action(self):
        blocker = _BlockingAction(self._controller)
        runner = self._runner([_action()])
        with patch.object(ScenarioRunner, "_build_action", lambda self, step: blocker):
            runner.start()
            blocker.started.wait(timeout=2.0)
            runner.cancel()
            blocker.proceed.set()
            runner.wait(timeout=2.0)
        self.assertEqual(blocker.state(), ActionState.CANCELLED)

    def test_cancel_during_parallel_propagates_to_all_threads(self):
        blockers = [_BlockingAction(self._controller), _BlockingAction(self._controller)]
        idx = [0]

        def factory(step):
            b = blockers[idx[0] % len(blockers)]
            idx[0] += 1
            return b

        runner = self._runner([_parallel([_action()], [_action()])])
        with patch.object(ScenarioRunner, "_build_action", lambda self, step: factory(step)):
            runner.start()
            for b in blockers:
                b.started.wait(timeout=2.0)
            runner.cancel()
            for b in blockers:
                b.proceed.set()
            runner.wait(timeout=2.0)

        self.assertEqual(runner.state(), ActionState.CANCELLED)
        for b in blockers:
            self.assertIn(b.state(), (ActionState.CANCELLED, ActionState.DONE))


# =========================================================================
# Pause / resume
# =========================================================================


class TestScenarioRunnerPauseResume(_Mixin, unittest.TestCase):
    def test_pause_propagates_to_active_action(self):
        blocker = _BlockingAction(self._controller)
        runner = self._runner([_action()])
        with patch.object(ScenarioRunner, "_build_action", lambda self, step: blocker):
            runner.start()
            blocker.started.wait(timeout=2.0)
            runner.pause()
            self.assertEqual(blocker.state(), ActionState.PAUSED)
            runner.resume()
            blocker.proceed.set()
            runner.wait(timeout=2.0)

    def test_resume_unpauses_active_action(self):
        blocker = _BlockingAction(self._controller)
        runner = self._runner([_action()])
        with patch.object(ScenarioRunner, "_build_action", lambda self, step: blocker):
            runner.start()
            blocker.started.wait(timeout=2.0)
            runner.pause()
            runner.resume()
            self.assertEqual(blocker.state(), ActionState.RUNNING)
            blocker.proceed.set()
            runner.wait(timeout=2.0)

    def test_pause_then_resume_runner_finishes_done(self):
        blocker = _BlockingAction(self._controller)
        runner = self._runner([_action()])
        with patch.object(ScenarioRunner, "_build_action", lambda self, step: blocker):
            runner.start()
            blocker.started.wait(timeout=2.0)
            runner.pause()
            runner.resume()
            blocker.proceed.set()
            self.assertTrue(runner.wait(timeout=2.0))
        self.assertEqual(runner.state(), ActionState.DONE)

    def test_pause_when_not_running_returns_false(self):
        self.assertFalse(self._runner([]).pause())

    def test_resume_when_not_paused_returns_false(self):
        self.assertFalse(self._runner([]).resume())

    def test_pause_during_parallel_propagates_to_thread_actions(self):
        blockers = [_BlockingAction(self._controller), _BlockingAction(self._controller)]
        idx = [0]

        def factory(step):
            b = blockers[idx[0] % len(blockers)]
            idx[0] += 1
            return b

        runner = self._runner([_parallel([_action()], [_action()])])
        with patch.object(ScenarioRunner, "_build_action", lambda self, step: factory(step)):
            runner.start()
            for b in blockers:
                b.started.wait(timeout=2.0)
            runner.pause()
            for b in blockers:
                self.assertEqual(b.state(), ActionState.PAUSED)
            runner.resume()
            for b in blockers:
                b.proceed.set()
            runner.wait(timeout=2.0)
        self.assertEqual(runner.state(), ActionState.DONE)


# =========================================================================
# Unknown action type
# =========================================================================


class TestScenarioRunnerUnknownAction(_Mixin, unittest.TestCase):
    def test_unknown_action_type_sets_runner_failed(self):
        runner = self._runner([_action("nonexistent")])
        runner.start()
        runner.wait(timeout=2.0)
        self.assertEqual(runner.state(), ActionState.FAILED)

    def test_unknown_action_type_error_message(self):
        runner = self._runner([_action("nonexistent")])
        runner.start()
        runner.wait(timeout=2.0)
        self.assertIn("nonexistent", str(runner.error()))


if __name__ == "__main__":
    unittest.main()
