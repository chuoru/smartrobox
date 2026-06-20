#!/usr/bin/env python3
##
# @file runner.py
#
# @brief ScenarioRunner: execute a Scenario as a pauseable/cancellable BaseAction.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/19.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import threading

# Internal library
from actions.base import ActionState, BaseAction
from actions.estimate_pose import EstimatePoseAction
from actions.grasp import GraspAction
from actions.massage import MassageAction
from actions.robot_program import RobotProgramAction
from actions.visual_servo import VisualServoAction
from app.controller import Controller
from scenarios.step import ActionStep, ParallelStep, Scenario, SequenceStep


def _make_grasp(
    controller: Controller, data_folder: str, params: dict
) -> GraspAction:
    return GraspAction(
        controller,
        device_name=params["device"],
        grasp_level=int(params["grasp_level"]),
        torque_limit=int(params["torque_limit"]),
    )


def _make_robot_program(
    controller: Controller, data_folder: str, params: dict
) -> RobotProgramAction:
    return RobotProgramAction(
        controller,
        program_name=params["program"],
        device_name=params["device"],
        data_folder=data_folder,
    )


def _make_estimate_pose(
    controller: Controller, data_folder: str, params: dict
) -> EstimatePoseAction:
    return EstimatePoseAction(
        controller,
        device_name=params["device"],
        model_name=params.get("model", "yolo11n-pose.pt"),
    )


def _make_massage(
    controller: Controller, data_folder: str, params: dict
) -> MassageAction:
    return MassageAction(
        controller,
        device_name=params["device"],
        cycles=int(params.get("cycles", 5)),
        half_close_duration=float(params.get("half_close_duration", 0.4)),
        open_duration=float(params.get("open_duration", 0.4)),
        torque_limit=int(params.get("torque_limit", 180)),
    )


def _make_visual_servo(
    controller: Controller, data_folder: str, params: dict
) -> VisualServoAction:
    return VisualServoAction(
        controller,
        left_robot_device=params["left_robot_device"],
        right_robot_device=params["right_robot_device"],
        camera_device=params["camera_device"],
        left_arm_extrinsic=params["left_arm_extrinsic"],
        right_arm_extrinsic=params["right_arm_extrinsic"],
        error_threshold=float(params["error_threshold"]),
        stable_ticks=int(params["stable_ticks"]),
        servo_gain=float(params.get("servo_gain", 0.5)),
        cmd_period=float(params.get("cmd_period", 0.016)),
        timeout=float(params.get("timeout", 30.0)),
        model_name=params.get("model", "yolo11n-pose.pt"),
        keypoint_conf_min=float(params.get("keypoint_conf_min", 0.5)),
    )


_ACTION_REGISTRY: dict = {
    "grasp": _make_grasp,
    "massage": _make_massage,
    "robot_program": _make_robot_program,
    "estimate_pose": _make_estimate_pose,
    "visual_servo": _make_visual_servo,
}


class ScenarioRunner(BaseAction):
    """! Executes a Scenario as a pauseable/cancellable BaseAction.

    Sequential steps run one after another; parallel steps all start at
    once and the runner waits for all to finish before continuing.

    Pause, resume, and cancel propagate to the currently active child
    action(s) so that hardware motion stops at the earliest safe opportunity.
    """

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def __init__(
        self,
        controller: Controller,
        scenario: Scenario,
        data_folder: str,
    ) -> None:
        """! Initialise the runner.

        @param controller<Controller>: Controller used to dispatch device calls.
        @param scenario<Scenario>: Parsed scenario to execute.
        @param data_folder<str>: Root data directory forwarded to actions that
            require it (e.g. RobotProgramAction).
        """
        super().__init__(controller)
        self._scenario = scenario
        self._data_folder = data_folder
        self._active_actions: list[BaseAction] = []
        self._action_lock = threading.Lock()

    def pause(self) -> bool:
        """! Pause the runner and all currently active child actions.

        @return<bool>: True if pause was requested, False if not running.
        """
        result = super().pause()
        if result:
            with self._action_lock:
                for action in list(self._active_actions):
                    action.pause()
        return result

    def resume(self) -> bool:
        """! Resume the runner and all currently paused child actions.

        Children are resumed before the runner's own checkpoint is unblocked
        so that the runner thread never races ahead of the hardware.

        @return<bool>: True if resumed, False if not paused.
        """
        with self._action_lock:
            for action in list(self._active_actions):
                action.resume()
        return super().resume()

    def cancel(self) -> bool:
        """! Cancel the runner and all currently active child actions.

        @return<bool>: True if cancellation was requested, False if already terminal.
        """
        result = super().cancel()
        with self._action_lock:
            for action in list(self._active_actions):
                action.cancel()
        return result

    def parameters(self) -> dict:
        """! Return the runner's configuration parameters.

        @return<dict>: {"scenario": <scenario name>}
        """
        return {"scenario": self._scenario.name}

    # =========================================================================
    # PROTECTED METHODS
    # =========================================================================

    def _run(self) -> bool:
        """! Execute all scenario steps in order.

        @return<bool>: True on full completion, False if cancelled.
        @raises RuntimeError: If any child action fails.
        """
        for step in self._scenario.steps:
            if isinstance(step, ParallelStep):
                ok = self._run_parallel(step)
            else:
                ok = self._run_step(step)
            if not ok:
                return False
            if not self._checkpoint():
                return False
        return True

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _run_step(self, step: ActionStep) -> bool:
        """! Run a single action step and wait for completion.

        @param step<ActionStep>: Step to execute.
        @return<bool>: True if DONE, False if CANCELLED.
        @raises RuntimeError: If the action reaches FAILED state.
        """
        action = self._build_action(step)
        with self._action_lock:
            self._active_actions = [action]
        action.start()
        action.wait()
        with self._action_lock:
            self._active_actions = []
        state = action.state()
        if state == ActionState.FAILED:
            raise RuntimeError(
                f"Action '{step.action_type}' failed: {action.error()}"
            )
        return state == ActionState.DONE

    def _run_parallel(self, step: ParallelStep) -> bool:
        """! Start each thread as a sub-ScenarioRunner and wait for all to finish.

        Each SequenceStep thread becomes its own ScenarioRunner so that its
        steps execute sequentially inside the thread.  Pause/cancel on this
        runner propagate into the sub-runners via _active_actions, which in
        turn propagate to their own active actions — no extra wiring needed.

        If any thread fails, remaining threads are cancelled before raising.

        @param step<ParallelStep>: Parallel step group to execute.
        @return<bool>: True if all threads DONE, False if any CANCELLED.
        @raises RuntimeError: If any thread reaches FAILED state.
        """
        sub_runners = [
            ScenarioRunner(
                self._controller,
                Scenario(name=self._scenario.name, steps=seq.steps),
                self._data_folder,
            )
            for seq in step.threads
        ]
        with self._action_lock:
            self._active_actions = sub_runners
        for sub in sub_runners:
            sub.start()
        errors: list[Exception] = []
        for sub in sub_runners:
            sub.wait()
            if sub.state() == ActionState.FAILED and not errors:
                errors.append(sub.error())
                _TERMINAL = (ActionState.DONE, ActionState.FAILED, ActionState.CANCELLED)
                with self._action_lock:
                    for other in self._active_actions:
                        if other.state() not in _TERMINAL:
                            other.cancel()
        with self._action_lock:
            self._active_actions = []
        if errors:
            raise RuntimeError(f"Parallel thread failed: {errors[0]}")
        return all(s.state() == ActionState.DONE for s in sub_runners)

    def _build_action(self, step: ActionStep) -> BaseAction:
        """! Instantiate the action described by a step spec.

        @param step<ActionStep>: Step describing the action type and params.
        @return<BaseAction>: Constructed action instance.
        @raises ValueError: If the action type is not in _ACTION_REGISTRY.
        """
        factory = _ACTION_REGISTRY.get(step.action_type)
        if factory is None:
            raise ValueError(f"Unknown action type: {step.action_type!r}")
        return factory(self._controller, self._data_folder, step.params)
