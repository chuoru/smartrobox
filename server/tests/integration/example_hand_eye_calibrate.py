#!/usr/bin/env python3
##
# @file example_hand_eye_calibrate.py
#
# @brief Equipment-free hand-eye calibration via robot TCP correspondences.
#
#        Jog each arm's TCP to N positions visible in the head camera,
#        then left-click on the TCP tip in the live window.  For each click
#        the script records the 3D position in both the camera frame
#        (pixel_to_world with a 5×5 depth patch median) and the robot base
#        frame (tpos() in mm → metres).  After ≥ 4 samples per arm, press C
#        to fit rigid-body transforms T_cam_to_base via SVD Procrustes
#        regression and print paste-ready _LEFT_ARM_EXTRINSIC and
#        _RIGHT_ARM_EXTRINSIC literals.
#
#        Controls
#            Tab         switch active arm (left ↔ right)
#            Left-click  record current TCP as a correspondence
#            U           undo the last sample for the active arm
#            C           compute and print both extrinsic transforms
#            Q           quit
#
# @section author Author(s)
# - Created by chuoru on 2026/06/20.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# External library
import cv2
import numpy as np

# Internal library
from app.config import Config
from app.controller import Controller


_LEFT_ARM = "left_arm"
_RIGHT_ARM = "right_arm"
_HEAD_CAMERA = "head_camera"
_MIN_SAMPLES = 4
_PATCH_RADIUS = 2  # sample (2r+1)^2 pixels around each click for depth median

_DEVICE_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "projects", "anlab", "device.yaml")
)

# Mutable container used by the OpenCV mouse callback.
_click_state: list = [None]  # [0]: (u, v) | None


def _on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_state[0] = (x, y)


# ---------------------------------------------------------------------------
# Depth sampling
# ---------------------------------------------------------------------------

def _sample_cam_pt(
    ctrl: Controller, u: int, v: int
) -> list[float] | None:
    """! Median pixel_to_world over a (2r+1)×(2r+1) patch to reduce depth noise.

    @param ctrl<Controller>: Active controller.
    @param u<int>: Click column (pixels).
    @param v<int>: Click row (pixels).
    @return<list[float]|None>: [X, Y, Z] in camera frame (metres), or None.
    """
    pts = []
    r = _PATCH_RADIUS
    for dv in range(-r, r + 1):
        for du in range(-r, r + 1):
            pt = ctrl.execute(_HEAD_CAMERA, "pixel_to_world", u + du, v + dv)
            if pt is not None:
                pts.append(pt)
    if not pts:
        return None
    return np.median(np.array(pts, dtype=np.float64), axis=0).tolist()


# ---------------------------------------------------------------------------
# SVD rigid-body solver
# ---------------------------------------------------------------------------

def _fit_transform(
    cam_pts: list[list[float]],
    base_pts: list[list[float]],
) -> np.ndarray:
    """! Fit T_cam_to_base by SVD Procrustes registration.

    Minimises ||base_pt - (R @ cam_pt + t)|| over N point pairs.

    @param cam_pts<list[list[float]]>: N camera-frame 3-D points (metres).
    @param base_pts<list[list[float]]>: N base-frame 3-D points (metres).
    @return<np.ndarray>: 4×4 homogeneous T_cam_to_base (float64).
    """
    C = np.array(cam_pts, dtype=np.float64)
    B = np.array(base_pts, dtype=np.float64)

    mu_c = C.mean(axis=0)
    mu_b = B.mean(axis=0)

    H = (C - mu_c).T @ (B - mu_b)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:  # reflection guard
        Vt[-1] *= -1
        R = Vt.T @ U.T

    t = mu_b - R @ mu_c

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _reprojection_errors_mm(
    T: np.ndarray,
    cam_pts: list[list[float]],
    base_pts: list[list[float]],
) -> list[float]:
    """! Per-point reprojection error in mm.

    @param T<np.ndarray>: 4×4 T_cam_to_base.
    @param cam_pts<list[list[float]]>: Camera-frame points (metres).
    @param base_pts<list[list[float]]>: Base-frame points (metres).
    @return<list[float]>: ||T @ p_cam - p_base|| in mm, one per point.
    """
    R, t = T[:3, :3], T[:3, 3]
    errors = []
    for p_c, p_b in zip(cam_pts, base_pts):
        err = np.linalg.norm(R @ np.array(p_c) + t - np.array(p_b))
        errors.append(err * 1000.0)
    return errors


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_ui(
    frame: np.ndarray,
    corr_left: list[tuple],
    corr_right: list[tuple],
    active: str,
) -> None:
    """! Overlay recorded sample markers and key-binding guide.

    @param frame: BGR numpy array; modified in-place.
    @param corr_left<list[tuple]>: Collected (u, v, cam_pt, base_pt) for left arm.
    @param corr_right<list[tuple]>: Collected (u, v, cam_pt, base_pt) for right arm.
    @param active<str>: Currently active arm device name.
    """
    h = frame.shape[0]
    n_left = len(corr_left)
    n_right = len(corr_right)

    # Draw left-arm sample markers in cyan, right-arm in magenta.
    for i, (u, v, _, _) in enumerate(corr_left):
        cv2.circle(frame, (u, v), 6, (0, 255, 255), 2)
        cv2.putText(frame, f"L{i + 1}", (u + 8, v - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    for i, (u, v, _, _) in enumerate(corr_right):
        cv2.circle(frame, (u, v), 6, (255, 0, 255), 2)
        cv2.putText(frame, f"R{i + 1}", (u + 8, v - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

    left_label = "LEFT (active)" if active == _LEFT_ARM else "LEFT"
    right_label = "RIGHT (active)" if active == _RIGHT_ARM else "RIGHT"
    left_color = (0, 255, 0) if n_left >= _MIN_SAMPLES else (0, 140, 255)
    right_color = (0, 255, 0) if n_right >= _MIN_SAMPLES else (0, 140, 255)

    cv2.putText(frame, f"{left_label}: {n_left} / {_MIN_SAMPLES} min",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, left_color, 2)
    cv2.putText(frame, f"{right_label}: {n_right} / {_MIN_SAMPLES} min",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, right_color, 2)

    guide = [
        "Tab: switch arm  |  Left-click: record TCP  |  U: undo  |  C: compute  |  Q: quit",
        "Spread poses: near/far, left/right, up/down inside camera view",
    ]
    y = h - 10 - 18
    for line in reversed(guide):
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
        y -= 18


# ---------------------------------------------------------------------------
# Result printing
# ---------------------------------------------------------------------------

def _print_arm_result(
    label: str,
    const_name: str,
    corr: list[tuple],
) -> None:
    """! Fit and print a paste-ready extrinsic literal for one arm.

    @param label<str>: Human-readable arm label for console output.
    @param const_name<str>: Python constant name to emit (e.g. '_LEFT_ARM_EXTRINSIC').
    @param corr<list[tuple]>: Collected (u, v, cam_pt, base_pt) entries.
    """
    n = len(corr)
    if n < _MIN_SAMPLES:
        print(f"[calibrate] {label}: need ≥ {_MIN_SAMPLES} samples (have {n}) — skipped.")
        return

    cam_pts = [c[2] for c in corr]
    base_pts = [c[3] for c in corr]
    T = _fit_transform(cam_pts, base_pts)
    errors = _reprojection_errors_mm(T, cam_pts, base_pts)
    mean_err = sum(errors) / len(errors)
    max_err = max(errors)

    print(f"\n[calibrate] ── {label} ──────────────────────────────────────")
    print(f"[calibrate] Samples : {n}")
    print(f"[calibrate] Error   : mean={mean_err:.1f} mm  max={max_err:.1f} mm")
    for i, e in enumerate(errors):
        print(f"[calibrate]   #{i+1:2d}: {e:.1f} mm")

    print()
    print(f"{const_name} = [")
    for row in T.tolist():
        vals = ", ".join(f"{x:11.7f}" for x in row)
        print(f"    [{vals}],")
    print("]")
    print(f"[calibrate] ─────────────────────────────────────────────\n")

    if mean_err > 20.0:
        print(f"[calibrate] WARNING ({label}): mean error > 20 mm.")
        print("[calibrate]   • Click more precisely on the TCP tip.")
        print("[calibrate]   • Ensure TCP has valid depth (not too near/far).")
        print("[calibrate]   • Add more samples with better spatial spread.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """! Collect TCP correspondences for both arms and print fitted extrinsics."""
    ctrl = Controller(Config(_DEVICE_FILE))

    if not ctrl.open(_HEAD_CAMERA):
        print(f"[calibrate] Failed to open '{_HEAD_CAMERA}'")
        return
    if not ctrl.open(_LEFT_ARM):
        print(f"[calibrate] Failed to open '{_LEFT_ARM}'")
        ctrl.close(_HEAD_CAMERA)
        return
    if not ctrl.open(_RIGHT_ARM):
        print(f"[calibrate] Failed to open '{_RIGHT_ARM}'")
        ctrl.close(_LEFT_ARM)
        ctrl.close(_HEAD_CAMERA)
        return

    # (u, v, cam_pt_m, base_pt_m) per recorded sample, one list per arm.
    corr_left: list[tuple[int, int, list[float], list[float]]] = []
    corr_right: list[tuple[int, int, list[float], list[float]]] = []
    active = _LEFT_ARM

    cv2.namedWindow("Hand-eye calibrate")
    cv2.setMouseCallback("Hand-eye calibrate", _on_mouse)

    print("[calibrate] Jog one arm's TCP to a position visible in the head camera,")
    print("[calibrate] then left-click on the TCP tip in the window.")
    print(f"[calibrate] Collect {_MIN_SAMPLES}+ spread poses per arm, then press C.")
    print("[calibrate] Press Tab to switch between left and right arm.")
    print("[calibrate] Tip: vary X/Y/Z spread — avoid collinear clusters.")

    try:
        while True:
            frame = ctrl.execute(_HEAD_CAMERA, "get_color_frame")
            if frame is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            corr_active = corr_left if active == _LEFT_ARM else corr_right

            # Handle new click
            click = _click_state[0]
            if click is not None:
                _click_state[0] = None
                u, v = click

                cam_pt = _sample_cam_pt(ctrl, u, v)
                if cam_pt is None:
                    print(f"[calibrate] No valid depth near ({u}, {v}) — retry.")
                else:
                    ret, tcp_mm = ctrl.execute(active, "tpos")
                    if ret != 0:
                        print(f"[calibrate] tpos() returned error {ret} — skipping.")
                    else:
                        base_pt = [tcp_mm[0] / 1000.0,
                                   tcp_mm[1] / 1000.0,
                                   tcp_mm[2] / 1000.0]
                        corr_active.append((u, v, cam_pt, base_pt))
                        n = len(corr_active)
                        arm_label = "LEFT" if active == _LEFT_ARM else "RIGHT"
                        print(
                            f"[calibrate] {arm_label} #{n:2d}  pixel=({u},{v})"
                            f"  cam=[{cam_pt[0]:.3f}, {cam_pt[1]:.3f}, {cam_pt[2]:.3f}] m"
                            f"  base=[{base_pt[0]:.3f}, {base_pt[1]:.3f}, {base_pt[2]:.3f}] m"
                        )

            display = frame.copy()
            _draw_ui(display, corr_left, corr_right, active)
            cv2.imshow("Hand-eye calibrate", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == 9:  # Tab
                active = _RIGHT_ARM if active == _LEFT_ARM else _LEFT_ARM
                arm_label = "LEFT" if active == _LEFT_ARM else "RIGHT"
                print(f"[calibrate] Active arm switched to {arm_label}.")

            if key == ord("u"):
                if corr_active:
                    corr_active.pop()
                    arm_label = "LEFT" if active == _LEFT_ARM else "RIGHT"
                    n_left = len(corr_left)
                    n_right = len(corr_right)
                    print(
                        f"[calibrate] {arm_label} undone — "
                        f"LEFT {n_left}, RIGHT {n_right} samples remain."
                    )
                else:
                    print("[calibrate] Nothing to undo for the active arm.")

            if key == ord("c"):
                print("\n# Paste into example_estimate_pose_action.py:")
                _print_arm_result("LEFT ARM", "_LEFT_ARM_EXTRINSIC", corr_left)
                _print_arm_result("RIGHT ARM", "_RIGHT_ARM_EXTRINSIC", corr_right)

    finally:
        ctrl.close(_RIGHT_ARM)
        ctrl.close(_LEFT_ARM)
        ctrl.close(_HEAD_CAMERA)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
