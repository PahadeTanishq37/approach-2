"""
realsense_measure/fullbody/capture_ui.py

Real-time operator preview and HUD rendering for Stage 27 CaptureController.
Strictly imports cv2, numpy, and standard library only.
Headless-testable rendering functions without GUI or blocking dependencies.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def colorize_depth_m(
    depth_m: np.ndarray,
    min_m: float = 0.30,
    max_m: float = 4.50,
) -> np.ndarray:
    """Convert float32 metric depth array to BGR JET colormap visualization."""
    valid_mask = (depth_m > 0.05) & np.isfinite(depth_m)
    depth_clean = np.where(valid_mask, depth_m, min_m)
    depth_clamped = np.clip(depth_clean, min_m, max_m)
    norm = ((depth_clamped - min_m) / max(1e-3, (max_m - min_m)) * 255.0).astype(np.uint8)
    # Invert so near is warm (red) and far is cool (blue)
    norm_inv = 255 - norm
    colorized = cv2.applyColorMap(norm_inv, cv2.COLORMAP_JET)
    colorized[~valid_mask] = [0, 0, 0]
    return colorized


def render_frame(
    state_snapshot: Dict[str, Any],
    color: np.ndarray,
    depth_filtered: Optional[np.ndarray],
) -> np.ndarray:
    """
    Pure headless rendering function creating operator HUD.
    Returns composite BGR image of shape (H, W*2, 3).
    """
    h, w = color.shape[:2]

    # Generate colorized depth
    if depth_filtered is not None:
        if depth_filtered.ndim == 3:
            depth_vis = depth_filtered if depth_filtered.shape[:2] == (h, w) else cv2.resize(depth_filtered, (w, h))
        else:
            if depth_filtered.shape[:2] != (h, w):
                depth_resized = cv2.resize(depth_filtered, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                depth_resized = depth_filtered
            depth_vis = colorize_depth_m(depth_resized)
    else:
        depth_vis = np.zeros_like(color)

    # Side-by-side composite
    composite = np.hstack([color.copy(), depth_vis])
    comp_w = composite.shape[1]

    # Extract state snapshot data
    state = str(state_snapshot.get("state", "WAIT_FOR_SUBJECT"))
    stop_idx = int(state_snapshot.get("stop_index", 0))
    n_stops = int(state_snapshot.get("n_stops", 8))
    turn_dir = str(state_snapshot.get("turn_direction", "cw")).lower()
    retake_idx = int(state_snapshot.get("current_retake", 0))

    dist_m = state_snapshot.get("distance_m")
    valid_pct = state_snapshot.get("valid_depth_pct")
    motion_mm = state_snapshot.get("motion_mm")
    head_ok = bool(state_snapshot.get("head_visible", True))
    feet_ok = bool(state_snapshot.get("feet_visible", True))
    head_margin_pct = state_snapshot.get("head_margin_pct")
    sole_margin_pct = state_snapshot.get("sole_margin_pct")
    clip_pct = state_snapshot.get("clip_pct")
    burst_prog = float(np.clip(state_snapshot.get("burst_progress", 0.0), 0.0, 1.0))
    rejection_reason = state_snapshot.get("last_rejection_reason")

    # 1. Foreground outline and Bounding Box on RGB feed (left side)
    fg_m = state_snapshot.get("fg_mask")
    if fg_m is not None and isinstance(fg_m, np.ndarray):
        fg_u8 = (fg_m > 0).astype(np.uint8)
        if fg_u8.shape[:2] != (h, w):
            fg_u8 = cv2.resize(fg_u8, (w, h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(fg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(composite[:, :w], contours, -1, (0, 255, 0), 2)

    bbox = state_snapshot.get("bbox")
    if bbox is not None and len(bbox) == 4:
        bx0, by0, bx1, by1 = bbox
        cv2.rectangle(composite[:, :w], (bx0, by0), (bx1, by1), (0, 255, 255), 2)

    # 4% Head and Feet margin guidelines on RGB feed (left side)
    y_head_4 = int(0.04 * h)
    y_sole_4 = int(0.96 * h)
    cv2.line(composite[:, :w], (0, y_head_4), (w, y_head_4), (0, 165, 255), 1)
    cv2.putText(composite[:, :w], "HEAD MARGIN (4%)", (15, max(12, y_head_4 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 165, 255), 1)
    cv2.line(composite[:, :w], (0, y_sole_4), (w, y_sole_4), (0, 165, 255), 1)
    cv2.putText(composite[:, :w], "SOLE MARGIN (4%)", (15, min(h - 5, y_sole_4 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 165, 255), 1)

    # 2. Header banner with semi-transparent overlay
    banner_h = 80
    overlay = composite.copy()
    cv2.rectangle(overlay, (0, 0), (comp_w, banner_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, composite, 0.25, 0, composite)

    # State colors
    state_colors = {
        "WAIT_FOR_SUBJECT": (0, 200, 255),  # Yellow
        "POSITION_CHECK": (0, 200, 255),
        "TURN_PROMPT": (255, 200, 0),     # Cyan
        "WAIT_STILL": (255, 150, 0),
        "CAPTURE_BURST": (0, 255, 255),   # Yellow
        "EVALUATE": (200, 200, 200),
        "ACCEPT": (0, 255, 0),            # Green
        "DONE": (0, 255, 0),
        "RETAKE": (0, 0, 255),            # Red
        "ABORT": (0, 0, 255),
        "PREVIEW_ONLY": (255, 200, 0),    # Cyan/Yellow
    }
    state_color = state_colors.get(state, (255, 255, 255))

    # Large state text
    cv2.putText(composite, f"STATE: {state}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)

    # Stop counter
    stop_text = f"Stop {stop_idx + 1} of {n_stops}"
    if retake_idx > 0:
        stop_text += f" (Retake {retake_idx})"
    cv2.putText(composite, stop_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

    # Turning instruction text (computed from turn_direction, turntable, and target angle)
    step_deg = 360.0 / max(1, n_stops)
    target_angle = (stop_idx * step_deg) % 360.0
    turntable = bool(state_snapshot.get("turntable", False))

    if turntable:
        instr_text = f"Turntable Target: {target_angle:.0f} deg (step {step_deg:.0f} deg)"
    else:
        if turn_dir == "cw":
            instr_text = f"Turn {step_deg:.0f} deg to RIGHT -> Target {target_angle:.0f} deg"
        else:
            instr_text = f"Turn {step_deg:.0f} deg to LEFT -> Target {target_angle:.0f} deg"

    (txt_w, txt_h), _ = cv2.getTextSize(instr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    instr_x = max(w + 5, comp_w - txt_w - 15)
    cv2.putText(composite, instr_text, (instr_x, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # Per-stop on-screen prompt: "hold still" countdown (2.0s) or turn prompt (45 deg)
    if state == "WAIT_STILL":
        still_cur = int(state_snapshot.get("stillness_frames_current", 0))
        still_tgt = int(state_snapshot.get("stillness_frames_target", 60))
        rem_frames = max(0, still_tgt - still_cur)
        rem_s = rem_frames / 30.0
        countdown_text = f"HOLD STILL: {rem_s:.1f}s ({still_cur}/{still_tgt})"
        (cd_w, cd_h), _ = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cd_x = max(20, (comp_w - cd_w) // 2)
        cv2.putText(composite, countdown_text, (cd_x, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)
    elif state == "TURN_PROMPT":
        turn_prompt_text = f"Stop {stop_idx + 1}/{n_stops}: turn to tick {stop_idx + 1} ({target_angle:.0f} deg) [Press SPACE or wait]"
        (tp_w, tp_h), _ = cv2.getTextSize(turn_prompt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
        tp_x = max(20, (comp_w - tp_w) // 2)
        cv2.putText(composite, turn_prompt_text, (tp_x, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)

    # Active orientation indicator (e.g. Orientation: PORTRAIT CCW or LANDSCAPE)
    orient = str(state_snapshot.get("orientation", "")).strip().lower()
    if orient in ("cw", "ccw"):
        orient_text = f"Orientation: PORTRAIT {orient.upper()}"
    elif orient == "none":
        orient_text = "Orientation: LANDSCAPE"
    elif orient:
        orient_text = f"Orientation: {orient.upper()}"
    else:
        orient_text = ""

    if orient_text:
        (o_w, o_h), _ = cv2.getTextSize(orient_text, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
        orient_x = max(w + 5, comp_w - o_w - 15)
        cv2.putText(composite, orient_text, (orient_x, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)

    # 3. Live metrics HUD bar
    hud_y = banner_h + 25
    x_off = 20

    # Distance metric [capture_dist_min_m - capture_dist_max_m] (default [1.70 - 2.40])
    d_min = float(state_snapshot.get("dist_min_m", 1.70))
    d_max = float(state_snapshot.get("dist_max_m", 2.40))
    if dist_m is not None:
        d_color = (0, 255, 0) if (d_min <= dist_m <= d_max) else (0, 0, 255)
        d_str = f"Dist: {dist_m:.2f}m"
    else:
        d_color = (180, 180, 180)
        d_str = "Dist: --"
    cv2.putText(composite, d_str, (x_off, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, d_color, 2)
    x_off += 140

    # Head Margin % (>= 4.0%)
    if head_margin_pct is not None:
        hm_color = (0, 255, 0) if head_margin_pct >= 4.0 else (0, 0, 255)
        hm_str = f"Head: {head_margin_pct:.1f}%"
    else:
        hm_color = (180, 180, 180)
        hm_str = f"Head: {'OK' if head_ok else 'CLIP'}"
    cv2.putText(composite, hm_str, (x_off, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hm_color, 2)
    x_off += 150

    # Sole Margin % (>= 4.0%)
    if sole_margin_pct is not None:
        sm_color = (0, 255, 0) if sole_margin_pct >= 4.0 else (0, 0, 255)
        sm_str = f"Sole: {sole_margin_pct:.1f}%"
    else:
        sm_color = (180, 180, 180)
        sm_str = f"Sole: {'OK' if feet_ok else 'CLIP'}"
    cv2.putText(composite, sm_str, (x_off, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, sm_color, 2)
    x_off += 150

    # Side Clip %
    if clip_pct is not None:
        cl_color = (0, 255, 0) if clip_pct <= 0.0 else (0, 0, 255)
        cl_str = f"Clip: {clip_pct:.1f}%"
    else:
        cl_color = (180, 180, 180)
        cl_str = "Clip: --"
    cv2.putText(composite, cl_str, (x_off, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cl_color, 2)
    x_off += 140

    # Motion mm (<= 10.0 mm)
    if motion_mm is not None:
        m_color = (0, 255, 0) if motion_mm <= 10.0 else (0, 0, 255)
        m_str = f"Motion: {motion_mm:.1f}mm"
    else:
        m_color = (180, 180, 180)
        m_str = "Motion: --"
    cv2.putText(composite, m_str, (x_off, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, m_color, 2)

    # 4. Last rejection reason
    if rejection_reason:
        cv2.putText(
            composite,
            f"REJECTED: {rejection_reason}",
            (20, h - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )

    # 5. Burst progress bar at bottom
    bar_y0, bar_y1 = h - 40, h - 25
    bar_x0, bar_x1 = 20, comp_w - 20
    bar_total_w = bar_x1 - bar_x0
    # Background frame
    cv2.rectangle(composite, (bar_x0, bar_y0), (bar_x1, bar_y1), (60, 60, 60), -1)
    if burst_prog > 0.0:
        fill_w = int(bar_total_w * burst_prog)
        cv2.rectangle(composite, (bar_x0, bar_y0), (bar_x0 + fill_w, bar_y1), (0, 255, 0), -1)
    cv2.rectangle(composite, (bar_x0, bar_y0), (bar_x1, bar_y1), (180, 180, 180), 1)

    # 6. Keybindings prompt
    keys_text = "[SPACE] Confirm  |  [R] Retake  |  [S] Skip  |  [Q/ESC] Abort"
    cv2.putText(composite, keys_text, (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return composite


def run_ui_loop(
    controller: Any,
    window_name: str = "Full-Body 8-Stop Turntable Capture",
    headless: bool = False,
    max_steps: Optional[int] = None,
) -> None:
    """
    Drive controller.step() with live UI preview and keyboard input handling.
    In headless mode, runs pure stepping without calling cv2.imshow or cv2.waitKey.
    """
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    step_count = 0
    last_console_log_t = 0.0
    try:
        while True:
            now = time.time()
            state = controller.step(now)
            step_count += 1

            if state in ("DONE", "ABORT"):
                break

            if max_steps is not None and step_count >= max_steps:
                break

            # 2-second periodic console log
            if (now - last_console_log_t) >= 2.0:
                last_console_log_t = now
                st_name = controller.state.value if hasattr(controller.state, "value") else str(controller.state)
                d_val = getattr(controller, "_latest_distance_m", None)
                d_str = f"{d_val:.2f}m" if d_val is not None else "--"
                hm_val = getattr(controller, "_latest_head_margin_pct", None)
                hm_str = f"{hm_val:.1f}%" if hm_val is not None else "--"
                sm_val = getattr(controller, "_latest_sole_margin_pct", None)
                sm_str = f"{sm_val:.1f}%" if sm_val is not None else "--"
                cl_val = getattr(controller, "_latest_clip_pct", None)
                cl_str = f"{cl_val:.1f}%" if cl_val is not None else "--"
                bbox_val = getattr(controller, "_latest_bbox", None)
                bbox_str = str(bbox_val) if bbox_val is not None else "--"
                reason_str = getattr(controller, "_latest_rejection_reason", None) or "None"
                print(
                    f"[{st_name:<16}] Dist: {d_str} | Head: {hm_str} | Sole: {sm_str} | "
                    f"Clip: {cl_str} | BBox: {bbox_str} | Reason: {reason_str}"
                )

            if not headless:
                # Build snapshot from controller
                snapshot = {
                    "state": controller.state.value if hasattr(controller.state, "value") else str(controller.state),
                    "stop_index": getattr(controller, "current_stop", 0),
                    "n_stops": getattr(controller, "n_stops", 8),
                    "turn_direction": getattr(controller.writer, "turn_direction", "cw"),
                    "turntable": getattr(controller, "turntable", False),
                    "stillness_frames_current": len(getattr(controller, "_stillness_window", [])),
                    "stillness_frames_target": getattr(controller.thresholds, "stillness_frames", 60),
                    "orientation": getattr(controller, "portrait_rotation", getattr(getattr(controller, "writer", None), "camera_portrait_rotation", "none")),
                    "current_retake": getattr(controller, "current_retake", 0),
                    "distance_m": getattr(controller, "_latest_distance_m", None),
                    "valid_depth_pct": getattr(controller, "_latest_valid_depth_pct", None),
                    "motion_mm": getattr(controller, "_latest_motion_mm", None),
                    "head_visible": getattr(controller, "_latest_head_visible", True),
                    "feet_visible": getattr(controller, "_latest_feet_visible", True),
                    "head_margin_pct": getattr(controller, "_latest_head_margin_pct", None),
                    "sole_margin_pct": getattr(controller, "_latest_sole_margin_pct", None),
                    "clip_pct": getattr(controller, "_latest_clip_pct", None),
                    "fg_mask": getattr(controller, "_latest_fg_mask", None),
                    "bbox": getattr(controller, "_latest_bbox", None),
                    "burst_progress": len(getattr(controller, "_burst_depths", [])) / max(1, controller.thresholds.burst_frames),
                    "last_rejection_reason": getattr(controller, "_latest_rejection_reason", None),
                }

                # Get latest frame for display if available
                color = getattr(controller, "_latest_color", None)
                depth = getattr(controller, "_latest_filtered", None)
                if color is None:
                    color = np.zeros((480, 270, 3), dtype=np.uint8)

                frame = render_frame(snapshot, color, depth)
                disp_frame = frame
                if frame.shape[1] > 1200:
                    scale = 1200.0 / frame.shape[1]
                    target_w = 1200
                    target_h = int(round(frame.shape[0] * scale))
                    disp_frame = cv2.resize(
                        frame,
                        (target_w, target_h),
                        interpolation=cv2.INTER_AREA,
                    )
                cv2.imshow(window_name, disp_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # SPACE
                    controller.send_command("confirm")
                elif key in (ord("r"), ord("R")):
                    controller.send_command("retake")
                elif key in (ord("s"), ord("S")):
                    controller.send_command("skip")
                elif key in (ord("q"), ord("Q"), 27):  # Q or ESC
                    controller.send_command("abort")

            time.sleep(0.01)

    finally:
        if not headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

