"""
realsense_measure/fullbody/capture_controller.py

Stage 27 Live Eight-Stop Capture Controller.
Pure logic state machine for orchestrating multi-stop full-body turntable capture.
Zero dependencies on Open3D, MediaPipe, or pyrealsense2 at import time.
No sleeping, no UI rendering (cv2.imshow), and no stdin input() inside the controller.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import cv2
import numpy as np

from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.orientation import nominal_angle_deg
from realsense_measure.fullbody.session_format import AttemptHandle, SessionWriter

logger = logging.getLogger(__name__)


class CaptureCamera(Protocol):
    """Protocol for capture cameras (hardware or synthetic simulation)."""

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def read_raw_filtered(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], float]:
        """Returns (color_bgr, raw_depth, filtered_depth|None, timestamp_s)."""
        ...

    @property
    def intrinsics(self) -> Dict[str, Any]:
        ...

    @property
    def depth_scale(self) -> float:
        ...

    @property
    def depth_units(self) -> str:
        ...

    def device_info(self) -> Dict[str, Any]:
        ...


@dataclass(frozen=True)
class CaptureThresholds:
    """
    Thresholds and timing parameters for full-body capture controller.
    Comments distinguish existing defaults from new/untested Stage 27 thresholds.
    """

    # Burst frame count: default 15 frames
    burst_frames: int = 15

    # Maximum retake attempts allowed per stop before aborting
    max_retakes: int = 3

    # Consecutive frames required under motion limit to trigger capture (2.0 s @ 30 fps = 60 frames)
    stillness_frames: int = 60

    # Motion/sway limit in mm (enforced at 10.0 mm)
    motion_limit_mm: float = 10.0

    # Existing default: capture_session.CaptureQualityChecker.depth_valid_frac_threshold (capture_session.py:364)
    valid_depth_fraction: float = 0.30

    # Acceptable subject standoff window [capture_dist_min_m, capture_dist_max_m] from FullBodyConfig
    dist_min_m: float = field(default_factory=lambda: FullBodyConfig().capture_dist_min_m)
    dist_max_m: float = field(default_factory=lambda: FullBodyConfig().capture_dist_max_m)

    # NEW/UNTESTED: Require head (top border) and feet (bottom border) to remain unclipped
    require_head_feet: bool = True

    # NEW/UNTESTED: Maximum allowable fraction of body pixels touching image borders
    max_border_clip_frac: float = 0.02

    # NEW/UNTESTED: Timeout for turn prompt before auto-proceeding to wait_still (seconds)
    turn_prompt_timeout_s: float = 10.0

    # NEW/UNTESTED: Timeout for position check before reverting to wait_for_subject (seconds)
    position_check_timeout_s: float = 15.0

    # NEW/UNTESTED: Timeout for burst acquisition (seconds)
    burst_timeout_s: float = 5.0


class ControllerState(str, Enum):
    """States of the Stage 27 capture state machine."""

    WAIT_FOR_SUBJECT = "WAIT_FOR_SUBJECT"
    POSITION_CHECK = "POSITION_CHECK"
    TURN_PROMPT = "TURN_PROMPT"
    WAIT_STILL = "WAIT_STILL"
    CAPTURE_BURST = "CAPTURE_BURST"
    EVALUATE = "EVALUATE"
    ACCEPT = "ACCEPT"
    RETAKE = "RETAKE"
    DONE = "DONE"
    ABORT = "ABORT"


def default_quality_fn(
    burst_colors: List[np.ndarray],
    burst_depths: List[np.ndarray],
    depth_scale: float = 1.0,
    depth_units: str = "m",
    thresholds: Optional[CaptureThresholds] = None,
    bg_depth: Optional[np.ndarray] = None,
    cfg: Optional[FullBodyConfig] = None,
    intrinsics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Numpy-only quality assessment for a captured burst:
    - Valid depth fraction in central ROI
    - Temporal depth standard deviation (mm)
    - Sway: median per-frame shift of the foreground centroid and torso-region depth (mm)
    - Border-touch fraction and clipping diagnostics
    """
    th = thresholds or CaptureThresholds()
    n_frames = len(burst_depths)
    if n_frames < th.burst_frames:
        return {
            "is_flagged": True,
            "reasons": [
                f"Burst frame count ({n_frames}) is less than required burst_frames ({th.burst_frames})"
            ],
            "sway_mm": 999.0,
            "motion_mm": 999.0,
            "temporal_depth_std_mm": 999.0,
            "valid_depth_fraction": 0.0,
            "border_clip_fraction": 0.0,
            "head_clipped": False,
            "feet_clipped": False,
        }

    # Normalize depth to metres float32 for computation
    depth_m_list: List[np.ndarray] = []
    for d in burst_depths:
        if depth_units == "sensor_units":
            depth_m_list.append(d.astype(np.float32) * depth_scale)
        else:
            depth_m_list.append(d.astype(np.float32))

    h, w = depth_m_list[0].shape[:2]
    # Central ROI (middle 60% of image)
    r_y0, r_y1 = int(0.20 * h), int(0.80 * h)
    r_x0, r_x1 = int(0.20 * w), int(0.80 * w)

    valid_fractions: List[float] = []
    border_touch_fractions: List[float] = []
    head_clipped_count = 0
    feet_clipped_count = 0

    # 1. Compute sway as the median per-frame shift of the foreground centroid
    # and torso-region depth (using compute_foreground_mask against background session)
    from realsense_measure.fullbody.fusion import compute_foreground_mask
    cfg_eval = cfg or FullBodyConfig()

    fx = float(intrinsics.get("fx", max(h, w))) if intrinsics else float(max(h, w))
    fy = float(intrinsics.get("fy", max(h, w))) if intrinsics else float(max(h, w))
    cx = float(intrinsics.get("cx", w / 2.0)) if intrinsics else float(w / 2.0)
    cy = float(intrinsics.get("cy", h / 2.0)) if intrinsics else float(h / 2.0)

    per_frame_centroids: List[Tuple[float, float, float]] = []
    per_frame_torso_z: List[float] = []

    m_y = max(1, int(0.02 * h))
    m_x = max(1, int(0.02 * w))
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:m_y, :] = True
    border_mask[-m_y:, :] = True
    border_mask[:, :m_x] = True
    border_mask[:, -m_x:] = True

    for d_m in depth_m_list:
        valid_mask = (d_m > 0.3) & (d_m < 3.8)
        roi_mask = valid_mask[r_y0:r_y1, r_x0:r_x1]
        roi_valid_frac = float(np.count_nonzero(roi_mask)) / float(roi_mask.size)
        valid_fractions.append(roi_valid_frac)

        total_valid = np.count_nonzero(valid_mask)
        if total_valid > 0:
            touching = np.count_nonzero(valid_mask & border_mask)
            border_touch_fractions.append(float(touching) / float(total_valid))
        else:
            border_touch_fractions.append(1.0)

        # Check top row (head) and bottom row (feet)
        if np.any(valid_mask[:m_y, :]):
            head_clipped_count += 1
        if np.any(valid_mask[-m_y:, :]):
            feet_clipped_count += 1

        # Foreground mask via background subtraction if background is available
        if bg_depth is not None and bg_depth.shape[:2] == (h, w):
            fg_m = compute_foreground_mask(d_m, bg_depth, cfg_eval)
        else:
            med_z = float(np.median(d_m[valid_mask])) if total_valid > 50 else 2.0
            fg_m = valid_mask & (np.abs(d_m - med_z) < 0.45)

        ys, xs = np.nonzero(fg_m)
        if len(ys) > 50:
            u_c = float(np.mean(xs))
            v_c = float(np.mean(ys))
            z_c = float(np.median(d_m[ys, xs]))
            x_m = (u_c - cx) * (z_c / fx)
            y_m = (v_c - cy) * (z_c / fy)
            per_frame_centroids.append((x_m, y_m, z_c))

            # Torso: 20% to 65% vertical span, central 60% horizontal span
            y_min, y_max = int(np.min(ys)), int(np.max(ys))
            x_min, x_max = int(np.min(xs)), int(np.max(xs))
            span_h = max(10, y_max - y_min)
            span_w = max(10, x_max - x_min)
            t_y0 = y_min + int(0.20 * span_h)
            t_y1 = y_min + int(0.65 * span_h)
            t_x0 = x_min + int(0.20 * span_w)
            t_x1 = x_min + int(0.80 * span_w)
            torso_sub = d_m[t_y0:t_y1, t_x0:t_x1]
            torso_val = torso_sub[(torso_sub > 0.4) & (torso_sub < 3.8)]
            if len(torso_val) > 20:
                per_frame_torso_z.append(float(np.median(torso_val)))
            else:
                per_frame_torso_z.append(z_c)
        else:
            per_frame_centroids.append((0.0, 0.0, 0.0))
            per_frame_torso_z.append(0.0)

    # Compute consecutive per-frame shifts (sway)
    per_frame_shifts_mm: List[float] = []
    if len(per_frame_centroids) > 1:
        for i in range(len(per_frame_centroids) - 1):
            c0 = per_frame_centroids[i]
            c1 = per_frame_centroids[i + 1]
            dx_mm = (c1[0] - c0[0]) * 1000.0
            dy_mm = (c1[1] - c0[1]) * 1000.0
            dz_torso_mm = (per_frame_torso_z[i + 1] - per_frame_torso_z[i]) * 1000.0
            shift_mm = float(np.sqrt(dx_mm**2 + dy_mm**2 + dz_torso_mm**2))
            per_frame_shifts_mm.append(shift_mm)
        sway_mm = float(np.median(per_frame_shifts_mm))
    else:
        sway_mm = 0.0

    # Temporal depth std in mm across stack
    depth_stack = np.stack(depth_m_list, axis=0)
    valid_stack = (depth_stack > 0.3) & (depth_stack < 3.8)
    valid_count_per_pix = np.sum(valid_stack, axis=0)
    stable_pix = valid_count_per_pix >= max(3, int(0.5 * n_frames))

    if np.any(stable_pix):
        masked_d = np.where(valid_stack, depth_stack, np.nan)
        temporal_std = np.nanstd(masked_d[:, stable_pix], axis=0)
        median_depth_std_mm = float(np.nanmedian(temporal_std) * 1000.0)
    else:
        median_depth_std_mm = 999.0

    mean_valid_frac = float(np.mean(valid_fractions))
    mean_border_clip = float(np.mean(border_touch_fractions))
    head_clipped = head_clipped_count > (n_frames // 2)
    feet_clipped = feet_clipped_count > (n_frames // 2)

    reasons: List[str] = []
    if sway_mm > th.motion_limit_mm:
        reasons.append(
            f"Subject sway {sway_mm:.1f} mm exceeds motion limit {th.motion_limit_mm:.1f} mm"
        )
    if mean_valid_frac < th.valid_depth_fraction:
        reasons.append(
            f"Valid depth fraction {mean_valid_frac:.2f} below threshold {th.valid_depth_fraction:.2f}"
        )
    if mean_border_clip > th.max_border_clip_frac:
        reasons.append(
            f"Border clipping fraction {mean_border_clip:.3f} exceeds limit {th.max_border_clip_frac:.3f}"
        )
    if th.require_head_feet and head_clipped:
        reasons.append("Head clipped by top image border")
    if th.require_head_feet and feet_clipped:
        reasons.append("Feet clipped by bottom image border")

    is_flagged = len(reasons) > 0

    return {
        "is_flagged": is_flagged,
        "reasons": reasons,
        "sway_mm": sway_mm,
        "motion_mm": sway_mm,
        "temporal_depth_std_mm": median_depth_std_mm,
        "valid_depth_fraction": mean_valid_frac,
        "border_clip_fraction": mean_border_clip,
        "head_clipped": head_clipped,
        "feet_clipped": feet_clipped,
    }


class CaptureController:
    """
    Pure logic state machine driving the 8-stop turntable capture session.
    No sleeping, no UI rendering, and no stdin input().
    """

    def __init__(
        self,
        camera: CaptureCamera,
        writer: SessionWriter,
        thresholds: Optional[CaptureThresholds] = None,
        landmark_fn: Optional[Callable[[np.ndarray, np.ndarray], Optional[Dict[str, Any]]]] = None,
        quality_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        bg_depth: Optional[np.ndarray] = None,
        cfg: Optional[FullBodyConfig] = None,
        turntable: bool = False,
    ):
        self.camera = camera
        self.writer = writer
        self.thresholds = thresholds or CaptureThresholds()
        self.landmark_fn = landmark_fn
        self.quality_fn = quality_fn or default_quality_fn
        self.bg_depth = bg_depth
        self.cfg = cfg or FullBodyConfig()
        self.turntable = bool(turntable)

        self.state = ControllerState.WAIT_FOR_SUBJECT
        self.current_stop = 0
        self.current_retake = 0
        self.n_stops = writer.n_stops
        self.portrait_rotation = getattr(
            camera, "portrait_rotation", getattr(writer, "camera_portrait_rotation", "none")
        )

        # Injected command queue ("confirm", "skip", "abort", "retake")
        self.command_queue: deque[str] = deque()

        # Internal state tracking
        self._state_entry_time: float = 0.0
        self._wait_subject_consecutive: int = 0
        self._stillness_window: deque[Tuple[float, float, float]] = deque(
            maxlen=self.thresholds.stillness_frames
        )
        self._burst_colors: List[np.ndarray] = []
        self._burst_depths: List[np.ndarray] = []
        self._burst_filtered: Optional[np.ndarray] = None
        self._burst_timestamps: List[float] = []
        self._latest_eval_report: Dict[str, Any] = {}
        self._latest_eval_landmarks: Optional[Dict[str, Any]] = None

        # Live telemetry cached for UI preview
        self._latest_color: Optional[np.ndarray] = None
        self._latest_filtered: Optional[np.ndarray] = None
        self._latest_distance_m: Optional[float] = None
        self._latest_valid_depth_pct: Optional[float] = None
        self._latest_motion_mm: Optional[float] = None
        self._latest_head_visible: bool = True
        self._latest_feet_visible: bool = True
        self._latest_rejection_reason: Optional[str] = None

    def send_command(self, cmd: str) -> None:
        """Inject an operator command into the controller queue."""
        clean_cmd = str(cmd).strip().lower()
        self.command_queue.append(clean_cmd)

    def _pop_command(self) -> Optional[str]:
        if self.command_queue:
            return self.command_queue.popleft()
        return None

    def _transition(self, new_state: ControllerState, now: float) -> None:
        logger.info(
            f"CaptureController transition: {self.state} -> {new_state} (stop {self.current_stop}, retake {self.current_retake})"
        )
        self.state = new_state
        self._state_entry_time = now

    def step(self, now: float) -> ControllerState:
        """
        Advance state machine logic by one step at current timestamp `now` (seconds).
        Returns the current state after step evaluation.
        """
        # Handle universal operator commands
        cmd = self._pop_command()
        if cmd == "abort":
            self._transition(ControllerState.ABORT, now)
            return self.state

        # Terminal states
        if self.state in (ControllerState.DONE, ControllerState.ABORT):
            return self.state

        # Read camera frame
        color, depth, filtered, ts = self.camera.read_raw_filtered()
        if color is not None:
            self._latest_color = color
        if filtered is not None:
            self._latest_filtered = filtered

        # State dispatch
        if self.state == ControllerState.WAIT_FOR_SUBJECT:
            self._step_wait_for_subject(now, color, depth, cmd)

        elif self.state == ControllerState.POSITION_CHECK:
            self._step_position_check(now, color, depth, cmd)

        elif self.state == ControllerState.TURN_PROMPT:
            self._step_turn_prompt(now, cmd)

        elif self.state == ControllerState.WAIT_STILL:
            self._step_wait_still(now, depth, cmd)

        elif self.state == ControllerState.CAPTURE_BURST:
            self._step_capture_burst(now, color, depth, filtered, ts)

        elif self.state == ControllerState.EVALUATE:
            self._step_evaluate(now)

        elif self.state == ControllerState.ACCEPT:
            self._step_accept(now)

        elif self.state == ControllerState.RETAKE:
            self._step_retake(now, cmd)

        return self.state

    # -------------------------------------------------------------------------
    # State handlers
    # -------------------------------------------------------------------------

    def _step_wait_for_subject(self, now: float, color: Optional[np.ndarray], depth: Optional[np.ndarray], cmd: Optional[str] = None) -> None:
        if cmd == "confirm":
            self._transition(ControllerState.POSITION_CHECK, now)
            return

        if depth is None:
            self._wait_subject_consecutive = 0
            return

        h, w = depth.shape[:2]
        d_m = depth.astype(np.float32) * (self.camera.depth_scale if self.camera.depth_units == "sensor_units" else 1.0)
        roi = d_m[int(0.25 * h) : int(0.75 * h), int(0.25 * w) : int(0.75 * w)]
        valid_frac = float(np.count_nonzero((roi > 0.3) & (roi < 3.5))) / float(roi.size)

        if valid_frac >= self.thresholds.valid_depth_fraction:
            self._wait_subject_consecutive += 1
            if self._wait_subject_consecutive >= 5:
                self._wait_subject_consecutive = 0
                self._transition(ControllerState.POSITION_CHECK, now)
        else:
            self._wait_subject_consecutive = 0

    def _step_position_check(self, now: float, color: Optional[np.ndarray], depth: Optional[np.ndarray], cmd: Optional[str] = None) -> None:
        if cmd == "confirm":
            self._transition(ControllerState.TURN_PROMPT, now)
            return

        if depth is None:
            if (now - self._state_entry_time) > self.thresholds.position_check_timeout_s:
                self._transition(ControllerState.WAIT_FOR_SUBJECT, now)
            return

        h, w = depth.shape[:2]
        d_m = depth.astype(np.float32) * (self.camera.depth_scale if self.camera.depth_units == "sensor_units" else 1.0)
        valid_mask = (d_m > 0.3) & (d_m < 3.5)
        valid_count = np.count_nonzero(valid_mask)

        if valid_count < 100:
            if (now - self._state_entry_time) > self.thresholds.position_check_timeout_s:
                self._transition(ControllerState.WAIT_FOR_SUBJECT, now)
            return

        # Distance window check
        med_dist = float(np.median(d_m[valid_mask]))
        self._latest_distance_m = med_dist
        self._latest_valid_depth_pct = float(valid_count) / float(d_m.size) * 100.0

        if med_dist < self.thresholds.dist_min_m or med_dist > self.thresholds.dist_max_m:
            if (now - self._state_entry_time) > self.thresholds.position_check_timeout_s:
                logger.warning(f"Position check timed out: subject distance {med_dist:.2f} m out of range")
                self._transition(ControllerState.WAIT_FOR_SUBJECT, now)
            return

        # Border clipping check
        m_y = max(1, int(0.02 * h))
        m_x = max(1, int(0.02 * w))
        border_mask = np.zeros_like(valid_mask, dtype=bool)
        border_mask[:m_y, :] = True
        border_mask[-m_y:, :] = True
        border_mask[:, :m_x] = True
        border_mask[:, -m_x:] = True

        touching_count = np.count_nonzero(valid_mask & border_mask)
        clip_frac = float(touching_count) / float(valid_count)

        if clip_frac > self.thresholds.max_border_clip_frac:
            if (now - self._state_entry_time) > self.thresholds.position_check_timeout_s:
                logger.warning(f"Position check timed out: border clip frac {clip_frac:.3f} exceeds threshold")
                self._transition(ControllerState.WAIT_FOR_SUBJECT, now)
            return

        # Head / feet clipping
        head_clipped = bool(np.any(valid_mask[:m_y, :]))
        feet_clipped = bool(np.any(valid_mask[-m_y:, :]))
        self._latest_head_visible = not head_clipped
        self._latest_feet_visible = not feet_clipped

        if self.thresholds.require_head_feet:
            if head_clipped or feet_clipped:
                if (now - self._state_entry_time) > self.thresholds.position_check_timeout_s:
                    logger.warning("Position check timed out: head or feet clipped by image boundary")
                    self._transition(ControllerState.WAIT_FOR_SUBJECT, now)
                return

        # All checks passed
        self._transition(ControllerState.TURN_PROMPT, now)

    def _step_turn_prompt(self, now: float, cmd: Optional[str]) -> None:
        elapsed = now - self._state_entry_time
        if cmd == "confirm" or elapsed >= self.thresholds.turn_prompt_timeout_s:
            self._stillness_window.clear()
            self._transition(ControllerState.WAIT_STILL, now)

    def _step_wait_still(self, now: float, depth: Optional[np.ndarray], cmd: Optional[str]) -> None:
        if cmd == "confirm":
            self._burst_colors.clear()
            self._burst_depths.clear()
            self._burst_filtered = None
            self._burst_timestamps.clear()
            self._transition(ControllerState.CAPTURE_BURST, now)
            return

        if depth is None:
            return

        h, w = depth.shape[:2]
        d_m = depth.astype(np.float32) * (self.camera.depth_scale if self.camera.depth_units == "sensor_units" else 1.0)
        valid_mask = (d_m > 0.3) & (d_m < 3.5)
        ys, xs = np.nonzero(valid_mask)
        if len(ys) > 50:
            c_x = float(np.mean(xs))
            c_y = float(np.mean(ys))
            c_z = float(np.mean(d_m[ys, xs]))
            self._stillness_window.append((c_x, c_y, c_z))

        if len(self._stillness_window) >= self.thresholds.stillness_frames:
            w_arr = np.array(self._stillness_window)
            med_z = float(np.median(w_arr[:, 2]))
            focal = float(max(h, w))
            dx_mm = (w_arr[:, 0] - np.mean(w_arr[:, 0])) * (med_z / focal) * 1000.0
            dy_mm = (w_arr[:, 1] - np.mean(w_arr[:, 1])) * (med_z / focal) * 1000.0
            dz_mm = (w_arr[:, 2] - np.mean(w_arr[:, 2])) * 1000.0
            disp_mm = np.max(np.sqrt(dx_mm**2 + dy_mm**2 + dz_mm**2))
            self._latest_motion_mm = float(disp_mm)

            if disp_mm < self.thresholds.motion_limit_mm:
                self._burst_colors.clear()
                self._burst_depths.clear()
                self._burst_filtered = None
                self._burst_timestamps.clear()
                self._transition(ControllerState.CAPTURE_BURST, now)

    def _step_capture_burst(
        self,
        now: float,
        color: Optional[np.ndarray],
        depth: Optional[np.ndarray],
        filtered: Optional[np.ndarray],
        ts: float,
    ) -> None:
        if color is not None and depth is not None:
            self._burst_colors.append(color)
            self._burst_depths.append(depth)
            self._burst_timestamps.append(ts)
            if filtered is not None:
                self._burst_filtered = filtered

        elapsed = now - self._state_entry_time
        if len(self._burst_depths) >= self.thresholds.burst_frames or elapsed >= self.thresholds.burst_timeout_s:
            self._transition(ControllerState.EVALUATE, now)

    def _step_evaluate(self, now: float) -> None:
        report = self.quality_fn(
            self._burst_colors,
            self._burst_depths,
            depth_scale=self.camera.depth_scale,
            depth_units=self.camera.depth_units,
            thresholds=self.thresholds,
            bg_depth=self.bg_depth,
            cfg=self.cfg,
            intrinsics=self.camera.intrinsics,
        )
        self._latest_eval_report = report
        sway_val = float(report.get("sway_mm", report.get("motion_mm", 0.0)))
        self._latest_motion_mm = sway_val

        # Optional landmarks
        if self.landmark_fn and len(self._burst_colors) > 0:
            mid_idx = len(self._burst_colors) // 2
            self._latest_eval_landmarks = self.landmark_fn(
                self._burst_colors[mid_idx], self._burst_depths[mid_idx]
            )
        else:
            self._latest_eval_landmarks = None

        if not report.get("is_flagged", False):
            handle = self.writer.begin_attempt(self.current_stop)
            self.writer.write_burst(
                handle,
                self._burst_colors,
                self._burst_depths,
                self._burst_timestamps,
            )
            if self._burst_filtered is not None:
                self.writer.write_filtered_depth(handle, self._burst_filtered)
            self.writer.write_quality(handle, self._latest_eval_report)
            if self._latest_eval_landmarks:
                self.writer.write_landmarks(handle, self._latest_eval_landmarks)

            self.writer.finish_attempt(handle, accepted=True, reason="Quality check passed", sway_mm=sway_val)
            self._transition(ControllerState.ACCEPT, now)
        else:
            handle = self.writer.begin_attempt(self.current_stop)
            self.writer.write_burst(
                handle,
                self._burst_colors,
                self._burst_depths,
                self._burst_timestamps,
            )
            self.writer.write_quality(handle, self._latest_eval_report)
            if self._latest_eval_landmarks:
                self.writer.write_landmarks(handle, self._latest_eval_landmarks)

            fail_reasons = "; ".join(self._latest_eval_report.get("reasons", ["Quality check failed"]))
            self._latest_rejection_reason = fail_reasons
            self.writer.finish_attempt(handle, accepted=False, reason=fail_reasons, sway_mm=sway_val)
            self._transition(ControllerState.RETAKE, now)

    def _step_accept(self, now: float) -> None:
        if self.current_stop + 1 >= self.n_stops:
            self.writer.finalize()
            self._transition(ControllerState.DONE, now)
        else:
            self.current_stop += 1
            self.current_retake = 0
            self._transition(ControllerState.TURN_PROMPT, now)

    def _step_retake(self, now: float, cmd: Optional[str]) -> None:
        self.current_retake += 1
        if self.current_retake > self.thresholds.max_retakes:
            if cmd == "skip":
                # Operator skipped stop: record skip in stop.json, do not finalize session
                stop_json_path = self.writer.root / f"stop_{self.current_stop:02d}" / "stop.json"
                if stop_json_path.exists():
                    with open(stop_json_path, "r", encoding="utf-8") as f:
                        s_meta = json.load(f)
                    s_meta["skipped"] = True
                    with open(stop_json_path, "w", encoding="utf-8") as f:
                        json.dump(s_meta, f, indent=2)

                logger.warning(f"Stop {self.current_stop} skipped by operator command.")
                if self.current_stop + 1 >= self.n_stops:
                    self._transition(ControllerState.DONE, now)
                else:
                    self.current_stop += 1
                    self.current_retake = 0
                    self._transition(ControllerState.TURN_PROMPT, now)
            else:
                logger.error(
                    f"Stop {self.current_stop} exceeded max retakes ({self.thresholds.max_retakes}). Aborting."
                )
                self._transition(ControllerState.ABORT, now)
        else:
            self._transition(ControllerState.TURN_PROMPT, now)
