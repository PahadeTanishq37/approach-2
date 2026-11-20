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

    # Burst frame count: default 60 frames
    burst_frames: int = 60

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
    burst_timeout_s: float = 8.0

    # Subject mode: "person" or "object"
    subject_type: str = "person"

    # Optional object diameter in mm for rigid object mode
    object_diameter_mm: Optional[float] = None

    # Audio cue beeps at hold-start and burst-end
    audio_cues: bool = True

    # Allow proceeding with depth-window segmentation when background depth map is missing
    allow_missing_background: bool = False


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


def play_audio_cue(cue: str, enabled: bool = True) -> None:
    """
    Play an audio cue asynchronously so it never blocks the control loop.
    cue: 'hold_start' ("hold still") -> 800 Hz, 150 ms
         'burst_end'  ("turn now")   -> 1200 Hz, 250 ms
    Safe fallback to terminal bell, ignores exceptions on headless/virtual systems.
    """
    if not enabled:
        return

    import sys
    import threading

    def _beep():
        try:
            if sys.platform == "win32":
                import winsound
                if cue == "hold_start":
                    winsound.Beep(800, 150)
                elif cue == "burst_end":
                    winsound.Beep(1200, 250)
                else:
                    winsound.Beep(1000, 200)
            else:
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception:
            pass

    t = threading.Thread(target=_beep, daemon=True)
    t.start()


def subject_mask(
    depth_m: np.ndarray,
    bg_depth: Optional[np.ndarray],
    cfg: Optional[FullBodyConfig] = None,
    allow_missing: bool = False,
) -> np.ndarray:
    """
    Extract foreground subject mask from depth map using background-depth subtraction.
    Strictly requires matching background depth map unless allow_missing=True.
    """
    h, w = depth_m.shape[:2]
    cfg_eval = cfg or FullBodyConfig()
    if bg_depth is None:
        if allow_missing:
            d_min = cfg_eval.capture_dist_min_m - 0.25
            d_max = cfg_eval.capture_dist_max_m + 0.25
            raw_mask = (depth_m >= d_min) & (depth_m <= d_max) & (depth_m > 0.1)
            if not np.any(raw_mask):
                return np.zeros((h, w), dtype=bool)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                raw_mask.astype(np.uint8), connectivity=8
            )
            if num_labels <= 1:
                return np.zeros((h, w), dtype=bool)
            largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            return labels == largest_label
        raise RuntimeError(f"No background depth map loaded. Run capture-background at {w}x{h} first.")
    if bg_depth.shape[:2] != (h, w):
        raise RuntimeError(
            f"Background resolution {bg_depth.shape[1]}x{bg_depth.shape[0]} "
            f"does not match live stream {w}x{h}. Run capture-background at {w}x{h} first."
        )
    from realsense_measure.fullbody.fusion import compute_foreground_mask
    return compute_foreground_mask(depth_m, bg_depth, cfg_eval)


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
    Numpy-only quality assessment for a captured burst using foreground subject mask:
    - Subject presence and foreground pixel count
    - Head margin and sole margin >= 4% from foreground bbox (person mode)
    - Border clipping from top/left/right border touch (person mode) or all 4 borders (object mode)
    - Sway: median per-frame shift of foreground centroid and torso depth (mm)
    - Temporal depth standard deviation (mm)
    """
    th = thresholds or CaptureThresholds()
    n_frames = len(burst_depths)
    min_required_frames = max(1, int(0.9 * th.burst_frames))
    if n_frames < min_required_frames:
        return {
            "is_flagged": True,
            "reasons": [
                f"Burst frame count ({n_frames}) is less than 90% of required burst_frames ({th.burst_frames})"
            ],
            "sway_mm": 999.0,
            "motion_mm": 999.0,
            "temporal_depth_std_mm": 999.0,
            "valid_depth_fraction": 0.0,
            "border_clip_fraction": 0.0,
            "head_clipped": False,
            "feet_clipped": False,
            "foreground_px": 0,
            "frames_requested": th.burst_frames,
            "frames_saved": n_frames,
        }

    # Normalize depth to metres float32 for computation
    depth_m_list: List[np.ndarray] = []
    for d in burst_depths:
        if depth_units == "sensor_units":
            depth_m_list.append(d.astype(np.float32) * depth_scale)
        else:
            depth_m_list.append(d.astype(np.float32))

    h, w = depth_m_list[0].shape[:2]
    cfg_eval = cfg or FullBodyConfig()

    fx = float(intrinsics.get("fx", max(h, w))) if intrinsics else float(max(h, w))
    fy = float(intrinsics.get("fy", max(h, w))) if intrinsics else float(max(h, w))
    cx = float(intrinsics.get("cx", w / 2.0)) if intrinsics else float(w / 2.0)
    cy = float(intrinsics.get("cy", h / 2.0)) if intrinsics else float(h / 2.0)

    per_frame_centroids: List[Tuple[float, float, float]] = []
    per_frame_torso_z: List[float] = []
    fg_px_counts: List[int] = []
    head_clipped_count = 0
    feet_clipped_count = 0
    border_clip_fractions: List[float] = []
    fg_masks: List[np.ndarray] = []
    is_object = (th.subject_type == "object")

    for d_m in depth_m_list:
        fg_m = subject_mask(d_m, bg_depth, cfg_eval, allow_missing=th.allow_missing_background)
        fg_masks.append(fg_m)
        ys, xs = np.nonzero(fg_m)
        fg_count = len(ys)
        fg_px_counts.append(fg_count)

        min_px = 50 if is_object else 100
        if fg_count < min_px:
            border_clip_fractions.append(0.0)
            if not is_object:
                head_clipped_count += 1
                feet_clipped_count += 1
            per_frame_centroids.append((0.0, 0.0, 0.0))
            per_frame_torso_z.append(0.0)
            continue

        y_min, y_max = int(np.min(ys)), int(np.max(ys))
        x_min, x_max = int(np.min(xs)), int(np.max(xs))

        top_touch = np.count_nonzero(fg_m[0, :])
        bottom_touch = np.count_nonzero(fg_m[-1, :])
        left_touch = np.count_nonzero(fg_m[:, 0])
        right_touch = np.count_nonzero(fg_m[:, -1])

        if not is_object:
            # Head margin and sole margin checks from foreground bbox (>= 4% required)
            head_margin_pct = float(y_min) / float(h) * 100.0
            sole_margin_pct = float((h - 1) - y_max) / float(h) * 100.0
            if head_margin_pct < 4.0:
                head_clipped_count += 1
            if sole_margin_pct < 4.0:
                feet_clipped_count += 1

            # Side clipping = top, left, right border touching (bottom border excluded)
            clip_px = int(top_touch + left_touch + right_touch)
        else:
            # In object mode, object must not touch any of the 4 borders
            clip_px = int(top_touch + bottom_touch + left_touch + right_touch)

        border_clip_fractions.append(float(clip_px) / float(fg_count))

        # Centroid
        u_c = float(np.mean(xs))
        v_c = float(np.mean(ys))
        z_c = float(np.median(d_m[ys, xs]))
        x_m = (u_c - cx) * (z_c / fx)
        y_m = (v_c - cy) * (z_c / fy)
        per_frame_centroids.append((x_m, y_m, z_c))

        # Torso depth (20%-65% vertical span, 25%-75% horizontal span of foreground bbox)
        span_h = max(10, y_max - y_min)
        span_w = max(10, x_max - x_min)
        t_y0 = y_min + int(0.20 * span_h)
        t_y1 = y_min + int(0.65 * span_h)
        t_x0 = x_min + int(0.25 * span_w)
        t_x1 = x_min + int(0.75 * span_w)
        torso_sub = fg_m[t_y0:t_y1, t_x0:t_x1]
        torso_val = d_m[t_y0:t_y1, t_x0:t_x1][torso_sub]
        if len(torso_val) > 20:
            per_frame_torso_z.append(float(np.median(torso_val)))
        else:
            per_frame_torso_z.append(z_c)

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

    mean_fg_px = int(np.mean(fg_px_counts)) if fg_px_counts else 0
    mean_border_clip = float(np.mean(border_clip_fractions)) if border_clip_fractions else 0.0

    # Temporal depth std on foreground
    depth_stack = np.stack(depth_m_list, axis=0)
    rep_fg = fg_masks[len(fg_masks) // 2]
    if np.any(rep_fg):
        std_map = np.std(depth_stack, axis=0) * 1000.0
        median_depth_std_mm = float(np.median(std_map[rep_fg]))
    else:
        median_depth_std_mm = 999.0

    head_clipped = head_clipped_count > 0
    feet_clipped = feet_clipped_count > 0

    reasons: List[str] = []
    if sway_mm > th.motion_limit_mm:
        reasons.append(
            f"Subject sway {sway_mm:.1f} mm exceeds motion limit {th.motion_limit_mm:.1f} mm"
        )
    if not is_object:
        if th.require_head_feet and head_clipped:
            reasons.append(f"Head margin < 4.0% in {head_clipped_count}/{n_frames} frames")
        if th.require_head_feet and feet_clipped:
            reasons.append(f"Sole margin < 4.0% in {feet_clipped_count}/{n_frames} frames")
    if is_object and mean_border_clip > 0.0:
        reasons.append(
            f"Object clipped by image border ({mean_border_clip * 100:.2f}% border clipping)"
        )
    elif mean_border_clip > th.max_border_clip_frac:
        reasons.append(
            f"Border clipping fraction {mean_border_clip:.3f} exceeds limit {th.max_border_clip_frac:.3f}"
        )
    min_fg_limit = 100 if is_object else 500
    if mean_fg_px < min_fg_limit:
        reasons.append("Insufficient foreground subject pixels")

    is_flagged = len(reasons) > 0

    return {
        "is_flagged": is_flagged,
        "reasons": reasons,
        "sway_mm": sway_mm,
        "motion_mm": sway_mm,
        "temporal_depth_std_mm": median_depth_std_mm,
        "valid_depth_fraction": float(mean_fg_px) / float(h * w),
        "border_clip_fraction": mean_border_clip,
        "head_clipped": head_clipped,
        "feet_clipped": feet_clipped,
        "foreground_px": mean_fg_px,
        "frames_requested": th.burst_frames,
        "frames_saved": n_frames,
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

        intr = camera.intrinsics if hasattr(camera, "intrinsics") else {}
        cam_w = int(intr.get("width", 848))
        cam_h = int(intr.get("height", 480))
        if self.bg_depth is None:
            if not self.thresholds.allow_missing_background:
                raise RuntimeError(
                    f"No background depth map loaded. Run capture-background at {cam_w}x{cam_h} first."
                )
            else:
                logger.warning(
                    "WARNING: No background depth map loaded. Proceeding with distance-gated foreground segmentation."
                )
        elif self.bg_depth.shape[:2] != (cam_h, cam_w):
            raise RuntimeError(
                f"Background resolution {self.bg_depth.shape[1]}x{self.bg_depth.shape[0]} "
                f"does not match camera stream {cam_w}x{cam_h}. Run capture-background at {cam_w}x{cam_h} first."
            )

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

        # Live telemetry cached for UI preview and console logging
        self._latest_color: Optional[np.ndarray] = None
        self._latest_filtered: Optional[np.ndarray] = None
        self._latest_distance_m: Optional[float] = None
        self._latest_valid_depth_pct: Optional[float] = None
        self._latest_motion_mm: Optional[float] = None
        self._latest_head_visible: bool = True
        self._latest_feet_visible: bool = True
        self._latest_rejection_reason: Optional[str] = None

        # Foreground telemetry
        self._latest_fg_mask: Optional[np.ndarray] = None
        self._latest_fg_count: int = 0
        self._latest_bbox: Optional[Tuple[int, int, int, int]] = None
        self._latest_bbox_h_pct: Optional[float] = None
        self._latest_head_margin_pct: Optional[float] = None
        self._latest_sole_margin_pct: Optional[float] = None
        self._latest_clip_pct: Optional[float] = None

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
        if new_state == ControllerState.WAIT_STILL:
            play_audio_cue("hold_start", getattr(self.thresholds, "audio_cues", True))

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

        fg_mask = subject_mask(d_m, self.bg_depth, self.cfg, allow_missing=self.thresholds.allow_missing_background)
        fg_count = int(np.count_nonzero(fg_mask))
        total_px = h * w
        fg_fraction = float(fg_count) / float(total_px)

        self._latest_fg_mask = fg_mask
        self._latest_fg_count = fg_count

        is_present = False
        is_object = (self.thresholds.subject_type == "object")
        if fg_count > 0:
            ys, xs = np.nonzero(fg_mask)
            y_min, y_max = int(np.min(ys)), int(np.max(ys))
            x_min, x_max = int(np.min(xs)), int(np.max(xs))
            bbox_h = y_max - y_min + 1
            bbox_h_ratio = float(bbox_h) / float(h)
            self._latest_bbox = (x_min, y_min, x_max, y_max)
            self._latest_bbox_h_pct = bbox_h_ratio * 100.0
            self._latest_distance_m = float(np.median(d_m[ys, xs]))

            if is_object:
                if fg_fraction >= 0.002 and bbox_h_ratio >= 0.10:
                    is_present = True
            else:
                # WAIT_FOR_SUBJECT: subject present when foreground pixels >= 0.5% and bbox height >= 40%
                if fg_fraction >= 0.005 and bbox_h_ratio >= 0.40:
                    is_present = True
        else:
            self._latest_bbox = None
            self._latest_bbox_h_pct = 0.0
            self._latest_distance_m = None

        if is_present:
            self._wait_subject_consecutive += 1
            self._latest_rejection_reason = None
            if self._wait_subject_consecutive >= 5:
                self._wait_subject_consecutive = 0
                self._transition(ControllerState.POSITION_CHECK, now)
        else:
            self._wait_subject_consecutive = 0
            if fg_count == 0:
                self._latest_rejection_reason = "Waiting for subject (no foreground)"
            elif is_object and fg_fraction < 0.002:
                self._latest_rejection_reason = f"Waiting for object (foreground {fg_fraction*100:.2f}% < 0.2%)"
            elif not is_object and fg_fraction < 0.005:
                self._latest_rejection_reason = f"Waiting for subject (foreground {fg_fraction*100:.2f}% < 0.5%)"
            elif is_object:
                self._latest_rejection_reason = f"Waiting for object (bbox height {self._latest_bbox_h_pct:.1f}% < 10%)"
            else:
                self._latest_rejection_reason = f"Waiting for subject (bbox height {self._latest_bbox_h_pct:.1f}% < 40%)"

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

        fg_mask = subject_mask(d_m, self.bg_depth, self.cfg, allow_missing=self.thresholds.allow_missing_background)
        ys, xs = np.nonzero(fg_mask)
        fg_count = len(ys)
        self._latest_fg_mask = fg_mask
        self._latest_fg_count = fg_count
        is_object = (self.thresholds.subject_type == "object")
        min_fg_pos = 50 if is_object else 100

        if fg_count < min_fg_pos:
            self._latest_rejection_reason = "No subject foreground detected"
            self._latest_bbox = None
            if (now - self._state_entry_time) > self.thresholds.position_check_timeout_s:
                self._transition(ControllerState.WAIT_FOR_SUBJECT, now)
            return

        y_min, y_max = int(np.min(ys)), int(np.max(ys))
        x_min, x_max = int(np.min(xs)), int(np.max(xs))
        self._latest_bbox = (x_min, y_min, x_max, y_max)

        # 1. Distance = median depth of foreground
        med_dist = float(np.median(d_m[ys, xs]))
        self._latest_distance_m = med_dist
        self._latest_valid_depth_pct = float(fg_count) / float(h * w) * 100.0

        bbox_h = y_max - y_min + 1
        bbox_h_pct = float(bbox_h) / float(h) * 100.0
        self._latest_bbox_h_pct = bbox_h_pct

        top_touch = np.count_nonzero(fg_mask[0, :])
        bottom_touch = np.count_nonzero(fg_mask[-1, :])
        left_touch = np.count_nonzero(fg_mask[:, 0])
        right_touch = np.count_nonzero(fg_mask[:, -1])

        reasons = []
        if med_dist < self.thresholds.dist_min_m:
            reasons.append(f"Subject too close ({med_dist:.2f} m < {self.thresholds.dist_min_m:.2f} m)")
        elif med_dist > self.thresholds.dist_max_m:
            reasons.append(f"Subject too far ({med_dist:.2f} m > {self.thresholds.dist_max_m:.2f} m)")

        if is_object:
            all_clip_px = int(top_touch + bottom_touch + left_touch + right_touch)
            self._latest_clip_pct = float(all_clip_px) / float(fg_count) * 100.0
            self._latest_head_margin_pct = float(y_min) / float(h) * 100.0
            self._latest_sole_margin_pct = float((h - 1) - y_max) / float(h) * 100.0
            self._latest_head_visible = (top_touch == 0)
            self._latest_feet_visible = (bottom_touch == 0)
            if all_clip_px > 0:
                reasons.append(f"Object border clipped ({all_clip_px} px)")
        else:
            # Person mode: head margin and sole margin >= 4% from foreground bbox
            head_margin_px = y_min
            head_margin_pct = float(head_margin_px) / float(h) * 100.0
            self._latest_head_margin_pct = head_margin_pct
            self._latest_head_visible = (head_margin_pct >= 4.0)

            sole_margin_px = (h - 1) - y_max
            sole_margin_pct = float(sole_margin_px) / float(h) * 100.0
            self._latest_sole_margin_pct = sole_margin_pct
            self._latest_feet_visible = (sole_margin_pct >= 4.0)

            # Side clipping = foreground pixels touching left/right/top borders
            # (bottom border counts only as sole margin test)
            side_clip_px = int(top_touch + left_touch + right_touch)
            side_clip_pct = float(side_clip_px) / float(fg_count) * 100.0
            self._latest_clip_pct = side_clip_pct

            if bbox_h_pct < 60.0:
                reasons.append(f"Bbox height too small ({bbox_h_pct:.1f}% < 60.0%)")

            if self.thresholds.require_head_feet and head_margin_pct < 4.0:
                reasons.append(f"Head margin too small ({head_margin_pct:.1f}% < 4.0%)")

            if self.thresholds.require_head_feet and sole_margin_pct < 4.0:
                reasons.append(f"Sole margin too small ({sole_margin_pct:.1f}% < 4.0%)")

            if side_clip_px > 0:
                reasons.append(f"Side/top border clipped ({side_clip_px} px)")

        if reasons:
            self._latest_rejection_reason = "; ".join(reasons)
            if (now - self._state_entry_time) > self.thresholds.position_check_timeout_s:
                logger.warning(f"Position check timed out: {self._latest_rejection_reason}")
                self._transition(ControllerState.WAIT_FOR_SUBJECT, now)
            return

        self._latest_rejection_reason = None
        self._transition(ControllerState.TURN_PROMPT, now)

    def _step_turn_prompt(self, now: float, cmd: Optional[str]) -> None:
        elapsed = now - self._state_entry_time
        if not hasattr(self, "_last_turn_prompt_stop") or self._last_turn_prompt_stop != self.current_stop:
            self._last_turn_prompt_stop = self.current_stop
            step_deg = 360.0 / max(1, self.n_stops)
            target_deg = (self.current_stop * step_deg) % 360.0
            print(f"Stop {self.current_stop + 1}/{self.n_stops}: turn to tick {self.current_stop + 1} ({target_deg:.0f} deg)")

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

        fg_mask = subject_mask(d_m, self.bg_depth, self.cfg, allow_missing=self.thresholds.allow_missing_background)
        ys, xs = np.nonzero(fg_mask)
        self._latest_fg_mask = fg_mask
        self._latest_fg_count = len(ys)

        if len(ys) > 50:
            c_x = float(np.mean(xs))
            c_y = float(np.mean(ys))

            y_min, y_max = int(np.min(ys)), int(np.max(ys))
            x_min, x_max = int(np.min(xs)), int(np.max(xs))
            self._latest_bbox = (x_min, y_min, x_max, y_max)
            span_h = max(10, y_max - y_min)
            span_w = max(10, x_max - x_min)
            t_y0 = y_min + int(0.20 * span_h)
            t_y1 = y_min + int(0.65 * span_h)
            t_x0 = x_min + int(0.25 * span_w)
            t_x1 = x_min + int(0.75 * span_w)

            torso_sub = fg_mask[t_y0:t_y1, t_x0:t_x1]
            torso_depths = d_m[t_y0:t_y1, t_x0:t_x1][torso_sub]
            if len(torso_depths) > 10:
                c_z = float(np.median(torso_depths))
            else:
                c_z = float(np.median(d_m[ys, xs]))

            self._latest_distance_m = c_z
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

        n_saved = len(self._burst_depths)
        fg_px = int(report.get("foreground_px", 0))
        play_audio_cue("burst_end", getattr(self.thresholds, "audio_cues", True))

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
            print(f"Stop {self.current_stop} accepted: {n_saved} frames saved, sway {sway_val:.1f} mm, foreground {fg_px} px")
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
            print(f"Stop {self.current_stop} retake: {n_saved} frames saved, sway {sway_val:.1f} mm, foreground {fg_px} px")
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
