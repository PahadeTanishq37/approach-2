"""
tests/fullbody/test_subject_gating.py

Requirement 9 verification tests for CaptureController and CLI:
1. Synthetic scene where floor, ceiling, and walls have valid depth: a subject at 2.0 m passes.
2. Subject at 1.0 m fails on distance.
3. Subject whose feet touch the bottom edge fails on sole margin.
4. No background or resolution-mismatched background refuses to start.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Tuple

import numpy as np
import pytest

from realsense_measure.fullbody.capture_controller import (
    CaptureCamera,
    CaptureController,
    CaptureThresholds,
    ControllerState,
)
from realsense_measure.fullbody.cli import main
from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.session_format import SessionWriter
from tests.fullbody.fake_camera import FakeCamera


class SyntheticClutteredSceneCamera:
    """
    Synthetic camera rendering a room where floor, ceiling, and walls have valid depth:
    - Back wall at 3.2 m
    - Floor plane at camera height 0.95 m (slanting closer towards bottom)
    - Ceiling plane at camera height 1.80 m (slanting closer towards top)
    - Side walls at left and right
    - Subject at configurable standoff distance and vertical position.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        subject_depth_m: float = 2.0,
        feet_touch_bottom: bool = False,
    ):
        self.width = width
        self.height = height
        self.depth_scale = 0.001
        self.depth_units = "sensor_units"
        self.subject_depth_m = float(subject_depth_m)
        self.feet_touch_bottom = bool(feet_touch_bottom)
        self.is_running = True
        self.frame_id = 0

    def start(self) -> None:
        self.is_running = True

    def stop(self) -> None:
        self.is_running = False

    @property
    def intrinsics(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fx": 640.0,
            "fy": 640.0,
            "cx": self.width / 2.0,
            "cy": self.height / 2.0,
        }

    def device_info(self) -> dict:
        return {"name": "SyntheticClutteredSceneCamera"}

    def render_background_depth(self) -> np.ndarray:
        """Render empty room depth map with valid floor, ceiling, and walls."""
        h, w = self.height, self.width
        cx, cy = w / 2.0, h / 2.0
        fx, fy = 640.0, 640.0

        depth = np.full((h, w), 3.20, dtype=np.float32)

        # Floor plane (Y down in camera coords, height 0.95m above floor)
        y_grid = np.arange(h, dtype=np.float32)[:, None]
        floor_rows = y_grid > (cy + 20)
        floor_z = (0.95 * fy) / np.maximum(y_grid - cy, 1e-3)
        valid_floor = floor_rows & (floor_z >= 0.4) & (floor_z <= 3.2)
        depth = np.where(valid_floor, floor_z, depth)

        # Ceiling plane (height 1.8m above camera)
        ceil_rows = y_grid < (cy - 40)
        ceil_z = (1.80 * fy) / np.maximum(cy - y_grid, 1e-3)
        valid_ceil = ceil_rows & (ceil_z >= 0.4) & (ceil_z <= 3.2)
        depth = np.where(valid_ceil, ceil_z, depth)

        return depth

    def read_raw_filtered(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        h, w = self.height, self.width
        cx, cy = w / 2.0, h / 2.0

        bg_depth = self.render_background_depth()
        depth = bg_depth.copy()

        y_coords, x_coords = np.ogrid[:h, :w]

        # Subject geometry
        if self.feet_touch_bottom:
            # Shift body down so feet touch row h - 1 (sole margin = 0%)
            body_cy = cy + h * 0.16
            torso_mask = (((x_coords - cx) / (w * 0.18)) ** 2 + ((y_coords - body_cy) / (h * 0.35)) ** 2) <= 1.0
            head_cy = body_cy - h * 0.34
            head_mask = (((x_coords - cx) / (w * 0.10)) ** 2 + ((y_coords - head_cy) / (h * 0.09)) ** 2) <= 1.0
        else:
            # Normal well-framed subject: head margin ~7%, sole margin ~15%, bbox height ~78%
            body_cy = cy
            torso_mask = (((x_coords - cx) / (w * 0.18)) ** 2 + ((y_coords - body_cy) / (h * 0.35)) ** 2) <= 1.0
            head_cy = cy - h * 0.34
            head_mask = (((x_coords - cx) / (w * 0.10)) ** 2 + ((y_coords - head_cy) / (h * 0.09)) ** 2) <= 1.0

        subj_mask = torso_mask | head_mask
        if self.feet_touch_bottom:
            # Feet extend in front of the floor all the way to the bottom border
            depth[subj_mask] = np.minimum(self.subject_depth_m, bg_depth[subj_mask] - 0.08)
        else:
            depth[subj_mask] = self.subject_depth_m

        ts = float(self.frame_id) / 30.0
        self.frame_id += 1

        color = np.full((h, w, 3), 40, dtype=np.uint8)
        color[subj_mask] = [200, 160, 120]
        raw_depth = (depth / self.depth_scale).astype(np.uint16)
        return color, raw_depth, depth.copy(), ts


def test_subject_at_2m_in_cluttered_scene_passes():
    """1. In a scene where floor, ceiling, and walls have valid depth, a subject at 2.0 m passes."""
    cam = SyntheticClutteredSceneCamera(subject_depth_m=2.0, feet_touch_bottom=False)
    bg_depth = cam.render_background_depth()

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SessionWriter(
            root=Path(tmpdir) / "sess_pass",
            subject_id="SUBJ_PASS",
            n_stops=1,
            turn_direction="cw",
            camera_portrait_rotation="none",
            intrinsics=cam.intrinsics,
            depth_scale=cam.depth_scale,
            depth_units=cam.depth_units,
        )
        thresholds = CaptureThresholds(
            burst_frames=3,
            stillness_frames=3,
            position_check_timeout_s=1.0,
            turn_prompt_timeout_s=0.1,
        )
        controller = CaptureController(
            camera=cam,
            writer=writer,
            thresholds=thresholds,
            bg_depth=bg_depth,
        )

        now = 0.0
        passed_position_check = False
        for _ in range(50):
            state = controller.step(now)
            now += 0.033
            if state in (ControllerState.TURN_PROMPT, ControllerState.WAIT_STILL, ControllerState.CAPTURE_BURST, ControllerState.DONE):
                passed_position_check = True
                break

        assert passed_position_check, f"Subject at 2.0m should pass position check, got state: {controller.state}"
        assert controller._latest_distance_m == pytest.approx(2.0, abs=0.05)
        assert controller._latest_head_margin_pct >= 4.0
        assert controller._latest_sole_margin_pct >= 4.0
        assert controller._latest_bbox_h_pct >= 60.0


def test_subject_at_1m_fails_on_distance():
    """2. A subject at 1.0 m fails on distance (< 1.70 m threshold)."""
    cam = SyntheticClutteredSceneCamera(subject_depth_m=1.0, feet_touch_bottom=False)
    bg_depth = cam.render_background_depth()

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SessionWriter(
            root=Path(tmpdir) / "sess_too_close",
            subject_id="SUBJ_CLOSE",
            n_stops=1,
            turn_direction="cw",
            camera_portrait_rotation="none",
            intrinsics=cam.intrinsics,
            depth_scale=cam.depth_scale,
            depth_units=cam.depth_units,
        )
        thresholds = CaptureThresholds(
            position_check_timeout_s=0.2,
        )
        controller = CaptureController(
            camera=cam,
            writer=writer,
            thresholds=thresholds,
            bg_depth=bg_depth,
        )

        now = 0.0
        for _ in range(30):
            state = controller.step(now)
            now += 0.033
            assert state in (ControllerState.WAIT_FOR_SUBJECT, ControllerState.POSITION_CHECK)

        assert controller._latest_distance_m == pytest.approx(1.0, abs=0.05)
        assert controller._latest_rejection_reason is not None
        assert "too close" in controller._latest_rejection_reason.lower()


def test_subject_feet_touching_bottom_fails_on_sole_margin():
    """3. A subject whose feet touch the bottom edge fails on sole margin (< 4.0%)."""
    cam = SyntheticClutteredSceneCamera(subject_depth_m=2.0, feet_touch_bottom=True)
    bg_depth = cam.render_background_depth()

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SessionWriter(
            root=Path(tmpdir) / "sess_sole_clip",
            subject_id="SUBJ_SOLE",
            n_stops=1,
            turn_direction="cw",
            camera_portrait_rotation="none",
            intrinsics=cam.intrinsics,
            depth_scale=cam.depth_scale,
            depth_units=cam.depth_units,
        )
        thresholds = CaptureThresholds(
            position_check_timeout_s=0.2,
        )
        controller = CaptureController(
            camera=cam,
            writer=writer,
            thresholds=thresholds,
            bg_depth=bg_depth,
        )

        now = 0.0
        for _ in range(30):
            state = controller.step(now)
            now += 0.033
            assert state in (ControllerState.WAIT_FOR_SUBJECT, ControllerState.POSITION_CHECK)

        assert controller._latest_sole_margin_pct < 4.0
        assert controller._latest_rejection_reason is not None
        assert "sole margin" in controller._latest_rejection_reason.lower()


def test_no_background_or_resolution_mismatched_background_refuses_to_start():
    """4. No background or resolution-mismatched background refuses to start."""
    cam = SyntheticClutteredSceneCamera(subject_depth_m=2.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SessionWriter(
            root=Path(tmpdir) / "sess_no_bg",
            subject_id="SUBJ_02",
            n_stops=1,
            turn_direction="cw",
            camera_portrait_rotation="none",
            intrinsics=cam.intrinsics,
            depth_scale=cam.depth_scale,
            depth_units=cam.depth_units,
        )

        # 1. No background depth map -> raises RuntimeError
        with pytest.raises(RuntimeError, match="No background depth map loaded"):
            CaptureController(camera=cam, writer=writer, bg_depth=None)

        # 2. Resolution mismatched background depth map -> raises RuntimeError
        wrong_res_bg = np.full((480, 848), 3.0, dtype=np.float32)
        with pytest.raises(RuntimeError, match="Background resolution"):
            CaptureController(camera=cam, writer=writer, bg_depth=wrong_res_bg)

        # 3. CLI refuses to start when background is missing
        argv_no_bg = [
            "capture",
            "--out", str(Path(tmpdir) / "cli_no_bg"),
            "--subject-id", "SUBJ_CLI_NO_BG",
            "--stops", "1",
            "--fake",
            "--background", str(Path(tmpdir) / "non_existent_bg_dir"),
            "--headless",
        ]
        ret = main(argv_no_bg)
        assert ret == 1
