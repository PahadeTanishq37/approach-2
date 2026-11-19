"""
tests/fullbody/fake_camera.py

Numpy-only synthetic camera generating humanoid-like depth blobs.
Implements the CaptureCamera protocol with scriptable faults:
- swaying (lateral motion > 10 mm)
- depth_dropout (valid fraction < 0.30)
- head_clipped (body pixels touching top image boundary)
- too_far (subject distance > 1.5 m)
- too_close (subject distance < 1.1 m)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np


class FakeCamera:
    """Synthetic camera implementing CaptureCamera protocol."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        depth_scale: float = 0.001,
        depth_units: str = "sensor_units",
        base_depth_m: float = 2.00,
        fps: float = 30.0,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        empty_scene: bool = False,
    ):
        self.width = width
        self.height = height
        self._depth_scale = float(depth_scale)
        self._depth_units = str(depth_units)
        self.base_depth_m = float(base_depth_m)
        self.fps = float(fps)
        self.fx = fx
        self.fy = fy
        self.empty_scene = bool(empty_scene)

        self.is_running = False
        self.frame_id = 0
        self.fault: Optional[str] = None

    def start(self) -> None:
        self.is_running = True

    def stop(self) -> None:
        self.is_running = False

    @property
    def intrinsics(self) -> Dict[str, Any]:
        fx = self.fx if self.fx is not None else (640.0 if self.width == 1280 else 425.0)
        fy = self.fy if self.fy is not None else fx
        return {
            "width": self.width,
            "height": self.height,
            "fx": fx,
            "fy": fy,
            "cx": self.width / 2.0,
            "cy": self.height / 2.0,
            "is_post_rotation": bool(self.width < self.height),
            "is_post_decimation": False,
        }

    @property
    def depth_scale(self) -> float:
        return self._depth_scale

    @property
    def depth_units(self) -> str:
        return self._depth_units

    def device_info(self) -> Dict[str, Any]:
        return {
            "name": "Synthetic FakeCamera D455f",
            "serial_number": "SIM_001",
            "firmware_version": "9.9.9",
        }

    def set_fault(self, fault: Optional[str]) -> None:
        self.fault = fault

    def read_raw_filtered(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], float]:
        if not self.is_running:
            return None, None, None, 0.0

        ts = float(self.frame_id) / self.fps
        self.frame_id += 1

        h, w = self.height, self.width
        # Grid coordinates
        y_coords, x_coords = np.ogrid[:h, :w]

        # Base body center
        cx = w / 2.0
        cy = h / 2.0
        current_depth_m = self.base_depth_m

        # Handle faults
        if self.fault == "too_far":
            current_depth_m = 2.70
        elif self.fault == "too_close":
            current_depth_m = 1.30

        if self.fault == "swaying":
            # Significant sway: +/- 40 pixels lateral shift oscillating at 1 Hz
            cx += 40.0 * math.sin(2.0 * math.pi * 1.0 * ts)

        is_empty = self.empty_scene or (self.fault == "empty_scene")
        if is_empty:
            # Empty background scene: wall at 3.0 m, floor plane at camera height 0.95 m
            fy_val = self.fy if self.fy is not None else (640.0 if self.width == 1280 else 425.0)
            depth_m = np.full((h, w), 3.00, dtype=np.float32)
            # Floor plane in camera coordinates (optical frame: +Y down, normal [0, -1, 0], height 0.95m)
            y_1d = np.arange(h, dtype=np.float32)
            floor_rows = y_1d > (cy + 20)
            floor_z_1d = (0.95 * fy_val) / np.maximum(y_1d - cy, 1e-3)
            valid_floor_rows = floor_rows & (floor_z_1d >= 0.5) & (floor_z_1d <= 3.0)
            for r in np.where(valid_floor_rows)[0]:
                depth_m[r, :] = floor_z_1d[r]
            body_mask = np.zeros((h, w), dtype=bool)
        else:
            # Body ellipsoid: torso + head + limbs
            torso_mask = (((x_coords - cx) / (w * 0.22)) ** 2 + ((y_coords - cy) / (h * 0.35)) ** 2) <= 1.0

            head_cy = cy - h * 0.38
            if self.fault == "head_clipped":
                # Shift body and head up so head hits row 0
                head_cy = 10.0
                torso_mask = (((x_coords - cx) / (w * 0.22)) ** 2 + ((y_coords - (cy - h * 0.25)) / (h * 0.35)) ** 2) <= 1.0

            head_mask = (((x_coords - cx) / (w * 0.12)) ** 2 + ((y_coords - head_cy) / (h * 0.10)) ** 2) <= 1.0
            body_mask = torso_mask | head_mask
            depth_m = np.zeros((h, w), dtype=np.float32)
            depth_m[body_mask] = current_depth_m

        if self.fault == "depth_dropout":
            # Severe dropout: zero out 85% of pixels
            random_drop = np.random.rand(h, w) > 0.15
            depth_m[random_drop] = 0.0

        # Convert to storage format
        if self.depth_units == "sensor_units":
            raw_depth = (depth_m / self.depth_scale).astype(np.uint16)
        else:
            raw_depth = depth_m.astype(np.float32)

        # Color buffer
        color_bgr = np.full((h, w, 3), 40, dtype=np.uint8)
        color_bgr[body_mask] = [180, 150, 120]

        filtered_depth_m = depth_m.copy()

        return color_bgr, raw_depth, filtered_depth_m, ts

