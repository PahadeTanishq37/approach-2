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
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        empty_scene: bool = False,
        floor: bool = False,
        floor_tilt_deg: float = 0.0,
        floor_height_m: float = 0.95,
    ):
        self.width = width
        self.height = height
        self._depth_scale = float(depth_scale)
        self._depth_units = str(depth_units)
        self.base_depth_m = float(base_depth_m)
        self.fps = float(fps)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.empty_scene = bool(empty_scene)
        self.floor = bool(floor)
        self.floor_tilt_deg = float(floor_tilt_deg)
        self.floor_height_m = float(floor_height_m)

        self.is_running = False
        self.frame_id = 0
        self.fault: Optional[str] = None
        self._last_colorized: Optional[np.ndarray] = None

    def start(self) -> None:
        self.is_running = True

    def stop(self) -> None:
        self.is_running = False

    @property
    def last_colorized(self) -> Optional[np.ndarray]:
        return self._last_colorized

    @property
    def stream_profiles(self) -> Dict[str, Any]:
        return {
            "depth": {"width": self.width, "height": self.height, "fps": int(self.fps), "format": "z16"},
            "color": {"width": self.width, "height": self.height, "fps": int(self.fps), "format": "bgr8"},
        }

    @property
    def intrinsics(self) -> Dict[str, Any]:
        fx = self.fx if self.fx is not None else (640.0 if (self.width >= 1280 or self.height >= 1280) else 425.0)
        fy = self.fy if self.fy is not None else fx
        cx = self.cx if self.cx is not None else (self.width / 2.0)
        cy = self.cy if self.cy is not None else (self.height / 2.0)
        return {
            "width": self.width,
            "height": self.height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
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
            "stream_profile": f"{self.width}x{self.height}@{int(self.fps)}fps",
            "stream_profiles": self.stream_profiles,
            "filter_settings": {
                "decimation": False,
                "spatial": False,
                "temporal": False,
                "hole_filling": False,
                "raw_only": True,
            },
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
        fy_val = self.fy if self.fy is not None else (640.0 if (self.width >= 1280 or self.height >= 1280) else 425.0)

        # Floor plane generation
        has_floor = self.floor or is_empty
        floor_mask = np.zeros((h, w), dtype=bool)
        floor_depth = np.zeros((h, w), dtype=np.float32)

        if has_floor:
            theta_rad = math.radians(self.floor_tilt_deg)
            cos_t = math.cos(theta_rad)
            sin_t = math.sin(theta_rad)
            y_1d = np.arange(h, dtype=np.float32)
            denom = cos_t * (y_1d - cy) / fy_val + sin_t
            valid_rows = denom > 1e-4
            z_1d = np.zeros(h, dtype=np.float32)
            z_1d[valid_rows] = self.floor_height_m / denom[valid_rows]
            valid_z_rows = valid_rows & (z_1d >= 0.5) & (z_1d <= 3.8)
            for r in np.where(valid_z_rows)[0]:
                floor_mask[r, :] = True
                floor_depth[r, :] = z_1d[r]

        if is_empty:
            depth_m = np.full((h, w), 3.00, dtype=np.float32)
            depth_m[floor_mask] = floor_depth[floor_mask]
            body_mask = np.zeros((h, w), dtype=bool)
        else:
            # Body ellipsoid: torso + head + limbs
            torso_mask = (((x_coords - cx) / (w * 0.22)) ** 2 + ((y_coords - cy) / (h * 0.35)) ** 2) <= 1.0

            head_cy = cy - h * 0.34
            if self.fault == "head_clipped":
                # Shift body and head up so head hits row 0
                head_cy = 5.0
                torso_mask = (((x_coords - cx) / (w * 0.22)) ** 2 + ((y_coords - (cy - h * 0.25)) / (h * 0.35)) ** 2) <= 1.0
            elif self.fault in ("sole_clipped", "feet_clipped"):
                # Shift torso down so feet hit bottom border
                torso_cy = cy + h * 0.16
                torso_mask = (((x_coords - cx) / (w * 0.22)) ** 2 + ((y_coords - torso_cy) / (h * 0.35)) ** 2) <= 1.0
                head_cy = torso_cy - h * 0.34
            elif self.fault == "side_clipped":
                # Shift body to right border
                cx = w - 10.0
                torso_mask = (((x_coords - cx) / (w * 0.22)) ** 2 + ((y_coords - cy) / (h * 0.35)) ** 2) <= 1.0

            head_mask = (((x_coords - cx) / (w * 0.12)) ** 2 + ((y_coords - head_cy) / (h * 0.09)) ** 2) <= 1.0
            body_mask = torso_mask | head_mask
            depth_m = np.zeros((h, w), dtype=np.float32)
            if has_floor:
                depth_m[floor_mask] = floor_depth[floor_mask]
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
        if has_floor:
            color_bgr[floor_mask] = [70, 70, 70]
        color_bgr[body_mask] = [180, 150, 120]

        filtered_depth_m = depth_m.copy()

        return color_bgr, raw_depth, filtered_depth_m, ts

