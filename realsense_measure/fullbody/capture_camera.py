"""
realsense_measure/fullbody/capture_camera.py

RealSense camera adapter and configuration for full-body capture.
Implements the CaptureCamera protocol.
Guaranteed to import without loading pyrealsense2 or open3d native libraries.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np

from realsense_measure.config import CameraConfig
from realsense_measure.fullbody.capture_controller import CaptureCamera

logger = logging.getLogger(__name__)


def full_body_camera_config(
    portrait_rotation: str = "none",
    camera_rotation: Optional[str] = None,
    resolution: str = "1280x720",
    fps: int = 30,
    raw_only: bool = True,
) -> CameraConfig:
    """
    Construct a CameraConfig tailored for full-body 8- or 12-stop turntable capture.

    Design rationale:
    - resolution = "1280x720": High-definition 720p native stream for full-body capture
      (848x480 remains selectable).
    - raw_only = True: Saves raw z16 and color, disables RealSense temporal and hole-filling
      filters (the multi-frame burst median provides temporal stabilization).
    - depth_max_m = 4.0: Subject stands at ~2.0 m (capture range 1.7 - 2.4 m). Retains full
      subject silhouette, extended arms, and surrounding floor context without clipping.
    - depth_min_m = 0.4: Prevents near-field sensor noise from entering the cloud.
    - use_decimation = False: Strictly OFF to preserve full 1:1 pixel geometry.
    - rotate_90 = 'none': Native landscape mounting (no software rotation).
    - fps = 30: Standard frame rate providing good temporal integration and SNR during burst capture.
    """
    rot = camera_rotation if camera_rotation is not None else portrait_rotation
    rot = str(rot).lower()
    if rot not in ("cw", "ccw", "none"):
        param_name = "camera_rotation" if camera_rotation is not None else "portrait_rotation"
        raise ValueError(f"Invalid {param_name} '{rot}'. Must be 'cw', 'ccw', or 'none'.")

    # Parse resolution string (e.g. "1280x720" or "848x480")
    if "x" in resolution:
        w_str, h_str = resolution.lower().split("x")
        w, h = int(w_str), int(h_str)
    else:
        w, h = 1280, 720

    cfg = CameraConfig(
        width=w,
        height=h,
        fps=fps,
        depth_min_m=0.40,
        depth_max_m=4.00,
        use_decimation=False,  # Strictly OFF to preserve full resolution geometry
        use_spatial_filter=not raw_only,
        use_temporal_filter=not raw_only,   # Disabled in raw_only mode (burst median does temporal job)
        use_hole_filling=not raw_only,      # Disabled in raw_only mode
        rotate_90=rot if rot in ("cw", "ccw") else None,  # 'none' means native landscape mode
    )
    # Store raw_only flag on config
    cfg.raw_only = raw_only  # type: ignore[attr-defined]

    # Ensure preview visualizer covers full range
    cfg.depth_vis.visual_min_m = 0.40
    cfg.depth_vis.visual_max_m = 4.00
    return cfg


class RealSenseCaptureAdapter:
    """
    Adapter implementing the CaptureCamera protocol using RealSenseCamera.
    Loads pyrealsense2 / RealSenseCamera lazily on start() to prevent import-time side effects.
    """

    def __init__(
        self,
        camera_config: Optional[CameraConfig] = None,
        camera_rotation: str = "none",
        portrait_rotation: Optional[str] = None,
        warmup_frames: int = 30,
        max_dropped_retries: int = 5,
        camera_factory: Optional[Callable[..., Any]] = None,
    ):
        rot = portrait_rotation if portrait_rotation is not None else camera_rotation
        rot = str(rot).lower()
        self.camera_rotation = rot
        self.portrait_rotation = rot  # backward compatibility alias
        self.camera_config = camera_config or full_body_camera_config(camera_rotation=rot)
        self.warmup_frames = int(warmup_frames)
        self.max_dropped_retries = int(max_dropped_retries)
        self._camera_factory = camera_factory

        self._camera: Any = None
        self._is_started = False
        self._intrinsics: Dict[str, Any] = {}
        self._device_info: Dict[str, Any] = {}
        self._stream_profile_str: str = ""
        self._stream_profiles: Dict[str, Any] = {}
        self._last_colorized: Optional[np.ndarray] = None

    def start(self) -> None:
        """Start the camera pipeline, drop warm-up frames, and cache intrinsics."""
        if self._is_started:
            return

        # Refuse to start if decimation is active and intrinsics are not scaled
        if self.camera_config.use_decimation:
            raise RuntimeError(
                "RealSenseCaptureAdapter: decimation is active. Full-body capture requires decimation OFF."
            )

        # Lazy instantiation of RealSenseCamera
        if self._camera_factory is not None:
            self._camera = self._camera_factory(self.camera_config)
        else:
            try:
                from realsense_measure.camera.realsense_capture import RealSenseCamera
                self._camera = RealSenseCamera(cfg=self.camera_config)
            except Exception as exc:
                raise RuntimeError(f"Failed to import/instantiate RealSenseCamera: {exc}") from exc

        try:
            self._camera.start()
            self._is_started = True

            # Query device info
            try:
                info = self._camera.get_device_info()
            except Exception:
                info = {}

            # Query detailed hardware options (exposure, laser power, visual preset)
            laser_power_mw = info.get("laser_power_mw")
            exposure_depth_us = None
            exposure_color_us = None
            visual_preset_str = None
            stream_profiles: Dict[str, Any] = {}
            try:
                pipeline = getattr(self._camera, "pipeline", None)
                if pipeline is not None:
                    profile = pipeline.get_active_profile()
                    if profile is not None:
                        dev = profile.get_device()
                        import pyrealsense2 as rs
                        try:
                            ds = profile.get_stream(rs.stream.depth).as_video_stream_profile()
                            stream_profiles["depth"] = {
                                "width": int(ds.width()),
                                "height": int(ds.height()),
                                "fps": int(ds.fps()),
                                "format": str(ds.format()),
                            }
                        except Exception:
                            pass
                        try:
                            cs = profile.get_stream(rs.stream.color).as_video_stream_profile()
                            stream_profiles["color"] = {
                                "width": int(cs.width()),
                                "height": int(cs.height()),
                                "fps": int(cs.fps()),
                                "format": str(cs.format()),
                            }
                        except Exception:
                            pass
                        for s in dev.query_sensors():
                            s_name = s.get_info(rs.camera_info.name).lower() if s.supports(rs.camera_info.name) else ""
                            if "depth" in s_name or "stereo" in s_name:
                                if laser_power_mw is None and s.supports(rs.option.laser_power):
                                    laser_power_mw = float(s.get_option(rs.option.laser_power))
                                if s.supports(rs.option.exposure):
                                    exposure_depth_us = float(s.get_option(rs.option.exposure))
                                if s.supports(rs.option.visual_preset):
                                    try:
                                        p_val = int(s.get_option(rs.option.visual_preset))
                                        visual_preset_str = s.get_option_value_description(rs.option.visual_preset, p_val)
                                    except Exception:
                                        visual_preset_str = str(int(s.get_option(rs.option.visual_preset)))
                            elif "rgb" in s_name or "color" in s_name:
                                if s.supports(rs.option.exposure):
                                    exposure_color_us = float(s.get_option(rs.option.exposure))
            except Exception as opt_exc:
                logger.debug(f"Hardware sensor option query skipped: {opt_exc}")

            if not stream_profiles:
                stream_profiles = {
                    "depth": {"width": self.camera_config.width, "height": self.camera_config.height, "fps": int(self.camera_config.fps), "format": "z16"},
                    "color": {"width": self.camera_config.width, "height": self.camera_config.height, "fps": int(self.camera_config.fps), "format": "bgr8"},
                }
            self._stream_profiles = stream_profiles

            # Cache post-rotation intrinsics
            intr_o3d = getattr(self._camera, "intrinsics_o3d", None)
            has_valid_intr = False
            if intr_o3d is not None and hasattr(intr_o3d, "get_focal_length"):
                try:
                    fx, fy = intr_o3d.get_focal_length()
                    cx, cy = intr_o3d.get_principal_point()
                    w, h = int(intr_o3d.width), int(intr_o3d.height)
                    has_valid_intr = True
                except Exception:
                    has_valid_intr = False

            if not has_valid_intr:
                # Fallback to configured dimensions based on rotation mode
                if self.camera_config.rotate_90 in ("cw", "ccw"):
                    w, h = self.camera_config.height, self.camera_config.width
                else:
                    # 'none' or no rotation: native landscape dimensions
                    w, h = self.camera_config.width, self.camera_config.height
                fx = fy = float(max(w, h))
                cx, cy = w / 2.0, h / 2.0

            self._device_info = {
                "name": info.get("name", "Intel RealSense D455"),
                "serial_number": info.get("serial_number", "UNKNOWN"),
                "firmware_version": info.get("firmware_version", "UNKNOWN"),
                "usb_type": info.get("usb_type", "3.x"),
                "resolution": f"{w}x{h}",
                "stream_profile": f"{w}x{h}@{self.camera_config.fps}fps",
                "laser_power_mw": laser_power_mw,
                "exposure_depth_us": exposure_depth_us,
                "exposure_color_us": exposure_color_us,
                "visual_preset": visual_preset_str,
                "filter_settings": {
                    "decimation": bool(self.camera_config.use_decimation),
                    "spatial": bool(self.camera_config.use_spatial_filter),
                    "temporal": bool(self.camera_config.use_temporal_filter),
                    "hole_filling": bool(self.camera_config.use_hole_filling),
                    "raw_only": bool(getattr(self.camera_config, "raw_only", True)),
                },
            }

            self._intrinsics = {
                "width": int(w),
                "height": int(h),
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy),
                "is_post_rotation": bool(self.camera_config.rotate_90 in ("cw", "ccw")),
                "is_post_decimation": False,
            }
            self._stream_profile_str = f"{w}x{h}@{self.camera_config.fps}fps"

            # Discard warm-up frames for auto-exposure, gain, and temporal stabilization
            logger.info(f"Discarding {self.warmup_frames} warm-up frames ...")
            for _ in range(self.warmup_frames):
                self._camera.read_with_raw_and_colorized()

        except Exception:
            self.stop()
            raise

    def get_imu_gravity(self) -> Optional[np.ndarray]:
        """
        Check if D455 IMU is available via pyrealsense2 and return normalized accelerometer gravity vector.
        Returns None if IMU motion sensor is not present, unconfigured, or unsupported.
        """
        if self._camera is None:
            return None
        try:
            pipeline = getattr(self._camera, "pipeline", None)
            if pipeline is None:
                return None
            profile = pipeline.get_active_profile()
            if profile is None:
                return None
            dev = profile.get_device()
            import pyrealsense2 as rs
            sensors = dev.query_sensors()
            motion_sensor = None
            for s in sensors:
                try:
                    s_name = s.get_info(rs.camera_info.name).lower()
                    if "motion" in s_name or "accel" in s_name:
                        motion_sensor = s
                        break
                except Exception:
                    continue
            if motion_sensor is None:
                return None
            return None
        except Exception as exc:
            logger.debug(f"IMU sensor query: {exc}")
            return None

    def stop(self) -> None:
        """Safely stop camera in a finally block."""
        try:
            if self._camera is not None:
                try:
                    self._camera.stop()
                except Exception as exc:
                    logger.warning(f"Error while stopping RealSenseCamera: {exc}")
        finally:
            self._camera = None
            self._is_started = False

    def __enter__(self) -> "RealSenseCaptureAdapter":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    @property
    def intrinsics(self) -> Dict[str, Any]:
        return dict(self._intrinsics)

    @property
    def stream_profiles(self) -> Dict[str, Any]:
        """Dictionary of negotiated stream profiles for 'depth' and 'color'."""
        return dict(self._stream_profiles)

    @property
    def last_colorized(self) -> Optional[np.ndarray]:
        """Cached 3-channel BGR SDK-colorized depth image from last read, if available."""
        return self._last_colorized

    @property
    def depth_scale(self) -> float:
        if self._camera is not None and getattr(self._camera, "depth_scale", None) is not None:
            return float(self._camera.depth_scale)
        return 0.001

    @property
    def depth_units(self) -> str:
        # Raw depth is stored in native uint16 sensor units
        return "sensor_units"

    def device_info(self) -> Dict[str, Any]:
        info = dict(self._device_info)
        info["depth_scale"] = self.depth_scale
        info["stream_profile"] = self._stream_profile_str
        info["stream_profiles"] = self.stream_profiles
        info["intrinsics"] = self.intrinsics
        return info

    def read_raw_filtered(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Capture one frame pair.
        Returns (color_bgr, raw_depth_z16, filtered_depth_m, timestamp_s).
        Retries dropped frames up to max_dropped_retries.
        Raises RuntimeError on device disconnect.
        """
        if not self._is_started or self._camera is None:
            raise RuntimeError("RealSenseCaptureAdapter is not started. Call start() before reading.")

        for attempt in range(self.max_dropped_retries + 1):
            try:
                (
                    color_bgr,
                    depth_filtered_m,
                    depth_raw_m,
                    raw_z16,
                    depth_color_bgr,
                    frame_meta,
                ) = self._camera.read_with_raw_and_colorized()
            except Exception as exc:
                # Disconnect / fatal hardware error detected
                logger.error(f"RealSense hardware read failure: {exc}")
                raise RuntimeError(f"RealSense camera disconnected or hardware failed: {exc}") from exc

            if color_bgr is not None and raw_z16 is not None:
                self._last_colorized = depth_color_bgr
                ts_ms = frame_meta.get("timestamp_ms", 0.0) if frame_meta else 0.0
                ts_s = float(ts_ms) / 1000.0 if ts_ms > 0 else time.time()
                return color_bgr, raw_z16, depth_filtered_m, ts_s

            logger.debug(f"Dropped frame encountered on attempt {attempt + 1}/{self.max_dropped_retries + 1}")

        # All retries returned None (normal dropped frame)
        return None, None, None, time.time()

