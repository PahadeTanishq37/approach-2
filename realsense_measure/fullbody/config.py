"""
realsense_measure/fullbody/config.py

Configuration parameters for full-body scanning, alignment, fusion,
and anthropometric measurement.
All thresholds are defined here; modules must not use hard-coded magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class FullBodyConfig:
    """
    Central configuration for the fullbody reconstruction and measurement pipeline.
    """
    # -------------------------------------------------------------------------
    # Protocol & Stops
    # -------------------------------------------------------------------------
    stop_count: int = 8                          # 8 or 12 stops (360 / stop_count nominal step)
    num_stops: int = 8                           # backward compatibility alias for stop_count
    expected_step_deg: float = 45.0              # 360.0 / stop_count
    turn_direction: str = "cw"                   # "cw" (clockwise from above) or "ccw"
    turntable: bool = False                      # True if subject stands on automated/manual turntable
    resolution: str = "1280x720"                 # "1280x720" (full-body default) or "848x480"
    camera_rotation: str = "none"                # "none" (native landscape), "cw", or "ccw"
    camera_portrait_rotation: str = "none"       # alias for backward compatibility
    max_angle_deviation_deg: float = 25.0
    burst_frame_count: int = 15                  # 15 frames per stop burst (default)
    motion_limit_mm: float = 10.0                # 10 mm sway limit enforced during burst
    max_retakes: int = 3                         # max retakes per stop
    raw_only: bool = True                        # save raw z16/color, disable HW temporal/hole filters
    camera_height_m: float = 0.95                # UNUSED: dynamically taken from fitted floor plane (abs(floor.coefficients[3]))
    camera_distance_m: float = 2.00
    capture_dist_min_m: float = 1.70
    capture_dist_max_m: float = 2.40
    capture_range_min_m: float = 1.70            # backward compatibility alias
    capture_range_max_m: float = 2.40            # backward compatibility alias
    expected_intrinsics: Optional[Dict[str, float]] = None  # Expected active stream intrinsics {width, height, fx, fy, cx, cy}

    # -------------------------------------------------------------------------
    # Burst processing
    # -------------------------------------------------------------------------
    burst_min_valid_frac: float = 0.50
    flying_pixel_depth_gradient_threshold_m: float = 0.04
    hole_fill_max_radius_px: int = 3
    temporal_std_outlier_sigma: float = 3.0
    depth_min_m: float = 0.40
    depth_max_m: float = 4.00

    # -------------------------------------------------------------------------
    # Floor detection
    # -------------------------------------------------------------------------
    floor_lower_image_fraction: float = 0.35  # bottom 35% of image rows
    floor_distance_threshold_m: float = 0.015  # 15 mm
    floor_ransac_iterations: int = 2000
    floor_normal_min_up_alignment: float = 0.70  # cos(45 deg) approx
    floor_max_angle_diff_deg: float = 4.0        # max normal discrepancy across stops

    # -------------------------------------------------------------------------
    # Confidence weighting
    # -------------------------------------------------------------------------
    min_confidence_threshold: float = 0.05
    incidence_angle_power: float = 2.0
    edge_distance_falloff_px: float = 8.0  # increased from 5 → 8 px: makes ring coverage robust to sensor noise at body edges
    noise_sigma_base_m: float = 0.0015           # 1.5 mm at 1.0 m (sigma_z = 0.0015 * (z / 1.0)^2)
    noise_reference_distance_m: float = 1.00

    # -------------------------------------------------------------------------
    # 3D Keypoints & Skeletal Estimation
    # -------------------------------------------------------------------------
    keypoint_window_size_px: int = 5
    keypoint_min_confidence: float = 0.40
    keypoint_min_visibility: float = 0.50
    keypoint_depth_mad_scale: float = 2.5

    # -------------------------------------------------------------------------
    # Body Part Segmentation (Capsules)
    # -------------------------------------------------------------------------
    capsule_radius_head_m: float = 0.13
    capsule_radius_torso_m: float = 0.22
    capsule_radius_pelvis_m: float = 0.18
    capsule_radius_upper_arm_m: float = 0.09
    capsule_radius_forearm_m: float = 0.07
    capsule_radius_thigh_m: float = 0.13
    capsule_radius_shank_m: float = 0.09

    # -------------------------------------------------------------------------
    # Multi-Stop Alignment
    # -------------------------------------------------------------------------
    alignment_max_yaw_correction_deg: float = 15.0
    alignment_max_translation_correction_m: float = 0.08
    alignment_icp_distance_threshold_m: float = 0.06
    alignment_max_iterations: int = 15
    alignment_convergence_eps: float = 1e-4
    alignment_min_inlier_fraction: float = 0.40
    rigid_parts_only: bool = True
    body_axis_drift_sigma_m: float = 0.02
    max_inter_stop_residual_mm: float = 15.0

    # -------------------------------------------------------------------------
    # TSDF Fusion
    # -------------------------------------------------------------------------
    tsdf_voxel_length_m: float = 0.005  # 5 mm
    tsdf_sdf_trunc_m: float = 0.020     # 20 mm (4x voxel)
    tsdf_depth_trunc_m: float = 4.00
    min_mesh_component_ratio: float = 0.02
    keep_dominant_mesh_component: bool = False

    # -------------------------------------------------------------------------
    # Anthropometric Measurements
    # -------------------------------------------------------------------------
    slice_thickness_m: float = 0.020    # 20 mm slab (increased from 15 mm: more points in slab reduces chest underestimate)
    num_angular_bins: int = 180         # 2 deg resolution
    min_angular_coverage_for_ok: float = 0.70
    min_angular_coverage_for_partial: float = 0.35
    crown_hole_threshold: float = 0.20  # hole map fraction above which crown is flagged unobserved
    bootstrap_iterations: int = 200
    crown_edge_margin_px: int = 12

    # -------------------------------------------------------------------------
    # Body Template (Optional Plugin)
    # -------------------------------------------------------------------------
    arap_max_iterations: int = 20
    arap_regularization_weight: float = 50.0

    # -------------------------------------------------------------------------
    # Background & Calibration Assets
    # -------------------------------------------------------------------------
    background_dir: Path = field(default_factory=lambda: Path("raw_captures/background"))

    # -------------------------------------------------------------------------
    # Output & Diagnostics
    # -------------------------------------------------------------------------
    output_dir: Path = field(default_factory=lambda: Path("scan_output/fullbody"))
    save_debug_visualizations: bool = False
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.turn_direction not in ("cw", "ccw"):
            raise ValueError(
                f"Invalid turn_direction '{self.turn_direction}'. Must be 'cw' or 'ccw'."
            )

        # Synchronize stop_count and num_stops
        if self.stop_count not in (8, 12):
            if self.num_stops in (8, 12):
                self.stop_count = self.num_stops
            else:
                raise ValueError(f"Invalid stop_count {self.stop_count}. Must be 8 or 12.")
        self.num_stops = self.stop_count
        self.expected_step_deg = 360.0 / float(self.stop_count)

        # Resolution validation
        res_norm = str(self.resolution).lower().strip()
        if res_norm not in ("1280x720", "848x480"):
            raise ValueError(f"Invalid resolution '{self.resolution}'. Must be '1280x720' or '848x480'.")
        self.resolution = res_norm

        # Synchronize camera_rotation and camera_portrait_rotation
        if self.camera_rotation != "none" and self.camera_portrait_rotation == "none":
            self.camera_portrait_rotation = self.camera_rotation
        elif self.camera_portrait_rotation != "none" and self.camera_rotation == "none":
            self.camera_rotation = self.camera_portrait_rotation

        if self.camera_rotation not in ("cw", "ccw", "none"):
            raise ValueError(
                f"Invalid camera_rotation '{self.camera_rotation}'. Must be 'cw', 'ccw', or 'none'."
            )
        if self.camera_portrait_rotation not in ("cw", "ccw", "none"):
            raise ValueError(
                f"Invalid camera_portrait_rotation '{self.camera_portrait_rotation}'. Must be 'cw', 'ccw', or 'none'."
            )

        self.capture_range_min_m = self.capture_dist_min_m
        self.capture_range_max_m = self.capture_dist_max_m


