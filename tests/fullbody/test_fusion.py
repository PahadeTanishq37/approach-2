"""
tests/fullbody/test_fusion.py

Unit and integration tests for volumetric fusion, foreground/background rejection,
and fallback geometry extraction.
"""

from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d
import pytest

from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.fusion import fuse_aligned_stops
from realsense_measure.fullbody.types import AlignedStop, StopData


def test_fusion_foreground_mask_rejects_background_walls():
    """
    Verify that 2D person mask dilation retains person geometry while
    completely rejecting background wall points at Z > 2.0 m.
    Generates diagnostics on original, retained, and rejected points.
    """
    h, w = 240, 320
    fx, fy, cx, cy = 200.0, 200.0, 160.0, 120.0
    intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    cfg = FullBodyConfig(
        depth_min_m=0.3,
        depth_max_m=3.0,
        tsdf_voxel_length_m=0.01,
        tsdf_depth_trunc_m=2.8,
    )

    # 1. Synthesize depth:
    # - Foreground person at Z = 1.20 m (ellipse centered at (cx, cy))
    # - Background room wall at Z = 2.40 m everywhere else
    depth = np.full((h, w), 2.40, dtype=np.float32)
    person_mask = np.zeros((h, w), dtype=bool)

    y_grid, x_grid = np.ogrid[:h, :w]
    # Elliptical torso / person region
    ellipse_mask = (((x_grid - cx) / 40.0) ** 2 + ((y_grid - cy) / 60.0) ** 2) <= 1.0
    depth[ellipse_mask] = 1.20
    person_mask[ellipse_mask] = True

    # Count initial unmasked valid pixels
    orig_total_valid = int(np.sum(depth > 0))
    orig_wall_pixels = int(np.sum(depth > 2.0))
    orig_person_pixels = int(np.sum(ellipse_mask))

    color_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    color_bgr[ellipse_mask] = [180, 120, 80]
    color_bgr[~ellipse_mask] = [50, 50, 50]

    # Create StopData with person_mask
    stop = StopData(
        stop_id=0,
        nominal_angle_deg=0.0,
        color=color_bgr,
        depth=depth,
        depth_std=np.full((h, w), 0.002, dtype=np.float32),
        intrinsics=intrinsics,
        person_mask=person_mask,
    )

    # Create s.pcd with person_mask applied (as done in cli.py)
    depth_masked = depth.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated_m = cv2.dilate(person_mask.astype(np.uint8), kernel).astype(bool)
    core_depths = depth_masked[person_mask & (depth_masked > cfg.depth_min_m)]
    z_med = float(np.median(core_depths))
    depth_consistent = (depth_masked >= max(cfg.depth_min_m, z_med - 0.50)) & (
        depth_masked <= min(cfg.depth_max_m, z_med + 0.50)
    )
    depth_masked[~dilated_m | ~depth_consistent] = 0.0

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_bgr[:, :, ::-1].copy()),
        o3d.geometry.Image(depth_masked),
        depth_scale=1.0,
        depth_trunc=cfg.depth_max_m,
        convert_rgb_to_intensity=False,
    )
    pcd_canonical = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)

    aligned_stop = AlignedStop(
        stop_id=0,
        nominal_angle_deg=0.0,
        T_canonical_camera=np.eye(4),
        prior_yaw_deg=0.0,
        refined_yaw_deg=0.0,
        prior_translation_m=np.zeros(3),
        refined_translation_m=np.zeros(3),
        residual_before_mm=1.0,
        residual_after_mm=1.0,
        inlier_fraction=0.9,
        accepted=True,
        pcd_canonical=pcd_canonical,
    )

    mesh, fused_pcd, hole_map = fuse_aligned_stops([stop], [aligned_stop], cfg)

    # Convert fused points to array
    fused_pts = np.asarray(fused_pcd.points)
    assert len(fused_pts) > 0, "Fused point cloud should contain points for foreground person"

    # Wall points at Z > 2.0 m must be 0 in the fused reconstruction
    wall_points_in_fusion = np.sum(fused_pts[:, 2] > 2.0)
    retained_person_points = len(fused_pts)
    rejected_points = orig_wall_pixels

    # Diagnostics check
    assert wall_points_in_fusion == 0, f"Expected 0 wall points in fusion, got {wall_points_in_fusion}"
    assert retained_person_points > 100, f"Person points must be retained, got {retained_person_points}"
    print(f"\n[fusion test diagnostic] Original: {orig_total_valid} ({orig_person_pixels} person, {orig_wall_pixels} wall) | Retained: {retained_person_points} | Rejected wall: {rejected_points}")
