"""
tests/fullbody/test_synthetic_background.py

Synthetic tests for background calibration, floor plane recovery, and foreground masking:
- Tilted-floor scene recovers gravity within 1 deg.
- Person occupying a region is masked cleanly via background subtraction.
- Wall behind the person never enters the point cloud.
- Mask pixel counts per stop on a multi-stop fake-camera session.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import cv2
import numpy as np
import open3d as o3d
import pytest

from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.floor import fit_floor_plane
from realsense_measure.fullbody.fusion import compute_foreground_mask
from tests.fullbody.fake_camera import FakeCamera


def test_tilted_floor_scene_recovers_gravity():
    """Verify that a synthetic tilted-floor scene recovers gravity normal within 1 deg."""
    h, w = 480, 848
    fx = fy = 428.0
    cx, cy = w / 2.0, h / 2.0
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    cfg = FullBodyConfig()

    # Ground truth: floor tilted backwards (camera pitch down by 6.0 degrees)
    # Floor plane in camera coordinates: n_cam dot P + d = 0
    pitch_rad = np.deg2rad(6.0)
    # Looking down slightly: floor normal tilts towards +Z (normal pointing upwards: [0, -cos(pitch), sin(pitch)])
    gt_normal = np.array([0.0, -np.cos(pitch_rad), np.sin(pitch_rad)], dtype=np.float64)
    gt_normal /= np.linalg.norm(gt_normal)
    # Floor height: 0.95 m below camera
    d_val = 0.95

    # Synthesize lower 40% image rows as floor plane
    depth = np.zeros((h, w), dtype=np.float32)
    u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    ray_x = (u_grid - cx) / fx
    ray_y = (v_grid - cy) / fy
    ray_z = np.ones((h, w), dtype=np.float32)

    # For points on plane: n dot (ray * z) + d = 0 => z = -d / (n_x*rx + n_y*ry + n_z*rz)
    denom = gt_normal[0] * ray_x + gt_normal[1] * ray_y + gt_normal[2] * ray_z
    z_floor = np.where(denom < -1e-4, -d_val / denom, 0.0)

    # Floor exists in lower rows where ray_y > 0 (downwards)
    row_start = int(0.55 * h)
    valid_floor_region = (z_floor > 0.5) & (z_floor < 4.0)
    depth[row_start:, :] = np.where(valid_floor_region[row_start:, :], z_floor[row_start:, :], 0.0)

    # Fit floor plane
    floor = fit_floor_plane(depth, intr, cfg)
    assert floor is not None, "Floor fitting failed on synthetic tilted floor"

    # Angle between estimated normal and ground truth normal
    cos_angle = abs(float(np.dot(floor.normal, gt_normal)))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angular_err_deg = float(np.rad2deg(np.arccos(cos_angle)))

    assert angular_err_deg < 1.0, f"Floor normal recovery error {angular_err_deg:.3f} deg exceeds 1 deg limit"


def test_foreground_mask_person_cleanly_masked_and_wall_rejected():
    """Verify person is masked cleanly by background subtraction while wall is completely rejected."""
    h, w = 480, 848
    cfg = FullBodyConfig()

    # 1. Background scene: wall at 3.2 m across the entire frame
    bg_depth = np.full((h, w), 3.20, dtype=np.float32)

    # 2. Stop frame: wall at 3.2 m with a standing person at 2.0 m in central region
    stop_depth = bg_depth.copy()
    # Person box: rows 60 to 420, columns 320 to 520
    person_slice = (slice(60, 420), slice(320, 520))
    stop_depth[person_slice] = 2.00

    # 3. Compute foreground mask
    fg_mask = compute_foreground_mask(stop_depth, bg_depth, cfg)

    # Person region must be masked True
    person_pixels = fg_mask[person_slice]
    assert np.all(person_pixels[10:-10, 10:-10]), "Interior of person should be 100% foreground"
    assert np.sum(person_pixels) > 50000

    # Wall outside person must be strictly False (never enters point cloud)
    wall_slice_left = fg_mask[:, :300]
    wall_slice_right = fg_mask[:, 540:]
    assert np.sum(wall_slice_left) == 0, "Wall on left must have 0 foreground pixels"
    assert np.sum(wall_slice_right) == 0, "Wall on right must have 0 foreground pixels"


def test_fake_camera_session_mask_counts(capsys):
    """Run a multi-stop fake camera session and report mask pixel counts per stop."""
    cfg = FullBodyConfig()
    cam = FakeCamera(width=848, height=480, base_depth_m=2.0)
    cam.start()

    # 1. Capture background (empty scene: person absent)
    # For synthetic background: wall at 3.0 m
    bg_depth = np.full((480, 848), 3.00, dtype=np.float32)

    # 2. Acquire 4 stops from FakeCamera
    mask_counts = []
    print("\n" + "=" * 50)
    print("FAKE-CAMERA SESSION: MASK PIXEL COUNTS PER STOP")
    print("=" * 50)
    print(f"{'Stop ID':<10} | {'Mask Pixel Count':<20} | {'Status':<10}")
    print("-" * 50)

    for stop_id in range(4):
        c, r, f, ts = cam.read_raw_filtered()
        scale = cam.depth_scale
        d_m = r.astype(np.float32) * scale

        fg_mask = compute_foreground_mask(d_m, bg_depth, cfg)
        px_count = int(np.sum(fg_mask))
        mask_counts.append(px_count)

        status = "OK" if px_count > 10000 else "LOW"
        print(f"Stop {stop_id:<5} | {px_count:<20d} | {status:<10}")
        assert px_count > 10000, f"Stop {stop_id} should have substantial foreground pixels"

    print("=" * 50 + "\n")
    cam.stop()

    assert len(mask_counts) == 4
    # All stops with synthetic humanoid ellipsoid should have consistent non-zero counts
    for cnt in mask_counts:
        assert cnt > 15000

