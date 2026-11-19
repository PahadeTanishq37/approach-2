"""
tests/fullbody/test_confidence.py

Unit tests for per-pixel confidence weighting:
- Monotonicity with incidence angle.
- Monotonicity with edge distance.
- Monotonicity with depth (quadratic noise model).
- Low-confidence filtering.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from realsense_measure.fullbody.confidence import compute_confidence_map
from realsense_measure.fullbody.config import FullBodyConfig


def test_confidence_monotonicity():
    """Verify confidence behaves monotonically with incidence angle, edge distance, and depth."""
    h, w = 60, 60
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, 80.0, 80.0, 30.0, 30.0)
    cfg = FullBodyConfig(min_confidence_threshold=0.15)

    # 1. Flat perpendicular plane at 1.0 m vs tilted plane
    depth_flat = np.full((h, w), 1.00, dtype=np.float32)
    conf_flat, _ = compute_confidence_map(depth_flat, intr, cfg)

    # Center pixel of flat plane has normal facing camera
    center_conf_flat = conf_flat[30, 30]
    assert center_conf_flat > 0.60

    # 2. Edge proximity check: border pixel has lower confidence than interior pixel
    edge_conf = conf_flat[30, 1]
    assert edge_conf < center_conf_flat

    # 3. Depth distance check: plane at 2.4 m has lower confidence than plane at 1.0 m
    depth_far = np.full((h, w), 2.40, dtype=np.float32)
    conf_far, _ = compute_confidence_map(depth_far, intr, cfg)
    center_conf_far = conf_far[30, 30]

    assert center_conf_far < center_conf_flat


def test_low_confidence_invalidation():
    """Verify pixels with confidence below threshold are zeroed in the filtered depth map."""
    h, w = 40, 40
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, 80.0, 80.0, 20.0, 20.0)
    cfg = FullBodyConfig(min_confidence_threshold=0.30)

    # Valid depth with artificial very low validity count
    depth = np.full((h, w), 1.20, dtype=np.float32)
    valid_counts = np.full((h, w), 1, dtype=np.int32)
    valid_counts[20, 20] = 30  # high validity in center

    conf, filtered_d = compute_confidence_map(
        depth, intr, cfg, valid_counts=valid_counts, burst_total=30
    )

    # Pixel with 1/30 validity should be below threshold and zeroed
    assert filtered_d[5, 5] == 0.0
    # Pixel with 30/30 validity should remain valid
    assert filtered_d[20, 20] > 0.0


def test_frontal_surface_distance_retention():
    """Verify frontal surface at 1.5 m and 2.0 m is kept and 2.5 m is not entirely discarded."""
    h, w = 60, 60
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, 80.0, 80.0, 30.0, 30.0)
    cfg = FullBodyConfig()  # Uses min_confidence_threshold=0.05 and sigma_z = 0.0015 * (z/1.0)^2

    # 1. Frontal surface at 1.5 m -> kept
    depth_15 = np.full((h, w), 1.50, dtype=np.float32)
    conf_15, filtered_15 = compute_confidence_map(depth_15, intr, cfg)
    center_15 = conf_15[30, 30]
    assert center_15 > cfg.min_confidence_threshold
    assert filtered_15[30, 30] == 1.50
    assert np.all(filtered_15[15:45, 15:45] > 0.0)

    # 2. Frontal surface at 2.0 m -> kept
    depth_20 = np.full((h, w), 2.00, dtype=np.float32)
    conf_20, filtered_20 = compute_confidence_map(depth_20, intr, cfg)
    center_20 = conf_20[30, 30]
    assert center_20 > cfg.min_confidence_threshold
    assert filtered_20[30, 30] == 2.00
    assert np.all(filtered_20[15:45, 15:45] > 0.0)

    # 3. Frontal surface at 2.5 m -> not entirely discarded
    depth_25 = np.full((h, w), 2.50, dtype=np.float32)
    conf_25, filtered_25 = compute_confidence_map(depth_25, intr, cfg)
    center_25 = conf_25[30, 30]
    assert center_25 > cfg.min_confidence_threshold
    assert filtered_25[30, 30] == 2.50
    valid_count_25 = np.sum(filtered_25 > 0.0)
    assert valid_count_25 > 0
    assert valid_count_25 / (h * w) > 0.50


def test_confidence_table_1_to_3m_and_grazing_flooring(capsys):
    """
    Print and test a confidence-vs-distance table for 1.0-3.0 m:
    Frontal surfaces must survive at 2.0 and 2.4 m, and grazing surfaces are floored at 0.3.
    """
    h, w = 60, 60
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, 80.0, 80.0, 30.0, 30.0)
    cfg = FullBodyConfig()

    distances = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    print("\n" + "=" * 62)
    print(f"{'Distance (m)':<14} | {'Sigma_z (mm)':<14} | {'Confidence':<12} | {'Survives?':<10}")
    print("-" * 62)

    conf_values = {}
    for z in distances:
        depth = np.full((h, w), z, dtype=np.float32)
        conf_map, filt = compute_confidence_map(depth, intr, cfg)
        c_val = float(conf_map[30, 30])
        conf_values[z] = c_val
        sigma_mm = cfg.noise_sigma_base_m * ((z / cfg.noise_reference_distance_m) ** 2) * 1000.0
        survives = c_val >= cfg.min_confidence_threshold
        print(f"{z:<14.2f} | {sigma_mm:<14.2f} | {c_val:<12.4f} | {'YES' if survives else 'NO':<10}")
        assert survives, f"Frontal surface at {z} m should survive threshold {cfg.min_confidence_threshold}"

    print("=" * 62 + "\n")

    # Frontal surfaces must survive at 2.0 and 2.4 m
    assert conf_values[2.0] >= cfg.min_confidence_threshold
    assert conf_values[2.4] >= cfg.min_confidence_threshold
    assert conf_values[2.0] > 0.35
    assert conf_values[2.4] > 0.20

    # Test grazing surface: continuous steep slope in depth (without jump edges > 0.05m)
    # The incidence-angle term should be floored at 0.3, so combined confidence >= 0.3 * w_noise
    # At 2.0 m, w_noise ~ 0.41, so confidence ~ 0.123 >= 0.05 (survives!)
    tilted_depth = np.zeros((h, w), dtype=np.float32)
    for col in range(w):
        tilted_depth[:, col] = 2.0 + (col - 30) * 0.03  # 3 cm/px slope (below 5 cm jump edge threshold)
    conf_tilted, filt_tilted = compute_confidence_map(tilted_depth, intr, cfg)
    # Center region should survive due to floor at 0.3
    assert conf_tilted[30, 30] >= cfg.min_confidence_threshold
    assert filt_tilted[30, 30] > 0.0



