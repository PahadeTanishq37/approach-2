"""
realsense_measure/fullbody/debug.py

Per-stop debugging artifacts and quantitative diagnostics:
- mask_overlay.png: Color frame with foreground mask contour and 4% margins
- depth_diff.png: Background minus stop depth clipped to 0-0.30 m with 3 cm contour
- cloud.ply: Foreground masked point cloud in the gravity frame
- stats.json: Pure data-derived metrics
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import open3d as o3d

from realsense_measure.fullbody.alignment import compute_gravity_transform
from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.floor import fit_floor_plane, load_floor_plane
from realsense_measure.fullbody.fusion import compute_foreground_mask
from realsense_measure.fullbody.io import load_capture_session
from realsense_measure.fullbody.types import FloorPlane, StopData

logger = logging.getLogger(__name__)


def debug_stop(
    session_dir: Union[str, Path],
    stop_index: int,
    out_dir: Optional[Union[str, Path]] = None,
    cfg: Optional[FullBodyConfig] = None,
) -> Dict[str, Any]:
    """
    Generate debugging visualisations and quantitative statistics for a single stop.

    Saves to <session_dir>/debug/stop_NN/ (or out_dir):
      - mask_overlay.png: Color frame with foreground mask edge and 4% margins
      - depth_diff.png: Background minus stop depth, clipped 0-0.30 m, with 3 cm contour
      - cloud.ply: Masked point cloud in the gravity frame (Y-up, floor at Y=0)
      - stats.json: Computed strictly from data only
    """
    if cfg is None:
        cfg = FullBodyConfig()

    sess_p = Path(session_dir)
    if not sess_p.exists():
        raise FileNotFoundError(f"Session directory not found: {sess_p}")

    # 1. Load capture session
    stops = load_capture_session(sess_p, cfg)
    if not stops:
        raise ValueError(f"No stops found in session '{sess_p}'")
    if stop_index < 0 or stop_index >= len(stops):
        raise IndexError(f"Stop index {stop_index} out of range (session has {len(stops)} stops)")

    stop: StopData = stops[stop_index]
    if stop.depth is None or stop.color is None:
        raise ValueError(f"Stop {stop_index} is missing color or depth data")

    # Determine destination directory
    if out_dir is not None:
        target_dir = Path(out_dir)
    else:
        target_dir = sess_p / "debug" / f"stop_{stop_index:02d}"
    target_dir.mkdir(parents=True, exist_ok=True)

    # 2. Resolve background depth map and floor plane
    bg_depth: Optional[np.ndarray] = None
    bg_floor: Optional[FloorPlane] = None

    # Check session background directory first, then fallback candidates
    bg_candidates = [
        sess_p / "background",
        sess_p,
        Path("raw_captures/background"),
        Path("background"),
    ]

    for cand in bg_candidates:
        if cand.exists():
            depth_file = cand / "background_depth.npy"
            if depth_file.exists() and bg_depth is None:
                try:
                    bg_depth = np.load(str(depth_file)).astype(np.float32)
                except Exception as e:
                    logger.warning(f"Failed loading {depth_file}: {e}")

            floor_file = cand / "background_floor.json"
            if floor_file.exists() and bg_floor is None:
                try:
                    bg_floor = load_floor_plane(floor_file)
                except Exception as e:
                    logger.warning(f"Failed loading {floor_file}: {e}")

    # Fallback floor if no pre-calibrated background floor exists
    if bg_floor is None:
        bg_floor = fit_floor_plane(stop.depth, stop.intrinsics, cfg)
        if bg_floor is None:
            # Synthetic horizontal plane under camera
            normal = np.array([0.0, -1.0, 0.0], dtype=np.float64)
            d = 0.95
            bg_floor = FloorPlane(
                coefficients=np.array([0.0, -1.0, 0.0, d]),
                normal=normal,
                point_on_plane=np.array([0.0, -d, 2.0]),
                inlier_count=1000,
                inlier_ratio=1.0,
                rmse_m=0.002,
                gravity_R_cam=np.diag([-1.0, -1.0, 1.0]),
            )

    # Fallback background depth if none found
    if bg_depth is None:
        # Create virtual background slightly behind subject standoff
        bg_depth = np.full_like(stop.depth, cfg.camera_distance_m + 0.50, dtype=np.float32)

    # Ensure background matches stop depth shape
    h, w = stop.depth.shape[:2]
    if bg_depth.shape[:2] != (h, w):
        bg_depth = cv2.resize(bg_depth, (w, h), interpolation=cv2.INTER_NEAREST)

    # 3. Compute foreground mask
    fg_mask = compute_foreground_mask(stop.depth, bg_depth, cfg)
    fg_px_count = int(np.sum(fg_mask))

    # 4. Generate mask_overlay.png (mask edge, 4% margins)
    overlay = stop.color.copy()

    # 4% margins
    margin_top = int(0.04 * h)
    margin_bottom = int(0.96 * h)
    margin_left = int(0.04 * w)
    margin_right = int(0.96 * w)

    # Draw margin guide lines
    cv2.line(overlay, (margin_left, 0), (margin_left, h), (255, 0, 255), 1, cv2.LINE_AA)
    cv2.line(overlay, (margin_right, 0), (margin_right, h), (255, 0, 255), 1, cv2.LINE_AA)
    cv2.line(overlay, (0, margin_top), (w, margin_top), (255, 0, 255), 1, cv2.LINE_AA)
    cv2.line(overlay, (0, margin_bottom), (w, margin_bottom), (255, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(overlay, "4% Margin", (margin_left + 6, margin_top + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

    # Foreground mask edge
    contours, _ = cv2.findContours(fg_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    mask_overlay_path = target_dir / "mask_overlay.png"
    cv2.imwrite(str(mask_overlay_path), overlay)

    # 5. Generate depth_diff.png (bg minus stop depth, clipped 0-0.30 m, 3 cm contour)
    depth_diff = bg_depth - stop.depth
    depth_diff_clipped = np.clip(depth_diff, 0.0, 0.30)
    norm_diff = (depth_diff_clipped / 0.30 * 255.0).astype(np.uint8)
    diff_color = cv2.applyColorMap(norm_diff, cv2.COLORMAP_TURBO)

    # 3 cm contour (foreground threshold boundary)
    contour_3cm_mask = (depth_diff >= 0.03).astype(np.uint8)
    contours_3cm, _ = cv2.findContours(contour_3cm_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(diff_color, contours_3cm, -1, (255, 255, 255), 2)
    cv2.putText(diff_color, "3 cm Contour (FG Threshold)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    depth_diff_path = target_dir / "depth_diff.png"
    cv2.imwrite(str(depth_diff_path), diff_color)

    # 6. De-project foreground into 3D gravity frame and save cloud.ply
    fx = stop.intrinsics.get_focal_length()[0]
    fy = stop.intrinsics.get_focal_length()[1]
    cx = stop.intrinsics.get_principal_point()[0]
    cy = stop.intrinsics.get_principal_point()[1]

    valid_mask = fg_mask & (stop.depth > 0) & np.isfinite(stop.depth)
    v_idx, u_idx = np.where(valid_mask)
    z_vals = stop.depth[v_idx, u_idx]
    x_vals = (u_idx - cx) * z_vals / fx
    y_vals = (v_idx - cy) * z_vals / fy
    pts_cam = np.column_stack([x_vals, y_vals, z_vals])

    # Convert BGR to RGB colors
    colors_rgb = stop.color[v_idx, u_idx, ::-1].astype(np.float64) / 255.0

    T_grav_cam = compute_gravity_transform(bg_floor)
    pts_grav = (pts_cam @ T_grav_cam[:3, :3].T) + T_grav_cam[:3, 3]

    pcd = o3d.geometry.PointCloud()
    if len(pts_grav) > 0:
        pcd.points = o3d.utility.Vector3dVector(pts_grav)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

    cloud_ply_path = target_dir / "cloud.ply"
    o3d.io.write_point_cloud(str(cloud_ply_path), pcd)

    # 7. Compute stats.json from data only
    if fg_px_count > 0:
        ymin = int(np.min(v_idx))
        ymax = int(np.max(v_idx))
        xmin = int(np.min(u_idx))
        xmax = int(np.max(u_idx))
        bbox = [ymin, xmin, ymax, xmax]

        # Torso region: vertical span 30% to 60% of bbox
        torso_ymin = ymin + int(0.30 * (ymax - ymin))
        torso_ymax = ymin + int(0.60 * (ymax - ymin))
        torso_mask = (v_idx >= torso_ymin) & (v_idx <= torso_ymax)
        if np.any(torso_mask):
            torso_med_d = float(np.median(z_vals[torso_mask]))
        else:
            torso_med_d = float(np.median(z_vals))

        # Torso temporal std (depth noise) in mm
        if stop.depth_std is not None and np.any(torso_mask):
            torso_std_mm = float(np.median(stop.depth_std[v_idx[torso_mask], u_idx[torso_mask]]) * 1000.0)
        else:
            torso_std_mm = 0.0

        # Lowest foreground row vs floor contact estimate
        # Floor contact row estimate at subject's torso median depth:
        # Floor equation in camera coordinates: n_x X + n_y Y + n_z Z + d = 0
        n_floor = bg_floor.normal
        d_floor = float(bg_floor.coefficients[3])
        if abs(n_floor[1]) > 1e-4 and torso_med_d > 0:
            # At X = 0 (central column), Z = torso_med_d:
            # Y = -(n_z * Z + d) / n_y = (v - cy) * Z / fy
            v_floor_est = cy - fy * (n_floor[2] * torso_med_d + d_floor) / (n_floor[1] * torso_med_d)
            v_floor_est = float(v_floor_est)
        else:
            v_floor_est = float(h)

        row_diff_floor = float(v_floor_est - ymax)

        # Floor-pixel leakage in bottom 8% of bbox
        bot_8pct_ymin = ymax - int(0.08 * (ymax - ymin))
        bot_8pct_sel = v_idx >= bot_8pct_ymin
        total_bot_px = int(np.sum(bot_8pct_sel))
        if total_bot_px > 0:
            # Distance to floor plane in camera coords
            denom = float(np.linalg.norm(n_floor))
            plane_dist = np.abs(
                n_floor[0] * pts_cam[bot_8pct_sel, 0]
                + n_floor[1] * pts_cam[bot_8pct_sel, 1]
                + n_floor[2] * pts_cam[bot_8pct_sel, 2]
                + d_floor
            ) / max(denom, 1e-6)
            leakage_px = int(np.sum(plane_dist < 0.02))  # within 2 cm of floor
            leakage_frac = float(leakage_px / total_bot_px)
        else:
            leakage_px = 0
            leakage_frac = 0.0

        # Points per height band (10 bands from Y = 0.0 to 2.0 m in gravity frame)
        band_edges = np.linspace(0.0, 2.0, 11)
        points_per_height_band = []
        for b_i in range(10):
            y_lo, y_hi = band_edges[b_i], band_edges[b_i + 1]
            c_band = int(np.sum((pts_grav[:, 1] >= y_lo) & (pts_grav[:, 1] < y_hi)))
            points_per_height_band.append(c_band)
    else:
        bbox = [0, 0, 0, 0]
        torso_med_d = 0.0
        torso_std_mm = 0.0
        v_floor_est = float(h)
        row_diff_floor = 0.0
        leakage_px = 0
        total_bot_px = 0
        leakage_frac = 0.0
        points_per_height_band = [0] * 10

    stats = {
        "stop_id": stop_index,
        "nominal_angle_deg": float(stop.nominal_angle_deg),
        "foreground_px_count": fg_px_count,
        "bbox": bbox,
        "lowest_foreground_row": bbox[2] if fg_px_count > 0 else 0,
        "floor_contact_row_estimate": round(v_floor_est, 1),
        "lowest_foreground_row_vs_floor_contact": round(row_diff_floor, 1),
        "torso_median_depth_m": round(torso_med_d, 4),
        "torso_temporal_std_mm": round(torso_std_mm, 2),
        "floor_rmse_mm": round(float(bg_floor.rmse_m * 1000.0), 2),
        "floor_pixel_leakage_in_bottom_8pct": {
            "leakage_pixel_count": leakage_px,
            "total_pixels_in_bottom_8pct": total_bot_px,
            "leakage_fraction": round(leakage_frac, 4),
        },
        "points_per_height_band": points_per_height_band,
    }

    stats_path = target_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved debug artifacts for stop {stop_index} to '{target_dir}'.")
    return stats
