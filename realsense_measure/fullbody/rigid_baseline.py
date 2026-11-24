"""
realsense_measure/fullbody/rigid_baseline.py

Layer 1 rigid baseline reconstruction pipeline:
- Robust temporal median of depth per stop (burst processing + flying pixel rejection)
- Background depth subtraction for foreground masking
- Gravity alignment from floor plane fitting
- Vertical rotation axis estimation from multi-stop horizontal cross-sections
- Nominal-angle alignment
- Constrained bounded refinement (per-stop yaw in +-5 deg, floor-plane translation in +-3 cm, 3-level point-to-plane ICP)
- TSDF fusion (ScalableTSDFVolume with UniformTSDFVolume fallback) and mesh component extraction
- Per-stop colored overlay and detailed JSON reporting
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d

from realsense_measure.fullbody.burst import process_stop_burst
from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.floor import build_gravity_rotation, fit_floor_plane
from realsense_measure.fullbody.orientation import nominal_angle_deg

logger = logging.getLogger(__name__)


@dataclass
class StopPointCloud:
    stop_index: int
    nominal_yaw_deg: float
    pcd_cam: o3d.geometry.PointCloud
    pcd_grav: o3d.geometry.PointCloud
    pcd_aligned: o3d.geometry.PointCloud
    color_rep: np.ndarray
    depth_fused: np.ndarray
    fg_mask: np.ndarray
    intrinsics: o3d.camera.PinholeCameraIntrinsic
    T_cam_to_grav: np.ndarray
    T_nom_align: np.ndarray
    T_refined: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    refine_yaw_deg: float = 0.0
    refine_trans_mm: Tuple[float, float] = (0.0, 0.0)
    residual_before_mm: float = 0.0
    residual_after_mm: float = 0.0
    bound_exceeded: bool = False
    yaw_observable: bool = True


@dataclass
class RigidBaselineConfig:
    voxel_size_m: float = 0.004
    sdf_trunc_m: float = 0.012
    refine: bool = True
    max_yaw_deg: float = 5.0
    max_trans_m: float = 0.03
    default_cam_height_m: float = 0.95
    fullbody_cfg: FullBodyConfig = field(default_factory=FullBodyConfig)


def compute_foreground_mask_simple(
    depth_m: np.ndarray,
    bg_depth_m: Optional[np.ndarray],
    cfg: FullBodyConfig,
) -> np.ndarray:
    """
    Foreground segmentation via background depth subtraction with morphological cleanup.
    """
    h, w = depth_m.shape[:2]
    valid_curr = (depth_m >= cfg.depth_min_m) & (depth_m <= cfg.depth_max_m) & np.isfinite(depth_m)

    if bg_depth_m is not None:
        if bg_depth_m.shape[:2] != (h, w):
            bg_depth_m = cv2.resize(bg_depth_m, (w, h), interpolation=cv2.INTER_NEAREST)
        valid_bg = (bg_depth_m >= cfg.depth_min_m) & (bg_depth_m <= cfg.depth_max_m) & np.isfinite(bg_depth_m)
        is_fg = valid_curr & valid_bg & ((bg_depth_m - depth_m) >= 0.03)
        is_fg |= (valid_curr & ~valid_bg)
    else:
        is_fg = valid_curr

    if not np.any(is_fg):
        return np.zeros((h, w), dtype=bool)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = is_fg.astype(np.uint8)
    opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num_labels <= 1:
        return np.zeros((h, w), dtype=bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    max_area = stats[largest_idx, cv2.CC_STAT_AREA]
    min_comp_area = max(100, int(0.01 * max_area))
    valid_labels = [
        lbl for lbl in range(1, num_labels)
        if stats[lbl, cv2.CC_STAT_AREA] >= min_comp_area
    ]
    return np.isin(labels, valid_labels)


def backproject_depth_to_pcd(
    depth_m: np.ndarray,
    color_bgr: np.ndarray,
    mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> o3d.geometry.PointCloud:
    """Backproject masked depth and color into an Open3D PointCloud."""
    h, w = depth_m.shape[:2]
    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    sel = mask & (depth_m > 0.0) & np.isfinite(depth_m)
    if not np.any(sel):
        return o3d.geometry.PointCloud()

    z = depth_m[sel].astype(np.float64)
    u = u_grid[sel].astype(np.float64)
    v = v_grid[sel].astype(np.float64)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)

    colors = (color_bgr[sel][:, ::-1] / 255.0).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def estimate_rotation_axis(
    stop_pcds: List[o3d.geometry.PointCloud],
    n_slices: int = 3,
    known_axis: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, float]:
    """
    Estimate vertical rotation axis (x0, z0) on floor plane by taking horizontal
    slices across all stops at several heights and fitting circle/arc geometry.
    
    Returns (x0, z0, spread_mm).
    """
    if known_axis is not None:
        return float(known_axis[0]), float(known_axis[1]), 0.0

    all_pts = []
    for p in stop_pcds:
        if len(p.points) > 0:
            all_pts.append(np.asarray(p.points))
    if not all_pts:
        return 0.0, 2.0, 0.0

    stacked = np.vstack(all_pts)
    y_min, y_max = float(np.min(stacked[:, 1])), float(np.max(stacked[:, 1]))
    h_span = max(0.1, y_max - y_min)

    # Sample heights in the middle 50% of the subject
    fracs = [0.35, 0.50, 0.65] if n_slices == 3 else np.linspace(0.3, 0.7, n_slices)
    slice_centers = []

    for frac in fracs:
        y_target = y_min + frac * h_span
        per_stop_centers = []
        for p in stop_pcds:
            pts = np.asarray(p.points)
            if len(pts) == 0:
                continue
            in_slice = pts[np.abs(pts[:, 1] - y_target) < 0.04]
            if len(in_slice) >= 20:
                xs = in_slice[:, 0]
                zs = in_slice[:, 2]
                # Algebraic circle/arc fit on visible front contour
                try:
                    A_mat = np.column_stack([2.0 * xs, 2.0 * zs, np.ones_like(xs)])
                    b_vec = xs ** 2 + zs ** 2
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    xc_fit, zc_fit = float(sol[0]), float(sol[1])
                    r_sq = sol[2] + xc_fit ** 2 + zc_fit ** 2
                    if r_sq > 0:
                        r_fit = math.sqrt(r_sq)
                        if 0.04 <= r_fit <= 0.60 and zc_fit >= np.min(zs):
                            per_stop_centers.append(np.array([xc_fit, zc_fit]))
                            continue
                except Exception:
                    pass

                # Fallback to midpoint and depth + half_width
                x_mid = 0.5 * (float(np.min(xs)) + float(np.max(xs)))
                r_half = 0.5 * (float(np.max(xs)) - float(np.min(xs)))
                z_cen = float(np.min(zs)) + max(0.05, r_half)
                per_stop_centers.append(np.array([x_mid, z_cen]))

        if per_stop_centers:
            slice_centers.append(np.median(per_stop_centers, axis=0))

    if not slice_centers:
        med_xz = np.median(stacked[:, [0, 2]], axis=0)
        return float(med_xz[0]), float(med_xz[1]), 0.0

    slice_centers_arr = np.array(slice_centers)
    axis_xz = np.median(slice_centers_arr, axis=0)
    diffs = slice_centers_arr - axis_xz
    spread_mm = float(np.max(np.linalg.norm(diffs, axis=1)) * 1000.0)

    return float(axis_xz[0]), float(axis_xz[1]), spread_mm


def compute_nominal_alignment_matrix(
    stop_idx: int,
    n_stops: int,
    turn_direction: str,
    axis_x: float,
    axis_z: float,
) -> Tuple[np.ndarray, float]:
    """
    Compute 4x4 matrix rotating stop k back into stop 0's orientation around the vertical axis.
    """
    nom_deg = nominal_angle_deg(stop_idx, n_stops)
    rot_deg = -nom_deg if turn_direction.lower() == "cw" else nom_deg
    rot_rad = math.radians(rot_deg)

    # 2D rotation in horizontal (X, Z) plane about (axis_x, axis_z)
    cos_a = math.cos(rot_rad)
    sin_a = math.sin(rot_rad)

    # In 3D: Y is up, X right, Z away
    # R_y(alpha): [cos 0 sin; 0 1 0; -sin 0 cos]
    R_3x3 = np.array([
        [cos_a, 0.0, sin_a],
        [0.0, 1.0, 0.0],
        [-sin_a, 0.0, cos_a],
    ], dtype=np.float64)

    # T = T_center @ R @ T_inv_center
    c = np.array([axis_x, 0.0, axis_z], dtype=np.float64)
    t = c - R_3x3 @ c

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_3x3
    T[:3, 3] = t
    return T, rot_deg


def constrained_icp_refine(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    axis_x: float,
    axis_z: float,
    max_yaw_deg: float = 5.0,
    max_trans_m: float = 0.03,
) -> Tuple[np.ndarray, float, Tuple[float, float], float, float, bool, bool]:
    """
    Constrained point-to-plane ICP refinement with observability check:
    - Analyzes 3-DOF Hessian eigenvalues. If yaw is weakly constrained (e.g. rotationally
      symmetric cylinder about own axis), freezes yaw at nominal and solves only 2-DOF translation.
    - If yaw is observable, refines bounded 3-DOF [dx, dz, dyaw].
    - Zero pitch, zero roll, zero vertical shift (dy = 0).
    
    Returns: (T_refine, yaw_deg, (dx_mm, dz_mm), residual_before_mm, residual_after_mm, bound_exceeded, yaw_observable)
    """
    if len(source_pcd.points) < 50 or len(target_pcd.points) < 50:
        return np.eye(4), 0.0, (0.0, 0.0), 0.0, 0.0, False, False

    # Ensure target has surface normals
    if not target_pcd.has_normals():
        target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))

    src_work = o3d.geometry.PointCloud(source_pcd)
    target_kdtree = o3d.geometry.KDTreeFlann(target_pcd)
    target_pts = np.asarray(target_pcd.points)
    target_norms = np.asarray(target_pcd.normals)

    # Initial residual
    res_before_sq = []
    src_pts_init = np.asarray(src_work.points)
    for p in src_pts_init[::max(1, len(src_pts_init) // 1000)]:
        _, idxs, dists = target_kdtree.search_knn_vector_3d(p, 1)
        if dists and dists[0] < 0.05 ** 2:
            q = target_pts[idxs[0]]
            n = target_norms[idxs[0]]
            res_before_sq.append((np.dot(p - q, n)) ** 2)
    residual_before_mm = float(np.sqrt(np.mean(res_before_sq)) * 1000.0) if res_before_sq else 0.0

    # 1. Observability check via Hessian conditioning of 3-DOF ICP problem
    src_eval = src_work.voxel_down_sample(0.008)
    eval_pts = np.asarray(src_eval.points)
    A_eval = np.zeros((3, 3), dtype=np.float64)
    r_sq_list = []
    eval_count = 0
    for p in eval_pts:
        _, idxs, dists = target_kdtree.search_knn_vector_3d(p, 1)
        if not dists or dists[0] > 0.04 ** 2:
            continue
        q = target_pts[idxs[0]]
        n = target_norms[idxs[0]]
        arm_z = p[2] - axis_z
        arm_x = p[0] - axis_x
        r_sq_list.append(arm_x ** 2 + arm_z ** 2)
        J_yaw = n[0] * arm_z - n[2] * arm_x
        J = np.array([n[0], n[2], J_yaw], dtype=np.float64)
        A_eval += np.outer(J, J)
        eval_count += 1

    r_char = math.sqrt(np.mean(r_sq_list)) if r_sq_list else 0.150
    D_mat = np.diag([1.0, 1.0, 1.0 / r_char])
    H_scaled = D_mat @ A_eval @ D_mat
    eigvals = np.sort(np.linalg.eigvalsh(H_scaled))[::-1]
    cond_ratio = float(eigvals[-1] / max(1e-12, eigvals[0])) if eval_count >= 20 else 0.0

    # Yaw is observable if smallest eigenvalue ratio of scaled Hessian >= 0.015
    yaw_observable = bool(cond_ratio >= 0.015 and eval_count >= 20)
    if not yaw_observable:
        logger.info(f"Yaw direction is weakly constrained (cond_ratio={cond_ratio:.2e}); freezing yaw at nominal.")

    # 3 pyramid levels (coarse to fine)
    levels = [
        {"voxel": 0.016, "dist_thresh": 0.05, "iters": 8},
        {"voxel": 0.008, "dist_thresh": 0.03, "iters": 8},
        {"voxel": 0.004, "dist_thresh": 0.015, "iters": 8},
    ]

    total_yaw_rad = 0.0
    total_dx = 0.0
    total_dz = 0.0
    bound_exceeded = False
    max_yaw_rad = math.radians(max_yaw_deg)
    T_accum = np.eye(4, dtype=np.float64)

    for lvl in levels:
        v_size = lvl["voxel"]
        d_thresh = lvl["dist_thresh"]
        d_thresh_sq = d_thresh ** 2

        src_sub = src_work.voxel_down_sample(v_size)
        pts = np.asarray(src_sub.points)
        if len(pts) < 30:
            continue
        if len(pts) > 2000:
            pts = pts[::max(1, len(pts) // 2000)]

        for _ in range(lvl["iters"]):
            if not yaw_observable:
                # Freeze yaw at nominal, solve only 2-DOF planar translation
                A_mat = np.zeros((2, 2), dtype=np.float64)
                b_vec = np.zeros(2, dtype=np.float64)
                count = 0
                for p in pts:
                    _, idxs, dists = target_kdtree.search_knn_vector_3d(p, 1)
                    if not dists or dists[0] > d_thresh_sq:
                        continue
                    q = target_pts[idxs[0]]
                    n = target_norms[idxs[0]]
                    J = np.array([n[0], n[2]], dtype=np.float64)
                    r = float(np.dot(p - q, n))
                    A_mat += np.outer(J, J)
                    b_vec += -r * J
                    count += 1

                if count < 20:
                    break

                A_mat += 1e-4 * np.eye(2)
                try:
                    delta = np.linalg.solve(A_mat, b_vec)
                except np.linalg.LinAlgError:
                    break

                step_dx, step_dz, step_yaw = float(delta[0]), float(delta[1]), 0.0
            else:
                # Solve full 3-DOF point-to-plane system: J @ [dx, dz, dyaw]^T = - r
                A_mat = np.zeros((3, 3), dtype=np.float64)
                b_vec = np.zeros(3, dtype=np.float64)
                count = 0

                for p in pts:
                    _, idxs, dists = target_kdtree.search_knn_vector_3d(p, 1)
                    if not dists or dists[0] > d_thresh_sq:
                        continue
                    q = target_pts[idxs[0]]
                    n = target_norms[idxs[0]]

                    arm_z = p[2] - axis_z
                    arm_x = p[0] - axis_x
                    J_yaw = n[0] * arm_z - n[2] * arm_x
                    J = np.array([n[0], n[2], J_yaw], dtype=np.float64)
                    r = float(np.dot(p - q, n))

                    A_mat += np.outer(J, J)
                    b_vec += -r * J
                    count += 1

                if count < 20:
                    break

                A_mat += 1e-4 * np.eye(3)
                try:
                    delta = np.linalg.solve(A_mat, b_vec)
                except np.linalg.LinAlgError:
                    break

                step_dx, step_dz, step_yaw = float(delta[0]), float(delta[1]), float(delta[2])

            # Damping to prevent overshoot
            step_dx = float(np.clip(step_dx, -0.01, 0.01))
            step_dz = float(np.clip(step_dz, -0.01, 0.01))
            step_yaw = float(np.clip(step_yaw, -math.radians(2.0), math.radians(2.0)))

            total_dx += step_dx
            total_dz += step_dz
            total_yaw_rad += step_yaw

            # Check and enforce bounds
            if abs(total_yaw_rad) > max_yaw_rad:
                bound_exceeded = True
                total_yaw_rad = math.copysign(max_yaw_rad, total_yaw_rad)

            trans_norm = math.hypot(total_dx, total_dz)
            if trans_norm > max_trans_m:
                bound_exceeded = True
                scale = max_trans_m / trans_norm
                total_dx *= scale
                total_dz *= scale

            # Apply incremental transform to current point cloud
            cos_y = math.cos(step_yaw)
            sin_y = math.sin(step_yaw)
            R_inc = np.array([
                [cos_y, 0.0, sin_y],
                [0.0, 1.0, 0.0],
                [-sin_y, 0.0, cos_y],
            ])
            t_inc = np.array([axis_x, 0.0, axis_z]) - R_inc @ np.array([axis_x, 0.0, axis_z]) + np.array([step_dx, 0.0, step_dz])
            T_inc = np.eye(4)
            T_inc[:3, :3] = R_inc
            T_inc[:3, 3] = t_inc

            src_work.transform(T_inc)
            T_accum = T_inc @ T_accum
            src_sub = src_work.voxel_down_sample(v_size)
            pts = np.asarray(src_sub.points)
            if len(pts) > 2000:
                pts = pts[::max(1, len(pts) // 2000)]

    # Final residual
    res_after_sq = []
    src_pts_fin = np.asarray(src_work.points)
    for p in src_pts_fin[::max(1, len(src_pts_fin) // 1000)]:
        _, idxs, dists = target_kdtree.search_knn_vector_3d(p, 1)
        if dists and dists[0] < 0.05 ** 2:
            q = target_pts[idxs[0]]
            n = target_norms[idxs[0]]
            res_after_sq.append((np.dot(p - q, n)) ** 2)
    residual_after_mm = float(np.sqrt(np.mean(res_after_sq)) * 1000.0) if res_after_sq else 0.0

    # Build final cumulative transformation from T_accum
    T_refine = T_accum
    recovered_yaw_deg = float(math.degrees(math.atan2(T_accum[0, 2], T_accum[0, 0])))
    recovered_dx_mm = float((T_accum[0, 3] - (axis_x - (T_accum[0, 0] * axis_x + T_accum[0, 2] * axis_z))) * 1000.0)
    recovered_dz_mm = float((T_accum[2, 3] - (axis_z - (T_accum[2, 0] * axis_x + T_accum[2, 2] * axis_z))) * 1000.0)

    return (
        T_refine,
        recovered_yaw_deg,
        (recovered_dx_mm, recovered_dz_mm),
        residual_before_mm,
        residual_after_mm,
        bound_exceeded,
        yaw_observable,
    )


def fuse_stops_tsdf(
    stop_pcds: List[StopPointCloud],
    voxel_size_m: float = 0.004,
    sdf_trunc_m: float = 0.012,
) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud, int]:
    """
    Fuse per-stop RGB-D depth frames with known aligned poses using ScalableTSDFVolume
    (with UniformTSDFVolume and point cloud union fallbacks).
    Returns (cleaned_mesh, fused_point_cloud, components_before_cleanup).
    """
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size_m,
        sdf_trunc=sdf_trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    integrated = 0
    for s in stop_pcds:
        # Full transform from camera optical coordinates to common world frame
        T_cam_to_world = s.T_refined @ s.T_nom_align @ s.T_cam_to_grav
        T_world_to_cam = np.linalg.inv(T_cam_to_world)

        masked_depth = s.depth_fused.copy()
        masked_depth[~s.fg_mask] = 0.0

        depth_im = o3d.geometry.Image(masked_depth.astype(np.float32))
        color_im = o3d.geometry.Image(s.color_rep[:, :, ::-1].copy().astype(np.uint8))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_im,
            depth_im,
            depth_scale=1.0,
            depth_trunc=3.8,
            convert_rgb_to_intensity=False,
        )

        try:
            volume.integrate(rgbd, s.intrinsics, T_world_to_cam)
            integrated += 1
        except Exception as exc:
            logger.warning(f"TSDF integration warning for stop {s.stop_index}: {exc}")

    mesh = volume.extract_triangle_mesh()
    pcd = volume.extract_point_cloud()

    # UniformTSDFVolume fallback if ScalableTSDF produced empty geometry
    if len(mesh.vertices) == 0:
        all_pts_list = [np.asarray(s.pcd_aligned.points) for s in stop_pcds if len(s.pcd_aligned.points) > 0]
        if all_pts_list:
            all_pts = np.vstack(all_pts_list)
            min_b = np.min(all_pts, axis=0) - 0.06
            max_b = np.max(all_pts, axis=0) + 0.06
            vol_len = float(np.max(max_b - min_b))
            res = int(np.clip(round(vol_len / voxel_size_m), 128, 384))
            vol_u = o3d.pipelines.integration.UniformTSDFVolume(
                length=vol_len,
                resolution=res,
                sdf_trunc=sdf_trunc_m,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
                origin=min_b.reshape(3, 1),
            )
            for s in stop_pcds:
                T_cam_to_world = s.T_refined @ s.T_nom_align @ s.T_cam_to_grav
                T_world_to_cam = np.linalg.inv(T_cam_to_world)
                masked_depth = s.depth_fused.copy()
                masked_depth[~s.fg_mask] = 0.0
                depth_im = o3d.geometry.Image(masked_depth.astype(np.float32))
                color_im = o3d.geometry.Image(s.color_rep[:, :, ::-1].copy().astype(np.uint8))
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_im,
                    depth_im,
                    depth_scale=1.0,
                    depth_trunc=3.8,
                    convert_rgb_to_intensity=False,
                )
                try:
                    vol_u.integrate(rgbd, s.intrinsics, T_world_to_cam)
                except Exception:
                    pass
            mesh = vol_u.extract_triangle_mesh()
            if len(pcd.points) == 0:
                pcd = vol_u.extract_point_cloud()

    # Point cloud union fallback
    if len(pcd.points) == 0:
        pcd = o3d.geometry.PointCloud()
        for s in stop_pcds:
            pcd += s.pcd_aligned
        pcd = pcd.voxel_down_sample(voxel_size_m)

    # Component counting and cleanup
    components_before = 0
    components_gt_1cm2 = 0
    largest_component_area_m2 = 0.0
    fraction_area_removed = 0.0

    if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        triangle_clusters, num_triangles, _ = mesh.cluster_connected_triangles()
        components_before = len(num_triangles)

        # Vectorized triangle surface area calculation
        v = np.asarray(mesh.vertices)
        t = np.asarray(mesh.triangles)
        v0 = v[t[:, 0]]
        v1 = v[t[:, 1]]
        v2 = v[t[:, 2]]
        cross_prod = np.cross(v1 - v0, v2 - v0)
        tri_areas = 0.5 * np.linalg.norm(cross_prod, axis=1)
        total_area = float(np.sum(tri_areas))

        cluster_arr = np.asarray(triangle_clusters)
        cluster_areas = np.zeros(components_before, dtype=np.float64)
        np.add.at(cluster_areas, cluster_arr, tri_areas)

        # 1 cm^2 = 0.0001 m^2
        components_gt_1cm2 = int(np.sum(cluster_areas > 0.0001))

        if components_before > 0:
            largest_cluster = int(np.argmax(cluster_areas))
            largest_component_area_m2 = float(cluster_areas[largest_cluster])
            area_removed = total_area - largest_component_area_m2
            fraction_area_removed = float(area_removed / total_area) if total_area > 0 else 0.0

            if components_before > 1:
                to_remove = [i for i, c in enumerate(triangle_clusters) if c != largest_cluster]
                mesh.remove_triangles_by_index(to_remove)
                mesh.remove_unreferenced_vertices()

        mesh.compute_vertex_normals()

    if len(pcd.points) > 0 and not pcd.has_normals():
        pcd.estimate_normals()

    component_stats = {
        "components_before_cleanup": components_before,
        "components_area_gt_1cm2_before_cleanup": components_gt_1cm2,
        "fraction_area_removed": round(fraction_area_removed, 6),
        "largest_component_area_m2": round(largest_component_area_m2, 6),
        "components_after_cleanup": 1 if components_before > 0 else 0,
    }
    return mesh, pcd, component_stats


def run_rigid_baseline(
    capture_dir: Path,
    out_dir: Path,
    bg_dir: Optional[Path] = None,
    voxel_size_m: float = 0.004,
    refine: Optional[bool] = None,
    known_axis: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """
    Execute complete Layer 1 rigid baseline reconstruction.
    """
    capture_dir = Path(capture_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir = out_dir / "aligned_stops"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    meta_path = capture_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in capture directory: {capture_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    subject_type = str(meta.get("subject_type", "person")).lower()
    is_object = (subject_type == "object")
    turn_direction = str(meta.get("turn_direction", "cw")).lower()
    n_stops = int(meta.get("n_stops", meta.get("stop_count", 8)))

    # Default refine flag: ON for objects, OFF for persons unless specified
    if refine is None:
        refine = is_object

    intr_dict = meta.get("intrinsics", {})
    w = int(intr_dict.get("width", 1280))
    h = int(intr_dict.get("height", 720))
    fx = float(intr_dict.get("fx", 640.0))
    fy = float(intr_dict.get("fy", fx))
    cx = float(intr_dict.get("cx", w / 2.0))
    cy = float(intr_dict.get("cy", h / 2.0))

    o3d_intr = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    cfg = FullBodyConfig()

    # 1. Load background depth map
    bg_depth_m: Optional[np.ndarray] = None
    bg_source = bg_dir
    if bg_source is None:
        if (capture_dir / "background" / "background_depth.npy").exists():
            bg_source = capture_dir / "background"
        elif meta.get("background", {}).get("source_path"):
            cand = Path(meta["background"]["source_path"])
            if cand.exists():
                bg_source = cand

    if bg_source and (bg_source / "background_depth.npy").exists():
        raw_bg = np.load(str(bg_source / "background_depth.npy")).astype(np.float32)
        scale = float(meta.get("depth_scale", meta.get("depth_storage", {}).get("depth_scale", 0.001)))
        units = meta.get("depth_units", meta.get("depth_storage", {}).get("units", "sensor_units"))
        bg_depth_m = raw_bg * scale if units == "sensor_units" else raw_bg
        logger.info(f"Loaded background calibration from {bg_source}")

    # 2. Fit floor plane & determine gravity alignment
    floor_plane = None
    if bg_depth_m is not None:
        floor_plane = fit_floor_plane(bg_depth_m, o3d_intr, cfg)

    if floor_plane is not None:
        R_grav = build_gravity_rotation(floor_plane.normal)
        y_floor = float((R_grav @ floor_plane.point_on_plane)[1])
        T_cam_to_grav = np.eye(4, dtype=np.float64)
        T_cam_to_grav[:3, :3] = R_grav
        T_cam_to_grav[1, 3] = -y_floor
        floor_source = "background_plane"
    else:
        # Default level camera setup
        rig_h = float(meta.get("rig_metadata", {}).get("cam_height_m", 0.95))
        T_cam_to_grav = np.array([
            [-1.0,  0.0, 0.0, 0.0],
            [ 0.0, -1.0, 0.0, rig_h],
            [ 0.0,  0.0, 1.0, 0.0],
            [ 0.0,  0.0, 0.0, 1.0],
        ], dtype=np.float64)
        floor_source = "rig_height_default"

    # 3. Process raw bursts per stop
    stops_data: List[StopPointCloud] = []
    scale = float(meta.get("depth_scale", meta.get("depth_storage", {}).get("depth_scale", 0.001)))
    units = meta.get("depth_units", meta.get("depth_storage", {}).get("units", "sensor_units"))

    for s_idx in range(n_stops):
        s_folder = capture_dir / f"stop_{s_idx:02d}"
        s_json = s_folder / "stop.json"
        accepted_att = 0
        if s_json.exists():
            try:
                with open(s_json, "r", encoding="utf-8") as sf:
                    sj = json.load(sf)
                acc = sj.get("accepted_attempt")
                if acc is not None:
                    accepted_att = int(acc)
            except Exception:
                pass

        attempt_dir = s_folder / f"attempt_{accepted_att:02d}"
        if not attempt_dir.exists():
            continue

        c_files = sorted(list(attempt_dir.glob("raw_color_*.png")) + list(attempt_dir.glob("frame_*_color.png")))
        d_files = sorted(list(attempt_dir.glob("raw_depth_*.npy")) + list(attempt_dir.glob("frame_*_depth.npy")))
        if not d_files:
            continue

        colors = [cv2.imread(str(cf)) for cf in c_files]
        raw_depths = [np.load(str(df)).astype(np.float32) for df in d_files]
        depths_m = [rd * scale if units == "sensor_units" else rd for rd in raw_depths]

        # Robust temporal burst median & flying pixel filter
        rep_col, fused_d, _, _ = process_stop_burst(colors, depths_m, cfg)
        fg_mask = compute_foreground_mask_simple(fused_d, bg_depth_m, cfg)

        pcd_cam = backproject_depth_to_pcd(fused_d, rep_col, fg_mask, fx, fy, cx, cy)
        if len(pcd_cam.points) >= 30:
            pcd_cam.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=25))
            pcd_cam.orient_normals_towards_camera_location([0.0, 0.0, 0.0])
        pcd_grav = o3d.geometry.PointCloud(pcd_cam)
        pcd_grav.transform(T_cam_to_grav)

        nom_deg = nominal_angle_deg(s_idx, n_stops)

        stops_data.append(StopPointCloud(
            stop_index=s_idx,
            nominal_yaw_deg=nom_deg,
            pcd_cam=pcd_cam,
            pcd_grav=pcd_grav,
            pcd_aligned=o3d.geometry.PointCloud(pcd_grav),
            color_rep=rep_col,
            depth_fused=fused_d,
            fg_mask=fg_mask,
            intrinsics=o3d_intr,
            T_cam_to_grav=T_cam_to_grav,
            T_nom_align=np.eye(4),
        ))

    if not stops_data:
        raise RuntimeError(f"No valid stops found in session: {capture_dir}")

    # 4. Estimate vertical rotation axis across stops
    grav_pcds = [s.pcd_grav for s in stops_data]
    axis_x, axis_z, spread_mm = estimate_rotation_axis(grav_pcds, known_axis=known_axis)
    logger.info(f"Estimated rotation axis: X={axis_x:.4f} m, Z={axis_z:.4f} m (spread: {spread_mm:.1f} mm)")

    # 5. Nominal alignment
    for s in stops_data:
        T_nom, rot_deg = compute_nominal_alignment_matrix(
            s.stop_index, n_stops, turn_direction, axis_x, axis_z
        )
        s.T_nom_align = T_nom
        s.pcd_aligned = o3d.geometry.PointCloud(s.pcd_grav)
        s.pcd_aligned.transform(T_nom)

    # 6. Bounded Refinement (if enabled)
    refine_results = []
    if refine and len(stops_data) > 1:
        logger.info("Running bounded 3-DOF refinement on aligned stops...")
        for s in stops_data:
            # Build target cloud from union of other stops
            target_pcd = o3d.geometry.PointCloud()
            for other in stops_data:
                if other.stop_index != s.stop_index and len(other.pcd_aligned.points) > 0:
                    target_pcd += other.pcd_aligned
            target_pcd = target_pcd.voxel_down_sample(0.008)

            T_ref, dyaw_deg, (dx_mm, dz_mm), res_bef, res_aft, bound_flag, yaw_obs = constrained_icp_refine(
                source_pcd=s.pcd_aligned,
                target_pcd=target_pcd,
                axis_x=axis_x,
                axis_z=axis_z,
                max_yaw_deg=5.0,
                max_trans_m=0.03,
            )

            s.T_refined = T_ref
            s.refine_yaw_deg = dyaw_deg
            s.refine_trans_mm = (dx_mm, dz_mm)
            s.residual_before_mm = res_bef
            s.residual_after_mm = res_aft
            s.bound_exceeded = bound_flag
            s.yaw_observable = yaw_obs

            # Apply refinement
            s.pcd_aligned.transform(T_ref)

            refine_results.append({
                "stop_index": s.stop_index,
                "nominal_yaw_deg": s.nominal_yaw_deg,
                "yaw_correction_deg": round(dyaw_deg, 3),
                "trans_correction_mm": [round(dx_mm, 2), round(dz_mm, 2)],
                "residual_before_mm": round(res_bef, 2),
                "residual_after_mm": round(res_aft, 2),
                "bound_exceeded": bound_flag,
                "yaw_observable": yaw_obs,
                "yaw_status": "observable" if yaw_obs else "unobservable",
            })
    else:
        for s in stops_data:
            refine_results.append({
                "stop_index": s.stop_index,
                "nominal_yaw_deg": s.nominal_yaw_deg,
                "yaw_correction_deg": 0.0,
                "trans_correction_mm": [0.0, 0.0],
                "residual_before_mm": 0.0,
                "residual_after_mm": 0.0,
                "bound_exceeded": False,
                "yaw_observable": True,
                "yaw_status": "unrefined",
            })

    # 7. TSDF Fusion
    sdf_trunc = 3.0 * voxel_size_m
    fused_mesh, fused_pcd, comp_stats = fuse_stops_tsdf(
        stops_data, voxel_size_m=voxel_size_m, sdf_trunc_m=sdf_trunc
    )

    # 8. Save outputs
    # Aligned PLY per stop
    for s in stops_data:
        stop_ply_path = aligned_dir / f"stop_{s.stop_index:02d}.ply"
        o3d.io.write_point_cloud(str(stop_ply_path), s.pcd_aligned)

    # Multi-color overlay PLY
    overlay_colors = [
        [1.0, 0.1, 0.1],  # 0: Red
        [1.0, 0.5, 0.0],  # 1: Orange
        [1.0, 0.9, 0.1],  # 2: Yellow
        [0.1, 0.9, 0.1],  # 3: Green
        [0.1, 0.9, 0.9],  # 4: Cyan
        [0.1, 0.3, 1.0],  # 5: Blue
        [0.7, 0.1, 1.0],  # 6: Purple
        [1.0, 0.1, 0.7],  # 7: Magenta
    ]
    overlay_pcd = o3d.geometry.PointCloud()
    for s in stops_data:
        col_rgb = overlay_colors[s.stop_index % len(overlay_colors)]
        stop_sub = s.pcd_aligned.voxel_down_sample(0.006)
        n_pts = len(stop_sub.points)
        if n_pts > 0:
            colored_stop = o3d.geometry.PointCloud(stop_sub)
            colored_stop.colors = o3d.utility.Vector3dVector(np.tile(col_rgb, (n_pts, 1)))
            overlay_pcd += colored_stop

    overlay_path = out_dir / "aligned_overlay.ply"
    o3d.io.write_point_cloud(str(overlay_path), overlay_pcd)

    mesh_path = out_dir / "fused_mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), fused_mesh)

    pcd_path = out_dir / "fused_cloud.ply"
    o3d.io.write_point_cloud(str(pcd_path), fused_pcd)

    # Report JSON
    report = {
        "pipeline": "rigid_baseline_v1",
        "subject_id": meta.get("subject_id", "UNKNOWN"),
        "subject_type": subject_type,
        "n_stops": len(stops_data),
        "turn_direction": turn_direction,
        "refine_enabled": refine,
        "floor_alignment": {
            "source": floor_source,
            "normal": list(floor_plane.normal) if floor_plane else [0.0, 1.0, 0.0],
            "rmse_mm": round(floor_plane.rmse_m * 1000.0, 2) if floor_plane else 0.0,
        },
        "rotation_axis": {
            "x0_m": round(axis_x, 4),
            "z0_m": round(axis_z, 4),
            "spread_mm": round(spread_mm, 2),
        },
        "refinement": refine_results,
        "tsdf": {
            "voxel_size_m": voxel_size_m,
            "sdf_trunc_m": sdf_trunc,
            "components_before_cleanup": comp_stats["components_before_cleanup"],
            "components_area_gt_1cm2_before_cleanup": comp_stats["components_area_gt_1cm2_before_cleanup"],
            "fraction_area_removed": comp_stats["fraction_area_removed"],
            "largest_component_area_m2": comp_stats["largest_component_area_m2"],
            "components_after_cleanup": comp_stats["components_after_cleanup"],
            "mesh_vertices": len(fused_mesh.vertices),
            "mesh_triangles": len(fused_mesh.triangles),
            "cloud_points": len(fused_pcd.points),
        },
        "output_files": {
            "fused_mesh": str(mesh_path.name),
            "fused_cloud": str(pcd_path.name),
            "aligned_overlay": str(overlay_path.name),
            "aligned_stops_dir": str(aligned_dir.name),
        },
    }

    report_path = out_dir / "rigid_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Rigid baseline reconstruction finalized at {out_dir}")
    print_rigid_summary(report)
    return report


def print_rigid_summary(report: Dict[str, Any]) -> None:
    """Print clean formatted console summary including honest component metrics."""
    subj = report.get("subject_id", "UNKNOWN")
    stype = report.get("subject_type", "unknown")
    n_stops = report.get("n_stops", 0)
    turn = report.get("turn_direction", "cw")
    axis = report.get("rotation_axis", {})
    tsdf = report.get("tsdf", {})
    ref = report.get("refinement", [])

    print("\n" + "=" * 78)
    print("RIGID BASELINE RECONSTRUCTION REPORT")
    print("=" * 78)
    print(f"Subject:        {subj} ({stype})")
    print(f"Protocol:       {n_stops} stops ({turn.upper()})")
    print(f"Rotation Axis:  X={axis.get('x0_m', 0.0):.4f} m, Z={axis.get('z0_m', 0.0):.4f} m (spread: {axis.get('spread_mm', 0.0):.2f} mm)")
    if ref:
        print("-" * 78)
        print(f"{'Stop':<5} | {'Nominal':<8} | {'Corrected':<10} | {'Status':<13} | {'Trans (dx, dz)':<18} | {'Residuals'}")
        print("-" * 78)
        for r in ref:
            s_idx = r.get("stop_index", 0)
            nom = r.get("nominal_yaw_deg", 0.0)
            dyaw = r.get("yaw_correction_deg", 0.0)
            status = r.get("yaw_status", "observable" if r.get("yaw_observable", True) else "unobservable")
            tr = r.get("trans_correction_mm", [0.0, 0.0])
            r_bef = r.get("residual_before_mm", 0.0)
            r_aft = r.get("residual_after_mm", 0.0)
            print(f"{s_idx:<5} | {nom:>6.1f} deg | {dyaw:>+7.2f} deg | {status:<13} | [{tr[0]:>5.1f}, {tr[1]:>5.1f}] mm | {r_bef:.1f} -> {r_aft:.1f} mm")

    print("-" * 78)
    print("COMPONENT & MESH METRICS (BEFORE & AFTER CLEANUP):")
    print(f"  Components Before Cleanup:         {tsdf.get('components_before_cleanup', 0):,}")
    print(f"  Components with Area > 1 cm^2:     {tsdf.get('components_area_gt_1cm2_before_cleanup', 0):,}")
    print(f"  Largest Component Area:            {tsdf.get('largest_component_area_m2', 0.0):.4f} m^2")
    frac_rem = tsdf.get('fraction_area_removed', 0.0)
    print(f"  Area Removed by Largest-Comp:      {frac_rem * 100:.2f}% ({frac_rem:.6f})")
    print(f"  Components After Cleanup:          {tsdf.get('components_after_cleanup', 1)}")
    print(f"  Final Mesh Vertices / Triangles:   {tsdf.get('mesh_vertices', 0):,} / {tsdf.get('mesh_triangles', 0):,}")
    print(f"  Final Point Cloud Points:          {tsdf.get('cloud_points', 0):,}")
    print("=" * 78 + "\n")
