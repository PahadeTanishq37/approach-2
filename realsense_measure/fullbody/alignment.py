"""
realsense_measure/fullbody/alignment.py

Multi-stop full-body alignment:
1. Floor-constrained gravity frame (locks height + pitch + roll to gravity).
2. Keypoint-based yaw and common vertical turning axis prior.
3. Multi-view Pose-Graph optimization with Levenberg-Marquardt.
4. Analytical 3-parameter motion (yaw about vertical + XZ translation) ICP on rigid body parts
   (head, torso, pelvis; arms excluded).
   NOTE: Per-limb articulated refinement for arms is not implemented here (known limitation).
5. 3D keypoint constraints on torso/head to resolve cylindrical yaw ambiguity.
6. Loop-closure error evaluation on closing edge before optimization.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.orientation import canonical_yaw_deg
from realsense_measure.fullbody.parts import (
    ARM_PARTS,
    RIGID_PARTS,
    BodyPart,
    is_arm_point,
    is_rigid_point,
    label_points_by_body_part,
)
from realsense_measure.fullbody.types import (
    AlignedStop,
    FloorPlane,
    Keypoints3D,
    StopData,
)

logger = logging.getLogger(__name__)


def compute_gravity_transform(floor: FloorPlane) -> np.ndarray:
    """
    Build 4x4 matrix T_grav_cam that aligns camera coords with gravity frame:
    floor plane is at Y = 0, normal is +Y = [0, 1, 0].
    Camera optical center projects to (0, d, 0) in gravity frame.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = floor.gravity_R_cam
    d = float(floor.coefficients[3])
    T[:3, 3] = np.array([0.0, d, 0.0], dtype=np.float64)
    return T


def compute_keypoint_prior_yaw_legacy(
    keypoints_grav: Keypoints3D,
    nominal_deg: float,
    cfg: FullBodyConfig,
) -> Tuple[float, np.ndarray, bool]:
    """Previous non-robust prior estimator (for benchmark comparison)."""
    l_sh, c_ls, _ = keypoints_grav.get("left_shoulder")
    r_sh, c_rs, _ = keypoints_grav.get("right_shoulder")
    l_hp, c_lh, _ = keypoints_grav.get("left_hip")
    r_hp, c_rh, _ = keypoints_grav.get("right_hip")
    nose, c_n, _ = keypoints_grav.get("nose")

    if (l_sh is None or r_sh is None) and (l_hp is None or r_hp is None):
        return nominal_deg, np.array([0.0, 0.9, 1.3], dtype=np.float64), False

    centers = []
    lat_vecs = []
    if l_sh is not None and r_sh is not None:
        centers.append(0.5 * (l_sh + r_sh))
        lat_vecs.append(l_sh - r_sh)
    if l_hp is not None and r_hp is not None:
        centers.append(0.5 * (l_hp + r_hp))
        lat_vecs.append(l_hp - r_hp)

    center = np.mean(centers, axis=0)
    lat = np.mean(lat_vecs, axis=0)
    lat[1] = 0.0
    norm_lat = np.linalg.norm(lat)
    if norm_lat < 1e-4:
        return nominal_deg, np.array([0.0, 0.9, 1.3], dtype=np.float64), False
    lat /= norm_lat
    ant = np.array([-lat[2], 0.0, lat[0]], dtype=np.float64)
    if nose is not None and c_n > 0.2:
        nose_dir = nose - center
        nose_dir[1] = 0.0
        if np.dot(ant, nose_dir) < 0:
            ant = -ant
    yaw_rad = np.arctan2(ant[0], ant[2])
    return float(np.rad2deg(yaw_rad)), center, True


def compute_keypoint_prior_yaw(
    keypoints_grav: Keypoints3D,
    nominal_deg: float,
    cfg: FullBodyConfig,
) -> Tuple[float, np.ndarray, bool]:
    """
    Robust prior estimator:
    1. Keypoint confidence weights (w = c * v).
    2. Joint-outlier rejection (anatomical spans and vertical spine alignment).
    3. Robust Huber M-estimator over torso joint candidates for body-axis (X, Z).
    4. Circular consensus for lateral orientation and yaw.
    """
    l_sh, c_ls, v_ls = keypoints_grav.get("left_shoulder")
    r_sh, c_rs, v_rs = keypoints_grav.get("right_shoulder")
    l_hp, c_lh, v_lh = keypoints_grav.get("left_hip")
    r_hp, c_rh, v_rh = keypoints_grav.get("right_hip")
    nose, c_n, _ = keypoints_grav.get("nose")

    # Joint candidate pairs and anatomical checks
    candidates = []
    # (pair_name, p_left, p_right, conf_weight, expected_span, min_span, max_span)
    if l_sh is not None and r_sh is not None:
        w_sh = float(c_ls * c_rs * v_ls * v_rs)
        if c_ls >= 0.2 and c_rs >= 0.2:
            candidates.append(("shoulder", l_sh, r_sh, w_sh, 0.36, 0.24, 0.48))

    if l_hp is not None and r_hp is not None:
        w_hp = float(c_lh * c_rh * v_lh * v_rh)
        if c_lh >= 0.2 and c_rh >= 0.2:
            candidates.append(("hip", l_hp, r_hp, w_hp, 0.20, 0.12, 0.36))

    if not candidates:
        return nominal_deg, np.array([0.0, 0.9, 1.3], dtype=np.float64), False

    # Outlier rejection on bone spans
    valid_pairs = []
    for name, pl, pr, w, exp_span, min_s, max_s in candidates:
        span = float(np.linalg.norm(pl - pr))
        if min_s <= span <= max_s:
            residual = abs(span - exp_span)
            span_weight = w / (1.0 + 10.0 * residual)
            c_mid = 0.5 * (pl + pr)
            v_lat = np.array([pl[0] - pr[0], 0.0, pl[2] - pr[2]], dtype=np.float64)
            norm_v = np.linalg.norm(v_lat)
            if norm_v > 1e-4:
                v_lat /= norm_v
                valid_pairs.append({
                    "name": name,
                    "center": c_mid,
                    "lateral": v_lat,
                    "weight": span_weight,
                    "span_residual": residual,
                })

    if not valid_pairs:
        # Fallback to nominal if all pairs failed anatomical span bounds
        return nominal_deg, np.array([0.0, 0.9, 1.3], dtype=np.float64), False

    # Joint-outlier rejection via spine verticality
    if len(valid_pairs) == 2:
        sh_pair = valid_pairs[0] if valid_pairs[0]["name"] == "shoulder" else valid_pairs[1]
        hp_pair = valid_pairs[0] if valid_pairs[0]["name"] == "hip" else valid_pairs[1]
        c_sh = sh_pair["center"]
        c_hp = hp_pair["center"]
        spine_disp_xz = float(np.linalg.norm([c_sh[0] - c_hp[0], c_sh[2] - c_hp[2]]))
        if spine_disp_xz > 0.06:
            # One joint pair contains an outlier; keep pair with lower span residual
            if sh_pair["span_residual"] <= hp_pair["span_residual"]:
                valid_pairs = [sh_pair]
            else:
                valid_pairs = [hp_pair]

    # Robust Huber body-axis center estimation in XZ
    pts_xz = np.array([[p["center"][0], p["center"][2]] for p in valid_pairs], dtype=np.float64)
    weights = np.array([p["weight"] for p in valid_pairs], dtype=np.float64)
    if np.sum(weights) < 1e-6:
        weights = np.ones(len(valid_pairs), dtype=np.float64)
    weights /= np.sum(weights)

    center_xz = np.average(pts_xz, axis=0, weights=weights)
    delta_huber = 0.02
    for _ in range(5):
        dists = np.linalg.norm(pts_xz - center_xz, axis=1)
        h_w = weights * np.where(dists <= delta_huber, 1.0, delta_huber / np.maximum(dists, 1e-6))
        if np.sum(h_w) > 1e-6:
            center_xz = np.average(pts_xz, axis=0, weights=h_w)

    y_center = float(np.mean([p["center"][1] for p in valid_pairs]))
    center = np.array([center_xz[0], y_center, center_xz[1]], dtype=np.float64)

    # Robust lateral consensus vector
    lat_vecs = np.array([p["lateral"] for p in valid_pairs])
    lat = np.average(lat_vecs, axis=0, weights=weights)
    lat[1] = 0.0
    norm_lat = np.linalg.norm(lat)
    if norm_lat < 1e-4:
        return nominal_deg, center, False
    lat /= norm_lat

    # Anterior vector: cross(lateral, UP) -> [-lz, 0, lx]
    ant = np.array([-lat[2], 0.0, lat[0]], dtype=np.float64)

    # Disambiguate front vs back using nose position
    if nose is not None and c_n > 0.2:
        nose_dir = nose - center
        nose_dir[1] = 0.0
        if np.dot(ant, nose_dir) < 0:
            ant = -ant
            lat = -lat

    yaw_rad = np.arctan2(ant[0], ant[2])
    yaw_deg = float(np.rad2deg(yaw_rad))

    return yaw_deg, center, True


def estimate_common_vertical_axis_and_drift(
    centers_grav: List[np.ndarray],
    weights: List[float],
    cfg: FullBodyConfig,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Model the turning subject as rotating about one shared vertical axis (X_axis, Z_axis)
    plus small per-stop physical drift: delta_i ~ N(0, sigma_drift^2) with sigma ~ 0.02 m.

    Jointly estimate the common axis across stops via weighted Huber/median, then
    regularize (shrink) per-stop centers toward the shared axis.
    """
    n = len(centers_grav)
    if n == 0:
        return np.zeros(2), []

    pts = np.array([[c[0], c[2]] for c in centers_grav], dtype=np.float64)
    w = np.array(weights, dtype=np.float64)
    if np.sum(w) < 1e-4:
        w = np.ones(n, dtype=np.float64)
    w /= np.sum(w)

    # Robust common axis (Huber M-estimator)
    axis_xz = np.average(pts, axis=0, weights=w)
    delta = 0.03
    for _ in range(10):
        res = np.linalg.norm(pts - axis_xz, axis=1)
        huber_w = w * np.where(res <= delta, 1.0, delta / np.maximum(res, 1e-6))
        if np.sum(huber_w) > 1e-6:
            new_axis = np.average(pts, axis=0, weights=huber_w)
            if np.linalg.norm(new_axis - axis_xz) < 1e-5:
                break
            axis_xz = new_axis

    sigma_drift = getattr(cfg, "body_axis_drift_sigma_m", 0.02)
    sigma_meas = 0.03
    lam = (sigma_meas / max(0.005, sigma_drift)) ** 2

    reg_centers = []
    for i, c in enumerate(centers_grav):
        # Regularized MAP shrinkage toward common vertical axis
        shrunk_offset = (1.0 / (1.0 + lam)) * (pts[i] - axis_xz)
        reg_xz = axis_xz + shrunk_offset
        reg_c = np.array([reg_xz[0], c[1], reg_xz[1]], dtype=np.float64)
        reg_centers.append(reg_c)

    return axis_xz, reg_centers


def check_and_enforce_loop_closure(
    stops: List[AlignedStop],
    cfg: FullBodyConfig,
) -> float:
    """
    Check loop closure drift across stops.
    Consumes the real closing edge residual if recorded on stops;
    otherwise computes drift from registered poses.
    Returns loop closure drift in degrees.
    """
    n = len(stops)
    if n < 4:
        return 0.0

    if stops and getattr(stops[0], "loop_closure_residual_deg", None) is not None:
        drift = float(stops[0].loop_closure_residual_deg)
        logger.info(f"  [align] Loop closure drift from closing edge = {drift:.2f} deg")
        if abs(drift) > 20.0:
            logger.warning(f"  [align] Loop closure drift {drift:.2f} deg exceeds 20 deg! Flagging review.")
            for s in stops:
                s.review_status = "NEEDS_REVIEW"
        return drift

    # Standalone fallback for mock AlignedStop instances without precomputed closing edge
    yaws = []
    for s in stops:
        if s.T_canonical_camera is not None and not np.allclose(s.T_canonical_camera[:3, :3], np.eye(3)):
            R = s.T_canonical_camera[:3, :3]
            yaw_val = float(np.rad2deg(np.arctan2(R[2, 0], R[0, 0])))
        else:
            yaw_val = float(s.refined_yaw_deg)
        yaws.append(yaw_val)
    yaws_arr = np.array(yaws, dtype=np.float64)

    step_deg = 360.0 / float(n)
    expected_step = -step_deg if cfg.turn_direction == "cw" else step_deg
    expected_span = (n - 1) * expected_step
    actual_span = (yaws_arr[-1] - yaws_arr[0] + 180.0) % 360.0 - 180.0
    drift = float((actual_span - expected_span + 180.0) % 360.0 - 180.0)

    logger.info(f"  [align] Loop closure drift = {drift:.2f} deg (expected span {expected_span:.1f} deg, actual {actual_span:.1f} deg)")

    if abs(drift) <= 20.0:
        for i, s in enumerate(stops):
            adj = -drift * (i / float(n))
            s.refined_yaw_deg += adj
            if s.T_canonical_camera is not None:
                rad = np.deg2rad(adj)
                c_y, s_y = np.cos(rad), np.sin(rad)
                R_adj = np.array([
                    [ c_y, 0.0, -s_y, 0.0],
                    [ 0.0, 1.0,  0.0, 0.0],
                    [ s_y, 0.0,  c_y, 0.0],
                    [ 0.0, 0.0,  0.0, 1.0],
                ], dtype=np.float64)
                s.T_canonical_camera = R_adj @ s.T_canonical_camera
                if s.pcd_canonical is not None:
                    s.pcd_canonical.transform(R_adj)
                if s.keypoints_canonical is not None:
                    s.keypoints_canonical = s.keypoints_canonical.transform(R_adj, "canonical")
    else:
        logger.warning(f"  [align] Loop closure drift {drift:.2f} deg exceeds 20 deg! Flagging review.")
        for s in stops:
            s.review_status = "NEEDS_REVIEW"

    return float(drift)


def compute_ghosting_metric_mm(
    aligned_stops: List[AlignedStop],
    fused_mesh_or_pcd: Any,
) -> float:
    """
    Compute ghosting metric: mean distance (in mm) between per-stop transformed clouds
    and the fused mesh/cloud.
    """
    if fused_mesh_or_pcd is None:
        return 0.0

    if hasattr(fused_mesh_or_pcd, "vertices"):
        mesh_pts = np.asarray(fused_mesh_or_pcd.vertices)
        if len(mesh_pts) == 0:
            return 0.0
        ref_pcd = o3d.geometry.PointCloud()
        ref_pcd.points = o3d.utility.Vector3dVector(mesh_pts)
    elif hasattr(fused_mesh_or_pcd, "points"):
        ref_pcd = fused_mesh_or_pcd
    else:
        return 0.0

    distances = []
    for s in aligned_stops:
        if s.pcd_canonical is not None and len(s.pcd_canonical.points) > 0:
            d = s.pcd_canonical.compute_point_cloud_distance(ref_pcd)
            distances.extend(d)

    if not distances:
        return 0.0
    return float(np.mean(distances) * 1000.0)


def align_stops(
    stops: List[StopData],
    consensus_floor: FloorPlane,
    cfg: FullBodyConfig,
) -> List[AlignedStop]:
    """
    Execute full multi-stop alignment pipeline:
    1. Robust keypoint prior per stop (Huber on torso joints, outlier rejection).
    2. Common vertical turning axis estimation with regularized per-stop drift.
    3. MAP-style 4-DoF ICP with multi-start and tangential sliding penalty.
    4. Loop closure check and enforcement.
    """
    n_stops = len(stops)
    if n_stops == 0:
        return []

    T_grav_cam = compute_gravity_transform(consensus_floor)

    # 1. Compute initial priors per stop
    raw_priors = []
    centers_grav = []
    weights_grav = []

    for s in stops:
        kpts_cam = s.keypoints_camera
        if kpts_cam is not None:
            kpts_grav = kpts_cam.transform(T_grav_cam, "gravity")
        else:
            kpts_grav = Keypoints3D([], np.zeros((0, 3)), np.zeros(0), np.zeros(0), "gravity")

        from realsense_measure.fullbody.orientation import canonical_yaw_deg
        nominal_canonical = canonical_yaw_deg(s.nominal_angle_deg, cfg.turn_direction)

        prior_yaw, center, is_kpt_valid = compute_keypoint_prior_yaw(kpts_grav, nominal_canonical, cfg)

        diff = (prior_yaw - nominal_canonical + 180.0) % 360.0 - 180.0
        if abs(diff) > cfg.max_angle_deviation_deg:
            logger.info(
                f"  [align] Stop {s.stop_id} yaw prior {prior_yaw:.1f} deg deviates by {diff:.1f} deg "
                f"from nominal {nominal_canonical:.1f} deg; clamping prior to nominal."
            )
            prior_yaw = nominal_canonical

        raw_priors.append((prior_yaw, center, is_kpt_valid, kpts_cam))
        centers_grav.append(center)
        weights_grav.append(1.0 if is_kpt_valid else 0.1)

    # 2. Joint common vertical axis estimation and drift shrinkage
    _, regularized_centers = estimate_common_vertical_axis_and_drift(centers_grav, weights_grav, cfg)

    aligned_stops: List[AlignedStop] = []
    for i, s in enumerate(stops):
        prior_yaw, _, is_kpt_valid, kpts_cam = raw_priors[i]
        center = regularized_centers[i] if regularized_centers else centers_grav[i]

        yaw_rad = np.deg2rad(prior_yaw)
        c_y, s_y = np.cos(yaw_rad), np.sin(yaw_rad)
        R_yaw = np.array([
            [c_y, 0.0, -s_y],
            [0.0, 1.0,  0.0],
            [s_y, 0.0,  c_y],
        ], dtype=np.float64)

        t_canon = np.array([-c_y * center[0] + s_y * center[2], 0.0, -s_y * center[0] - c_y * center[2]])
        T_canon_grav = np.eye(4, dtype=np.float64)
        T_canon_grav[:3, :3] = R_yaw
        T_canon_grav[:3, 3] = t_canon

        T_canon_cam = T_canon_grav @ T_grav_cam

        kpts_can = kpts_cam.transform(T_canon_cam, "canonical") if kpts_cam else None

        pcd_can = None
        labels = None
        if s.pcd is not None:
            pcd_can = o3d.geometry.PointCloud(s.pcd)
            pcd_can.transform(T_canon_cam)
            pts_np = np.asarray(pcd_can.points)
            if kpts_can:
                labels = label_points_by_body_part(pts_np, kpts_can, cfg)

        aligned_stops.append(
            AlignedStop(
                stop_id=s.stop_id,
                nominal_angle_deg=s.nominal_angle_deg,
                T_canonical_camera=T_canon_cam,
                prior_yaw_deg=prior_yaw,
                refined_yaw_deg=prior_yaw,
                prior_translation_m=t_canon.copy(),
                refined_translation_m=t_canon.copy(),
                residual_before_mm=0.0,
                residual_after_mm=0.0,
                inlier_fraction=1.0,
                accepted=True,
                pcd_canonical=pcd_can,
                keypoints_canonical=kpts_can,
                body_part_labels=labels,
                prior_T_canonical_camera=T_canon_cam.copy(),
                hit_clamp=False,
                review_status="ACCEPTED",
            )
        )

    # 3. Refinement loop (Pose-Graph optimization on rigid parts)
    if n_stops > 1:
        _refine_alignment_multiview(aligned_stops, cfg, raw_stops=stops)

    # 4. Enforce loop consistency
    if n_stops >= 4:
        check_and_enforce_loop_closure(aligned_stops, cfg)

    return aligned_stops


def solve_3param_point_to_plane_icp(
    src_pcd: o3d.geometry.PointCloud,
    ref_pcd: o3d.geometry.PointCloud,
    voxel_scales: Tuple[float, ...] = (0.02, 0.01, 0.005),
    max_iter_per_scale: int = 15,
    distance_threshold_m: float = 0.06,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Solve relative transform between two clouds using analytical 3-parameter motion ICP:
    parameters are xi = (theta, tx, tz) for rotation about vertical (+Y) and XZ translation.
    Does NOT solve 6-DoF and project afterwards.

    Supports multi-scale voxels (20 mm, 10 mm, 5 mm) with colored ICP (point-to-plane fallback).

    Returns:
      (T_delta, fitness, rmse_mm, rmse_before_mm)
    """
    if len(src_pcd.points) < 10 or len(ref_pcd.points) < 10:
        return np.eye(4, dtype=np.float64), 0.0, 999.0, 999.0

    pts_src_orig = np.asarray(src_pcd.points)
    pts_ref_orig = np.asarray(ref_pcd.points)

    # Initial evaluation
    tree_init = cKDTree(pts_ref_orig)
    dists_init, _ = tree_init.query(pts_src_orig, k=1)
    inliers_init = dists_init < distance_threshold_m
    rmse_before_mm = (
        float(np.sqrt(np.mean(dists_init[inliers_init] ** 2)) * 1000.0)
        if np.any(inliers_init)
        else 999.0
    )

    curr_theta = 0.0
    curr_tx = 0.0
    curr_tz = 0.0

    for vox in voxel_scales:
        src_down = src_pcd.voxel_down_sample(vox) if vox > 0 else src_pcd
        ref_down = ref_pcd.voxel_down_sample(vox) if vox > 0 else ref_pcd

        if len(src_down.points) < 10 or len(ref_down.points) < 10:
            continue

        src_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=max(3.0 * vox, 0.03), max_nn=20)
        )
        ref_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=max(3.0 * vox, 0.03), max_nn=20)
        )

        pts_s = np.asarray(src_down.points)
        pts_r = np.asarray(ref_down.points)
        nrms_s = np.asarray(src_down.normals)
        nrms_r = np.asarray(ref_down.normals)

        tree = cKDTree(pts_r)
        d_max = max(2.5 * vox, distance_threshold_m)

        # Check for color variance for colored ICP
        has_color = False
        color_grad_r = None
        gray_s = None
        gray_r = None
        if (
            src_down.has_colors()
            and ref_down.has_colors()
            and len(src_down.colors) == len(pts_s)
            and len(ref_down.colors) == len(pts_r)
        ):
            c_s = np.asarray(src_down.colors)
            c_r = np.asarray(ref_down.colors)
            if float(np.var(c_r)) > 0.005:
                gray_s = 0.299 * c_s[:, 0] + 0.587 * c_s[:, 1] + 0.114 * c_s[:, 2]
                gray_r = 0.299 * c_r[:, 0] + 0.587 * c_r[:, 1] + 0.114 * c_r[:, 2]
                color_grad_r = np.zeros_like(pts_r)
                dists_knn, idxs_knn = tree.query(pts_r, k=min(7, len(pts_r)))
                for k_idx in range(len(pts_r)):
                    nn = idxs_knn[k_idx, 1:]
                    dp = pts_r[nn] - pts_r[k_idx]
                    dc = gray_r[nn] - gray_r[k_idx]
                    if len(nn) >= 3:
                        g, _, _, _ = np.linalg.lstsq(dp, dc, rcond=None)
                        color_grad_r[k_idx] = g
                has_color = True

        for _ in range(max_iter_per_scale):
            c = np.cos(curr_theta)
            s = np.sin(curr_theta)
            R_cur = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=np.float64)
            t_cur = np.array([curr_tx, 0.0, curr_tz], dtype=np.float64)

            p_trans = (pts_s @ R_cur.T) + t_cur
            n_trans = nrms_s @ R_cur.T

            dists, idxs = tree.query(p_trans, k=1)
            valid = (dists < d_max) & (np.sum(n_trans * nrms_r[idxs], axis=1) > 0.4)
            if np.sum(valid) < 10:
                break

            p = p_trans[valid]
            q = pts_r[idxs[valid]]
            n = nrms_r[idxs[valid]]

            r_geom = np.sum(n * (p - q), axis=1)
            p_rot_x = p[:, 0] - curr_tx
            p_rot_z = p[:, 2] - curr_tz
            J_geom = np.column_stack([-n[:, 0] * p_rot_z + n[:, 2] * p_rot_x, n[:, 0], n[:, 2]])

            huber_k = vox * 1.5
            abs_r = np.abs(r_geom)
            w_huber = np.where(abs_r <= huber_k, 1.0, huber_k / np.maximum(abs_r, 1e-6))

            A_geom = (J_geom.T * w_huber) @ J_geom
            b_geom = (J_geom.T * w_huber) @ (-r_geom)

            if has_color and color_grad_r is not None and gray_s is not None and gray_r is not None:
                g = color_grad_r[idxs[valid]]
                r_col = gray_s[valid] - gray_r[idxs[valid]]
                J_col = np.column_stack([-g[:, 0] * p_rot_z + g[:, 2] * p_rot_x, g[:, 0], g[:, 2]])
                A_col = J_col.T @ J_col
                b_col = J_col.T @ (-r_col)
                A = 0.6 * A_geom + 0.4 * A_col
                b = 0.6 * b_geom + 0.4 * b_col
            else:
                A = A_geom
                b = b_geom

            delta = np.linalg.solve(A + 1e-5 * np.eye(3), b)
            curr_theta += delta[0]
            curr_tx += delta[1]
            curr_tz += delta[2]

            if abs(delta[0]) < 1e-5 and math.hypot(delta[1], delta[2]) < 1e-5:
                break

    c_final = np.cos(curr_theta)
    s_final = np.sin(curr_theta)
    T_delta = np.array([
        [c_final, 0.0, -s_final, curr_tx],
        [0.0,     1.0,  0.0,     0.0],
        [s_final, 0.0,  c_final, curr_tz],
        [0.0,     0.0,  0.0,     1.0],
    ], dtype=np.float64)

    pts_final = (pts_src_orig @ T_delta[:3, :3].T) + T_delta[:3, 3]
    dists_final, _ = tree_init.query(pts_final, k=1)
    inliers = dists_final < distance_threshold_m
    fitness = float(np.mean(inliers)) if len(dists_final) > 0 else 0.0
    rmse_mm = (
        float(np.sqrt(np.mean(dists_final[inliers] ** 2)) * 1000.0)
        if np.any(inliers)
        else 999.0
    )

    return T_delta, fitness, rmse_mm, rmse_before_mm


def refine_with_keypoint_constraints(
    curr_theta: float,
    curr_tx: float,
    curr_tz: float,
    kpts_i: Optional[Keypoints3D],
    kpts_j: Optional[Keypoints3D],
    min_conf: float = 0.25,
) -> Tuple[float, float, float, str]:
    """
    Pass 2: Refine yaw and horizontal translation using 3D keypoint distance residuals.
    Constrains near-cylindrical torso surfaces using shoulders, hips, ears, and nose.

    Returns:
      (refined_theta, refined_tx, refined_tz, dominant_constraint: "icp" | "keypoints")
    """
    if kpts_i is None or kpts_j is None:
        return curr_theta, curr_tx, curr_tz, "icp"

    candidate_names = [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_ear",
        "right_ear",
        "nose",
    ]
    pts_i = []
    pts_j = []
    weights = []

    for name in candidate_names:
        pi, ci, vi = kpts_i.get(name)
        pj, cj, vj = kpts_j.get(name)
        if pi is not None and pj is not None and ci >= min_conf and cj >= min_conf:
            w = float(min(ci * vi, cj * vj))
            if w > 0.05:
                pts_i.append(pi)
                pts_j.append(pj)
                weights.append(w)

    if not pts_i:
        return curr_theta, curr_tx, curr_tz, "icp"

    theta = curr_theta
    tx = curr_tx
    tz = curr_tz

    icp_w = 10.0

    for _ in range(5):
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=np.float64)
        t = np.array([tx, 0.0, tz], dtype=np.float64)

        # Scale translation regularizer appropriately with meters vs radians
        A = np.diag([icp_w, icp_w * 50.0, icp_w * 50.0])
        b = -A @ np.array([theta - curr_theta, tx - curr_tx, tz - curr_tz], dtype=np.float64)

        for pi, pj, w in zip(pts_i, pts_j, weights):
            p_trans = R @ pi + t
            rx = p_trans[0] - pj[0]
            rz = p_trans[2] - pj[2]

            p_rot_x = p_trans[0] - tx
            p_rot_z = p_trans[2] - tz
            Jx = np.array([-p_rot_z, 1.0, 0.0])
            Jz = np.array([ p_rot_x, 0.0, 1.0])

            kpt_scale = 500.0 * w
            A += kpt_scale * (np.outer(Jx, Jx) + np.outer(Jz, Jz))
            b += kpt_scale * (Jx * (-rx) + Jz * (-rz))

        delta = np.linalg.solve(A + 1e-4 * np.eye(3), b)
        theta += delta[0]
        tx += delta[1]
        tz += delta[2]
        if abs(delta[0]) < 1e-5 and math.hypot(delta[1], delta[2]) < 1e-5:
            break

    yaw_shift_deg = float(np.rad2deg(abs(theta - curr_theta)))
    dominant = "keypoints" if yaw_shift_deg > 0.5 else "icp"
    return theta, tx, tz, dominant


def _extract_rigid_clouds(
    stop: AlignedStop,
    cfg: FullBodyConfig,
) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """
    Extract rigid body parts (head, torso, pelvis), excluding arms (ARM_PARTS).
    Returns (pcd_cam_rigid, pcd_canon_rigid).
    """
    if stop.pcd_canonical is None or len(stop.pcd_canonical.points) == 0:
        return o3d.geometry.PointCloud(), o3d.geometry.PointCloud()

    pts_canon = np.asarray(stop.pcd_canonical.points)
    colors = np.asarray(stop.pcd_canonical.colors) if stop.pcd_canonical.has_colors() else None

    # Apply rigid mask and arm exclusion mask
    if stop.body_part_labels is not None and cfg.rigid_parts_only:
        labels = stop.body_part_labels
        rigid_mask = is_rigid_point(labels) & ~is_arm_point(labels)
        if np.sum(rigid_mask) >= 50:
            pts_canon = pts_canon[rigid_mask]
            if colors is not None and len(colors) == len(labels):
                colors = colors[rigid_mask]

    pcd_canon_rigid = o3d.geometry.PointCloud()
    pcd_canon_rigid.points = o3d.utility.Vector3dVector(pts_canon)
    if colors is not None and len(colors) == len(pts_canon):
        pcd_canon_rigid.colors = o3d.utility.Vector3dVector(colors)

    # Transform back to camera coordinates
    T_cam_canon = np.linalg.inv(stop.prior_T_canonical_camera)
    pcd_cam_rigid = copy.deepcopy(pcd_canon_rigid).transform(T_cam_canon)

    return pcd_cam_rigid, pcd_canon_rigid


def _refine_alignment_multiview(
    aligned_stops: List[AlignedStop],
    cfg: FullBodyConfig,
    raw_stops: Optional[List[StopData]] = None,
) -> None:
    """
    Multi-view pose-graph registration across stops:
    1. Extract rigid body parts (head, torso, pelvis), excluding arms (ARM_PARTS).
       NOTE: Per-limb articulated refinement for arms is not implemented here (known limitation).
    2. Build PoseGraph with nodes = stops, initial poses = nominal canonical poses.
    3. Register consecutive edges (i, i+1) and loop-closing edge (N-1, 0) via 3-parameter motion ICP
       (yaw about vertical + XZ translation) followed by 3D keypoint constraint refinement.
    4. Reject edges with low fitness or excessive yaw correction, falling back to nominal edge.
    5. Information matrices from o3d.pipelines.registration.get_information_matrix_from_point_clouds.
    6. Measure loop-closure error BEFORE optimization (residual of closing edge in deg and mm).
    7. Global optimization (Levenberg-Marquardt) updates T_canonical_camera for every stop.
    8. Record per-stop edge diagnostics: edge fitness, rmse mm, yaw correction deg, source ("icp"|"nominal").
    """
    n = len(aligned_stops)
    if n < 2:
        return

    # 1. Extract rigid clouds per stop
    rigid_clouds_cam: List[o3d.geometry.PointCloud] = []
    rigid_clouds_canon: List[o3d.geometry.PointCloud] = []
    for s in aligned_stops:
        p_cam, p_can = _extract_rigid_clouds(s, cfg)
        rigid_clouds_cam.append(p_cam)
        rigid_clouds_canon.append(p_can)

    # 2. Define edge pairs: consecutive (i, i+1) and loop closing (N-1, 0) for N >= 3
    if n == 2:
        edge_pairs = [(0, 1)]
    else:
        edge_pairs = [(i, (i + 1) % n) for i in range(n)]

    edge_results: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for (i, j) in edge_pairs:
        src_canon = rigid_clouds_canon[i]
        tgt_canon = rigid_clouds_canon[j]

        if len(src_canon.points) < 20 or len(tgt_canon.points) < 20:
            edge_results[(i, j)] = {
                "T_delta": np.eye(4, dtype=np.float64),
                "fitness": 0.0,
                "rmse_mm": 999.0,
                "rmse_before_mm": 999.0,
                "yaw_corr": 0.0,
                "source": "nominal",
                "dominant": "nominal",
                "accepted": False,
                "reason": "Insufficient rigid points for ICP registration (<20 points)",
            }
            if j != 0:
                aligned_stops[j].accepted = False
                aligned_stops[j].review_status = "NEEDS_REVIEW"
                aligned_stops[j].rejection_reason = edge_results[(i, j)]["reason"]
            continue

        T_delta_icp, fitness, rmse_mm, rmse_before_mm = solve_3param_point_to_plane_icp(
            src_canon,
            tgt_canon,
            voxel_scales=(0.02, 0.01, 0.005),
            distance_threshold_m=cfg.alignment_icp_distance_threshold_m,
        )

        theta_icp = float(np.arctan2(T_delta_icp[2, 0], T_delta_icp[0, 0]))
        tx_icp = float(T_delta_icp[0, 3])
        tz_icp = float(T_delta_icp[2, 3])

        # Pass 2: 3D keypoint refinement
        theta_ref, tx_ref, tz_ref, dominant = refine_with_keypoint_constraints(
            theta_icp,
            tx_icp,
            tz_icp,
            aligned_stops[i].keypoints_canonical,
            aligned_stops[j].keypoints_canonical,
        )

        c = np.cos(theta_ref)
        s = np.sin(theta_ref)
        T_delta = np.array([
            [c,   0.0, -s,  tx_ref],
            [0.0, 1.0,  0.0, 0.0],
            [s,   0.0,  c,   tz_ref],
            [0.0, 0.0,  0.0, 1.0],
        ], dtype=np.float64)

        yaw_corr = float(np.rad2deg(abs(theta_ref)))

        # Edge acceptance/rejection criteria (fitness and yaw bounds)
        if fitness < cfg.alignment_min_inlier_fraction:
            edge_accepted = False
            rejection_reason = (
                f"ICP fitness {fitness:.3f} < threshold {cfg.alignment_min_inlier_fraction:.2f}"
            )
        elif yaw_corr > cfg.alignment_max_yaw_correction_deg:
            edge_accepted = False
            rejection_reason = (
                f"Yaw correction {yaw_corr:.1f} deg > threshold {cfg.alignment_max_yaw_correction_deg:.1f} deg"
            )
        else:
            edge_accepted = True
            rejection_reason = ""

        if not edge_accepted:
            T_delta = np.eye(4, dtype=np.float64)
            edge_source = "nominal"
            dominant = "nominal"
            if j != 0:
                aligned_stops[j].accepted = False
                aligned_stops[j].review_status = "NEEDS_REVIEW"
                aligned_stops[j].rejection_reason = rejection_reason
        else:
            edge_source = "icp"

        edge_results[(i, j)] = {
            "T_delta": T_delta,
            "fitness": fitness,
            "rmse_mm": rmse_mm,
            "rmse_before_mm": rmse_before_mm,
            "yaw_corr": yaw_corr,
            "source": edge_source,
            "dominant": dominant,
            "accepted": edge_accepted,
            "reason": rejection_reason,
        }

    # 3. Compute loop-closure error BEFORE optimization (closing edge residual)
    if n >= 3 and (n - 1, 0) in edge_results:
        T_chain = np.eye(4, dtype=np.float64)
        for k in range(n - 1):
            T_chain = edge_results[(k, k + 1)]["T_delta"] @ T_chain

        T_closing = edge_results[(n - 1, 0)]["T_delta"]
        T_loop = T_closing @ T_chain
        R_loop = T_loop[:3, :3]
        delta_theta_loop = float(np.rad2deg(np.arctan2(R_loop[2, 0], R_loop[0, 0])))
        delta_t_loop_mm = float(math.hypot(T_loop[0, 3], T_loop[2, 3]) * 1000.0)

        logger.info(
            f"  [align] Loop-closure error before optimization: yaw {delta_theta_loop:.2f} deg, "
            f"trans {delta_t_loop_mm:.1f} mm"
        )
        for s in aligned_stops:
            s.loop_closure_residual_deg = delta_theta_loop
            s.loop_closure_residual_mm = delta_t_loop_mm

    # 4. Construct PoseGraph
    pg = o3d.pipelines.registration.PoseGraph()
    for s in aligned_stops:
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(s.prior_T_canonical_camera))

    for (i, j) in edge_pairs:
        res = edge_results[(i, j)]
        T_i_init = aligned_stops[i].prior_T_canonical_camera
        T_j_init = aligned_stops[j].prior_T_canonical_camera
        T_delta = res["T_delta"]

        # Relative transform mapping camera i -> camera j in Open3D PoseGraph convention:
        # P_j = T_edge_cam @ P_i.
        # Since T_delta aligns canonical cloud i to canonical cloud j: P_canon,j ~ T_delta @ P_canon,i
        # target pose satisfies T_j = T_delta @ T_j_init, so T_edge = inv(T_j) @ T_i = inv(T_delta @ T_j_init) @ T_i_init.
        T_edge_cam = np.linalg.inv(T_delta @ T_j_init) @ T_i_init

        if res["source"] == "icp" and len(rigid_clouds_cam[i].points) >= 10 and len(rigid_clouds_cam[j].points) >= 10:
            pcd_i_down = rigid_clouds_cam[i].voxel_down_sample(0.02)
            pcd_j_down = rigid_clouds_cam[j].voxel_down_sample(0.02)
            info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                pcd_i_down,
                pcd_j_down,
                cfg.alignment_icp_distance_threshold_m,
                T_edge_cam,
            )
            if not np.all(np.isfinite(info)) or np.trace(info) < 1.0:
                info = np.eye(6, dtype=np.float64) * 1.0
            # Odometry chain is certain; loop-closing edge (N-1 -> 0) is uncertain
            uncertain = (j == 0) or (not res["accepted"])
        else:
            info = np.eye(6, dtype=np.float64) * 1e-4
            uncertain = True

        pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(i, j, T_edge_cam, info, uncertain=uncertain))

    # 5. Global Levenberg-Marquardt optimization
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=cfg.alignment_icp_distance_threshold_m,
        edge_prune_threshold=0.25,
        preference_loop_closure=0.1,
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(
        pg,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )

    # 6. Update stop poses and canonical geometries
    for i in range(n):
        s = aligned_stops[i]
        T_opt = pg.nodes[i].pose
        s.T_canonical_camera = T_opt

        # Canonical yaw: prior yaw updated by pose-graph optimized rotation
        T_delta_opt = T_opt @ np.linalg.inv(s.prior_T_canonical_camera)
        delta_yaw = float(np.rad2deg(np.arctan2(T_delta_opt[2, 0], T_delta_opt[0, 0])))
        s.refined_yaw_deg = float((s.prior_yaw_deg + delta_yaw + 360.0) % 360.0)
        s.refined_translation_m = T_opt[:3, 3].copy()

        # Update point cloud in canonical frame
        if raw_stops and raw_stops[i].pcd is not None:
            s.pcd_canonical = o3d.geometry.PointCloud(raw_stops[i].pcd)
            s.pcd_canonical.transform(T_opt)
        elif s.pcd_canonical is not None:
            T_upd = T_opt @ np.linalg.inv(s.prior_T_canonical_camera)
            s.pcd_canonical.transform(T_upd)

        # Update keypoints in canonical frame
        if raw_stops and raw_stops[i].keypoints_camera is not None:
            s.keypoints_canonical = raw_stops[i].keypoints_camera.transform(T_opt, "canonical")
        elif s.keypoints_canonical is not None:
            T_upd = T_opt @ np.linalg.inv(s.prior_T_canonical_camera)
            s.keypoints_canonical = s.keypoints_canonical.transform(T_upd, "canonical")

        if s.pcd_canonical is not None and s.keypoints_canonical is not None:
            s.body_part_labels = label_points_by_body_part(
                np.asarray(s.pcd_canonical.points), s.keypoints_canonical, cfg
            )

        # Store edge diagnostics on stop
        if i > 0:
            diag = edge_results.get((i - 1, i))
        else:
            diag = edge_results.get((n - 1, 0)) or edge_results.get((0, 1))

        if diag:
            s.edge_fitness = diag["fitness"]
            s.edge_rmse_mm = diag["rmse_mm"]
            s.yaw_correction_deg = diag["yaw_corr"]
            s.registration_source = diag["source"]
            s.dominant_constraint = diag["dominant"]
            s.residual_before_mm = diag["rmse_before_mm"]
            s.residual_after_mm = diag["rmse_mm"]
            s.inlier_fraction = diag["fitness"]
            if diag["source"] == "icp":
                s.accepted = True
                s.review_status = "ACCEPTED"
                if diag["rmse_mm"] > cfg.max_inter_stop_residual_mm:
                    s.review_status = "NEEDS_REVIEW"


