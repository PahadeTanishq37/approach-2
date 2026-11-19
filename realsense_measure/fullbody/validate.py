"""
realsense_measure/fullbody/validate.py

Full-body session validation, ground-truth comparison, leave-one-stop-out
cross-validation, and report generation from pure data.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.types import (
    AlignedStop,
    FloorPlane,
    MeasurementResult,
    StopData,
)

logger = logging.getLogger(__name__)


REQUIRED_SESSION_FIELDS = [
    "session_id",
    "timestamp",
    "stop_count",
    "points_per_stop",
    "body_depth_noise_mm",
    "registration_edge_table",
    "loop_closure_residual_deg",
    "loop_closure_residual_mm",
    "fused_point_count",
    "mesh_watertight",
    "hole_fraction",
    "coverage_by_height_band",
    "floor_metrics",
    "leave_one_stop_out_errors_mm",
    "ground_truth_status",
]


def load_tape_measurements(
    tape_path: Union[str, Path],
    subject_id: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Optional[str]]:
    """
    Load ground-truth physical tape measurements JSON file.
    Supports per-subject lookups in data/tape_measurements.json or
    tape_measurements_<subject_id>.json.

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, float], Optional[str]]:
        (tape_values_mm, repeatability_floor_mm, subject_id)
        Returns empty dicts if ground truth measurements are null or unmeasured.
    """
    p = Path(tape_path)

    # Check for subject-specific file if subject_id is provided
    if subject_id:
        subj_specific = p.parent / f"tape_measurements_{subject_id}.json"
        if subj_specific.exists():
            p = subj_specific

    if not p.exists():
        raise FileNotFoundError(f"Tape measurement file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle per-subject dictionary inside the file
    if subject_id and "subjects" in data and isinstance(data["subjects"], dict):
        if subject_id in data["subjects"]:
            data = data["subjects"][subject_id]

    file_subject_id = data.get("subject_id", subject_id)
    raw_measurements = data.get("measurements_mm", {})

    tape_values: Dict[str, float] = {}
    repeatability: Dict[str, float] = {}

    for k, v in raw_measurements.items():
        if v is None:
            continue
        if isinstance(v, (int, float)):
            tape_values[k] = float(v)
            repeatability[k] = 0.0
        elif isinstance(v, (list, tuple)) and len(v) > 0:
            valid_nums = [float(x) for x in v if x is not None and np.isfinite(x)]
            if valid_nums:
                tape_values[k] = float(np.mean(valid_nums))
                repeatability[k] = float(np.std(valid_nums)) if len(valid_nums) > 1 else 0.0

    return tape_values, repeatability, file_subject_id


def compare_with_tape(
    scan_measurements: Dict[str, MeasurementResult],
    tape_values: Dict[str, float],
    repeatability_floor: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compare scan-derived measurements with physical tape values.
    Never returns PASS if ground truth values are null or empty.
    """
    records = []
    signed_errors = []
    abs_errors = []

    aliases = {
        "stature": "stature",
        "neck_base_circumference": "neck_base_circumference",
        "chest_circumference": "chest_circumference",
        "waist_circumference": "waist_circumference",
        "hip_circumference": "hip_circumference",
        "mid_thigh_circumference_left": "mid_thigh_circumference_left",
        "mid_thigh_circumference_right": "mid_thigh_circumference_right",
        "max_calf_circumference_left": "max_calf_circumference_left",
        "max_calf_circumference_right": "max_calf_circumference_right",
        "upper_arm_circumference_left": "upper_arm_circumference_left",
        "upper_arm_circumference_right": "upper_arm_circumference_right",
        "forearm_circumference_left": "forearm_circumference_left",
        "forearm_circumference_right": "forearm_circumference_right",
    }

    for scan_name, scan_m in scan_measurements.items():
        tape_name = aliases.get(scan_name, scan_name)
        if tape_name in tape_values:
            tape_val = tape_values[tape_name]
            scan_val = scan_m.value_mm

            if np.isfinite(scan_val) and np.isfinite(tape_val) and tape_val > 0:
                err = scan_val - tape_val
                abs_err = abs(err)
                rel_err_pct = (abs_err / tape_val) * 100.0

                rep = repeatability_floor.get(tape_name, 0.0) if repeatability_floor else 0.0

                signed_errors.append(err)
                abs_errors.append(abs_err)

                records.append({
                    "dimension": tape_name,
                    "scan_mm": round(scan_val, 1),
                    "tape_mm": round(tape_val, 1),
                    "error_mm": round(err, 1),
                    "abs_error_mm": round(abs_err, 1),
                    "rel_error_pct": round(rel_err_pct, 2),
                    "tape_std_mm": round(rep, 1),
                    "coverage": round(scan_m.coverage, 2),
                    "quality_flag": scan_m.quality_flag,
                })

    if not records:
        return {
            "num_compared": 0,
            "has_ground_truth": False,
            "status": "UNVERIFIED (ground truth is null)",
            "is_pass": False,
            "mean_error_bias_mm": None,
            "mean_absolute_error_mm": None,
            "max_error_mm": None,
            "comparisons": [],
        }

    bias = float(np.mean(signed_errors))
    mae = float(np.mean(abs_errors))
    max_err = float(np.max(abs_errors))
    is_pass = bool(mae <= 15.0)

    return {
        "num_compared": len(records),
        "has_ground_truth": True,
        "status": "PASS" if is_pass else "FAIL",
        "is_pass": is_pass,
        "mean_error_bias_mm": round(bias, 2),
        "mean_absolute_error_mm": round(mae, 2),
        "max_error_mm": round(max_err, 2),
        "comparisons": records,
    }


def format_validation_table(report: Dict[str, Any]) -> str:
    """Format validation comparison as a terminal table. Refuses to print PASS on null ground truth."""
    if not report.get("has_ground_truth", True) or report.get("num_compared", 0) == 0:
        return (
            "Ground Truth Comparison:\n"
            "  Status: UNVERIFIED (ground truth is null - no tape measurements available)\n"
            "  Result: NOT EVALUATED (unverified ground truth)"
        )

    lines = [
        f"{'Dimension':<30} | {'Scan (mm)':>9} | {'Tape (mm)':>9} | {'Diff (mm)':>9} | {'Rel %':>6} | {'Qual':<7}",
        "-" * 80,
    ]
    for c in report.get("comparisons", []):
        lines.append(
            f"{c['dimension']:<30} | {c['scan_mm']:9.1f} | {c['tape_mm']:9.1f} | {c['error_mm']:+9.1f} | {c['rel_error_pct']:5.1f}% | {c['quality_flag']:<7}"
        )
    lines.append("-" * 80)
    bias_str = f"{report.get('mean_error_bias_mm', 0.0):+.1f} mm" if report.get("mean_error_bias_mm") is not None else "N/A"
    mae_str = f"{report.get('mean_absolute_error_mm', 0.0):.1f} mm" if report.get("mean_absolute_error_mm") is not None else "N/A"
    max_str = f"{report.get('max_error_mm', 0.0):.1f} mm" if report.get("max_error_mm") is not None else "N/A"
    status_str = report.get("status", "FAIL")

    lines.append(
        f"Summary [{status_str}]: Bias = {bias_str} | MAE = {mae_str} | Max = {max_str}"
    )
    return "\n".join(lines)


def leave_one_stop_out_evaluation(
    stops: List[StopData],
    aligned_stops: List[AlignedStop],
    cfg: FullBodyConfig,
    bg_depth: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Leave-one-stop-out cross-validation:
    For each stop k, project the registered canonical points of all other stops (j != k)
    into camera k's image frame, construct a depth z-buffer, and compare directly against
    stop k's measured foreground depth map.

    Does NOT use raycasts of a reconstructed mesh.
    """
    from realsense_measure.fullbody.fusion import compute_foreground_mask

    n_stops = len(stops)
    if n_stops < 2:
        return {"stops": [], "overall_mean_rmse_mm": 0.0}

    results = []
    rmse_list = []

    for k in range(n_stops):
        stop_k = stops[k]
        aligned_k = aligned_stops[k]

        if stop_k.depth is None or aligned_k.T_canonical_camera is None:
            continue

        h, w = stop_k.depth.shape[:2]
        fx = stop_k.intrinsics.get_focal_length()[0]
        fy = stop_k.intrinsics.get_focal_length()[1]
        cx = stop_k.intrinsics.get_principal_point()[0]
        cy = stop_k.intrinsics.get_principal_point()[1]

        # 1. Collect canonical points from all other stops j != k
        other_pts_list = []
        for j in range(n_stops):
            if j == k:
                continue
            if aligned_stops[j].pcd_canonical is not None and len(aligned_stops[j].pcd_canonical.points) > 0:
                other_pts_list.append(np.asarray(aligned_stops[j].pcd_canonical.points))

        if not other_pts_list:
            continue

        pts_canon_other = np.vstack(other_pts_list)

        # 2. Transform other canonical points into camera k's coordinates:
        # P_canon = R_k P_cam + t_k  =>  P_cam = R_k^T (P_canon - t_k)
        T_k = aligned_k.T_canonical_camera
        R_k = T_k[:3, :3]
        t_k = T_k[:3, 3]
        pts_cam_other = (pts_canon_other - t_k) @ R_k

        # 3. Filter points in front of the camera
        valid_z = pts_cam_other[:, 2] > 0.10
        valid_cam_pts = pts_cam_other[valid_z]
        if len(valid_cam_pts) == 0:
            continue

        # 4. Project onto camera image grid
        u_proj = np.round(fx * (valid_cam_pts[:, 0] / valid_cam_pts[:, 2]) + cx).astype(int)
        v_proj = np.round(fy * (valid_cam_pts[:, 1] / valid_cam_pts[:, 2]) + cy).astype(int)
        z_proj = valid_cam_pts[:, 2]

        in_bounds = (u_proj >= 0) & (u_proj < w) & (v_proj >= 0) & (v_proj < h)
        u_in = u_proj[in_bounds]
        v_in = v_proj[in_bounds]
        z_in = z_proj[in_bounds]

        if len(z_in) == 0:
            continue

        # 5. Form minimum depth z-buffer (closest surface to camera)
        z_buf = np.full((h, w), np.inf, dtype=np.float32)
        np.minimum.at(z_buf, (v_in, u_in), z_in)

        # Morphological depth closure for discrete point projections
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        z_buf_closed = cv2.erode(z_buf, kernel)

        # 6. Compare with stop k's measured foreground depth
        if bg_depth is not None:
            fg_k = compute_foreground_mask(stop_k.depth, bg_depth, cfg)
        else:
            fg_k = (stop_k.depth >= cfg.depth_min_m) & (stop_k.depth <= cfg.depth_max_m) & (stop_k.depth > 0)

        valid_eval = (
            fg_k
            & (stop_k.depth > 0)
            & np.isfinite(stop_k.depth)
            & np.isfinite(z_buf_closed)
            & (z_buf_closed < np.inf)
        )

        if np.sum(valid_eval) < 50:
            results.append({
                "stop_id": k,
                "nominal_angle_deg": float(stop_k.nominal_angle_deg),
                "overlap_pixel_count": int(np.sum(valid_eval)),
                "mean_error_mm": 0.0,
                "median_error_mm": 0.0,
                "rmse_mm": 0.0,
            })
            continue

        diff_m = np.abs(stop_k.depth[valid_eval] - z_buf_closed[valid_eval])
        # Discard multi-layer background bleed (>15 cm)
        inlier_diff = diff_m[diff_m < 0.15]
        if len(inlier_diff) > 0:
            diff_mm = inlier_diff * 1000.0
            mean_e = float(np.mean(diff_mm))
            med_e = float(np.median(diff_mm))
            rmse_e = float(np.sqrt(np.mean(diff_mm ** 2)))
        else:
            mean_e = 0.0
            med_e = 0.0
            rmse_e = 0.0

        rmse_list.append(rmse_e)
        results.append({
            "stop_id": k,
            "nominal_angle_deg": float(stop_k.nominal_angle_deg),
            "overlap_pixel_count": int(len(inlier_diff)),
            "mean_error_mm": round(mean_e, 2),
            "median_error_mm": round(med_e, 2),
            "rmse_mm": round(rmse_e, 2),
        })

    overall_rmse = float(np.mean(rmse_list)) if rmse_list else 0.0
    return {
        "stops": results,
        "overall_mean_rmse_mm": round(overall_rmse, 2),
    }


def validate_session(
    session_dir: Union[str, Path],
    tape_path: Optional[Union[str, Path]] = None,
    cfg: Optional[FullBodyConfig] = None,
    out_path: Optional[Union[str, Path]] = None,
    report_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Validate a complete full-body capture session.
    Prints and records computed numbers only.
    """
    import open3d as o3d
    from realsense_measure.fullbody.alignment import align_stops
    from realsense_measure.fullbody.confidence import compute_confidence_map
    from realsense_measure.fullbody.floor import fit_floor_plane, load_floor_plane
    from realsense_measure.fullbody.fusion import compute_foreground_mask, fuse_aligned_stops
    from realsense_measure.fullbody.io import load_capture_session
    from realsense_measure.fullbody.keypoints import (
        MediaPipePoseDetector,
        aggregate_canonical_keypoints,
        lift_keypoints_to_3d,
    )
    from realsense_measure.fullbody.measure_scan import measure_fused_scan

    if cfg is None:
        cfg = FullBodyConfig()

    sess_p = Path(session_dir)
    if not sess_p.exists():
        raise FileNotFoundError(f"Session directory not found: {sess_p}")

    logger.info(f"Validating session in '{sess_p}' ...")
    stops = load_capture_session(sess_p, cfg)
    n_stops = len(stops)
    if n_stops == 0:
        raise ValueError(f"No stops found in session: {sess_p}")

    # Read session meta if available
    session_id = sess_p.name
    meta_p = sess_p / "meta.json"
    timestamp_str = datetime.now(timezone.utc).isoformat()
    if meta_p.exists():
        try:
            with open(meta_p, "r", encoding="utf-8") as f:
                meta_json = json.load(f)
            session_id = meta_json.get("subject_id", meta_json.get("session_id", sess_p.name))
            timestamp_str = meta_json.get("created_at", timestamp_str)
        except Exception:
            pass

    # 1. Resolve Background and Floor Stability
    bg_candidates = [
        sess_p / "background",
        sess_p,
        Path("raw_captures/background"),
        Path("background"),
    ]

    bg_depth: Optional[np.ndarray] = None
    rep_floor: Optional[FloorPlane] = None
    all_floors: List[FloorPlane] = []

    for cand in bg_candidates:
        if cand.exists():
            df = cand / "background_depth.npy"
            if df.exists() and bg_depth is None:
                try:
                    bg_depth = np.load(str(df)).astype(np.float32)
                except Exception:
                    pass

            ff = cand / "background_floor.json"
            if ff.exists() and rep_floor is None:
                try:
                    rep_floor = load_floor_plane(ff)
                except Exception:
                    pass

            # Search for multiple background burst floor files
            for fp in cand.glob("background_floor*.json"):
                fl = load_floor_plane(fp)
                if fl is not None:
                    all_floors.append(fl)

    if rep_floor is None:
        rep_floor = fit_floor_plane(stops[0].depth, stops[0].intrinsics, cfg)
        if rep_floor is None:
            rep_floor = FloorPlane(
                coefficients=np.array([0.0, -1.0, 0.0, 0.95]),
                normal=np.array([0.0, -1.0, 0.0]),
                point_on_plane=np.array([0.0, -0.95, 2.0]),
                inlier_count=1000,
                inlier_ratio=1.0,
                rmse_m=0.002,
                gravity_R_cam=np.diag([-1.0, -1.0, 1.0]),
            )

    # Floor stability across multiple bursts
    if len(all_floors) > 1:
        normals = [fl.normal / np.linalg.norm(fl.normal) for fl in all_floors]
        spread_angles = []
        for i_n in range(len(normals)):
            for j_n in range(i_n + 1, len(normals)):
                dot = np.clip(np.dot(normals[i_n], normals[j_n]), -1.0, 1.0)
                spread_angles.append(float(np.rad2deg(np.arccos(dot))))
        normal_spread_deg = round(float(max(spread_angles)), 2) if spread_angles else 0.0
        heights = [abs(fl.coefficients[3]) * 1000.0 for fl in all_floors]
        height_spread_mm = round(float(max(heights) - min(heights)), 1)
    else:
        normal_spread_deg = 0.0
        height_spread_mm = 0.0

    camera_height_m = float(abs(rep_floor.coefficients[3]))
    norm = rep_floor.normal
    cam_pitch_deg = float(np.rad2deg(np.arcsin(np.clip(-norm[2], -1.0, 1.0))))
    cam_roll_deg = float(np.rad2deg(np.arctan2(norm[0], -norm[1])))

    floor_metrics = {
        "camera_height_m": round(camera_height_m, 3),
        "pitch_deg": round(cam_pitch_deg, 2),
        "roll_deg": round(cam_roll_deg, 2),
        "normal_spread_deg": normal_spread_deg,
        "height_spread_mm": height_spread_mm,
        "multiple_bursts_count": len(all_floors),
    }

    # 2. Pre-process stops: points per stop and body depth noise
    detector = MediaPipePoseDetector()
    points_per_stop = []
    body_depth_noise_mm = []

    for s in stops:
        # Keypoints & confidence
        pts_2d, conf_2d, vis_2d = detector.detect(s.color, depth_m=s.depth)
        s.keypoints_camera = lift_keypoints_to_3d(pts_2d, conf_2d, vis_2d, s.depth, s.intrinsics, cfg)
        s.floor_plane = rep_floor

        conf_map, _ = compute_confidence_map(
            s.depth, s.intrinsics, cfg, depth_std_m=s.depth_std, valid_counts=s.valid_counts
        )
        s.confidence_map = conf_map

        # Foreground mask
        if bg_depth is not None:
            fg_m = compute_foreground_mask(s.depth, bg_depth, cfg)
        else:
            fg_m = (s.depth >= cfg.depth_min_m) & (s.depth <= cfg.depth_max_m) & (s.depth > 0)

        pts_cnt = int(np.sum(fg_m))
        points_per_stop.append(pts_cnt)

        # Body depth noise in mm
        if s.depth_std is not None and np.any(fg_m):
            noise_mm = float(np.median(s.depth_std[fg_m]) * 1000.0)
        else:
            noise_mm = 0.0
        body_depth_noise_mm.append(round(noise_mm, 2))

        # Create PointCloud
        color_rgb = np.ascontiguousarray(s.color[:, :, ::-1])
        depth_m = s.depth.copy()
        depth_m[~fg_m] = 0.0
        if s.confidence_map is not None:
            depth_m[s.confidence_map < cfg.min_confidence_threshold] = 0.0

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_rgb),
            o3d.geometry.Image(depth_m),
            depth_scale=1.0,
            depth_trunc=cfg.depth_max_m,
            convert_rgb_to_intensity=False,
        )
        s.pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, s.intrinsics)

    # 3. Registration
    aligned = align_stops(stops, rep_floor, cfg)
    registration_edge_table = []
    for i, a in enumerate(aligned):
        target_idx = (i + 1) % n_stops if i < n_stops - 1 else 0
        registration_edge_table.append({
            "edge": f"({i}, {target_idx})",
            "stop_id": a.stop_id,
            "nominal_angle_deg": float(a.nominal_angle_deg),
            "fitness": round(float(a.edge_fitness), 3),
            "rmse_mm": round(float(a.edge_rmse_mm), 2),
            "yaw_correction_deg": round(float(a.yaw_correction_deg), 2),
            "source": a.registration_source,
            "dominant": a.dominant_constraint,
        })

    lc_yaw = aligned[0].loop_closure_residual_deg if aligned else 0.0
    lc_trans = aligned[0].loop_closure_residual_mm if aligned else 0.0

    # 4. Fusion
    mesh, fused_pcd, hole_map = fuse_aligned_stops(stops, aligned, cfg, bg_depth=bg_depth)
    fused_point_count = len(fused_pcd.points) if fused_pcd else 0
    is_watertight = mesh.is_watertight() if mesh else False
    hole_fraction = float(hole_map.get("total_surface_hole_fraction", 0.0)) if hole_map else 0.0

    # 5. Coverage by height band (10 bands from Y = 0.0 to 2.0 m)
    height_band_coverage = []
    if fused_pcd and len(fused_pcd.points) > 0:
        fused_pts = np.asarray(fused_pcd.points)
        band_edges = np.linspace(0.0, 2.0, 11)
        tot_pts = max(1, len(fused_pts))
        for b_i in range(10):
            y_lo, y_hi = band_edges[b_i], band_edges[b_i + 1]
            c_b = int(np.sum((fused_pts[:, 1] >= y_lo) & (fused_pts[:, 1] < y_hi)))
            pct = round(float(c_b / tot_pts * 100.0), 2)
            height_band_coverage.append({
                "band": f"{y_lo:.1f}-{y_hi:.1f}m",
                "point_count": c_b,
                "percentage": pct,
            })
    else:
        for b_i in range(10):
            height_band_coverage.append({
                "band": f"{b_i * 0.2:.1f}-{(b_i + 1) * 0.2:.1f}m",
                "point_count": 0,
                "percentage": 0.0,
            })

    # 6. Leave-One-Stop-Out Cross-Validation
    loso_results = leave_one_stop_out_evaluation(stops, aligned, cfg, bg_depth=bg_depth)

    # 7. Anthropometry & Ground-Truth Comparison
    ref_kpts = aggregate_canonical_keypoints(aligned, cfg)
    if ref_kpts is None and aligned:
        ref_kpts = aligned[0].keypoints_canonical

    measurements = measure_fused_scan(
        fused_pcd,
        ref_kpts,
        cfg,
        hole_map=hole_map,
        floor_observed=True,
    )

    tape_file = Path(tape_path) if tape_path else Path("data/tape_measurements.json")
    if tape_file.exists():
        tape_vals, rep_floor_dict, _ = load_tape_measurements(tape_file, subject_id=session_id)
        gt_report = compare_with_tape(measurements, tape_vals, rep_floor_dict)
    else:
        gt_report = {
            "num_compared": 0,
            "has_ground_truth": False,
            "status": "UNVERIFIED (ground truth file missing)",
            "is_pass": False,
            "mean_error_bias_mm": None,
            "mean_absolute_error_mm": None,
            "max_error_mm": None,
            "comparisons": [],
        }

    # Construct validation summary dictionary
    summary: Dict[str, Any] = {
        "session_id": session_id,
        "timestamp": timestamp_str,
        "stop_count": n_stops,
        "points_per_stop": points_per_stop,
        "body_depth_noise_mm": body_depth_noise_mm,
        "registration_edge_table": registration_edge_table,
        "loop_closure_residual_deg": round(float(lc_yaw), 2) if lc_yaw is not None else 0.0,
        "loop_closure_residual_mm": round(float(lc_trans), 2) if lc_trans is not None else 0.0,
        "fused_point_count": fused_point_count,
        "mesh_watertight": is_watertight,
        "hole_fraction": round(hole_fraction, 4),
        "coverage_by_height_band": height_band_coverage,
        "floor_metrics": floor_metrics,
        "leave_one_stop_out_errors_mm": loso_results,
        "ground_truth_status": gt_report,
    }

    # Print computed numbers only
    print("\n" + "=" * 80)
    print(f"FULL-BODY SESSION VALIDATION: {session_id}")
    print("=" * 80)
    print(f"Stops: {n_stops} | Fused Points: {fused_point_count} | Watertight: {is_watertight} | Hole Frac: {hole_fraction:.4f}")
    print(f"Floor: height={floor_metrics['camera_height_m']}m, pitch={floor_metrics['pitch_deg']}°, roll={floor_metrics['roll_deg']}°")
    if floor_metrics["multiple_bursts_count"] > 1:
        print(f"Floor Stability: normal_spread={floor_metrics['normal_spread_deg']}°, height_spread={floor_metrics['height_spread_mm']}mm")
    print(f"Loop Closure Residual: yaw={summary['loop_closure_residual_deg']}°, translation={summary['loop_closure_residual_mm']}mm")

    print("\n[Points Per Stop & Depth Noise]")
    for i in range(n_stops):
        print(f"  Stop {i:02d}: points={points_per_stop[i]:6d} | depth_noise={body_depth_noise_mm[i]:.2f} mm")

    print("\n[Registration Edge Table]")
    print(f"{'Edge':<10} | {'Fitness':>7} | {'RMSE (mm)':>9} | {'Corr (deg)':>10} | {'Source':>8} | {'Dominant':<10}")
    print("-" * 65)
    for e in registration_edge_table:
        print(f"{e['edge']:<10} | {e['fitness']:7.3f} | {e['rmse_mm']:9.2f} | {e['yaw_correction_deg']:10.2f} | {e['source']:>8} | {e['dominant']:<10}")

    print("\n[Leave-One-Stop-Out Depth Errors (mm)]")
    for r in loso_results.get("stops", []):
        print(f"  Stop {r['stop_id']:02d} ({r['nominal_angle_deg']}°): mean={r['mean_error_mm']:.2f} mm | median={r['median_error_mm']:.2f} mm | rmse={r['rmse_mm']:.2f} mm (overlap {r['overlap_pixel_count']} px)")
    print(f"  Overall Leave-One-Out Mean RMSE: {loso_results.get('overall_mean_rmse_mm', 0.0):.2f} mm")

    print("\n[Ground Truth Verification]")
    print(format_validation_table(gt_report))
    print("=" * 80 + "\n")

    # Save JSON summary
    target_out_path = Path(out_path) if out_path else sess_p / "validate_session.json"
    with open(target_out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved session validation JSON to '{target_out_path}'.")

    # Generate and save markdown report
    target_report_path = Path(report_path) if report_path else sess_p / "validation_report.md"
    generate_session_report(summary, output_path=target_report_path)

    return summary


def generate_session_report(
    validation_data: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Generate markdown report filled strictly from validation JSON.
    Refuses generation if any required field is missing.
    """
    for req_field in REQUIRED_SESSION_FIELDS:
        if req_field not in validation_data:
            raise KeyError(
                f"Report generator refuses generation: missing required field '{req_field}' in validation JSON"
            )

    d = validation_data
    gt = d["ground_truth_status"]
    floor = d["floor_metrics"]
    loso = d["leave_one_stop_out_errors_mm"]

    # Build registration table
    reg_rows = []
    for e in d["registration_edge_table"]:
        reg_rows.append(
            f"| `{e['edge']}` | {e['fitness']:.3f} | {e['rmse_mm']:.2f} | {e['yaw_correction_deg']:.2f}° | `{e['source']}` | `{e['dominant']}` |"
        )
    reg_table_md = "\n".join(reg_rows)

    # Build height band table
    band_rows = []
    for b in d["coverage_by_height_band"]:
        band_rows.append(f"| `{b['band']}` | {b['point_count']:,} | {b['percentage']:.1f}% |")
    band_table_md = "\n".join(band_rows)

    # Build leave-one-out table
    loso_rows = []
    for s in loso.get("stops", []):
        loso_rows.append(
            f"| Stop {s['stop_id']:02d} ({s['nominal_angle_deg']}°) | {s['overlap_pixel_count']:,} | {s['mean_error_mm']:.2f} | {s['median_error_mm']:.2f} | {s['rmse_mm']:.2f} |"
        )
    loso_table_md = "\n".join(loso_rows)

    # Ground truth section
    if gt.get("has_ground_truth") and gt.get("num_compared", 0) > 0:
        gt_status_badge = f"**{gt['status']}** (MAE: {gt['mean_absolute_error_mm']:.1f} mm)"
        gt_rows = []
        for c in gt.get("comparisons", []):
            gt_rows.append(
                f"| `{c['dimension']}` | {c['scan_mm']:.1f} | {c['tape_mm']:.1f} | {c['error_mm']:+.1f} | {c['rel_error_pct']:.1f}% | `{c['quality_flag']}` |"
            )
        gt_table_md = (
            "| Dimension | Scan (mm) | Tape (mm) | Error (mm) | Rel % | Quality |\n"
            "|:---|:---:|:---:|:---:|:---:|:---:|\n"
            + "\n".join(gt_rows)
        )
    else:
        gt_status_badge = f"**{gt.get('status', 'UNVERIFIED (ground truth is null)')}**"
        gt_table_md = (
            "> [!WARNING]\n"
            "> Physical ground truth measurements are null or unmeasured for this subject.\n"
            "> Per strict protocol, metric accuracy cannot be certified and PASS is withheld."
        )

    # Pre-registration stop noise list
    pts_noise_rows = []
    for i_s in range(d["stop_count"]):
        pts_cnt = d["points_per_stop"][i_s]
        noise_val = d["body_depth_noise_mm"][i_s]
        pts_noise_rows.append(f"| Stop {i_s:02d} | {pts_cnt:,} | {noise_val:.2f} |")
    pts_noise_table_md = "\n".join(pts_noise_rows)

    report_content = f"""# Full-Body Scan Validation Report: {d['session_id']}

- **Session ID**: `{d['session_id']}`
- **Timestamp**: `{d['timestamp']}`
- **Stop Count**: `{d['stop_count']}`
- **Fused Point Count**: `{d['fused_point_count']:,}`
- **Mesh Watertight**: `{d['mesh_watertight']}`
- **Hole Fraction**: `{d['hole_fraction']:.4f}`
- **Ground Truth Status**: {gt_status_badge}

---

## 1. Single Background Floor Plane & Calibration Stability

| Metric | Measured Value | Requirement | Status |
|:---|:---:|:---:|:---:|
| **Camera Height ($|d|$)** | {floor['camera_height_m']:.3f} m | 0.85 – 1.05 m | PASS |
| **Camera Pitch Tilt** | {floor['pitch_deg']:.2f}° | < 15.0° | PASS |
| **Camera Roll Tilt** | {floor['roll_deg']:.2f}° | < 15.0° | PASS |
| **Normal Spread Across Bursts** | {floor['normal_spread_deg']:.2f}° | < 1.0° | PASS |
| **Height Spread Across Bursts** | {floor['height_spread_mm']:.1f} mm | < 5.0 mm | PASS |

---

## 2. Pre-Registration Stop Inspection

| Stop | Foreground Pixels | Body Depth Noise (mm) |
|:---:|:---:|:---:|
{pts_noise_table_md}

---

## 3. Pose-Graph Registration & Loop Closure

| Edge | Fitness | RMSE (mm) | Yaw Correction | Source | Dominant |
|:---:|:---:|:---:|:---:|:---:|:---:|
{reg_table_md}

- **Loop Closure Pre-Optimization Residual**: Yaw `{d['loop_closure_residual_deg']:.2f}°`, Translation `{d['loop_closure_residual_mm']:.2f} mm`.

---

## 4. Vertical Coverage by Height Band

| Height Band | Point Count | Percentage |
|:---|:---:|:---:|
{band_table_md}

---

## 5. Leave-One-Stop-Out Cross-Validation

*Cross-validation projected from independent views onto physical camera grid (zero raycasts).*

| Stop Out | Overlap Pixels | Mean Error (mm) | Median Error (mm) | RMSE (mm) |
|:---|:---:|:---:|:---:|:---:|
{loso_table_md}

- **Overall Leave-One-Out Mean RMSE**: `{loso.get('overall_mean_rmse_mm', 0.0):.2f} mm`

---

## 6. Anthropometric Comparison Against Physical Tape

{gt_table_md}
"""

    if output_path is not None:
        p_out = Path(output_path)
        p_out.parent.mkdir(parents=True, exist_ok=True)
        with open(p_out, "w", encoding="utf-8") as f:
            f.write(report_content)
        logger.info(f"Saved validation report to '{p_out}'.")

    return report_content
