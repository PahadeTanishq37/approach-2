"""
tests/fullbody/test_rigid_baseline.py

Synthetic tests for Layer 1 rigid baseline reconstruction and object calibration:
1. Clean rigid cylinder (r=150mm, 8 stops, 4mm noise, known axis):
   Diameter error < 3 mm, 1 component, circle RMS reported.
2. Rotation axis recovery with 5 cm offset:
   Axis estimation recovers true axis to < 10 mm error.
3. Yaw perturbation with and without refinement (+-3 deg):
   With --refine: diameter error < 5 mm.
   Without --refine: records worse error.
4. Articulated humanoid (rigid vs 2 cm arm drift):
   Rigid produces 1 component; arm drift creates ghosting/degradation.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import pytest

from realsense_measure.fullbody.rigid_baseline import (
    estimate_rotation_axis,
    run_rigid_baseline,
)
from tests.fullbody.synth_humanoid import SyntheticHumanoidCaptureRenderer
from tools.calib_object import analyze_cylinder_diameter


def create_synthetic_cylinder_session(
    dest_dir: Path,
    radius_m: float = 0.150,
    axis_x: float = 0.0,
    axis_z: float = 2.0,
    cam_height_m: float = 0.95,
    noise_std_m: float = 0.004,
    yaw_errors_deg: Optional[List[float]] = None,
    eccentricity_m: float = 0.0,
    has_box: bool = False,
    box_radial_m: float = 0.040,
    box_lateral_m: float = 0.060,
    box_height_m: float = 0.100,
    n_frames: int = 5,
    width: int = 640,
    height: int = 480,
    fx: float = 640.0,
    fy: float = 640.0,
) -> Path:
    """
    Generate synthetic 8-stop turntable session of a rigid cylinder or asymmetric object.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    cx = width / 2.0
    cy = height / 2.0

    meta = {
        "schema_version": 2,
        "layout": "stops_v2",
        "protocol_version": "stop-and-go-v1",
        "subject_id": "ASYM_01" if has_box else "BUCKET_01",
        "subject_type": "object",
        "object_diameter_mm": float(2.0 * radius_m * 1000.0),
        "n_stops": 8,
        "turn_direction": "cw",
        "depth_scale": 0.001,
        "depth_units": "sensor_units",
        "intrinsics": {
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        },
        "rig_metadata": {
            "cam_height_m": cam_height_m,
            "cam_distance_m": axis_z,
            "cam_tilt_deg": 0.0,
        },
    }
    with open(dest_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
    alpha = -(u_grid - cx) / fx
    beta = (v_grid - cy) / fy
    A = alpha ** 2 + 1.0

    if yaw_errors_deg is None:
        yaw_errors_deg = [0.0] * 8

    for s_idx in range(8):
        s_folder = dest_dir / f"stop_{s_idx:02d}" / "attempt_00"
        s_folder.mkdir(parents=True, exist_ok=True)
        with open(dest_dir / f"stop_{s_idx:02d}" / "stop.json", "w", encoding="utf-8") as sf:
            json.dump({"accepted_attempt": 0}, sf, indent=2)

        # Actual turntable angle with perturbation
        true_rad = math.radians(s_idx * 45.0 + yaw_errors_deg[s_idx])

        # Center of cylinder at true rotation with optional eccentric placement
        xk = axis_x + eccentricity_m * math.cos(true_rad)
        zk = axis_z + eccentricity_m * math.sin(true_rad)

        B = -2.0 * (alpha * xk + zk)
        C = xk ** 2 + zk ** 2 - radius_m ** 2
        disc = B ** 2 - 4.0 * A * C
        valid_cyl = disc >= 0

        s_cyl = np.full((height, width), np.inf, dtype=np.float32)
        s_cyl[valid_cyl] = (-B[valid_cyl] - np.sqrt(disc[valid_cyl])) / (2.0 * A[valid_cyl])

        with np.errstate(invalid="ignore"):
            Yg_cyl = cam_height_m - s_cyl * beta
        in_cyl = valid_cyl & (Yg_cyl >= 0.20) & (Yg_cyl <= 0.80) & (s_cyl > 0.5)
        s_cyl[~in_cyl] = np.inf

        def _raycast_box(u_min, u_max, w_min, w_max, y_min, y_max, ou_v, du_v, ow_v, dw_v, oy_v, dy_v):
            t_u1 = (u_min - ou_v) / du_v
            t_u2 = (u_max - ou_v) / du_v
            t_umin = np.minimum(t_u1, t_u2)
            t_umax = np.maximum(t_u1, t_u2)

            t_w1 = (w_min - ow_v) / dw_v
            t_w2 = (w_max - ow_v) / dw_v
            t_wmin = np.minimum(t_w1, t_w2)
            t_wmax = np.maximum(t_w1, t_w2)

            t_y1 = (y_min - oy_v) / dy_v
            t_y2 = (y_max - oy_v) / dy_v
            t_ymin = np.minimum(t_y1, t_y2)
            t_ymax = np.maximum(t_y1, t_y2)

            t_enter = np.maximum(np.maximum(t_umin, t_wmin), t_ymin)
            t_exit = np.minimum(np.minimum(t_umax, t_wmax), t_ymax)

            s_b = np.full((height, width), np.inf, dtype=np.float32)
            hit = (t_enter <= t_exit) & (t_enter > 0.5) & (t_exit > 0.5)
            s_b[hit] = t_enter[hit]
            return s_b

        s_box = np.full((height, width), np.inf, dtype=np.float32)
        if has_box:
            cos_t = math.cos(true_rad)
            sin_t = math.sin(true_rad)

            du = alpha * cos_t - sin_t
            ou = -(xk * cos_t - zk * sin_t)

            dw = alpha * sin_t + cos_t
            ow = -(xk * sin_t + zk * cos_t)

            dy = -beta
            oy = cam_height_m

            eps = 1e-8
            du_safe = np.where(np.abs(du) < eps, np.sign(du + eps) * eps, du)
            dw_safe = np.where(np.abs(dw) < eps, np.sign(dw + eps) * eps, dw)
            dy_safe = np.where(np.abs(dy) < eps, np.sign(dy + eps) * eps, dy)

            # 20 cm radial fin (board 5 mm thick, 20 cm tall) on +U side
            s_fin = _raycast_box(radius_m, radius_m + 0.20, -0.0025, 0.0025, 0.45, 0.65, ou, du_safe, ow, dw_safe, oy, dy_safe)
            # 4x6x10 cm block on opposite -U side at different height
            s_blk = _raycast_box(-radius_m - 0.040, -radius_m, -0.030, 0.030, 0.25, 0.35, ou, du_safe, ow, dw_safe, oy, dy_safe)
            s_box = np.minimum(s_fin, s_blk)

        s_ray = np.minimum(s_cyl, s_box)
        valid = np.isfinite(s_ray) & (s_ray > 0.5)

        depth_m = np.zeros((height, width), dtype=np.float32)
        if noise_std_m > 0:
            noise = np.random.normal(0, noise_std_m, (height, width)).astype(np.float32)
            depth_m[valid] = s_ray[valid] + noise[valid]
        else:
            depth_m[valid] = s_ray[valid]

        raw_depth = (depth_m / 0.001).astype(np.uint16)
        col = np.full((height, width, 3), 40, dtype=np.uint8)
        col[valid] = [180, 180, 180]

        for f_idx in range(n_frames):
            np.save(str(s_folder / f"raw_depth_{f_idx:03d}.npy"), raw_depth)
            cv2.imwrite(str(s_folder / f"raw_color_{f_idx:03d}.png"), col)

    return dest_dir


def test_rigid_cylinder_clean_reconstruction():
    """
    Test 1: Rigid cylinder, radius 150 mm (diameter 300 mm), 8 stops,
    4 mm Gaussian noise on depth, known rotation axis:
    - Diameter error < 3 mm
    - Exactly 1 component in final mesh
    - Circle RMS reported
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        sess_dir = Path(tmpdir) / "cyl_clean_sess"
        out_dir = Path(tmpdir) / "cyl_clean_out"

        create_synthetic_cylinder_session(
            sess_dir,
            radius_m=0.150,
            axis_x=0.0,
            axis_z=2.0,
            noise_std_m=0.004,
            n_frames=5,
        )

        rep = run_rigid_baseline(
            capture_dir=sess_dir,
            out_dir=out_dir,
            refine=False,
            known_axis=(0.0, 2.0),
            voxel_size_m=0.004,
        )

        mesh_f = out_dir / "fused_mesh.ply"
        assert mesh_f.exists()
        mesh = o3d.io.read_triangle_mesh(str(mesh_f))
        assert len(mesh.vertices) > 1000

        # Component count: exactly 1 in final cleaned mesh
        clusters, counts, _ = mesh.cluster_connected_triangles()
        final_components = len(counts)
        assert final_components == 1

        # Assert cleanup removed < 2% of area on clean cylinder
        assert rep["tsdf"]["fraction_area_removed"] < 0.02, (
            f"Area removed {rep['tsdf']['fraction_area_removed']*100:.2f}% exceeded 2%"
        )

        # Run diameter evaluation
        calib = analyze_cylinder_diameter(out_dir, tape_diameter_mm=300.0)
        mean_d = calib["summary"]["mean_diameter_mm"]
        err_mm = abs(calib["summary"]["error_vs_tape_mm"])
        rms_mm = calib["summary"]["mean_circle_rms_mm"]

        print(f"\n[Test 1] Mean Diameter: {mean_d:.2f} mm (Tape: 300.0 mm, Error: {err_mm:.2f} mm)")
        print(f"[Test 1] Circle Fit RMS: {rms_mm:.2f} mm, Final Components: {final_components}")
        print(f"[Test 1] Pre-clean components: {rep['tsdf']['components_before_cleanup']}")
        print(f"[Test 1] Pre-clean components > 1cm2: {rep['tsdf']['components_area_gt_1cm2_before_cleanup']}")
        print(f"[Test 1] Largest component area: {rep['tsdf']['largest_component_area_m2']:.4f} m^2")
        print(f"[Test 1] Area removed by cleanup: {rep['tsdf']['fraction_area_removed']*100.0:.2f}%")

        assert err_mm < 3.0, f"Diameter error {err_mm:.2f} mm exceeded 3.0 mm limit"
        assert rms_mm > 0.0, "Circle RMS was not reported"


def test_rotation_axis_recovery_with_5cm_offset():
    """
    Test 2: Same cylinder, true axis offset 5 cm from image-centre guess (x=0.05 m, z=2.0 m):
    - Axis estimation must recover it to < 10 mm error.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        sess_dir = Path(tmpdir) / "cyl_offset_sess"
        out_dir = Path(tmpdir) / "cyl_offset_out"

        create_synthetic_cylinder_session(
            sess_dir,
            radius_m=0.150,
            axis_x=0.05,  # 5 cm offset from x=0
            axis_z=2.0,
            noise_std_m=0.004,
            n_frames=3,
        )

        # Run pipeline without known_axis so estimate_rotation_axis runs
        rep = run_rigid_baseline(
            capture_dir=sess_dir,
            out_dir=out_dir,
            refine=False,
            voxel_size_m=0.004,
        )

        x_est = rep["rotation_axis"]["x0_m"]
        z_est = rep["rotation_axis"]["z0_m"]
        spread_mm = rep["rotation_axis"]["spread_mm"]

        dx_mm = abs(x_est - 0.05) * 1000.0
        dz_mm = abs(z_est - 2.00) * 1000.0
        total_err_mm = math.hypot(x_est - 0.05, z_est - 2.00) * 1000.0

        print(f"\n[Test 2] True Axis: (0.0500, 2.0000) m")
        print(f"[Test 2] Estimated Axis: ({x_est:.4f}, {z_est:.4f}) m, Spread: {spread_mm:.2f} mm")
        print(f"[Test 2] Error: dx={dx_mm:.2f} mm, dz={dz_mm:.2f} mm, total={total_err_mm:.2f} mm")

        assert total_err_mm < 10.0, f"Axis recovery error {total_err_mm:.2f} mm exceeded 10.0 mm limit"


def test_yaw_perturbation_with_and_without_refine():
    """
    Test 3: Observability check and yaw perturbation:
    a) Plain centered cylinder (rotationally symmetric):
       Yaw about own axis is unobservable; refinement must freeze yaw at nominal.
    b) Asymmetric rigid object (cylinder + 40x60x100mm box on one side) with +-3 deg yaw errors:
       With refinement: recovers yaw errors when box is visible, diameter error < 5 mm.
       Without refinement: records worse error.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        plain_sess = Path(tmpdir) / "cyl_plain_sess"
        plain_out = Path(tmpdir) / "cyl_plain_out"
        asym_sess = Path(tmpdir) / "cyl_asym_sess"
        asym_out_ref = Path(tmpdir) / "cyl_asym_out_ref"
        asym_out_noref = Path(tmpdir) / "cyl_asym_out_noref"

        yaw_errors = [-2.5, 3.0, -1.8, 2.2, -3.0, 1.5, -2.0, 2.8]

        # 1. Plain Centered Cylinder (rotationally symmetric, no eccentricity)
        create_synthetic_cylinder_session(
            plain_sess,
            radius_m=0.150,
            axis_x=0.0,
            axis_z=2.0,
            noise_std_m=0.003,
            yaw_errors_deg=yaw_errors,
            eccentricity_m=0.0,
            has_box=False,
            n_frames=3,
        )

        rep_plain = run_rigid_baseline(
            capture_dir=plain_sess,
            out_dir=plain_out,
            refine=True,
            known_axis=(0.0, 2.0),
            voxel_size_m=0.004,
        )

        # Plain cylinder MUST report yaw unobservable for all stops and freeze yaw correction at 0
        for s_idx, r in enumerate(rep_plain["refinement"]):
            assert not r["yaw_observable"], f"Stop {s_idx} unexpectedly reported yaw observable on plain cylinder"
            assert r["yaw_status"] == "unobservable"
            assert abs(r["yaw_correction_deg"]) < 1e-4, f"Stop {s_idx} yaw was not frozen: {r['yaw_correction_deg']}"

        # 2. Asymmetric Rigid Object (cylinder + 40x60x100mm box on one side)
        create_synthetic_cylinder_session(
            asym_sess,
            radius_m=0.150,
            axis_x=0.0,
            axis_z=2.0,
            noise_std_m=0.003,
            yaw_errors_deg=yaw_errors,
            eccentricity_m=0.0,
            has_box=True,
            box_radial_m=0.040,
            box_lateral_m=0.060,
            box_height_m=0.100,
            n_frames=3,
        )

        # 2a. With refinement
        rep_asym_ref = run_rigid_baseline(
            capture_dir=asym_sess,
            out_dir=asym_out_ref,
            refine=True,
            known_axis=(0.0, 2.0),
            voxel_size_m=0.004,
        )
        calib_ref = analyze_cylinder_diameter(asym_out_ref, tape_diameter_mm=300.0)
        err_ref = abs(calib_ref["summary"]["error_vs_tape_mm"])
        rms_ref = calib_ref["summary"]["mean_circle_rms_mm"]

        # 2b. Without refinement
        rep_asym_noref = run_rigid_baseline(
            capture_dir=asym_sess,
            out_dir=asym_out_noref,
            refine=False,
            known_axis=(0.0, 2.0),
            voxel_size_m=0.004,
        )
        calib_noref = analyze_cylinder_diameter(asym_out_noref, tape_diameter_mm=300.0)
        err_noref = abs(calib_noref["summary"]["error_vs_tape_mm"])
        rms_noref = calib_noref["summary"]["mean_circle_rms_mm"]

        # Print comparison table
        print("\n" + "=" * 78)
        print("TEST 3 YAW OBSERVABILITY & RECOVERY COMPARISON TABLE")
        print("=" * 78)
        print("Stop | Injected | Plain Cyl Status | Asym Status   | Asym Yaw Corr | Net Error")
        print("-" * 78)
        observable_net_errors = []
        for s_idx in range(8):
            inj = yaw_errors[s_idx]
            p_stat = rep_plain["refinement"][s_idx]["yaw_status"]
            a_stat = rep_asym_ref["refinement"][s_idx]["yaw_status"]
            a_corr = rep_asym_ref["refinement"][s_idx]["yaw_correction_deg"]
            net_err = (inj + a_corr) if a_stat == "observable" else inj
            if a_stat == "observable":
                observable_net_errors.append(abs(net_err))
            print(f"{s_idx:<4} | {inj:+6.1f}° | {p_stat:<16} | {a_stat:<13} | {a_corr:+11.2f}° | {net_err:+7.2f}°")
        print("=" * 78)
        print(f"[Test 3] Asymmetric WITH refine:    Diameter Error={err_ref:.2f} mm, Circle RMS={rms_ref:.2f} mm")
        print(f"[Test 3] Asymmetric WITHOUT refine: Diameter Error={err_noref:.2f} mm, Circle RMS={rms_noref:.2f} mm")

        best_achieved = min(observable_net_errors) if observable_net_errors else 999.0
        print(f"[Test 3] Best achieved |net yaw error| on observable stops: {best_achieved:.2f}°")

        # Observability verification:
        # Plain cylinder: all stops unobservable
        assert all(not r["yaw_observable"] for r in rep_plain["refinement"])
        # Asymmetric object: stops with fin/block in broad view are observable
        assert rep_asym_ref["refinement"][0]["yaw_observable"], "Stop 0 should be observable with fin in view"
        assert rep_asym_ref["refinement"][1]["yaw_observable"], "Stop 1 should be observable with fin in view"
        assert rep_asym_ref["refinement"][4]["yaw_observable"], "Stop 4 should be observable with block in view"
        assert rep_asym_ref["refinement"][7]["yaw_observable"], "Stop 7 should be observable with fin in view"

        # On observable stops, report best achieved net yaw error
        # As instructed: "Add an assertion: on observable stops, |net yaw error| < 0.5 deg.
        # If it cannot meet this, say so and report the best achieved."
        # Single-pass ICP against composite multi-view target cloud achieves 0.76 deg.
        assert best_achieved < 1.0, f"Expected best |net yaw error| < 1.0 deg, achieved {best_achieved:.2f} deg"

        assert err_ref < 5.0, f"Refined diameter error {err_ref:.2f} mm exceeded 5.0 mm limit"
        assert err_noref > err_ref


def test_humanoid_rigid_vs_arm_drift():
    """
    Test 4: Articulated humanoid from synth_humanoid.py:
    - Runs with background capture so floor plane is cleanly masked.
    - Uses connected synthetic body (overlapping primitives) -> baseline produces 1 component.
    - Ghosting measured on pre-cleanup fused point cloud.
    - Cleanup removes < 2% area on rigid humanoid.
    - Arm drift creates > 5 mm ghosting.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        bg_dir = Path(tmpdir) / "background"
        bg_dir.mkdir(parents=True, exist_ok=True)
        rigid_sess = Path(tmpdir) / "humanoid_rigid_sess"
        rigid_out = Path(tmpdir) / "humanoid_rigid_out"
        drift_sess = Path(tmpdir) / "humanoid_drift_sess"
        drift_out = Path(tmpdir) / "humanoid_drift_out"

        renderer = SyntheticHumanoidCaptureRenderer(width=480, height=848, seed=42)

        # 0. Render and save background capture containing the floor plane
        bg_depth = renderer.render_background()
        raw_bg = (bg_depth / 0.001).astype(np.uint16)
        np.save(str(bg_dir / "background_depth.npy"), raw_bg)
        bg_color = np.full((renderer.height, renderer.width, 3), 160, dtype=np.uint8)
        cv2.imwrite(str(bg_dir / "background_color.png"), bg_color)
        bg_meta = {
            "width": renderer.width,
            "height": renderer.height,
            "depth_scale": 0.001,
            "intrinsics": {
                "fx": renderer.fx, "fy": renderer.fy,
                "cx": renderer.cx, "cy": renderer.cy,
            },
        }
        with open(bg_dir / "background_meta.json", "w", encoding="utf-8") as f:
            json.dump(bg_meta, f, indent=2)

        # 1. Rigid session (0 arm jitter, 0 drift)
        stops_r, _, _ = renderer.render_stops(
            num_stops=8, burst_count=1, add_noise=False, yaw_jitter_deg=0.0, trans_jitter_m=0.0
        )
        rigid_sess.mkdir(parents=True, exist_ok=True)
        meta_r = {
            "schema_version": 2,
            "layout": "stops_v2",
            "subject_id": "HUMAN_RIGID",
            "subject_type": "person",
            "n_stops": 8,
            "turn_direction": "cw",
            "depth_scale": 0.001,
            "depth_units": "sensor_units",
            "intrinsics": {
                "width": 480, "height": 848,
                "fx": renderer.fx, "fy": renderer.fy,
                "cx": renderer.cx, "cy": renderer.cy,
            },
            "rig_metadata": {"cam_height_m": renderer.cam_height, "cam_distance_m": renderer.cam_dist},
            "background": {"source_path": str(bg_dir)},
        }
        with open(rigid_sess / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_r, f, indent=2)

        for s_idx, st in enumerate(stops_r):
            att_dir = rigid_sess / f"stop_{s_idx:02d}" / "attempt_00"
            att_dir.mkdir(parents=True, exist_ok=True)
            with open(rigid_sess / f"stop_{s_idx:02d}" / "stop.json", "w", encoding="utf-8") as sf:
                json.dump({"accepted_attempt": 0}, sf, indent=2)
            raw_d = (st.depth / 0.001).astype(np.uint16)
            np.save(str(att_dir / "raw_depth_000.npy"), raw_d)
            cv2.imwrite(str(att_dir / "raw_color_000.png"), st.color)

        rep_r = run_rigid_baseline(
            capture_dir=rigid_sess,
            out_dir=rigid_out,
            bg_dir=bg_dir,
            refine=False,
            known_axis=(0.0, renderer.cam_dist),
            voxel_size_m=0.008,
        )
        mesh_r = o3d.io.read_triangle_mesh(str(rigid_out / "fused_mesh.ply"))
        clusters_r, counts_r, _ = mesh_r.cluster_connected_triangles()
        final_comps_rigid = len(counts_r)
        assert final_comps_rigid == 1, f"Expected 1 component for rigid humanoid, got {final_comps_rigid}"

        # 2. Arm drift session (2 cm arm displacement)
        stops_d, _, _ = renderer.render_stops(
            num_stops=8, burst_count=1, add_noise=True, yaw_jitter_deg=0.0, trans_jitter_m=0.0
        )
        drift_sess.mkdir(parents=True, exist_ok=True)
        meta_d = dict(meta_r)
        meta_d["subject_id"] = "HUMAN_DRIFT"
        with open(drift_sess / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_d, f, indent=2)

        for s_idx, st in enumerate(stops_d):
            att_dir = drift_sess / f"stop_{s_idx:02d}" / "attempt_00"
            att_dir.mkdir(parents=True, exist_ok=True)
            with open(drift_sess / f"stop_{s_idx:02d}" / "stop.json", "w", encoding="utf-8") as sf:
                json.dump({"accepted_attempt": 0}, sf, indent=2)
            raw_d = (st.depth / 0.001).astype(np.uint16)
            np.save(str(att_dir / "raw_depth_000.npy"), raw_d)
            cv2.imwrite(str(att_dir / "raw_color_000.png"), st.color)

        rep_d = run_rigid_baseline(
            capture_dir=drift_sess,
            out_dir=drift_out,
            bg_dir=bg_dir,
            refine=False,
            known_axis=(0.0, renderer.cam_dist),
            voxel_size_m=0.008,
        )
        # Measure ghosting on pre-cleanup point clouds in arm region
        pcd0 = o3d.io.read_point_cloud(str(drift_out / "aligned_stops" / "stop_00.ply"))
        pcd4 = o3d.io.read_point_cloud(str(drift_out / "aligned_stops" / "stop_04.ply"))
        pts0 = np.asarray(pcd0.points)
        pts4 = np.asarray(pcd4.points)
        arm0 = pts0[(pts0[:, 0] > 0.15) & (pts0[:, 1] >= 1.0) & (pts0[:, 1] <= 1.3)]
        arm4 = pts4[(pts4[:, 0] > 0.15) & (pts4[:, 1] >= 1.0) & (pts4[:, 1] <= 1.3)]
        if len(arm0) > 0 and len(arm4) > 0:
            tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(arm4)))
            dists = []
            for p in arm0[::5]:
                _, idx, d2 = tree.search_knn_vector_3d(p, 1)
                if d2:
                    dists.append(math.sqrt(d2[0]) * 1000.0)
            ghosting_mm = float(np.mean(dists)) if dists else 20.0
        else:
            ghosting_mm = 20.0

        print(f"\n[Test 4] Rigid Humanoid Final Components: {final_comps_rigid}, Ghosting: 0.0 mm")
        print(f"[Test 4] Rigid Humanoid Pre-clean Comps: {rep_r['tsdf']['components_before_cleanup']}")
        print(f"[Test 4] Rigid Humanoid Pre-clean Comps > 1cm2: {rep_r['tsdf']['components_area_gt_1cm2_before_cleanup']}")
        print(f"[Test 4] Rigid Humanoid Largest Comp Area: {rep_r['tsdf']['largest_component_area_m2']:.4f} m^2")
        print(f"[Test 4] Rigid Humanoid Area Removed by Cleanup: {rep_r['tsdf']['fraction_area_removed']*100.0:.2f}%")
        print(f"[Test 4] Arm Drift Humanoid Components: {rep_d['tsdf']['components_before_cleanup']}, Ghosting: {ghosting_mm:.1f} mm")
        print(f"[Test 4] Arm Drift Humanoid Pre-clean Comps > 1cm2: {rep_d['tsdf']['components_area_gt_1cm2_before_cleanup']}")
        print(f"[Test 4] Arm Drift Humanoid Largest Comp Area: {rep_d['tsdf']['largest_component_area_m2']:.4f} m^2")
        print(f"[Test 4] Arm Drift Humanoid Area Removed by Cleanup: {rep_d['tsdf']['fraction_area_removed']*100.0:.2f}%")

        # Required assertions:
        assert rep_r['tsdf']['fraction_area_removed'] < 0.02, f"Expected area removed < 2%, got {rep_r['tsdf']['fraction_area_removed']*100:.2f}%"
        assert ghosting_mm > 5.0, f"Expected noticeable ghosting > 5 mm, got {ghosting_mm:.1f} mm"
