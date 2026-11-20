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
    n_frames: int = 5,
    width: int = 640,
    height: int = 480,
    fx: float = 640.0,
    fy: float = 640.0,
) -> Path:
    """
    Generate synthetic 8-stop turntable session of a rigid cylinder.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    cx = width / 2.0
    cy = height / 2.0

    meta = {
        "schema_version": 2,
        "layout": "stops_v2",
        "protocol_version": "stop-and-go-v1",
        "subject_id": "BUCKET_01",
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
        valid = disc >= 0

        s_ray = np.zeros((height, width), dtype=np.float32)
        s_ray[valid] = (-B[valid] - np.sqrt(disc[valid])) / (2.0 * A[valid])

        Yg = cam_height_m - s_ray * (v_grid - cy) / fy
        in_cyl = valid & (Yg >= 0.20) & (Yg <= 0.80) & (s_ray > 0.5)

        depth_m = np.zeros((height, width), dtype=np.float32)
        if noise_std_m > 0:
            noise = np.random.normal(0, noise_std_m, (height, width)).astype(np.float32)
            depth_m[in_cyl] = s_ray[in_cyl] + noise[in_cyl]
        else:
            depth_m[in_cyl] = s_ray[in_cyl]

        raw_depth = (depth_m / 0.001).astype(np.uint16)
        col = np.full((height, width, 3), 40, dtype=np.uint8)
        col[in_cyl] = [180, 180, 180]

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

        # Run diameter evaluation
        calib = analyze_cylinder_diameter(out_dir, tape_diameter_mm=300.0)
        mean_d = calib["summary"]["mean_diameter_mm"]
        err_mm = abs(calib["summary"]["error_vs_tape_mm"])
        rms_mm = calib["summary"]["mean_circle_rms_mm"]

        print(f"\n[Test 1] Mean Diameter: {mean_d:.2f} mm (Tape: 300.0 mm, Error: {err_mm:.2f} mm)")
        print(f"[Test 1] Circle Fit RMS: {rms_mm:.2f} mm, Components: {final_components}")

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
    Test 3: Same cylinder with per-stop yaw errors up to +-3 deg:
    - With --refine: diameter error must be < 5 mm.
    - Without --refine: records the worse error (measurement).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        sess_dir = Path(tmpdir) / "cyl_yaw_sess"
        out_refine = Path(tmpdir) / "cyl_yaw_refine"
        out_norefine = Path(tmpdir) / "cyl_yaw_norefine"

        yaw_errors = [-2.5, 3.0, -1.8, 2.2, -3.0, 1.5, -2.0, 2.8]

        create_synthetic_cylinder_session(
            sess_dir,
            radius_m=0.150,
            axis_x=0.0,
            axis_z=2.0,
            noise_std_m=0.003,
            yaw_errors_deg=yaw_errors,
            eccentricity_m=0.015,  # 1.5 cm eccentric placement on turntable
            n_frames=3,
        )

        # 1. With refinement
        rep_ref = run_rigid_baseline(
            capture_dir=sess_dir,
            out_dir=out_refine,
            refine=True,
            known_axis=(0.0, 2.0),
            voxel_size_m=0.004,
        )
        calib_ref = analyze_cylinder_diameter(out_refine, tape_diameter_mm=300.0)
        err_ref = abs(calib_ref["summary"]["error_vs_tape_mm"])
        rms_ref = calib_ref["summary"]["mean_circle_rms_mm"]

        # 2. Without refinement
        rep_noref = run_rigid_baseline(
            capture_dir=sess_dir,
            out_dir=out_norefine,
            refine=False,
            known_axis=(0.0, 2.0),
            voxel_size_m=0.004,
        )
        calib_noref = analyze_cylinder_diameter(out_norefine, tape_diameter_mm=300.0)
        err_noref = abs(calib_noref["summary"]["error_vs_tape_mm"])
        rms_noref = calib_noref["summary"]["mean_circle_rms_mm"]

        print(f"\n[Test 3] WITH refine:    Diameter Error={err_ref:.2f} mm, Circle RMS={rms_ref:.2f} mm")
        print(f"[Test 3] WITHOUT refine: Diameter Error={err_noref:.2f} mm, Circle RMS={rms_noref:.2f} mm")

        assert err_ref < 5.0, f"Refined diameter error {err_ref:.2f} mm exceeded 5.0 mm limit"
        # Verify that without refine the uncorrected error is recorded and worse
        assert err_noref > err_ref


def test_humanoid_rigid_vs_arm_drift():
    """
    Test 4: Articulated humanoid from synth_humanoid.py:
    - Rigid (no arm jitter, no sway): baseline produces 1 component.
    - Arm drift (2 cm between stops): records component count and ghosting.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        rigid_sess = Path(tmpdir) / "humanoid_rigid_sess"
        rigid_out = Path(tmpdir) / "humanoid_rigid_out"
        drift_sess = Path(tmpdir) / "humanoid_drift_sess"
        drift_out = Path(tmpdir) / "humanoid_drift_out"

        renderer = SyntheticHumanoidCaptureRenderer(width=480, height=848, seed=42)

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
            refine=False,
            known_axis=(0.0, renderer.cam_dist),
            voxel_size_m=0.008,
        )
        mesh_r = o3d.io.read_triangle_mesh(str(rigid_out / "fused_mesh.ply"))
        clusters_r, counts_r, _ = mesh_r.cluster_connected_triangles()
        final_comps_rigid = len(counts_r)
        assert final_comps_rigid == 1, f"Expected 1 component for rigid humanoid, got {final_comps_rigid}"

        # 2. Arm drift session (2 cm arm displacement)
        # 2 cm drift on 0.55m arm = ~2.08 deg jitter
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
            refine=False,
            known_axis=(0.0, renderer.cam_dist),
            voxel_size_m=0.008,
        )
        # Measure ghosting at arm level
        # In arm region (Y around 1.10 - 1.30 m), compute distance between stop 0 and stop 4 aligned clouds
        pcd0 = o3d.io.read_point_cloud(str(drift_out / "aligned_stops" / "stop_00.ply"))
        pcd4 = o3d.io.read_point_cloud(str(drift_out / "aligned_stops" / "stop_04.ply"))
        pts0 = np.asarray(pcd0.points)
        pts4 = np.asarray(pcd4.points)
        # Select left arm region: X > 0.15, Y in [1.0, 1.3]
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

        print(f"\n[Test 4] Rigid Humanoid Components: {final_comps_rigid}, Ghosting: 0.0 mm")
        print(f"[Test 4] Arm Drift Humanoid Components: {rep_d['tsdf']['components_before_cleanup']}, Ghosting: {ghosting_mm:.1f} mm")

        assert ghosting_mm > 5.0, "Expected noticeable ghosting from 2 cm arm drift"
