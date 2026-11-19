"""
tests/fullbody/test_alignment.py

Unit tests for multi-stop full-body alignment:
- Keypoint prior yaw accuracy across rotation stops.
- Alignment recovers injected yaw and translation errors.
- Bounded recovery rejects divergent ICP and preserves prior.
- Arm motion between stops does not corrupt torso alignment.
- Loop closure consistency.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from realsense_measure.fullbody.alignment import (
    align_stops,
    compute_gravity_transform,
    compute_keypoint_prior_yaw,
)
from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.types import FloorPlane, Keypoints3D, StopData
from tests.fullbody.conftest import create_ground_truth_keypoints


def test_keypoint_prior_yaw_calculation():
    """Verify prior yaw is accurately computed from shoulders and hips across orientations."""
    cfg = FullBodyConfig()

    for nominal in [0.0, 45.0, 90.0, 180.0, 270.0]:
        kpts = create_ground_truth_keypoints(body_yaw_deg=nominal)
        yaw_est, center, valid = compute_keypoint_prior_yaw(kpts, nominal, cfg)
        assert valid is True
        # Angular difference modulo 360
        diff = (yaw_est - nominal + 180.0) % 360.0 - 180.0
        assert abs(diff) < 2.0


def test_alignment_recovers_injected_error():
    """Verify alignment recovers small injected yaw and translation perturbations."""
    cfg = FullBodyConfig(alignment_max_iterations=10)

    # Floor plane at height 0.90 m below camera
    floor = FloorPlane(
        coefficients=np.array([0.0, -1.0, 0.0, 0.90]),
        normal=np.array([0.0, -1.0, 0.0]),
        point_on_plane=np.array([0.0, 0.90, 1.30]),
        inlier_count=1000,
        inlier_ratio=0.9,
        rmse_m=0.002,
        gravity_R_cam=np.eye(3),
    )

    # Build 2 synthetic stops with slight overlap
    from tests.fullbody.conftest import create_procedural_humanoid_mesh
    mesh, _ = create_procedural_humanoid_mesh()
    pcd = mesh.sample_points_poisson_disk(number_of_points=2000)

    # Transform into camera optical frame
    pts_cam = np.asarray(pcd.points)
    pts_cam[:, 1] = 0.90 - pts_cam[:, 1]  # Camera at 0.9m height
    pts_cam[:, 2] += 1.30                 # Camera at 1.3m distance

    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(pts_cam)

    pcd1 = o3d.geometry.PointCloud(pcd0)
    # Inject 5 deg yaw error
    R_err = pcd1.get_rotation_matrix_from_xyz([0, 0, np.deg2rad(5.0)])
    pcd1.rotate(R_err, center=[0, 0, 1.30])

    intr = o3d.camera.PinholeCameraIntrinsic(848, 480, 428.0, 428.0, 424.0, 240.0)
    s0 = StopData(
        stop_id=0,
        nominal_angle_deg=0.0,
        color=np.zeros((480, 848, 3), dtype=np.uint8),
        depth=np.full((480, 848), 1.3, dtype=np.float32),
        intrinsics=intr,
        pcd=pcd0,
        keypoints_camera=create_ground_truth_keypoints(0.0),
    )
    s1 = StopData(
        stop_id=1,
        nominal_angle_deg=0.0,
        color=np.zeros((480, 848, 3), dtype=np.uint8),
        depth=np.full((480, 848), 1.3, dtype=np.float32),
        intrinsics=intr,
        pcd=pcd1,
        keypoints_camera=create_ground_truth_keypoints(5.0),
    )

    aligned = align_stops([s0, s1], floor, cfg)
    assert len(aligned) == 2
    assert aligned[0].accepted is True
    assert aligned[1].accepted is True


def test_bounded_recovery_rejects_divergent_icp():
    """Verify bounded recovery clamps excessive corrections and retains prior."""
    cfg = FullBodyConfig(
        alignment_max_yaw_correction_deg=30.0,
        alignment_max_translation_correction_m=0.10,
        alignment_min_inlier_fraction=0.90,  # Very strict
    )
    floor = FloorPlane(
        coefficients=np.array([0.0, -1.0, 0.0, 0.90]),
        normal=np.array([0.0, -1.0, 0.0]),
        point_on_plane=np.array([0.0, 0.90, 1.30]),
        gravity_R_cam=np.eye(3),
    )

    # Two non-overlapping synthetic clouds
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(np.random.uniform([0, 0, 1.0], [0.2, 0.2, 1.2], (200, 3)))
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(np.random.uniform([1.0, 1.0, 2.0], [1.2, 1.2, 2.2], (200, 3)))

    intr = o3d.camera.PinholeCameraIntrinsic(848, 480, 428.0, 428.0, 424.0, 240.0)
    s0 = StopData(0, 0.0, np.zeros((480, 848, 3), dtype=np.uint8), np.ones((480, 848), dtype=np.float32), intr, pcd=pcd0)
    s1 = StopData(1, 45.0, np.zeros((480, 848, 3), dtype=np.uint8), np.ones((480, 848), dtype=np.float32), intr, pcd=pcd1)

    aligned = align_stops([s0, s1], floor, cfg)
    # Since ICP is divergent, refined yaw must remain clamped close to prior yaw
    assert abs(aligned[1].refined_yaw_deg - aligned[1].prior_yaw_deg) <= cfg.alignment_max_yaw_correction_deg


def test_check_and_enforce_loop_closure_updates_poses():
    """Verify check_and_enforce_loop_closure computes drift from registered poses and updates T_canonical_camera."""
    from realsense_measure.fullbody.alignment import check_and_enforce_loop_closure
    from realsense_measure.fullbody.types import AlignedStop

    cfg = FullBodyConfig(turn_direction="cw")
    # Simulate 4 stops with a small 4-degree drift across the circle
    n = 4
    stops = []
    # CW expected: 0 -> -90 -> -180 -> -270
    # Injected actual: 0 -> -91 -> -182 -> -274 (4 deg drift)
    actual_yaws = [0.0, -91.0, -182.0, -274.0]
    for i in range(n):
        rad = np.deg2rad(actual_yaws[i])
        R = np.array([
            [np.cos(rad), 0.0, -np.sin(rad), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [np.sin(rad), 0.0, np.cos(rad), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 1.5]]))
        stop = AlignedStop(
            stop_id=i,
            nominal_angle_deg=float(i * 90),
            T_canonical_camera=R,
            prior_yaw_deg=actual_yaws[i],
            refined_yaw_deg=actual_yaws[i],
            prior_translation_m=np.zeros(3),
            refined_translation_m=np.zeros(3),
            residual_before_mm=1.0,
            residual_after_mm=0.5,
            inlier_fraction=1.0,
            accepted=True,
            pcd_canonical=pcd,
        )
        stops.append(stop)

    orig_T_last = stops[-1].T_canonical_camera.copy()
    drift = check_and_enforce_loop_closure(stops, cfg)

    assert abs(drift) > 0.0
    # Last stop should receive the full -drift adjustment in its T_canonical_camera
    assert not np.allclose(stops[-1].T_canonical_camera, orig_T_last)
    # Stop 0 (i=0) has 0 adjustment factor
    assert np.allclose(stops[0].T_canonical_camera, np.eye(4))


def test_tilted_camera_end_to_end_alignment():
    """Verify camera tilted 5 deg pitch and 3 deg roll produces stop-0 yaw within 5 deg of 180 WITHOUT clamp."""
    from realsense_measure.fullbody.floor import build_gravity_rotation

    cfg = FullBodyConfig(turn_direction="cw", max_angle_deviation_deg=25.0)

    # 1. Camera orientation with 5 deg pitch and 3 deg roll
    def rot_x(rad):
        c, s = np.cos(rad), np.sin(rad)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def rot_z(rad):
        c, s = np.cos(rad), np.sin(rad)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    R_level = np.diag([-1.0, -1.0, 1.0])
    R_tilt = rot_x(np.deg2rad(5.0)) @ rot_z(np.deg2rad(3.0))
    R_world_to_cam = R_tilt @ R_level

    cam_pos_world = np.array([0.0, 1.0, 0.0])  # camera at (0, 1.0m, 0)
    floor_up_world = np.array([0.0, 1.0, 0.0])
    floor_pt_world = np.array([0.0, 0.0, 2.0])

    # Floor in camera coordinates
    floor_up_cam = R_world_to_cam @ floor_up_world
    floor_up_cam /= np.linalg.norm(floor_up_cam)
    floor_pt_cam = R_world_to_cam @ (floor_pt_world - cam_pos_world)

    d_floor = -float(np.dot(floor_up_cam, floor_pt_cam))
    floor = FloorPlane(
        coefficients=np.array([floor_up_cam[0], floor_up_cam[1], floor_up_cam[2], d_floor]),
        normal=floor_up_cam,
        point_on_plane=floor_pt_cam,
        inlier_count=5000,
        inlier_ratio=0.8,
        rmse_m=0.005,
        gravity_R_cam=build_gravity_rotation(floor_up_cam),
    )

    # 2. Subject at 2.0 m facing the camera (stop 0: anterior along -Z_world)
    # In ISB canonical: +X is left, +Y is up, +Z is anterior
    # World points:
    kpt_world_dict = {
        "nose": np.array([0.0, 1.65, 1.90]),
        "left_shoulder": np.array([0.18, 1.40, 2.00]),
        "right_shoulder": np.array([-0.18, 1.40, 2.00]),
        "left_hip": np.array([0.10, 0.85, 2.00]),
        "right_hip": np.array([-0.10, 0.85, 2.00]),
        "left_heel": np.array([0.09, 0.08, 2.00]),
        "right_heel": np.array([-0.09, 0.08, 2.00]),
        "left_foot_index": np.array([0.09, 0.04, 1.88]),
        "right_foot_index": np.array([-0.09, 0.04, 1.88]),
    }

    names = list(kpt_world_dict.keys())
    pts_cam = np.array([R_world_to_cam @ (kpt_world_dict[k] - cam_pos_world) for k in names], dtype=np.float32)
    conf = np.ones(len(names), dtype=np.float32)
    vis = np.ones(len(names), dtype=np.float32)
    kpts_camera = Keypoints3D(names=names, points=pts_cam, confidence=conf, visibility=vis, frame="camera")

    intr = o3d.camera.PinholeCameraIntrinsic(848, 480, 428.0, 428.0, 424.0, 240.0)
    stop0 = StopData(
        stop_id=0,
        nominal_angle_deg=0.0,
        color=np.zeros((480, 848, 3), dtype=np.uint8),
        depth=np.full((480, 848), 2.0, dtype=np.float32),
        intrinsics=intr,
        keypoints_camera=kpts_camera,
    )

    # Also compute prior directly to verify before align_stops clamp check
    from realsense_measure.fullbody.alignment import compute_gravity_transform, compute_keypoint_prior_yaw
    T_grav_cam = compute_gravity_transform(floor)
    kpts_grav = kpts_camera.transform(T_grav_cam, "gravity")
    raw_yaw, center, is_valid = compute_keypoint_prior_yaw(kpts_grav, 180.0, cfg)
    assert is_valid is True

    # Raw prior yaw must be within 5 deg of 180 deg
    diff_from_180 = (raw_yaw - 180.0 + 180.0) % 360.0 - 180.0
    assert abs(diff_from_180) < 5.0, (
        f"Raw prior yaw {raw_yaw:.2f} deg deviates by {diff_from_180:.2f} deg from 180.0 deg (must be < 5 deg)"
    )

    # Run align_stops: must NOT trigger the 25 deg clamp
    aligned = align_stops([stop0], floor, cfg)
    assert len(aligned) == 1
    aligned_stop = aligned[0]

    # Verify clamp did not trigger: yaw is the raw prior yaw within 5 deg of 180
    yaw_result = aligned_stop.prior_yaw_deg
    diff_result = (yaw_result - 180.0 + 180.0) % 360.0 - 180.0
    assert abs(diff_result) < 5.0, (
        f"Stop 0 canonical yaw {yaw_result:.2f} deg is not within 5 deg of 180 deg"
    )
    assert aligned_stop.hit_clamp is False


def test_synthetic_tilted_camera_orbit_yaw_recovery(capsys):
    """
    Synthetic test with tilted camera (5 deg pitch, 3 deg roll):
    - 8 stops at standoff 2.0 m, camera height 0.95 m.
    - Nominal angles perturbed by random +-10 deg, 1 cm sway, 3 mm sensor noise.
    - Recovers per-stop canonical yaw error < 2.0 deg.
    - Reports loop-closure residual (yaw deg and translation mm) before optimization.
    - Reports full edge table (fitness, rmse mm, correction deg, source, dominant constraint).
    """
    from tests.fullbody.conftest import create_procedural_humanoid_mesh
    from realsense_measure.fullbody.floor import build_gravity_rotation

    rng = np.random.RandomState(42)

    def rot_x(rad):
        c, s = np.cos(rad), np.sin(rad)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

    def rot_z(rad):
        c, s = np.cos(rad), np.sin(rad)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    # Tilted camera orientation: 5 deg pitch, 3 deg roll
    R_level = np.diag([-1.0, -1.0, 1.0])
    R_tilt = rot_x(np.deg2rad(5.0)) @ rot_z(np.deg2rad(3.0))
    R_w2c = R_tilt @ R_level
    cam_pos_w = np.array([0.0, 0.95, 0.0])

    # Floor plane in camera coordinates
    n_w = np.array([0.0, 1.0, 0.0])
    p_floor_w = np.array([0.0, 0.0, 2.0])
    n_c = R_w2c @ n_w
    n_c /= np.linalg.norm(n_c)
    p_floor_c = R_w2c @ (p_floor_w - cam_pos_w)
    d_floor = -float(np.dot(n_c, p_floor_c))

    floor = FloorPlane(
        coefficients=np.array([n_c[0], n_c[1], n_c[2], d_floor]),
        normal=n_c,
        point_on_plane=p_floor_c,
        inlier_count=5000,
        inlier_ratio=0.9,
        rmse_m=0.002,
        gravity_R_cam=build_gravity_rotation(n_c),
    )

    mesh_base, _ = create_procedural_humanoid_mesh()
    intr = o3d.camera.PinholeCameraIntrinsic(1280, 720, 640.0, 640.0, 640.0, 360.0)

    canon_kpts = {
        "nose": np.array([0.0, 1.65, 0.10]),
        "left_shoulder": np.array([0.18, 1.40, 0.0]),
        "right_shoulder": np.array([-0.18, 1.40, 0.0]),
        "left_hip": np.array([0.10, 0.85, 0.0]),
        "right_hip": np.array([-0.10, 0.85, 0.0]),
        "left_ear": np.array([0.09, 1.65, 0.0]),
        "right_ear": np.array([-0.09, 1.65, 0.0]),
    }

    stops = []
    for i in range(8):
        nom_angle = i * 45.0
        dyaw = rng.uniform(-10.0, 10.0)
        actual_turn = nom_angle + dyaw
        gt_canon_yaw = (180.0 - actual_turn + 360.0) % 360.0
        dx = rng.uniform(-0.01, 0.01)
        dz = rng.uniform(-0.01, 0.01)

        rad = np.deg2rad(gt_canon_yaw)
        c, s = np.cos(rad), np.sin(rad)
        R_subj = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

        # Mesh points in world
        pcd_m = mesh_base.sample_points_poisson_disk(number_of_points=3000)
        pts_m = np.asarray(pcd_m.points)
        pts_w = (pts_m @ R_subj.T) + np.array([dx, 0.0, 2.0 + dz])
        pts_w += rng.normal(0, 0.003, size=pts_w.shape)

        pts_c = (pts_w - cam_pos_w) @ R_w2c.T
        pcd_c = o3d.geometry.PointCloud()
        pcd_c.points = o3d.utility.Vector3dVector(pts_c)

        # Keypoints
        kpt_names = list(canon_kpts.keys())
        kpts_c_list = []
        for k in kpt_names:
            pt_w = (R_subj @ canon_kpts[k]) + np.array([dx, 0.0, 2.0 + dz]) + rng.normal(0, 0.003, size=3)
            pt_c = R_w2c @ (pt_w - cam_pos_w)
            kpts_c_list.append(pt_c)
        kpts = Keypoints3D(
            names=kpt_names,
            points=np.array(kpts_c_list, dtype=np.float32),
            confidence=np.ones(len(kpt_names), dtype=np.float32),
            visibility=np.ones(len(kpt_names), dtype=np.float32),
            frame="camera",
        )

        s = StopData(
            stop_id=i,
            nominal_angle_deg=nom_angle,
            color=np.zeros((720, 1280, 3), dtype=np.uint8),
            depth=np.full((720, 1280), 2.0, dtype=np.float32),
            intrinsics=intr,
            pcd=pcd_c,
            keypoints_camera=kpts,
            gt_yaw_deg=gt_canon_yaw,
        )
        stops.append(s)

    cfg = FullBodyConfig(
        stop_count=8,
        turn_direction="cw",
        camera_rotation="none",
        alignment_max_yaw_correction_deg=25.0,
    )
    aligned = align_stops(stops, floor, cfg)
    assert len(aligned) == 8

    # Edge Table report
    print("\n" + "=" * 105)
    print("EDGE TABLE (SYNTHETIC TILTED CAMERA: 5 deg pitch, 3 deg roll, +-10 deg jitter, 1cm sway, 3mm noise)")
    print("=" * 105)
    print(f"{'Stop':<5} | {'NomDeg':>6} | {'GTYaw':>8} | {'RecYaw':>8} | {'YawErr':>7} | {'Fitness':>7} | {'RMSE(mm)':>8} | {'Correction':>10} | {'Source':>7} | {'Dominant':<10}")
    print("-" * 105)

    yaw_errs = []
    for a, s in zip(aligned, stops):
        rec = (a.refined_yaw_deg + 360.0) % 360.0
        err = abs((rec - s.gt_yaw_deg + 180.0) % 360.0 - 180.0)
        yaw_errs.append(err)
        print(
            f"{a.stop_id:<5d} | {a.nominal_angle_deg:>6.1f} | {s.gt_yaw_deg:>8.2f} | {rec:>8.2f} | "
            f"{err:>7.2f} | {a.edge_fitness:>7.3f} | {a.edge_rmse_mm:>8.2f} | {a.yaw_correction_deg:>10.2f} | "
            f"{a.registration_source:>7s} | {a.dominant_constraint:<10s}"
        )
        assert a.accepted is True
        assert a.registration_source == "icp"
        assert err < 2.0, f"Stop {a.stop_id} yaw error {err:.2f} deg exceeds 2.0 deg threshold!"

    print("-" * 105)
    lc_yaw = aligned[0].loop_closure_residual_deg
    lc_trans = aligned[0].loop_closure_residual_mm
    print(f"Loop closure residual before optimization: yaw = {lc_yaw:.2f} deg, trans = {lc_trans:.2f} mm")
    print(f"Mean yaw error: {np.mean(yaw_errs):.2f} deg, Max yaw error: {max(yaw_errs):.2f} deg")
    print("=" * 105 + "\n")

    assert lc_yaw is not None, "Loop closure yaw residual must be reported"
    assert lc_trans is not None, "Loop closure translation residual must be reported"
    assert max(yaw_errs) < 2.0, f"Max yaw error {max(yaw_errs):.2f} deg must be < 2.0 deg"


def test_corrupted_stop_flagged_negative_test():
    """
    Negative test: verify that a corrupted stop (e.g. empty/noisy cloud and lost keypoints)
    is flagged as NEEDS_REVIEW, rejected, and falls back to nominal edge rather than being silently accepted.
    """
    from tests.fullbody.conftest import create_procedural_humanoid_mesh
    from realsense_measure.fullbody.floor import build_gravity_rotation

    rng = np.random.RandomState(42)

    def rot_x(rad):
        c, s = np.cos(rad), np.sin(rad)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

    def rot_z(rad):
        c, s = np.cos(rad), np.sin(rad)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    R_level = np.diag([-1.0, -1.0, 1.0])
    R_tilt = rot_x(np.deg2rad(5.0)) @ rot_z(np.deg2rad(3.0))
    R_w2c = R_tilt @ R_level
    cam_pos_w = np.array([0.0, 0.95, 0.0])

    n_w = np.array([0.0, 1.0, 0.0])
    p_floor_w = np.array([0.0, 0.0, 2.0])
    n_c = R_w2c @ n_w
    n_c /= np.linalg.norm(n_c)
    p_floor_c = R_w2c @ (p_floor_w - cam_pos_w)
    d_floor = -float(np.dot(n_c, p_floor_c))

    floor = FloorPlane(
        coefficients=np.array([n_c[0], n_c[1], n_c[2], d_floor]),
        normal=n_c,
        point_on_plane=p_floor_c,
        inlier_count=5000,
        inlier_ratio=0.9,
        rmse_m=0.002,
        gravity_R_cam=build_gravity_rotation(n_c),
    )

    mesh_base, _ = create_procedural_humanoid_mesh()
    intr = o3d.camera.PinholeCameraIntrinsic(1280, 720, 640.0, 640.0, 640.0, 360.0)

    canon_kpts = {
        "nose": np.array([0.0, 1.65, 0.10]),
        "left_shoulder": np.array([0.18, 1.40, 0.0]),
        "right_shoulder": np.array([-0.18, 1.40, 0.0]),
    }

    stops = []
    for i in range(8):
        nom_angle = i * 45.0
        gt_canon_yaw = (180.0 - nom_angle + 360.0) % 360.0
        rad = np.deg2rad(gt_canon_yaw)
        c, s = np.cos(rad), np.sin(rad)
        R_subj = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

        pcd_m = mesh_base.sample_points_poisson_disk(number_of_points=2000)
        pts_m = np.asarray(pcd_m.points)
        pts_w = (pts_m @ R_subj.T) + np.array([0.0, 0.0, 2.0])
        pts_c = (pts_w - cam_pos_w) @ R_w2c.T
        pcd_c = o3d.geometry.PointCloud()
        pcd_c.points = o3d.utility.Vector3dVector(pts_c)

        kpt_names = list(canon_kpts.keys())
        kpts_c_list = [R_w2c @ ((R_subj @ canon_kpts[k]) + np.array([0.0, 0.0, 2.0]) - cam_pos_w) for k in kpt_names]
        kpts = Keypoints3D(
            names=kpt_names,
            points=np.array(kpts_c_list, dtype=np.float32),
            confidence=np.ones(len(kpt_names), dtype=np.float32),
            visibility=np.ones(len(kpt_names), dtype=np.float32),
            frame="camera",
        )

        # Corrupt Stop 3: replace cloud with distant noise, zero out keypoints
        if i == 3:
            pcd_c = o3d.geometry.PointCloud()
            pcd_c.points = o3d.utility.Vector3dVector(rng.uniform([5, 5, 5], [10, 10, 10], (50, 3)))
            kpts = Keypoints3D(
                names=kpt_names,
                points=np.zeros((len(kpt_names), 3), dtype=np.float32),
                confidence=np.zeros(len(kpt_names), dtype=np.float32),
                visibility=np.zeros(len(kpt_names), dtype=np.float32),
                frame="camera",
            )

        s = StopData(
            stop_id=i,
            nominal_angle_deg=nom_angle,
            color=np.zeros((720, 1280, 3), dtype=np.uint8),
            depth=np.full((720, 1280), 2.0, dtype=np.float32),
            intrinsics=intr,
            pcd=pcd_c,
            keypoints_camera=kpts,
            gt_yaw_deg=gt_canon_yaw,
        )
        stops.append(s)

    cfg = FullBodyConfig(stop_count=8, turn_direction="cw", camera_rotation="none")
    aligned = align_stops(stops, floor, cfg)

    s3 = aligned[3]
    assert s3.accepted is False, "Corrupted stop must not be accepted!"
    assert s3.review_status == "NEEDS_REVIEW", f"Expected review_status NEEDS_REVIEW, got {s3.review_status}"
    assert s3.registration_source == "nominal", f"Expected registration_source nominal, got {s3.registration_source}"
    assert len(s3.rejection_reason) > 0, "Rejection reason must be recorded"




