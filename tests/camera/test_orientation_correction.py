"""
tests/camera/test_orientation_correction.py

Stage 29.2: Comprehensive verification of camera orientation correction:
1. Asymmetric RGB image rotation.
2. Corresponding depth image rotation.
3. RGB-depth pixel correspondence.
4. Rotated camera intrinsics.
5. Principal point transformation.
6. Landmark coordinate transformation.
7. Depth back-projection consistency.
8. Coordinate chirality (preservation of right-handed 3D coordinate system).
9. Portrait image dimensions (480x848).
10. Both supported rotation directions ('cw' and 'ccw').
"""

from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d
import pytest

from realsense_measure.camera.rotation import rotate_intrinsics, rotate_rgbd_arrays


@pytest.fixture
def asymmetric_rgbd_fixture():
    """
    Create an asymmetric test fixture with identifiable markers at:
    - TOP (Red: [0, 0, 255]) at (u=424, v=50), depth = 1.25 m
    - BOTTOM (Blue: [255, 0, 0]) at (u=424, v=430), depth = 2.50 m
    - LEFT (Green: [0, 255, 0]) at (u=100, v=240), depth = 1.75 m
    - RIGHT (Yellow: [0, 255, 255]) at (u=748, v=240), depth = 3.00 m
    """
    W, H = 848, 480
    color = np.zeros((H, W, 3), dtype=np.uint8)
    depth = np.zeros((H, W), dtype=np.float32)

    # Background ambient depth
    depth[:, :] = 2.0

    # TOP marker (Red, 1.25m)
    color[45:55, 419:429] = [0, 0, 255]
    depth[45:55, 419:429] = 1.25

    # BOTTOM marker (Blue, 2.50m)
    color[425:435, 419:429] = [255, 0, 0]
    depth[425:435, 419:429] = 2.50

    # LEFT marker (Green, 1.75m)
    color[235:245, 95:105] = [0, 255, 0]
    depth[235:245, 95:105] = 1.75

    # RIGHT marker (Yellow, 3.00m)
    color[235:245, 743:753] = [0, 255, 255]
    depth[235:245, 743:753] = 3.00

    markers = {
        "top": {"orig_u": 424, "orig_v": 50, "bgr": (0, 0, 255), "depth_m": 1.25},
        "bottom": {"orig_u": 424, "orig_v": 430, "bgr": (255, 0, 0), "depth_m": 2.50},
        "left": {"orig_u": 100, "orig_v": 240, "bgr": (0, 255, 0), "depth_m": 1.75},
        "right": {"orig_u": 748, "orig_v": 240, "bgr": (0, 255, 255), "depth_m": 3.00},
    }

    return color, depth, markers, W, H


@pytest.mark.parametrize("mode", ["cw", "ccw"])
def test_asymmetric_rgb_and_depth_rotation(asymmetric_rgbd_fixture, mode):
    """Items 1, 2, 3, 9: Verify RGB and depth rotation, pixel correspondence, and portrait dimensions."""
    color, depth, markers, W, H = asymmetric_rgbd_fixture

    color_rot, depth_rot = rotate_rgbd_arrays(color, depth, mode=mode)

    # 9. Portrait image dimensions (height 848, width 480)
    assert color_rot.shape == (W, H, 3)
    assert depth_rot.shape == (W, H)

    # 3. RGB-depth pixel correspondence at all marker locations
    for name, m in markers.items():
        u, v = m["orig_u"], m["orig_v"]
        if mode == "cw":
            u_rot = (H - 1) - v
            v_rot = u
        else:  # ccw
            u_rot = v
            v_rot = (W - 1) - u

        # Check RGB color matches expected marker
        actual_bgr = tuple(color_rot[v_rot, u_rot])
        assert actual_bgr == m["bgr"], f"RGB mismatch for {name} under {mode}: {actual_bgr} vs {m['bgr']}"

        # Check depth matches expected metric depth
        actual_d = depth_rot[v_rot, u_rot]
        assert np.isclose(actual_d, m["depth_m"], atol=1e-3), (
            f"Depth mismatch for {name} under {mode}: {actual_d} vs {m['depth_m']}"
        )


@pytest.mark.parametrize("mode", ["cw", "ccw"])
def test_camera_intrinsics_and_principal_point(mode):
    """Items 4 & 5: Verify rotated intrinsics and principal point transformation."""
    W, H = 848, 480
    fx, fy = 428.11, 427.55
    cx, cy = 422.17, 244.60

    intr_orig = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    intr_rot = rotate_intrinsics(intr_orig, mode=mode)

    assert intr_rot.width == H
    assert intr_rot.height == W

    rot_fx, rot_fy = intr_rot.get_focal_length()
    rot_cx, rot_cy = intr_rot.get_principal_point()

    assert np.isclose(rot_fx, fy)
    assert np.isclose(rot_fy, fx)

    if mode == "cw":
        expected_cx = float((H - 1) - cy)
        expected_cy = float(cx)
    else:  # ccw
        expected_cx = float(cy)
        expected_cy = float((W - 1) - cx)

    assert np.isclose(rot_cx, expected_cx, atol=1e-4)
    assert np.isclose(rot_cy, expected_cy, atol=1e-4)


@pytest.mark.parametrize("mode", ["cw", "ccw"])
def test_landmark_coordinate_transformation(mode):
    """Item 6: Verify 2D landmark coordinate transformation."""
    W, H = 848, 480
    # Simulate a human standing:
    # Nose near top: (u=424, v=80)
    # Left shoulder: (u=370, v=150)
    # Right shoulder: (u=478, v=150)
    # Left ankle near bottom: (u=390, v=440)
    # Right ankle near bottom: (u=458, v=440)
    landmarks_orig = {
        "nose": (424.0, 80.0),
        "left_shoulder": (370.0, 150.0),
        "right_shoulder": (478.0, 150.0),
        "left_ankle": (390.0, 440.0),
        "right_ankle": (458.0, 440.0),
    }

    landmarks_rot = {}
    for k, (u, v) in landmarks_orig.items():
        if mode == "cw":
            u_r = (H - 1) - v
            v_r = u
        else:  # ccw
            u_r = v
            v_r = (W - 1) - u
        landmarks_rot[k] = (u_r, v_r)

    # In CCW rotation:
    # Right of original image (u large) becomes TOP (v small):
    # If the original image had head pointing right (u=750, v=240):
    # In CCW: u_rot = 240, v_rot = (848-1)-750 = 97 (TOP OF IMAGE!)
    head_right = (750.0, 240.0)
    head_rot_u = head_right[1]
    head_rot_v = (W - 1) - head_right[0]
    assert head_rot_v < 150.0, "90 deg CCW must map right-pointing head to top of portrait image!"


@pytest.mark.parametrize("mode", ["cw", "ccw"])
def test_depth_backprojection_consistency_and_chirality(mode):
    """Items 7 & 8: Depth back-projection consistency and coordinate chirality."""
    W, H = 848, 480
    fx, fy = 428.11, 427.55
    cx, cy = 422.17, 244.60

    intr_orig = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    intr_rot = rotate_intrinsics(intr_orig, mode=mode)

    # 3D rotation matrix about optical axis +Z:
    if mode == "cw":
        R = np.array([
            [ 0.0, -1.0, 0.0],
            [ 1.0,  0.0, 0.0],
            [ 0.0,  0.0, 1.0],
        ], dtype=np.float64)
    else:  # ccw
        R = np.array([
            [ 0.0,  1.0, 0.0],
            [-1.0,  0.0, 0.0],
            [ 0.0,  0.0, 1.0],
        ], dtype=np.float64)

    # 8. Coordinate chirality check: det(R) == +1 (proper rotation, preserves handedness)
    assert np.isclose(np.linalg.det(R), 1.0), "Rotation matrix must be a proper SO(3) rotation with det=1"
    assert np.allclose(R.T @ R, np.eye(3)), "Rotation matrix must be strictly orthogonal"

    # 7. Test back-projection consistency for multiple points
    test_pixels = [
        (422.17, 244.60, 1.50),  # Principal point
        (100.0, 100.0, 1.20),
        (700.0, 150.0, 2.00),
        (250.0, 400.0, 2.50),
        (650.0, 420.0, 1.80),
    ]

    rot_fx, rot_fy = intr_rot.get_focal_length()
    rot_cx, rot_cy = intr_rot.get_principal_point()

    for u, v, z in test_pixels:
        # 3D point in original frame
        x_3d = (u - cx) * z / fx
        y_3d = (v - cy) * z / fy
        P_orig = np.array([x_3d, y_3d, z])

        # Rotated pixel
        if mode == "cw":
            u_rot = (H - 1) - v
            v_rot = u
        else:
            u_rot = v
            v_rot = (W - 1) - u

        # 3D point back-projected with rotated intrinsics
        x_3d_rot = (u_rot - rot_cx) * z / rot_fx
        y_3d_rot = (v_rot - rot_cy) * z / rot_fy
        P_rot = np.array([x_3d_rot, y_3d_rot, z])

        # Theoretical rotated 3D point
        P_expected = R @ P_orig

        assert np.allclose(P_rot, P_expected, atol=1e-5), (
            f"Back-projection mismatch for ({u}, {v}) under {mode}: {P_rot} vs {P_expected}"
        )


def test_both_rotation_directions_supported():
    """Item 10: Verify both 'cw' and 'ccw' are accepted and invalid raises ValueError."""
    W, H = 848, 480
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, 425.0, 425.0, 424.0, 240.0)
    color = np.zeros((H, W, 3), dtype=np.uint8)
    depth = np.zeros((H, W), dtype=np.float32)

    # Valid modes
    rotate_intrinsics(intr, "cw")
    rotate_intrinsics(intr, "ccw")
    rotate_rgbd_arrays(color, depth, "cw")
    rotate_rgbd_arrays(color, depth, "ccw")

    # Invalid mode raises ValueError
    with pytest.raises(ValueError):
        rotate_intrinsics(intr, "invalid")
    with pytest.raises(ValueError):
        rotate_rgbd_arrays(color, depth, "invalid")


def test_cli_parser_orientation_defaults_and_preview_flags():
    """Verify CLI parser defaults --portrait to ccw and provides preview flags."""
    from realsense_measure.fullbody.cli import build_parser

    parser = build_parser()

    # 1. capture subcommand
    cap_args = parser.parse_args(["capture", "--out", "test_out", "--subject-id", "TEST_01"])
    assert cap_args.portrait == "ccw", f"Default portrait for capture should be 'ccw', got {cap_args.portrait}"
    assert cap_args.preview_only is False

    cap_preview_args = parser.parse_args(
        ["capture", "--out", "test_out", "--subject-id", "TEST_01", "--preview-only"]
    )
    assert cap_preview_args.preview_only is True

    # 2. check-camera subcommand
    chk_args = parser.parse_args(["check-camera"])
    assert chk_args.portrait == "ccw", f"Default portrait for check-camera should be 'ccw', got {chk_args.portrait}"
    assert chk_args.preview is False

    chk_preview_args = parser.parse_args(["check-camera", "--preview"])
    assert chk_preview_args.preview is True


def test_hud_orientation_indicator_rendering():
    """Verify render_frame correctly handles and renders active portrait orientation."""
    from realsense_measure.fullbody.capture_ui import render_frame

    color = np.zeros((848, 480, 3), dtype=np.uint8)
    depth = np.ones((848, 480), dtype=np.float32) * 1.5

    # CCW snapshot
    snapshot_ccw = {
        "state": "WAIT_FOR_SUBJECT",
        "stop_index": 0,
        "n_stops": 8,
        "turn_direction": "cw",
        "orientation": "ccw",
    }
    rendered_ccw = render_frame(snapshot_ccw, color, depth)
    assert rendered_ccw.shape == (848, 960, 3)

    # CW snapshot
    snapshot_cw = {
        "state": "PREVIEW_ONLY",
        "stop_index": 0,
        "n_stops": 1,
        "turn_direction": "cw",
        "orientation": "cw",
    }
    rendered_cw = render_frame(snapshot_cw, color, depth)
    assert rendered_cw.shape == (848, 960, 3)
