"""
realsense_measure/fullbody/floor.py

Floor plane detection via RANSAC on the lower image region,
cross-stop floor consistency verification, and gravity-aligned
camera-to-world coordinate transform computation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d

from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.types import FloorPlane

logger = logging.getLogger(__name__)


def fit_floor_plane(
    depth_m: np.ndarray,
    intrinsics: o3d.camera.PinholeCameraIntrinsic,
    cfg: FullBodyConfig,
    color_bgr: Optional[np.ndarray] = None,
    mask_exclude: Optional[np.ndarray] = None,
) -> Optional[FloorPlane]:
    """
    Fit a floor plane to the lower portion of a depth image.

    Parameters
    ----------
    depth_m:
        HxW metric depth map.
    intrinsics:
        Camera pinhole intrinsics.
    cfg:
        FullBodyConfig instance.
    color_bgr:
        Optional color image (for debug visualization).
    mask_exclude:
        Optional boolean mask of pixels to exclude (e.g. subject feet).

    Returns
    -------
    Optional[FloorPlane]:
        Fitted FloorPlane instance or None if fitting fails.
    """
    h, w = depth_m.shape[:2]
    row_start = int(h * (1.0 - cfg.floor_lower_image_fraction))

    # Mask lower region
    lower_mask = np.zeros((h, w), dtype=bool)
    lower_mask[row_start:, :] = True
    valid = (depth_m > 0.0) & (depth_m >= cfg.depth_min_m) & (depth_m <= cfg.depth_max_m)
    if mask_exclude is not None:
        valid &= (~mask_exclude)
    
    roi_mask = lower_mask & valid
    if np.sum(roi_mask) < 100:
        logger.warning("  [floor] Insufficient valid depth pixels in lower region.")
        return None

    # Backproject to 3D points
    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    u_sel = u_grid[roi_mask].astype(np.float64)
    v_sel = v_grid[roi_mask].astype(np.float64)
    z_sel = depth_m[roi_mask].astype(np.float64)

    fx, fy = intrinsics.get_focal_length()
    cx, cy = intrinsics.get_principal_point()

    x_sel = (u_sel - cx) * z_sel / fx
    y_sel = (v_sel - cy) * z_sel / fy
    pts = np.stack([x_sel, y_sel, z_sel], axis=1)

    # RANSAC plane fitting using Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    try:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=cfg.floor_distance_threshold_m,
            ransac_n=3,
            num_iterations=cfg.floor_ransac_iterations,
        )
    except Exception as exc:
        logger.warning(f"  [floor] Open3D plane segmentation failed: {exc}")
        return None

    if len(inliers) < 50:
        logger.warning("  [floor] RANSAC found too few plane inliers.")
        return None

    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=np.float64)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-8:
        return None
    normal /= norm_len
    d /= norm_len

    # Camera optical frame: +Y is downwards. Floor normal pointing UPWARDS
    # towards the ceiling must have negative Y component in camera frame.
    expected_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    if np.dot(normal, expected_up) < 0:
        normal = -normal
        d = -d

    alignment = np.dot(normal, expected_up)
    if alignment < cfg.floor_normal_min_up_alignment:
        logger.warning(
            f"  [floor] Fitted plane normal [a={normal[0]:.2f}, b={normal[1]:.2f}, c={normal[2]:.2f}] "
            f"deviates from expected vertical (alignment={alignment:.2f} < {cfg.floor_normal_min_up_alignment:.2f})"
        )
        return None

    inlier_pts = pts[inliers]
    point_on_plane = np.mean(inlier_pts, axis=0)
    # Project exactly onto plane
    dist_to_plane = np.dot(point_on_plane, normal) + d
    point_on_plane -= dist_to_plane * normal

    # In camera optical frame (+Y downwards), the floor must be well below the camera.
    # An elevated surface (table, desk, box) with point_on_plane[1] < 0.50 m is rejected.
    if point_on_plane[1] < 0.50:
        logger.warning(
            f"  [floor] Fitted plane is elevated (Y={point_on_plane[1]:.2f} m < 0.50 m below camera); "
            f"rejected as table/desk/elevated surface."
        )
        return None

    rmse = float(np.sqrt(np.mean((inlier_pts @ normal + d) ** 2)))
    inlier_ratio = float(len(inliers) / len(pts))

    # Compute gravity alignment rotation R_grav_cam: normal -> [0, 1, 0]
    # preserving camera forward heading along +Z
    R_grav_cam = build_gravity_rotation(normal)

    return FloorPlane(
        coefficients=np.array([normal[0], normal[1], normal[2], d], dtype=np.float64),
        normal=normal,
        point_on_plane=point_on_plane,
        inlier_count=len(inliers),
        inlier_ratio=inlier_ratio,
        rmse_m=rmse,
        gravity_R_cam=R_grav_cam,
    )


def build_gravity_rotation(up_cam: np.ndarray) -> np.ndarray:
    """
    Build 3x3 rotation matrix R from camera optical coordinates to gravity frame:
      Y_g = up_cam (unit vector pointing upwards)
      Z_g = camera forward (0, 0, 1) minus its component along Y_g, normalized
      X_g = cross(Y_g, Z_g), normalized
      R = stack of rows [X_g, Y_g, Z_g]

    Properties:
      - det(R) = +1 (proper orthonormal rotation)
      - R @ up_cam = [0, 1, 0] (+Y is upwards in gravity frame)
      - R @ [0, 0, 1] has X=0 and Z>0 (camera forward maps to azimuth 0 along +Z)
      - For level camera where up_cam = [0, -1, 0], R = diag(-1, -1, 1).
    """
    up = np.asarray(up_cam, dtype=np.float64).reshape(3)
    norm_up = np.linalg.norm(up)
    if norm_up < 1e-8:
        raise ValueError("up_cam must be a non-zero vector")
    Y_g = up / norm_up

    fwd = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    Z_proj = fwd - np.dot(fwd, Y_g) * Y_g
    norm_z = np.linalg.norm(Z_proj)
    if norm_z < 1e-8:
        # Singular case: camera pointing straight up or straight down
        alt = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        Z_proj = alt - np.dot(alt, Y_g) * Y_g
        norm_z = np.linalg.norm(Z_proj)
    Z_g = Z_proj / norm_z

    X_g = np.cross(Y_g, Z_g)
    norm_x = np.linalg.norm(X_g)
    if norm_x < 1e-8:
        X_g = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        X_g /= norm_x

    R = np.vstack([X_g, Y_g, Z_g])
    return R


def compute_rotation_aligning_vectors(
    src_vec: np.ndarray,
    dst_vec: np.ndarray,
) -> np.ndarray:
    """
    Compute minimal 3x3 rotation matrix R such that R @ src_vec = dst_vec.
    NOTE: When src_vec and dst_vec are nearly antipodal (dot ~ -1),
    the rotation axis is unconstrained in the orthogonal plane.
    For gravity frames, use build_gravity_rotation(up_cam) instead.
    """
    v1 = src_vec / np.linalg.norm(src_vec)
    v2 = dst_vec / np.linalg.norm(dst_vec)

    dot = np.dot(v1, v2)
    if dot > 0.9999999:
        return np.eye(3, dtype=np.float64)
    if dot < -0.9999999:
        # 180 degree flip around any orthogonal axis
        axis = np.array([1.0, 0.0, 0.0]) if abs(v1[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(v1, axis)
        axis /= np.linalg.norm(axis)
        # Rodrigues formula for 180 degrees
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        return np.eye(3) + 2.0 * (K @ K)

    cross = np.cross(v1, v2)
    s = np.linalg.norm(cross)
    kmat = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0],
    ])
    R = np.eye(3) + kmat + (kmat @ kmat) * ((1.0 - dot) / (s ** 2))
    return R


def verify_floor_consistency(
    planes: List[FloorPlane],
    cfg: FullBodyConfig,
    min_planes: int = 2,
) -> Tuple[bool, FloorPlane]:
    """
    Verify that floor planes detected across stops agree on orientation and height.

    Returns
    -------
    Tuple[bool, FloorPlane]:
        (is_consistent, consensus_plane)
    """
    if not planes:
        raise ValueError("Cannot verify consistency of empty floor planes list")

    normals = np.array([p.normal for p in planes])  # (S, 3)
    # Average normal
    mean_normal = np.mean(normals, axis=0)
    mean_normal /= np.linalg.norm(mean_normal)

    # Angles to mean normal
    dots = np.clip(normals @ mean_normal, -1.0, 1.0)
    angles_deg = np.rad2deg(np.arccos(dots))
    max_angle_diff = float(np.max(angles_deg))

    is_consistent = (max_angle_diff <= cfg.floor_max_angle_diff_deg) and (len(planes) >= min_planes)
    if len(planes) < min_planes:
        logger.warning(
            f"  [floor] Insufficient floor detections across stops ({len(planes)} < {min_planes}); floor consistency rejected."
        )
    elif not is_consistent:
        logger.warning(
            f"  [floor] Floor normals across stops differ by up to {max_angle_diff:.2f} deg "
            f"(threshold: {cfg.floor_max_angle_diff_deg:.1f} deg)"
        )

    # Use median distance and mean normal for robust consensus plane
    mean_d = float(np.median([p.coefficients[3] for p in planes]))
    consensus_point = -mean_d * mean_normal
    R_grav_cam = build_gravity_rotation(mean_normal)

    consensus_plane = FloorPlane(
        coefficients=np.array([mean_normal[0], mean_normal[1], mean_normal[2], mean_d]),
        normal=mean_normal,
        point_on_plane=consensus_point,
        inlier_count=sum(p.inlier_count for p in planes),
        inlier_ratio=float(np.mean([p.inlier_ratio for p in planes])),
        rmse_m=float(np.mean([p.rmse_m for p in planes])),
        gravity_R_cam=R_grav_cam,
    )
    return is_consistent, consensus_plane


def save_floor_plane(floor: FloorPlane, path: Path) -> None:
    """Save FloorPlane attributes to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "coefficients": floor.coefficients.tolist(),
        "normal": floor.normal.tolist(),
        "point_on_plane": floor.point_on_plane.tolist(),
        "inlier_count": int(floor.inlier_count),
        "inlier_ratio": float(floor.inlier_ratio),
        "rmse_m": float(floor.rmse_m),
        "gravity_R_cam": floor.gravity_R_cam.tolist(),
        "is_observed": bool(floor.is_observed),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_floor_plane(path: Path) -> Optional[FloorPlane]:
    """Load FloorPlane from a JSON file."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        normal = np.array(data["normal"], dtype=np.float64)
        # Recompute gravity_R_cam via build_gravity_rotation to ensure consistent forward heading
        gravity_R = build_gravity_rotation(normal)
        return FloorPlane(
            coefficients=np.array(data["coefficients"], dtype=np.float64),
            normal=normal,
            point_on_plane=np.array(data["point_on_plane"], dtype=np.float64),
            inlier_count=int(data.get("inlier_count", 0)),
            inlier_ratio=float(data.get("inlier_ratio", 0.0)),
            rmse_m=float(data.get("rmse_m", 0.0)),
            gravity_R_cam=gravity_R,
            is_observed=bool(data.get("is_observed", True)),
        )
    except Exception as exc:
        logger.warning(f"Failed to load floor plane from {path}: {exc}")
        return None

