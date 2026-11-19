"""
realsense_measure/fullbody/fusion.py

Volumetric TSDF fusion of aligned stops into a unified canonical body scan.
Extracts triangle mesh and point cloud, filters detached fragments,
and builds an anatomical hole map of unobserved regions.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d

from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.types import AlignedStop, Keypoints3D, StopData

logger = logging.getLogger(__name__)


def compute_foreground_mask(
    depth_m: np.ndarray,
    bg_depth_m: Optional[np.ndarray],
    cfg: FullBodyConfig,
) -> np.ndarray:
    """
    Extract foreground subject mask from depth map using background-depth subtraction.
    A pixel is foreground if its burst-median depth is >= 3 cm closer than the
    background median at that pixel (and valid), followed by morphological open/close
    and keeping the largest connected component.

    Parameters
    ----------
    depth_m:
        HxW float32 metric depth of the stop (e.g. burst median).
    bg_depth_m:
        HxW float32 metric depth of the empty scene background, or None.
    cfg:
        FullBodyConfig instance.

    Returns
    -------
    np.ndarray:
        HxW boolean mask of foreground subject pixels.
    """
    h, w = depth_m.shape[:2]
    valid_curr = (depth_m >= cfg.depth_min_m) & (depth_m <= cfg.depth_max_m) & np.isfinite(depth_m)

    if bg_depth_m is not None:
        if bg_depth_m.shape[:2] != (h, w):
            bg_depth_m = cv2.resize(bg_depth_m, (w, h), interpolation=cv2.INTER_NEAREST)
        valid_bg = (bg_depth_m >= cfg.depth_min_m) & (bg_depth_m <= cfg.depth_max_m) & np.isfinite(bg_depth_m)
        # Pixel is foreground if depth is >= 3 cm closer than background
        is_fg = valid_curr & valid_bg & ((bg_depth_m - depth_m) >= 0.03)
        # If background pixel was missing or invalid, retain valid depth
        is_fg |= (valid_curr & ~valid_bg)
    else:
        # Fallback when no background is provided (e.g. standalone synthetic test)
        is_fg = valid_curr

    if not np.any(is_fg):
        return np.zeros((h, w), dtype=bool)

    # Morphological open then close
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = is_fg.astype(np.uint8)
    opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    # Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num_labels <= 1:
        return np.zeros((h, w), dtype=bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    max_area = stats[largest_idx, cv2.CC_STAT_AREA]
    min_comp_area = max(300, int(0.01 * max_area))
    valid_labels = [
        lbl for lbl in range(1, num_labels)
        if stats[lbl, cv2.CC_STAT_AREA] >= min_comp_area
    ]
    fg_mask = np.isin(labels, valid_labels)
    return fg_mask


def fuse_aligned_stops(
    stops: List[StopData],
    aligned_stops: List[AlignedStop],
    cfg: FullBodyConfig,
    bg_depth: Optional[np.ndarray] = None,
) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud, Dict[str, float]]:
    """
    Integrate multi-stop RGB-D frames into a single ScalableTSDFVolume.

    Parameters
    ----------
    stops:
        List of StopData instances.
    aligned_stops:
        List of AlignedStop instances.
    cfg:
        FullBodyConfig instance.
    bg_depth:
        Optional background depth array in metres for background-subtraction foreground gating.

    Returns
    -------
    Tuple[TriangleMesh, PointCloud, Dict[str, float]]:
        (cleaned_mesh, fused_pcd, hole_map)
    """
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=cfg.tsdf_voxel_length_m,
        sdf_trunc=cfg.tsdf_sdf_trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    integrated_count = 0
    for s, a in zip(stops, aligned_stops):
        if not a.accepted or a.T_canonical_camera is None:
            continue

        extrinsic = np.linalg.inv(a.T_canonical_camera)

        # Prepare RGB-D with shared background-depth subtraction foreground mask
        color_rgb = np.ascontiguousarray(s.color[:, :, ::-1])
        fg_mask = compute_foreground_mask(s.depth, bg_depth, cfg)
        depth = s.depth.copy()
        depth[~fg_mask] = 0.0
        if s.confidence_map is not None:
            depth[s.confidence_map < cfg.min_confidence_threshold] = 0.0

        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
        color_o3d = o3d.geometry.Image(color_rgb.astype(np.uint8))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=cfg.tsdf_depth_trunc_m,
            convert_rgb_to_intensity=False,
        )

        try:
            volume.integrate(rgbd, s.intrinsics, extrinsic)
            integrated_count += 1
        except Exception as exc:
            logger.warning(f"  [fusion] Stop {s.stop_id} integration failed: {exc}")

    mesh = volume.extract_triangle_mesh()
    pcd = volume.extract_point_cloud()

    # Fallback to bounded UniformTSDFVolume if ScalableTSDF yielded empty geometry
    if len(mesh.vertices) == 0:
        logger.info("  [fusion] ScalableTSDF yielded 0 vertices; attempting bounded UniformTSDFVolume.")
        pts_list = [
            np.asarray(a.pcd_canonical.points)
            for a in aligned_stops
            if a.pcd_canonical is not None and len(a.pcd_canonical.points) > 0
        ]
        if pts_list:
            all_pts = np.vstack(pts_list)
            min_b = np.min(all_pts, axis=0) - 0.05
            max_b = np.max(all_pts, axis=0) + 0.05
            length = float(np.max(max_b - min_b))
            res = int(np.clip(round(length / cfg.tsdf_voxel_length_m), 128, 384))
            vol_u = o3d.pipelines.integration.UniformTSDFVolume(
                length=length,
                resolution=res,
                sdf_trunc=cfg.tsdf_sdf_trunc_m,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
                origin=min_b.reshape(3, 1),
            )
            for s, a in zip(stops, aligned_stops):
                if not a.accepted or a.T_canonical_camera is None:
                    continue
                extrinsic = np.linalg.inv(a.T_canonical_camera)
                color_rgb = np.ascontiguousarray(s.color[:, :, ::-1])
                depth = s.depth.copy()
                if s.confidence_map is not None:
                    depth[s.confidence_map < cfg.min_confidence_threshold] = 0.0
                fg_mask = compute_foreground_mask(depth, bg_depth, cfg)
                depth[~fg_mask] = 0.0
                depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
                color_o3d = o3d.geometry.Image(color_rgb.astype(np.uint8))
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_o3d,
                    depth_o3d,
                    depth_scale=1.0,
                    depth_trunc=cfg.tsdf_depth_trunc_m,
                    convert_rgb_to_intensity=False,
                )
                try:
                    vol_u.integrate(rgbd, s.intrinsics, extrinsic)
                except Exception as exc:
                    logger.debug(f"  [fusion uniform fallback] Stop {s.stop_id} failed: {exc}")
            mesh = vol_u.extract_triangle_mesh()
            if len(pcd.points) == 0:
                pcd = vol_u.extract_point_cloud()

    # Fallback to concatenated point clouds if volume extraction is empty
    if len(pcd.points) == 0:
        logger.info("  [fusion] TSDF yielded 0 points, falling back to direct point union.")
        pcd = o3d.geometry.PointCloud()
        for a in aligned_stops:
            if a.pcd_canonical is not None:
                pcd += a.pcd_canonical
        pcd = pcd.voxel_down_sample(cfg.tsdf_voxel_length_m)

    # Clean up small disconnected components from mesh
    if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh = _clean_mesh_components(
            mesh,
            min_ratio=cfg.min_mesh_component_ratio,
            keep_dominant_only=cfg.keep_dominant_mesh_component,
        )
        mesh.compute_vertex_normals()

    if len(pcd.points) > 0:
        pcd.estimate_normals()

    hole_map = compute_hole_map(pcd, aligned_stops)
    return mesh, pcd, hole_map


def _clean_mesh_components(
    mesh: o3d.geometry.TriangleMesh,
    min_ratio: float = 0.02,
    keep_dominant_only: bool = False,
) -> o3d.geometry.TriangleMesh:
    """Remove disconnected triangle clusters smaller than min_ratio of the largest cluster."""
    try:
        triangle_clusters, num_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        num_triangles = np.asarray(num_triangles)

        if len(num_triangles) <= 1:
            return mesh

        largest_cluster_idx = int(np.argmax(num_triangles))
        largest_cluster_size = float(num_triangles[largest_cluster_idx])
        min_size = largest_cluster_size * min_ratio

        if keep_dominant_only:
            triangles_to_remove = [
                i for i, cluster_idx in enumerate(triangle_clusters)
                if cluster_idx != largest_cluster_idx
            ]
        else:
            triangles_to_remove = [
                i for i, cluster_idx in enumerate(triangle_clusters)
                if num_triangles[cluster_idx] < min_size
            ]
        mesh.remove_triangles_by_index(triangles_to_remove)
        mesh.remove_unreferenced_vertices()
    except Exception as exc:
        logger.debug(f"  [fusion] Mesh component filtering skipped: {exc}")

    return mesh


def compute_hole_map(
    pcd: o3d.geometry.PointCloud,
    aligned_stops: List[AlignedStop],
) -> Dict[str, float]:
    """
    Compute missing fraction (0.0 = fully covered, 1.0 = completely missing)
    for anatomically vulnerable regions.
    """
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return {
            "crown": 1.0,
            "crotch": 1.0,
            "underarms": 1.0,
            "feet": 1.0,
            "back_of_arms": 1.0,
        }

    y_min, y_max = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
    stature = max(0.5, y_max - y_min)

    # 1. Crown coverage (top 4 cm)
    crown_pts = np.sum(pts[:, 1] >= y_max - 0.04)
    crown_missing = float(np.clip(1.0 - (crown_pts / 300.0), 0.0, 1.0))

    # 2. Feet coverage (bottom 6 cm)
    feet_pts = np.sum(pts[:, 1] <= y_min + 0.06)
    feet_missing = float(np.clip(1.0 - (feet_pts / 400.0), 0.0, 1.0))

    # 3. Crotch region (roughly 45-52% of stature, central lateral)
    crotch_mask = (
        (pts[:, 1] >= y_min + 0.45 * stature) &
        (pts[:, 1] <= y_min + 0.53 * stature) &
        (np.abs(pts[:, 0]) <= 0.08)
    )
    crotch_pts = np.sum(crotch_mask)
    crotch_missing = float(np.clip(1.0 - (crotch_pts / 200.0), 0.0, 1.0))

    # 4. Underarms / axilla
    underarm_mask = (
        (pts[:, 1] >= y_min + 0.70 * stature) &
        (pts[:, 1] <= y_min + 0.78 * stature) &
        (np.abs(pts[:, 0]) >= 0.12) & (np.abs(pts[:, 0]) <= 0.25)
    )
    underarm_pts = np.sum(underarm_mask)
    underarm_missing = float(np.clip(1.0 - (underarm_pts / 350.0), 0.0, 1.0))

    # 5. Back of arms
    back_arm_mask = (
        (pts[:, 1] >= y_min + 0.60 * stature) &
        (pts[:, 1] <= y_min + 0.80 * stature) &
        (pts[:, 2] < -0.05) &
        (np.abs(pts[:, 0]) >= 0.15)
    )
    back_arm_pts = np.sum(back_arm_mask)
    back_arm_missing = float(np.clip(1.0 - (back_arm_pts / 250.0), 0.0, 1.0))

    return {
        "crown": round(crown_missing, 3),
        "crotch": round(crotch_missing, 3),
        "underarms": round(underarm_missing, 3),
        "feet": round(feet_missing, 3),
        "back_of_arms": round(back_arm_missing, 3),
    }

