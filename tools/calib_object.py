"""
tools/calib_object.py

Rigid object (cylinder/bucket) diameter and roundness validation tool.
- Slices fused mesh or point cloud at 20 heights across the middle 60% of height.
- Robust RANSAC + least-squares circle fitting per slice.
- Reports diameter, circle RMS, angular coverage, roundness (max - min radius),
  and error vs tape measurement.
- Generates 360-degree surface-to-circle residual map (binned per 10 deg)
  highlighting inter-stop seams, saved as PNG plot and JSON report.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)


def fit_circle_ransac_ls(
    pts_2d: np.ndarray,
    max_iters: int = 150,
    inlier_thresh_m: float = 0.005,
) -> Optional[Tuple[float, float, float, np.ndarray]]:
    """
    Fit circle (xc, zc, R) to 2D horizontal points using RANSAC followed by
    geometric least-squares refinement.
    
    Returns: (xc, zc, R, inlier_mask) or None.
    """
    n_pts = len(pts_2d)
    if n_pts < 15:
        return None

    best_inliers = np.zeros(n_pts, dtype=bool)
    best_count = 0
    best_circle = (0.0, 0.0, 0.0)

    # 1. RANSAC loop
    for _ in range(max_iters):
        sample_idxs = np.random.choice(n_pts, 3, replace=False)
        p1, p2, p3 = pts_2d[sample_idxs]

        # Circumcenter formula
        d = 2.0 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
        if abs(d) < 1e-7:
            continue

        p1_sq = p1[0] ** 2 + p1[1] ** 2
        p2_sq = p2[0] ** 2 + p2[1] ** 2
        p3_sq = p3[0] ** 2 + p3[1] ** 2

        xc = (p1_sq * (p2[1] - p3[1]) + p2_sq * (p3[1] - p1[1]) + p3_sq * (p1[1] - p2[1])) / d
        zc = (p1_sq * (p3[0] - p2[0]) + p2_sq * (p1[0] - p3[0]) + p3_sq * (p2[0] - p1[0])) / d
        r = math.sqrt((p1[0] - xc) ** 2 + (p1[1] - zc) ** 2)

        # Sanity range: radius between 4 cm and 60 cm
        if not (0.04 <= r <= 0.60):
            continue

        dists = np.abs(np.hypot(pts_2d[:, 0] - xc, pts_2d[:, 1] - zc) - r)
        inliers = dists < inlier_thresh_m
        count = int(np.sum(inliers))

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_circle = (xc, zc, r)

    if best_count < 10:
        return None

    # 2. Algebraic least-squares on inliers (Kasa circle fit)
    in_pts = pts_2d[best_inliers]
    x_i = in_pts[:, 0]
    z_i = in_pts[:, 1]

    A = np.column_stack([2.0 * x_i, 2.0 * z_i, np.ones_like(x_i)])
    b = x_i ** 2 + z_i ** 2
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        xc_ls, zc_ls, c_val = float(sol[0]), float(sol[1]), float(sol[2])
        r_sq = c_val + xc_ls ** 2 + zc_ls ** 2
        if r_sq <= 0:
            return best_circle[0], best_circle[1], best_circle[2], best_inliers
        r_ls = math.sqrt(r_sq)
    except Exception:
        xc_ls, zc_ls, r_ls = best_circle

    # 3. Geometric refinement (Gauss-Newton)
    xc_opt, zc_opt, r_opt = xc_ls, zc_ls, r_ls
    for _ in range(5):
        dx = in_pts[:, 0] - xc_opt
        dz = in_pts[:, 1] - zc_opt
        di = np.hypot(dx, dz)
        di = np.maximum(di, 1e-6)
        ri = di - r_opt

        # Jacobian: [-dx/di, -dz/di, -1]
        J = np.column_stack([-dx / di, -dz / di, -np.ones_like(di)])
        try:
            delta, _, _, _ = np.linalg.lstsq(J, -ri, rcond=None)
            xc_opt += float(delta[0])
            zc_opt += float(delta[1])
            r_opt += float(delta[2])
        except Exception:
            break

    # Recompute inliers with refined circle
    final_dists = np.abs(np.hypot(pts_2d[:, 0] - xc_opt, pts_2d[:, 1] - zc_opt) - r_opt)
    refined_inliers = final_dists < inlier_thresh_m

    return xc_opt, zc_opt, r_opt, refined_inliers


def analyze_cylinder_diameter(
    scan_path: Path,
    tape_diameter_mm: Optional[float] = None,
    n_slices: int = 20,
    slice_half_thickness_m: float = 0.005,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Main evaluation routine: loads mesh/cloud, takes 20 slices, fits circles,
    and produces summary metrics + residual map.
    """
    scan_path = Path(scan_path)
    if scan_path.is_dir():
        # Look for fused_mesh.ply or fused_cloud.ply in directory
        mesh_f = scan_path / "fused_mesh.ply"
        pcd_f = scan_path / "fused_cloud.ply"
        scan_file = mesh_f if mesh_f.exists() else pcd_f
        if out_dir is None:
            out_dir = scan_path
    else:
        scan_file = scan_path
        if out_dir is None:
            out_dir = scan_path.parent

    if not scan_file.exists():
        raise FileNotFoundError(f"Could not find scan file at {scan_file}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Try reading tape diameter from meta.json if not passed
    if tape_diameter_mm is None:
        meta_candidates = [
            out_dir / "meta.json",
            out_dir.parent / "meta.json",
            scan_path / "meta.json",
        ]
        for mc in meta_candidates:
            if mc.exists():
                try:
                    with open(mc, "r", encoding="utf-8") as f:
                        m_data = json.load(f)
                    val = m_data.get("object_diameter_mm")
                    if val is not None:
                        tape_diameter_mm = float(val)
                        break
                except Exception:
                    pass

    # Load geometry points
    if str(scan_file).lower().endswith(".ply") or str(scan_file).lower().endswith(".obj"):
        mesh = o3d.io.read_triangle_mesh(str(scan_file))
        if len(mesh.vertices) > 0:
            pts = np.asarray(mesh.vertices)
        else:
            pcd = o3d.io.read_point_cloud(str(scan_file))
            pts = np.asarray(pcd.points)
    else:
        pcd = o3d.io.read_point_cloud(str(scan_file))
        pts = np.asarray(pcd.points)

    if len(pts) < 100:
        raise ValueError(f"Too few points ({len(pts)}) in scan file: {scan_file}")

    # Bounding box along Y (height in gravity-aligned frame)
    y_vals = pts[:, 1]
    y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))
    total_h = y_max - y_min

    # Middle 60% of object height
    y_start = y_min + 0.20 * total_h
    y_end = y_min + 0.80 * total_h
    slice_heights = np.linspace(y_start, y_end, n_slices)

    slice_reports = []
    all_res_deg = []

    for idx, yh in enumerate(slice_heights):
        mask = np.abs(y_vals - yh) <= slice_half_thickness_m
        slice_pts = pts[mask][:, [0, 2]]  # (X, Z) coordinates

        fit = fit_circle_ransac_ls(slice_pts)
        if fit is None:
            continue

        xc, zc, r, inliers = fit
        in_pts = slice_pts[inliers]
        d_pts = np.hypot(in_pts[:, 0] - xc, in_pts[:, 1] - zc)
        residuals_m = d_pts - r
        rms_mm = float(np.sqrt(np.mean(residuals_m ** 2)) * 1000.0)

        # Angular coverage (binned per 10 degrees)
        angles_rad = np.arctan2(in_pts[:, 1] - zc, in_pts[:, 0] - xc)
        angles_deg = np.rad2deg(angles_rad) % 360.0
        bins = np.floor(angles_deg / 10.0).astype(int)
        unique_bins = len(np.unique(bins))
        coverage_deg = float(unique_bins * 10.0)

        # Roundness (max radius - min radius in mm)
        roundness_mm = float((np.max(d_pts) - np.min(d_pts)) * 1000.0)
        diameter_mm = float(2.0 * r * 1000.0)

        slice_reports.append({
            "slice_index": idx,
            "y_m": round(float(yh), 4),
            "diameter_mm": round(diameter_mm, 2),
            "circle_rms_mm": round(rms_mm, 2),
            "angular_coverage_deg": round(coverage_deg, 1),
            "roundness_mm": round(roundness_mm, 2),
            "inlier_points": int(len(in_pts)),
            "center_xz_m": [round(float(xc), 4), round(float(zc), 4)],
        })

        # Collect for full 360 residual map
        for ang, res in zip(angles_deg, residuals_m):
            all_res_deg.append((float(ang), float(res * 1000.0)))

    if not slice_reports:
        raise RuntimeError("Circle fitting failed on all 20 slices of object.")

    # Aggregate metrics across the slices
    diameters = [s["diameter_mm"] for s in slice_reports]
    rmses = [s["circle_rms_mm"] for s in slice_reports]
    roundnesses = [s["roundness_mm"] for s in slice_reports]
    coverages = [s["angular_coverage_deg"] for s in slice_reports]

    mean_diam = float(np.mean(diameters))
    med_diam = float(np.median(diameters))
    std_diam = float(np.std(diameters))
    mean_rms = float(np.mean(rmses))
    mean_roundness = float(np.mean(roundnesses))
    mean_coverage = float(np.mean(coverages))

    err_tape_mm = None
    err_tape_pct = None
    if tape_diameter_mm is not None and tape_diameter_mm > 0:
        err_tape_mm = float(mean_diam - tape_diameter_mm)
        err_tape_pct = float((err_tape_mm / tape_diameter_mm) * 100.0)

    # 360-degree residual map (binned per 10 degrees)
    binned_residuals = []
    bin_centers = [i * 10.0 + 5.0 for i in range(36)]
    bin_means = []
    bin_stds = []

    if all_res_deg:
        ang_arr = np.array([x[0] for x in all_res_deg])
        res_arr = np.array([x[1] for x in all_res_deg])
        for b_idx in range(36):
            b_min = b_idx * 10.0
            b_max = (b_idx + 1) * 10.0
            in_b = (ang_arr >= b_min) & (ang_arr < b_max)
            if np.any(in_b):
                m_val = float(np.mean(res_arr[in_b]))
                s_val = float(np.std(res_arr[in_b]))
                count = int(np.sum(in_b))
            else:
                m_val = 0.0
                s_val = 0.0
                count = 0
            bin_means.append(m_val)
            bin_stds.append(s_val)
            binned_residuals.append({
                "bin_index": b_idx,
                "angle_center_deg": bin_centers[b_idx],
                "mean_residual_mm": round(m_val, 2),
                "std_residual_mm": round(s_val, 2),
                "point_count": count,
            })

        # Save residual map PNG plot
        fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
        ax.plot(bin_centers, bin_means, marker="o", color="#1f77b4", label="Mean Surface Residual (mm)")
        ax.fill_between(
            bin_centers,
            np.array(bin_means) - np.array(bin_stds),
            np.array(bin_means) + np.array(bin_stds),
            color="#1f77b4",
            alpha=0.2,
            label="+- 1 Std Dev",
        )
        # Mark the 8 nominal stop angles (every 45 deg)
        for stop_angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            ax.axvline(stop_angle, color="red", linestyle="--", alpha=0.4, linewidth=1.0)
        ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
        ax.set_xlabel("Surface Azimuth Angle (deg)")
        ax.set_ylabel("Radial Residual (mm)")
        ax.set_title("Surface-to-Circle Residual vs Azimuth (Dashed Red = Nominal Stops)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")
        plot_path = out_dir / "residual_map.png"
        fig.tight_layout()
        fig.savefig(str(plot_path))
        plt.close(fig)
        plot_saved = str(plot_path.name)
    else:
        plot_saved = None

    report = {
        "scan_file": str(scan_file.name),
        "tape_diameter_mm": tape_diameter_mm,
        "summary": {
            "mean_diameter_mm": round(mean_diam, 2),
            "median_diameter_mm": round(med_diam, 2),
            "std_across_heights_mm": round(std_diam, 2),
            "mean_circle_rms_mm": round(mean_rms, 2),
            "mean_roundness_mm": round(mean_roundness, 2),
            "mean_coverage_deg": round(mean_coverage, 1),
            "error_vs_tape_mm": round(err_tape_mm, 2) if err_tape_mm is not None else None,
            "error_vs_tape_pct": round(err_tape_pct, 2) if err_tape_pct is not None else None,
        },
        "slice_count": len(slice_reports),
        "slices": slice_reports,
        "residual_bins_10deg": binned_residuals,
        "residual_map_plot": plot_saved,
    }

    report_json = out_dir / "calib_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def print_calib_summary(report: Dict[str, Any]) -> None:
    """Print clean formatted table and summary to console."""
    summ = report["summary"]
    tape = report.get("tape_diameter_mm")
    print("\n" + "=" * 78)
    print("OBJECT CALIBRATION & DIAMETER REPORT")
    print("=" * 78)
    print(f"{'Slice':<6} | {'Height (m)':<11} | {'Diam (mm)':<10} | {'RMS (mm)':<9} | {'Coverage':<10} | {'Roundness'}")
    print("-" * 78)
    for s in report.get("slices", []):
        print(
            f"{s['slice_index']:<6} | {s['y_m']:<11.3f} | {s['diameter_mm']:<10.2f} | "
            f"{s['circle_rms_mm']:<9.2f} | {s['angular_coverage_deg']:<8.0f} deg | {s['roundness_mm']:.2f} mm"
        )
    print("-" * 78)
    print("SUMMARY:")
    print(f"  Mean Diameter:       {summ['mean_diameter_mm']:.2f} mm")
    print(f"  Median Diameter:     {summ['median_diameter_mm']:.2f} mm")
    print(f"  Std Across Heights:  {summ['std_across_heights_mm']:.2f} mm")
    print(f"  Circle-Fit RMS:      {summ['mean_circle_rms_mm']:.2f} mm")
    print(f"  Roundness:           {summ['mean_roundness_mm']:.2f} mm")
    print(f"  Angular Coverage:    {summ['mean_coverage_deg']:.1f} deg")
    if tape is not None:
        err_mm = summ["error_vs_tape_mm"]
        err_pct = summ["error_vs_tape_pct"]
        print(f"  Tape Reference:      {tape:.2f} mm")
        print(f"  Error vs Tape:       {err_mm:+.2f} mm ({err_pct:+.2f}%)")
    print("=" * 78 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze cylinder diameter from reconstructed mesh or cloud.")
    parser.add_argument("--scan", required=True, type=Path, help="Directory containing fused_mesh.ply or PLY file")
    parser.add_argument("--tape-mm", type=float, default=None, help="Known ground-truth tape diameter in mm")
    args = parser.parse_args()

    try:
        report = analyze_cylinder_diameter(scan_path=args.scan, tape_diameter_mm=args.tape_mm)
        print_calib_summary(report)
        return 0
    except Exception as exc:
        print(f"Calibration error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
