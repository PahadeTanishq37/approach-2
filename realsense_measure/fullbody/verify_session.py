"""
realsense_measure/fullbody/verify_session.py

Static session verification tool for full-body multi-stop capture sessions.
Performs integrity, geometry, framing, stillness, and floor-level checks
without running registration, TSDF fusion, or anthropometric measurement.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.floor import fit_floor_plane

logger = logging.getLogger(__name__)


def colorize_depth(depth_m: np.ndarray, min_m: float = 0.5, max_m: float = 3.0) -> np.ndarray:
    """Colorize float32 metric depth array with JET colormap."""
    valid = (depth_m > 0.05) & (depth_m <= max_m)
    clamped = np.clip(depth_m, min_m, max_m)
    norm = ((clamped - min_m) / max(1e-3, (max_m - min_m)) * 255.0).astype(np.uint8)
    colorized = cv2.applyColorMap(255 - norm, cv2.COLORMAP_JET)
    colorized[~valid] = [0, 0, 0]
    return colorized


def verify_sha256_manifest(manifest_path: Path) -> Tuple[bool, List[str]]:
    """Verify all files listed in files.sha256 match their stored checksums."""
    if not manifest_path.exists():
        return False, ["files.sha256 missing"]

    errors = []
    base_dir = manifest_path.parent
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                expected_hash, rel_name = parts
                target_f = base_dir / rel_name
                if not target_f.exists():
                    errors.append(f"Missing file: {rel_name}")
                    continue
                actual_hash = hashlib.sha256(target_f.read_bytes()).hexdigest()
                if actual_hash.lower() != expected_hash.lower():
                    errors.append(f"SHA256 mismatch for {rel_name}")
    except Exception as exc:
        errors.append(f"Failed to read manifest: {exc}")

    return len(errors) == 0, errors


def verify_session(session_dir: Path) -> int:
    """
    Verify a saved capture session and print a summary table.
    Writes verify_report.json and contact_sheet.png inside session_dir.
    Returns 0 on PASS/WARN, 1 on FAIL.
    """
    session_path = Path(session_dir)
    if not session_path.exists() or not session_path.is_dir():
        print(f"Error: Session directory '{session_path}' does not exist.", file=sys.stderr)
        return 1

    meta_file = session_path / "meta.json"
    if not meta_file.exists():
        print(f"Error: Session missing meta.json at '{meta_file}'.", file=sys.stderr)
        return 1

    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as exc:
        print(f"Error: Failed to parse meta.json: {exc}", file=sys.stderr)
        return 1

    n_stops = int(meta.get("n_stops", meta.get("stop_count", 8)))
    subject_id = meta.get("subject_id", "UNKNOWN")
    subject_type = meta.get("subject_type", "person").lower()
    is_object = (subject_type == "object")
    frames_requested = int(meta.get("frames_requested", 60))
    intrinsics = meta.get("intrinsics", {})
    width = int(intrinsics.get("width", 1280))
    height = int(intrinsics.get("height", 720))
    orientation = str(meta.get("camera_portrait_rotation", "none")).lower()

    # Session-level checks
    session_checks: Dict[str, Any] = {}

    # Check 1: 8 stops accepted
    stops_present_count = 0
    stops_accepted_count = 0

    # Check 2: Background calibration
    bg_meta = meta.get("background")
    bg_dir = session_path / "background"
    bg_calib_present = False
    if bg_meta and bg_meta.get("present", False):
        bg_calib_present = True
    elif bg_dir.exists() and (bg_dir / "background_depth.npy").exists():
        bg_calib_present = True
    elif bg_meta and bg_meta.get("source_path"):
        bg_calib_present = True  # Linked external background

    # Check 3: Orientation and resolution
    orientation_check_pass = (orientation == "none")
    resolution_check_pass = (width == 1280 and height == 720)

    # Per-stop evaluation
    stop_reports: List[Dict[str, Any]] = []
    contact_tiles: List[np.ndarray] = []
    fatal_errors: List[str] = []
    warnings_list: List[str] = []
    floor_tilts_deg: List[float] = []

    # Open3D intrinsics for floor fitting if available
    try:
        import open3d as o3d
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            float(intrinsics.get("fx", max(width, height))),
            float(intrinsics.get("fy", max(width, height))),
            float(intrinsics.get("cx", width / 2.0)),
            float(intrinsics.get("cy", height / 2.0)),
        )
    except Exception:
        o3d_intrinsics = None

    cfg = FullBodyConfig()

    for s_idx in range(n_stops):
        s_folder = session_path / f"stop_{s_idx:02d}"
        s_report: Dict[str, Any] = {
            "stop_index": s_idx,
            "status": "FAIL",
            "frames_saved": 0,
            "frames_requested": frames_requested,
            "sha256_ok": False,
            "median_distance_m": None,
            "clipped": True,
            "valid_depth_pct": 0.0,
            "sway_mm": None,
            "floor_visible": False,
            "floor_residual_mm": None,
            "floor_tilt_deg": None,
            "reasons": [],
        }

        if not s_folder.exists():
            s_report["reasons"].append(f"Directory stop_{s_idx:02d} missing")
            stop_reports.append(s_report)
            fatal_errors.append(f"Stop {s_idx} folder missing")
            contact_tiles.append(np.zeros((270, 240, 3), dtype=np.uint8))
            continue

        stops_present_count += 1
        s_json = s_folder / "stop.json"
        if not s_json.exists():
            s_report["reasons"].append("stop.json missing")
            stop_reports.append(s_report)
            fatal_errors.append(f"Stop {s_idx} metadata missing")
            contact_tiles.append(np.zeros((270, 240, 3), dtype=np.uint8))
            continue

        try:
            with open(s_json, "r", encoding="utf-8") as sf:
                stop_data = json.load(sf)
        except Exception as exc:
            s_report["reasons"].append(f"Corrupt stop.json: {exc}")
            stop_reports.append(s_report)
            fatal_errors.append(f"Stop {s_idx} metadata corrupt")
            contact_tiles.append(np.zeros((270, 240, 3), dtype=np.uint8))
            continue

        accepted_attempt = stop_data.get("accepted_attempt")
        if accepted_attempt is None:
            # Check attempts if any was accepted
            attempts = stop_data.get("attempts", [])
            for att in attempts:
                if att.get("accepted", False):
                    accepted_attempt = att.get("attempt_index", 0)
                    break

        if accepted_attempt is None:
            s_report["reasons"].append("No accepted attempt")
            s_report["status"] = "FAIL"
            fatal_errors.append(f"Stop {s_idx} has no accepted attempt")
            stop_reports.append(s_report)
            contact_tiles.append(np.zeros((270, 240, 3), dtype=np.uint8))
            continue

        stops_accepted_count += 1
        attempt_dir = s_folder / f"attempt_{int(accepted_attempt):02d}"
        if not attempt_dir.exists():
            s_report["reasons"].append(f"Attempt folder {attempt_dir.name} missing")
            fatal_errors.append(f"Stop {s_idx} attempt dir missing")
            stop_reports.append(s_report)
            contact_tiles.append(np.zeros((270, 240, 3), dtype=np.uint8))
            continue

        # SHA256 integrity check
        manifest_f = attempt_dir / "files.sha256"
        sha_ok, sha_errs = verify_sha256_manifest(manifest_f)
        s_report["sha256_ok"] = sha_ok
        if not sha_ok:
            s_report["reasons"].extend(sha_errs)
            fatal_errors.append(f"Stop {s_idx} SHA256 integrity failed")

        # Frame counting
        color_files = sorted(list(attempt_dir.glob("raw_color_*.png")) + list(attempt_dir.glob("frame_*_color.png")))
        depth_files = sorted(list(attempt_dir.glob("raw_depth_*.npy")) + list(attempt_dir.glob("frame_*_depth.npy")))
        n_saved = min(len(color_files), len(depth_files))
        s_report["frames_saved"] = n_saved

        if n_saved < int(0.9 * frames_requested):
            s_report["reasons"].append(f"Frames saved ({n_saved}) < 90% of requested ({frames_requested})")
            fatal_errors.append(f"Stop {s_idx} saved only {n_saved}/{frames_requested} frames (< 90%)")
        elif n_saved < frames_requested:
            s_report["reasons"].append(f"Frames saved ({n_saved}) < 100% of requested ({frames_requested})")
            warnings_list.append(f"Stop {s_idx} saved {n_saved}/{frames_requested} frames (< 100%)")

        # Read quality.json
        q_file = attempt_dir / "quality.json"
        if q_file.exists():
            try:
                with open(q_file, "r", encoding="utf-8") as qf:
                    q_data = json.load(qf)
                s_report["sway_mm"] = float(q_data.get("sway_mm", q_data.get("motion_mm", 0.0)))
                s_report["valid_depth_pct"] = float(q_data.get("valid_depth_fraction", 0.0)) * 100.0
                clipped_flag = False
                if not is_object:
                    if q_data.get("head_clipped", False) or q_data.get("feet_clipped", False):
                        clipped_flag = True
                if float(q_data.get("border_clip_fraction", 0.0)) > 0.02:
                    clipped_flag = True
                s_report["clipped"] = clipped_flag
                if clipped_flag:
                    s_report["reasons"].append("Subject clipped by frame border")
                    fatal_errors.append(f"Stop {s_idx} subject clipped")
                if s_report["sway_mm"] is not None and s_report["sway_mm"] > 10.0:
                    s_report["reasons"].append(f"Sway {s_report['sway_mm']:.1f} mm > 10.0 mm")
                    warnings_list.append(f"Stop {s_idx} sway {s_report['sway_mm']:.1f} mm exceeds limit")
            except Exception as q_exc:
                s_report["reasons"].append(f"Could not read quality.json: {q_exc}")

        # Load representative middle frame
        rep_color = None
        rep_depth = None
        if color_files and depth_files:
            mid_idx = len(depth_files) // 2
            try:
                rep_color = cv2.imread(str(color_files[mid_idx]))
                rep_depth = np.load(str(depth_files[mid_idx])).astype(np.float32)
                scale = float(meta.get("depth_scale", meta.get("depth_storage", {}).get("depth_scale", 0.001)))
                units = meta.get("depth_units", meta.get("depth_storage", {}).get("units", "sensor_units"))
                if units == "sensor_units":
                    rep_depth = rep_depth * scale

                # Median distance
                val_depth = rep_depth[(rep_depth > 0.3) & (rep_depth < 3.8)]
                if len(val_depth) > 0:
                    s_report["median_distance_m"] = float(np.median(val_depth))

                # Floor fit
                if o3d_intrinsics is not None:
                    floor_plane = fit_floor_plane(rep_depth, o3d_intrinsics, cfg)
                    if floor_plane is not None:
                        s_report["floor_visible"] = True
                        s_report["floor_residual_mm"] = float(floor_plane.rmse_m * 1000.0)
                        # Compute tilt from expected vertical up [0, -1, 0]
                        normal = floor_plane.normal
                        cos_up = np.dot(normal, np.array([0.0, -1.0, 0.0]))
                        tilt_deg = float(np.rad2deg(np.arccos(np.clip(abs(cos_up), -1.0, 1.0))))
                        s_report["floor_tilt_deg"] = tilt_deg
                        floor_tilts_deg.append(tilt_deg)
                        if tilt_deg > 2.0:
                            s_report["reasons"].append(f"Floor normal tilted {tilt_deg:.1f} deg (> 2.0 deg)")
                            warnings_list.append(f"Stop {s_idx} camera tilt {tilt_deg:.1f} deg")
            except Exception as rep_exc:
                logger.debug(f"Stop {s_idx} frame analysis failed: {rep_exc}")

        # Status for this stop
        if s_report["reasons"]:
            if any(
                "clipped" in r.lower()
                or "missing" in r.lower()
                or "sha256" in r.lower()
                or "< 90%" in r.lower()
                for r in s_report["reasons"]
            ):
                s_report["status"] = "FAIL"
            else:
                s_report["status"] = "WARN"
        else:
            s_report["status"] = "PASS"

        stop_reports.append(s_report)

        # Build contact sheet tile (240x270: 240x135 RGB top, 240x135 Depth bottom)
        tile_w, tile_h_half = 240, 135
        if rep_color is not None and rep_depth is not None:
            rgb_tile = cv2.resize(rep_color, (tile_w, tile_h_half), interpolation=cv2.INTER_AREA)
            depth_col = colorize_depth(rep_depth)
            depth_tile = cv2.resize(depth_col, (tile_w, tile_h_half), interpolation=cv2.INTER_NEAREST)
        else:
            rgb_tile = np.zeros((tile_h_half, tile_w, 3), dtype=np.uint8)
            depth_tile = np.zeros((tile_h_half, tile_w, 3), dtype=np.uint8)

        # Overlay text
        status_color = (0, 255, 0) if s_report["status"] == "PASS" else ((0, 200, 255) if s_report["status"] == "WARN" else (0, 0, 255))
        cv2.putText(rgb_tile, f"Stop {s_idx} ({s_report['status']})", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
        dist_txt = f"{s_report['median_distance_m']:.2f}m" if s_report['median_distance_m'] else "--"
        cv2.putText(rgb_tile, f"d={dist_txt}", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        tile_combined = np.vstack([rgb_tile, depth_tile])
        # Border between tiles
        cv2.rectangle(tile_combined, (0, 0), (tile_w - 1, tile_h_half * 2 - 1), (50, 50, 50), 1)
        contact_tiles.append(tile_combined)

    # Combine into 1-row contact sheet
    if contact_tiles:
        contact_sheet = np.hstack(contact_tiles)
        contact_path = session_path / "contact_sheet.png"
        cv2.imwrite(str(contact_path), contact_sheet)
    else:
        contact_path = None

    # Camera tilt evaluation across stops
    mean_tilt_deg = float(np.mean(floor_tilts_deg)) if floor_tilts_deg else None
    camera_level_pass = True
    if mean_tilt_deg is not None and mean_tilt_deg > 2.0:
        camera_level_pass = False
        warnings_list.append(f"Camera tilted {mean_tilt_deg:.1f} deg (> 2.0 deg)")

    all_stops_pass = (stops_accepted_count == n_stops == 8)
    if not all_stops_pass:
        fatal_errors.append(f"Only {stops_accepted_count} of 8 stops accepted")

    if not bg_calib_present:
        warnings_list.append("Background calibration missing or unlinked")

    # Overall Verdict
    if fatal_errors or any(s["status"] == "FAIL" for s in stop_reports):
        verdict = "FAIL"
    elif warnings_list or any(s["status"] == "WARN" for s in stop_reports):
        verdict = "WARN"
    else:
        verdict = "PASS"

    # Write verify_report.json
    report_doc = {
        "verdict": verdict,
        "subject_id": subject_id,
        "subject_type": subject_type,
        "n_stops": n_stops,
        "stops_accepted": stops_accepted_count,
        "protocol_version": meta.get("protocol_version", "stop-and-go-v1"),
        "session_checks": {
            "all_stops_accepted": all_stops_pass,
            "background_present": bg_calib_present,
            "camera_level": camera_level_pass,
            "mean_floor_tilt_deg": mean_tilt_deg,
            "orientation_landscape": orientation_check_pass,
            "resolution_1280x720": resolution_check_pass,
        },
        "stop_reports": stop_reports,
        "fatal_errors": fatal_errors,
        "warnings": warnings_list,
        "contact_sheet_saved": str(contact_path.name) if contact_path else None,
    }

    report_json_path = session_path / "verify_report.json"
    with open(report_json_path, "w", encoding="utf-8") as rf:
        json.dump(report_doc, rf, indent=2)

    # Print Formatted Verification Table
    print("\n" + "=" * 98)
    print(f"SESSION VERIFICATION REPORT: {session_path}")
    print("=" * 98)
    print(f"{'Stop':<5} | {'Status':<6} | {'Frames (Saved/Req)':<18} | {'SHA256':<7} | {'Dist (m)':<8} | {'Clipped':<7} | {'Valid %':<7} | {'Sway (mm)':<9} | {'Floor Res (mm)':<14}")
    print("-" * 98)
    for s in stop_reports:
        s_idx = s["stop_index"]
        st = s["status"]
        fr_str = f"{s['frames_saved']} / {s['frames_requested']}"
        sha_str = "OK" if s["sha256_ok"] else "FAIL"
        d_str = f"{s['median_distance_m']:.2f}" if s["median_distance_m"] is not None else "--"
        cl_str = "YES" if s["clipped"] else "No"
        v_str = f"{s['valid_depth_pct']:.1f}%"
        sw_str = f"{s['sway_mm']:.1f}" if s["sway_mm"] is not None else "--"
        fl_str = f"{s['floor_residual_mm']:.1f}" if s["floor_residual_mm"] is not None else "--"
        print(f"{s_idx:<5} | {st:<6} | {fr_str:<18} | {sha_str:<7} | {d_str:<8} | {cl_str:<7} | {v_str:<7} | {sw_str:<9} | {fl_str:<14}")
    print("-" * 98)

    print("SESSION CHECKS:")
    st_pass = "PASS" if all_stops_pass else "FAIL"
    print(f"  - All 8 stops present and accepted: {st_pass} ({stops_accepted_count}/{n_stops})")

    bg_pass = "PASS" if bg_calib_present else "WARN"
    print(f"  - Background calibration:           {bg_pass} ({'present' if bg_calib_present else 'missing'})")

    if mean_tilt_deg is not None:
        tilt_pass = "PASS" if camera_level_pass else "WARN"
        print(f"  - Camera level:                     {tilt_pass} (floor normal tilt {mean_tilt_deg:.1f} deg <= 2.0 deg)")
    else:
        print(f"  - Camera level:                     N/A (no floor plane detected)")

    res_pass = "PASS" if (orientation_check_pass and resolution_check_pass) else "FAIL"
    orient_str = "landscape" if orientation_check_pass else f"portrait({orientation})"
    print(f"  - Camera orientation & resolution:  {res_pass} ({orient_str}, {width}x{height})")

    print("-" * 98)
    print(f"OVERALL VERDICT: {verdict}")
    if contact_path and contact_path.exists():
        print(f"Contact sheet saved to: {contact_path}")
    print(f"Verification report saved to: {report_json_path}")
    print("=" * 98 + "\n")

    return 0 if verdict in ("PASS", "WARN") else 1
