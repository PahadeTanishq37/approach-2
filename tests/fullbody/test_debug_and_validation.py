"""
tests/fullbody/test_debug_and_validation.py

Unit and integration tests for:
1. fullbody debug-stop (mask_overlay.png, depth_diff.png, cloud.ply, stats.json)
2. fullbody validate-session (computed numbers only, leave-one-out, floor stability)
3. Ground truth handling (refusal to print PASS when ground truth is null)
4. Leave-one-stop-out cross-validation without mesh raycasts
5. Report generator (refusal of missing fields)
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import cv2
import numpy as np
import open3d as o3d
import pytest

from realsense_measure.fullbody.cli import main
from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.debug import debug_stop
from realsense_measure.fullbody.types import MeasurementResult
from realsense_measure.fullbody.validate import (
    REQUIRED_SESSION_FIELDS,
    compare_with_tape,
    format_validation_table,
    generate_session_report,
    load_tape_measurements,
    validate_session,
)


@pytest.fixture(scope="module")
def fake_capture_session(tmp_path_factory):
    """Create a fast 4-stop fake capture session with background floor calibration."""
    base_dir = tmp_path_factory.mktemp("fake_session_suite")
    bg_dir = base_dir / "bg_calib"
    bg_argv = [
        "capture-background",
        "--out", str(bg_dir),
        "--burst-frames", "2",
        "--fake",
    ]
    assert main(bg_argv) == 0

    sess_dir = base_dir / "sess_4stops"
    argv = [
        "capture",
        "--out", str(sess_dir),
        "--subject-id", "SUBJ_TEST01",
        "--stops", "4",
        "--fake",
        "--burst-frames", "2",
        "--turn-prompt-timeout", "0.1",
        "--hold-still-s", "0.1",
        "--background", str(bg_dir),
        "--headless",
        "--no-head-feet-check",
    ]
    assert main(argv) == 0
    return sess_dir, bg_dir


def test_debug_stop_generates_all_artifacts_and_pure_data_stats(fake_capture_session):
    """Verify debug-stop generates mask_overlay, depth_diff, cloud.ply, and pure data stats.json."""
    sess_dir, _ = fake_capture_session
    out_dbg = sess_dir / "debug" / "stop_00"

    # Run via CLI
    argv = [
        "debug-stop",
        str(sess_dir),
        "--stop", "0",
    ]
    assert main(argv) == 0
    assert out_dbg.exists()

    # 1. mask_overlay.png
    mask_png = out_dbg / "mask_overlay.png"
    assert mask_png.exists()
    img_overlay = cv2.imread(str(mask_png))
    assert img_overlay is not None
    assert img_overlay.shape[0] > 0 and img_overlay.shape[1] > 0

    # 2. depth_diff.png
    diff_png = out_dbg / "depth_diff.png"
    assert diff_png.exists()
    img_diff = cv2.imread(str(diff_png))
    assert img_diff is not None
    assert img_diff.shape[0] > 0 and img_diff.shape[1] > 0

    # 3. cloud.ply
    ply_file = out_dbg / "cloud.ply"
    assert ply_file.exists()
    pcd = o3d.io.read_point_cloud(str(ply_file))
    assert len(pcd.points) > 0

    # 4. stats.json
    stats_json_p = out_dbg / "stats.json"
    assert stats_json_p.exists()
    with open(stats_json_p, "r", encoding="utf-8") as f:
        stats = json.load(f)

    # Check required data-derived keys
    assert stats["stop_id"] == 0
    assert "foreground_px_count" in stats and stats["foreground_px_count"] > 0
    assert "bbox" in stats and len(stats["bbox"]) == 4
    assert "lowest_foreground_row" in stats
    assert "floor_contact_row_estimate" in stats
    assert "lowest_foreground_row_vs_floor_contact" in stats
    assert "torso_median_depth_m" in stats and stats["torso_median_depth_m"] > 0.0
    assert "torso_temporal_std_mm" in stats
    assert "floor_rmse_mm" in stats
    assert "floor_pixel_leakage_in_bottom_8pct" in stats
    leakage = stats["floor_pixel_leakage_in_bottom_8pct"]
    assert "leakage_pixel_count" in leakage
    assert "total_pixels_in_bottom_8pct" in leakage
    assert "leakage_fraction" in leakage
    assert "points_per_height_band" in stats and len(stats["points_per_height_band"]) == 10


def test_validate_session_computed_numbers_and_report(fake_capture_session):
    """Verify validate-session computes pure data numbers, runs leave-one-out, and generates report."""
    sess_dir, _ = fake_capture_session

    argv = [
        "validate-session",
        str(sess_dir),
    ]
    assert main(argv) == 0

    val_json_p = sess_dir / "validate_session.json"
    assert val_json_p.exists()
    with open(val_json_p, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Verify computed numbers
    assert summary["stop_count"] == 4
    assert len(summary["points_per_stop"]) == 4
    assert all(p > 0 for p in summary["points_per_stop"])
    assert len(summary["body_depth_noise_mm"]) == 4
    assert "registration_edge_table" in summary and len(summary["registration_edge_table"]) >= 4
    assert "loop_closure_residual_deg" in summary
    assert "loop_closure_residual_mm" in summary
    assert summary["fused_point_count"] > 0
    assert "mesh_watertight" in summary
    assert "hole_fraction" in summary
    assert len(summary["coverage_by_height_band"]) == 10
    assert "floor_metrics" in summary
    assert "leave_one_stop_out_errors_mm" in summary

    # Leave-one-out metrics
    loso = summary["leave_one_stop_out_errors_mm"]
    assert "stops" in loso and len(loso["stops"]) == 4
    assert "overall_mean_rmse_mm" in loso
    for s_res in loso["stops"]:
        assert "mean_error_mm" in s_res
        assert "median_error_mm" in s_res
        assert "rmse_mm" in s_res
        assert "overlap_pixel_count" in s_res

    # Check generated markdown report
    rep_md_p = sess_dir / "validation_report.md"
    assert rep_md_p.exists()
    content = rep_md_p.read_text(encoding="utf-8")
    assert "# Full-Body Scan Validation Report:" in content
    assert "Registration & Loop Closure" in content
    assert "Leave-One-Stop-Out Cross-Validation" in content


def test_ground_truth_never_prints_pass_when_null():
    """Verify ground truth comparison refuses to claim PASS when measurements are null."""
    scan_measurements = {
        "stature": MeasurementResult(name="stature", value_mm=1750.0, method="slice_contour", coverage=1.0, interpolated_fraction=0.0, quality_flag="OK"),
        "chest_circumference": MeasurementResult(name="chest_circumference", value_mm=890.0, method="slice_contour", coverage=0.9, interpolated_fraction=0.0, quality_flag="OK"),
    }

    # All null tape measurements (default template)
    tape_vals = {}
    report = compare_with_tape(scan_measurements, tape_vals)

    assert report["has_ground_truth"] is False
    assert report["is_pass"] is False
    assert report["status"] == "UNVERIFIED (ground truth is null)"
    assert report["mean_absolute_error_mm"] is None

    table_str = format_validation_table(report)
    assert "UNVERIFIED" in table_str
    assert "PASS" not in table_str


def test_ground_truth_evaluates_when_measurements_provided(tmp_path):
    """Verify ground truth comparison calculates MAE and PASS when physical values exist."""
    scan_measurements = {
        "stature": MeasurementResult(name="stature", value_mm=1755.0, method="slice_contour", coverage=1.0, interpolated_fraction=0.0, quality_flag="OK"),
        "chest_circumference": MeasurementResult(name="chest_circumference", value_mm=890.0, method="slice_contour", coverage=0.9, interpolated_fraction=0.0, quality_flag="OK"),
    }

    tape_json = tmp_path / "tape_valid.json"
    data = {
        "subject_id": "SUBJ_01",
        "measurements_mm": {
            "stature": 1750.0,
            "chest_circumference": 885.0,
        }
    }
    tape_json.write_text(json.dumps(data), encoding="utf-8")

    tape_vals, rep_floor, subj_id = load_tape_measurements(tape_json, subject_id="SUBJ_01")
    report = compare_with_tape(scan_measurements, tape_vals, rep_floor)

    assert report["has_ground_truth"] is True
    assert report["num_compared"] == 2
    assert report["mean_absolute_error_mm"] == pytest.approx(5.0)
    assert report["is_pass"] is True
    assert report["status"] == "PASS"

    table_str = format_validation_table(report)
    assert "Summary [PASS]" in table_str


def test_report_generator_refuses_missing_fields():
    """Verify report generator refuses generation and raises KeyError if any required field is missing."""
    complete_summary = {
        "session_id": "SESS_TEST",
        "timestamp": "2026-11-18T12:00:00Z",
        "stop_count": 4,
        "points_per_stop": [1000, 1000, 1000, 1000],
        "body_depth_noise_mm": [2.5, 2.4, 2.6, 2.5],
        "registration_edge_table": [
            {"edge": "(0, 1)", "fitness": 1.0, "rmse_mm": 15.0, "yaw_correction_deg": 0.5, "source": "icp", "dominant": "icp"}
        ],
        "loop_closure_residual_deg": 0.1,
        "loop_closure_residual_mm": 1.2,
        "fused_point_count": 50000,
        "mesh_watertight": True,
        "hole_fraction": 0.01,
        "coverage_by_height_band": [{"band": "0.0-0.2m", "point_count": 100, "percentage": 2.0}],
        "floor_metrics": {
            "camera_height_m": 0.95, "pitch_deg": 1.0, "roll_deg": 0.5, "normal_spread_deg": 0.0, "height_spread_mm": 0.0
        },
        "leave_one_stop_out_errors_mm": {"stops": [], "overall_mean_rmse_mm": 5.2},
        "ground_truth_status": {"status": "UNVERIFIED (ground truth is null)", "has_ground_truth": False, "num_compared": 0},
    }

    # Should generate cleanly with all fields present
    report_md = generate_session_report(complete_summary)
    assert "# Full-Body Scan Validation Report: SESS_TEST" in report_md

    # Missing a required field must raise KeyError
    for f in REQUIRED_SESSION_FIELDS:
        corrupted = dict(complete_summary)
        del corrupted[f]
        with pytest.raises(KeyError, match=f"missing required field '{f}'"):
            generate_session_report(corrupted)
