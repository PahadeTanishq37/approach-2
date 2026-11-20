"""
tests/fullbody/test_stop_and_go_v1.py

Test suite for stop-and-go-v1 capture protocol:
- Subject ID validation for test/rigid objects (BUCKET_01, OBJ_01, etc.)
- 60-frame burst saving and verification with FakeCamera
- Object mode gating (short/compact object passes without person-height or sole-margin requirement)
- Rig and protocol metadata persistence and round-trip in meta.json
- verify-session subcommand (PASS on good session, FAIL on missing stop, FAIL on clipped stop)
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Dict, List

import cv2
import numpy as np
import pytest

from realsense_measure.body.capture_session import validate_anonymous_subject_id
from realsense_measure.fullbody.capture_controller import (
    CaptureController,
    CaptureThresholds,
    ControllerState,
    default_quality_fn,
)
from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.session_format import SessionWriter
from realsense_measure.fullbody.verify_session import verify_session
from tests.fullbody.fake_camera import FakeCamera


def test_subject_id_accepts_bucket_and_rigid_objects():
    """Verify subject_id validator accepts BUCKET_01, TEST_42, OBJ_01, CYLINDER_99 while rejecting PII."""
    assert validate_anonymous_subject_id("BUCKET_01") == "BUCKET_01"
    assert validate_anonymous_subject_id("bucket_01") == "bucket_01"
    assert validate_anonymous_subject_id("TEST_42") == "TEST_42"
    assert validate_anonymous_subject_id("OBJ_99") == "OBJ_99"
    assert validate_anonymous_subject_id("OBJECT_A") == "OBJECT_A"
    assert validate_anonymous_subject_id("CYLINDER_05") == "CYLINDER_05"
    assert validate_anonymous_subject_id("S02") == "S02"

    with pytest.raises(ValueError):
        validate_anonymous_subject_id("John Doe")
    with pytest.raises(ValueError):
        validate_anonymous_subject_id("user@example.com")


def test_60_frame_burst_saving_fake_camera():
    """Verify 60 raw color PNG and 60 depth NPY frames are written per burst without drops."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "burst_60"
        cam = FakeCamera(width=320, height=240, depth_scale=0.001, depth_units="sensor_units")
        cam.start()

        writer = SessionWriter(
            root=root,
            subject_id="BUCKET_01",
            n_stops=1,
            turn_direction="cw",
            camera_portrait_rotation="none",
            intrinsics=cam.intrinsics,
            depth_scale=cam.depth_scale,
            depth_units=cam.depth_units,
            frames_requested=60,
        )

        thresholds = CaptureThresholds(
            burst_frames=60,
            stillness_frames=3,
            turn_prompt_timeout_s=0.05,
            burst_timeout_s=5.0,
            allow_missing_background=True,
        )

        controller = CaptureController(
            camera=cam,
            writer=writer,
            thresholds=thresholds,
            bg_depth=None,
        )

        now = 0.0
        # Step through controller until 1 stop is accepted and finalized
        for _ in range(500):
            st = controller.step(now)
            now += 0.033
            if st == ControllerState.DONE:
                break

        assert controller.state == ControllerState.DONE
        attempt_dir = root / "stop_00" / "attempt_00"
        assert attempt_dir.exists()

        colors = list(attempt_dir.glob("raw_color_*.png"))
        depths = list(attempt_dir.glob("raw_depth_*.npy"))
        assert len(colors) == 60
        assert len(depths) == 60

        q_json = attempt_dir / "quality.json"
        assert q_json.exists()
        with open(q_json, "r", encoding="utf-8") as qf:
            q_data = json.load(qf)
        assert q_data["frames_requested"] == 60
        assert q_data["frames_saved"] == 60
        assert not q_data["is_flagged"]


def test_object_mode_passes_short_object():
    """Verify object mode passes a small object that fails person height and margin checks."""
    h, w = 480, 640
    d_m = np.zeros((h, w), dtype=np.float32)
    # Cylinder object near bottom: rows 350 to 470 (height 121px = 25.2% < 60%),
    # sole margin = (479 - 470)/480 = 1.8% < 4%, doesn't touch bottom row 479
    d_m[350:471, 270:370] = 2.00
    bg_depth = np.full((h, w), 3.20, dtype=np.float32)

    # 1. In person mode: fails in quality check because sole margin < 4%
    th_person = CaptureThresholds(burst_frames=10, subject_type="person", dist_min_m=1.70, dist_max_m=2.40)
    rep_person = default_quality_fn(
        [np.zeros((h, w, 3), dtype=np.uint8)] * 10,
        [d_m] * 10,
        depth_scale=1.0,
        depth_units="m",
        thresholds=th_person,
        bg_depth=bg_depth,
    )
    assert rep_person["is_flagged"]
    assert any("Sole margin" in r for r in rep_person["reasons"])

    # 2. In object mode: passes because object is completely inside frame and in distance window
    th_object = CaptureThresholds(burst_frames=10, subject_type="object", dist_min_m=1.70, dist_max_m=2.40)
    rep_object = default_quality_fn(
        [np.zeros((h, w, 3), dtype=np.uint8)] * 10,
        [d_m] * 10,
        depth_scale=1.0,
        depth_units="m",
        thresholds=th_object,
        bg_depth=bg_depth,
    )
    assert not rep_object["is_flagged"]
    assert rep_object["border_clip_fraction"] == 0.0

    # 3. In object mode, touching bottom border triggers clipping flag
    d_m_clipped = np.zeros((h, w), dtype=np.float32)
    d_m_clipped[380:480, 270:370] = 2.00  # touches bottom row 479
    rep_clipped = default_quality_fn(
        [np.zeros((h, w, 3), dtype=np.uint8)] * 10,
        [d_m_clipped] * 10,
        depth_scale=1.0,
        depth_units="m",
        thresholds=th_object,
        bg_depth=bg_depth,
    )
    assert rep_clipped["is_flagged"]
    assert rep_clipped["border_clip_fraction"] > 0.0


def test_rig_and_protocol_metadata_roundtrip():
    """Verify rig metadata and protocol fields are stored in meta.json and loaded properly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "rig_meta_test"
        cam = FakeCamera(width=320, height=240, depth_scale=0.001, depth_units="sensor_units")

        writer = SessionWriter(
            root=root,
            subject_id="BUCKET_01",
            n_stops=8,
            turn_direction="cw",
            camera_portrait_rotation="none",
            intrinsics=cam.intrinsics,
            depth_scale=cam.depth_scale,
            depth_units=cam.depth_units,
            cam_height_m=1.02,
            cam_distance_m=2.05,
            cam_tilt_deg=0.5,
            clothing="none",
            pose="A-pose",
            notes="rig calibration test",
            subject_type="object",
            object_diameter_mm=290.0,
            protocol_version="stop-and-go-v1",
            frames_requested=60,
        )

        # Create 8 dummy accepted stops
        for s_idx in range(8):
            handle = writer.begin_attempt(s_idx)
            colors = [np.full((240, 320, 3), 50, dtype=np.uint8) for _ in range(60)]
            depths = [np.full((240, 320), 2000, dtype=np.uint16) for _ in range(60)]
            ts = [i * 0.033 for i in range(60)]
            writer.write_burst(handle, colors, depths, ts)
            writer.write_quality(handle, {"is_flagged": False, "sway_mm": 1.2, "foreground_px": 5000})
            writer.finish_attempt(handle, accepted=True, reason="Pass", sway_mm=1.2)

        meta_path = writer.finalize()
        assert meta_path.exists()

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["protocol_version"] == "stop-and-go-v1"
        assert meta["subject_type"] == "object"
        assert meta["object_diameter_mm"] == 290.0
        assert meta["frames_requested"] == 60
        assert meta["nominal_stop_angles"] == [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
        rig = meta["rig_metadata"]
        assert rig["cam_height_m"] == 1.02
        assert rig["cam_distance_m"] == 2.05
        assert rig["cam_tilt_deg"] == 0.5
        assert rig["clothing"] == "none"
        assert rig["pose"] == "A-pose"
        assert rig["notes"] == "rig calibration test"


def test_verify_session_good_missing_and_clipped():
    """Verify verify_session returns PASS for valid 8-stop session and FAIL for broken sessions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)

        # 1. Good 8-stop session
        good_dir = base_dir / "good_session"
        cam = FakeCamera(width=1280, height=720, depth_scale=0.001, depth_units="sensor_units")
        writer = SessionWriter(
            root=good_dir,
            subject_id="S01",
            n_stops=8,
            turn_direction="cw",
            camera_portrait_rotation="none",
            intrinsics=cam.intrinsics,
            depth_scale=cam.depth_scale,
            depth_units=cam.depth_units,
            frames_requested=60,
        )

        for s_idx in range(8):
            handle = writer.begin_attempt(s_idx)
            # Create dummy color and depth with centered subject
            col = np.full((720, 1280, 3), 40, dtype=np.uint8)
            dep = np.full((720, 1280), 2000, dtype=np.uint16)
            # Leave borders zeroed out
            dep[0:20, :] = 0
            dep[-20:, :] = 0
            dep[:, 0:20] = 0
            dep[:, -20:] = 0
            writer.write_burst(handle, [col] * 60, [dep] * 60, [i * 0.033 for i in range(60)])
            writer.write_quality(handle, {
                "is_flagged": False,
                "sway_mm": 2.1,
                "valid_depth_fraction": 0.45,
                "head_clipped": False,
                "feet_clipped": False,
                "border_clip_fraction": 0.0,
                "foreground_px": 50000,
            })
            writer.finish_attempt(handle, accepted=True, reason="Pass", sway_mm=2.1)
        writer.finalize()

        code = verify_session(good_dir)
        assert code == 0
        rep_file = good_dir / "verify_report.json"
        assert rep_file.exists()
        sheet_file = good_dir / "contact_sheet.png"
        assert sheet_file.exists()
        with open(rep_file, "r", encoding="utf-8") as rf:
            rep = json.load(rf)
        assert rep["verdict"] in ("PASS", "WARN")

        # 2. Missing stop session
        missing_dir = base_dir / "missing_session"
        writer_m = SessionWriter(
            root=missing_dir,
            subject_id="S02",
            n_stops=8,
            turn_direction="cw",
            camera_portrait_rotation="none",
            intrinsics=cam.intrinsics,
            depth_scale=cam.depth_scale,
            depth_units=cam.depth_units,
            frames_requested=60,
        )
        for s_idx in range(7):  # Only 7 stops
            handle = writer_m.begin_attempt(s_idx)
            writer_m.write_burst(handle, [col] * 60, [dep] * 60, [i * 0.033 for i in range(60)])
            writer_m.write_quality(handle, {"is_flagged": False, "sway_mm": 1.0, "foreground_px": 50000})
            writer_m.finish_attempt(handle, accepted=True, reason="Pass", sway_mm=1.0)
        # Note: finalize would fail, but verify-session must catch missing stop and return 1 (FAIL)
        # Manually write meta.json
        meta_m = {
            "schema_version": 2,
            "layout": "stops_v2",
            "subject_id": "S02",
            "n_stops": 8,
            "intrinsics": cam.intrinsics,
            "camera_portrait_rotation": "none",
            "frames_requested": 60,
        }
        with open(missing_dir / "meta.json", "w", encoding="utf-8") as mf:
            json.dump(meta_m, mf)

        code_m = verify_session(missing_dir)
        assert code_m == 1
        with open(missing_dir / "verify_report.json", "r", encoding="utf-8") as rf:
            rep_m = json.load(rf)
        assert rep_m["verdict"] == "FAIL"

        # 3. Clipped stop session
        clipped_dir = base_dir / "clipped_session"
        writer_c = SessionWriter(
            root=clipped_dir,
            subject_id="S03",
            n_stops=8,
            turn_direction="cw",
            camera_portrait_rotation="none",
            intrinsics=cam.intrinsics,
            depth_scale=cam.depth_scale,
            depth_units=cam.depth_units,
            frames_requested=60,
        )
        for s_idx in range(8):
            handle = writer_c.begin_attempt(s_idx)
            writer_c.write_burst(handle, [col] * 60, [dep] * 60, [i * 0.033 for i in range(60)])
            # Mark stop 3 as head_clipped
            is_clipped = (s_idx == 3)
            writer_c.write_quality(handle, {
                "is_flagged": is_clipped,
                "sway_mm": 1.0,
                "head_clipped": is_clipped,
                "feet_clipped": False,
                "border_clip_fraction": 0.05 if is_clipped else 0.0,
                "foreground_px": 50000,
            })
            writer_c.finish_attempt(handle, accepted=True, reason="Quality check passed", sway_mm=1.0)
        writer_c.finalize()

        code_c = verify_session(clipped_dir)
        assert code_c == 1
        with open(clipped_dir / "verify_report.json", "r", encoding="utf-8") as rf:
            rep_c = json.load(rf)
        assert rep_c["verdict"] == "FAIL"


def test_verify_session_56_frames_warn():
    """Verify session with 56/60 frames gives WARN (>=90% and <100%) and exits 0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sess_dir = Path(tmpdir) / "sess_56"
        cam = FakeCamera(width=1280, height=720, depth_scale=0.001, depth_units="sensor_units")
        writer = SessionWriter(
            root=sess_dir,
            subject_id="S56",
            n_stops=8,
            turn_direction="cw",
            camera_portrait_rotation="none",
            intrinsics=cam.intrinsics,
            depth_scale=cam.depth_scale,
            depth_units=cam.depth_units,
            frames_requested=60,
        )

        col = np.full((720, 1280, 3), 40, dtype=np.uint8)
        dep = np.full((720, 1280), 2000, dtype=np.uint16)
        dep[0:20, :] = 0
        dep[-20:, :] = 0
        dep[:, 0:20] = 0
        dep[:, -20:] = 0

        for s_idx in range(8):
            handle = writer.begin_attempt(s_idx)
            # Write 56 frames instead of 60
            n_frames = 56 if s_idx == 0 else 60
            writer.write_burst(handle, [col] * n_frames, [dep] * n_frames, [i * 0.033 for i in range(n_frames)])
            writer.write_quality(handle, {
                "is_flagged": False,
                "sway_mm": 1.5,
                "valid_depth_fraction": 0.40,
                "head_clipped": False,
                "feet_clipped": False,
                "border_clip_fraction": 0.0,
                "foreground_px": 45000,
            })
            writer.finish_attempt(handle, accepted=True, reason="Pass", sway_mm=1.5)
        writer.finalize()

        code = verify_session(sess_dir)
        assert code == 0
        with open(sess_dir / "verify_report.json", "r", encoding="utf-8") as rf:
            rep = json.load(rf)
        assert rep["verdict"] == "WARN"
        assert rep["stop_reports"][0]["status"] == "WARN"
        assert rep["stop_reports"][0]["frames_saved"] == 56
        assert any("< 100%" in r for r in rep["stop_reports"][0]["reasons"])


def test_verify_session_floor_residual_and_3deg_tilt_warn():
    """Verify FakeCamera floor plane reports residual and flags 3 deg tilt as WARN."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Case A: Level camera (0 deg tilt)
        dir_0 = Path(tmpdir) / "floor_0deg"
        cam_0 = FakeCamera(width=1280, height=720, floor=True, floor_tilt_deg=0.0)
        cam_0.start()
        writer_0 = SessionWriter(
            root=dir_0,
            subject_id="S00",
            n_stops=8,
            turn_direction="cw",
            camera_portrait_rotation="none",
            intrinsics=cam_0.intrinsics,
            depth_scale=cam_0.depth_scale,
            depth_units=cam_0.depth_units,
            frames_requested=60,
        )
        c_frame, d_raw, _, _ = cam_0.read_raw_filtered()
        for s_idx in range(8):
            h = writer_0.begin_attempt(s_idx)
            writer_0.write_burst(h, [c_frame] * 60, [d_raw] * 60, [i * 0.033 for i in range(60)])
            writer_0.write_quality(h, {"is_flagged": False, "sway_mm": 1.0, "valid_depth_fraction": 0.3})
            writer_0.finish_attempt(h, accepted=True, reason="Pass", sway_mm=1.0)
        writer_0.finalize()

        code_0 = verify_session(dir_0)
        assert code_0 == 0
        with open(dir_0 / "verify_report.json", "r", encoding="utf-8") as rf:
            rep_0 = json.load(rf)
        s0 = rep_0["stop_reports"][0]
        assert s0["floor_visible"] is True
        assert s0["floor_residual_mm"] is not None
        assert s0["floor_residual_mm"] < 10.0  # Clean planar fit
        assert rep_0["session_checks"]["camera_level"] is True

        # Case B: 3 deg tilt camera
        dir_3 = Path(tmpdir) / "floor_3deg"
        cam_3 = FakeCamera(width=1280, height=720, floor=True, floor_tilt_deg=3.0)
        cam_3.start()
        writer_3 = SessionWriter(
            root=dir_3,
            subject_id="S03",
            n_stops=8,
            turn_direction="cw",
            camera_portrait_rotation="none",
            intrinsics=cam_3.intrinsics,
            depth_scale=cam_3.depth_scale,
            depth_units=cam_3.depth_units,
            frames_requested=60,
        )
        c_frame3, d_raw3, _, _ = cam_3.read_raw_filtered()
        for s_idx in range(8):
            h = writer_3.begin_attempt(s_idx)
            writer_3.write_burst(h, [c_frame3] * 60, [d_raw3] * 60, [i * 0.033 for i in range(60)])
            writer_3.write_quality(h, {"is_flagged": False, "sway_mm": 1.0, "valid_depth_fraction": 0.3})
            writer_3.finish_attempt(h, accepted=True, reason="Pass", sway_mm=1.0)
        writer_3.finalize()

        code_3 = verify_session(dir_3)
        assert code_3 == 0  # WARN still exits 0
        with open(dir_3 / "verify_report.json", "r", encoding="utf-8") as rf:
            rep_3 = json.load(rf)
        s3 = rep_3["stop_reports"][0]
        assert s3["floor_visible"] is True
        assert s3["floor_residual_mm"] is not None
        assert abs(s3["floor_tilt_deg"] - 3.0) < 0.5  # Matches ~3 deg
        assert rep_3["session_checks"]["camera_level"] is False
        assert rep_3["verdict"] == "WARN"
