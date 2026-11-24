"""
tests/fullbody/test_stage27_adapter_ui_cli.py

Unit tests for Stage 27 Change 5:
- RealSenseCaptureAdapter lifecycle, warmup frame discarding, dropped frame retry,
  disconnect handling, decimation refusal.
- full_body_camera_config parameters.
- capture_ui render_frame pure headless HUD rendering.
- cli import isolation (no open3d or pyrealsense2 in sys.modules).
- cli capture --fake end-to-end and failure handling.
- cli check-camera --fake snapshot output.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from realsense_measure.config import CameraConfig
from realsense_measure.fullbody.capture_camera import RealSenseCaptureAdapter, full_body_camera_config
from realsense_measure.fullbody.capture_controller import ControllerState
from realsense_measure.fullbody.capture_ui import render_frame
from realsense_measure.fullbody.cli import main
from realsense_measure.fullbody.config import FullBodyConfig
from realsense_measure.fullbody.io import load_capture_session


def test_full_body_camera_config():
    """Verify full_body_camera_config sets required Stage 27 parameters."""
    cfg_cw = full_body_camera_config("cw", resolution="848x480", fps=30)
    assert cfg_cw.width == 848
    assert cfg_cw.height == 480
    assert cfg_cw.fps == 30
    assert cfg_cw.depth_max_m >= 4.0
    assert cfg_cw.depth_min_m == pytest.approx(0.40)
    assert cfg_cw.use_decimation is False
    assert cfg_cw.rotate_90 == "cw"

    cfg_ccw = full_body_camera_config("ccw")
    assert cfg_ccw.rotate_90 == "ccw"
    assert cfg_ccw.use_decimation is False

    with pytest.raises(ValueError, match="Invalid portrait_rotation"):
        full_body_camera_config("invalid")


def test_adapter_warmup_and_raw_depth_units():
    """Verify adapter drops warmup frames and returns uint16 raw depth with correct depth_scale."""
    mock_cam = MagicMock()
    mock_cam.depth_scale = 0.001

    class MockIntr:
        width = 480
        height = 848
        def get_focal_length(self):
            return 425.0, 425.0
        def get_principal_point(self):
            return 240.0, 424.0

    mock_cam.intrinsics_o3d = MockIntr()
    mock_cam.get_device_info.return_value = {
        "name": "Mock D455",
        "serial_number": "MOCK_123",
        "firmware_version": "5.13.0",
    }

    call_count = [0]
    dummy_c = np.full((848, 480, 3), 100, dtype=np.uint8)
    dummy_filt = np.full((848, 480), 1.25, dtype=np.float32)
    dummy_raw_m = np.full((848, 480), 1.25, dtype=np.float32)
    dummy_z16 = np.full((848, 480), 1250, dtype=np.uint16)
    dummy_meta = {"timestamp_ms": 1000.0}

    def mock_read():
        call_count[0] += 1
        return dummy_c, dummy_filt, dummy_raw_m, dummy_z16, None, dummy_meta

    mock_cam.read_with_raw_and_colorized.side_effect = mock_read

    adapter = RealSenseCaptureAdapter(
        portrait_rotation="cw",
        warmup_frames=5,
        camera_factory=lambda cfg: mock_cam,
    )

    adapter.start()
    assert call_count[0] == 5  # 5 warmup frames discarded
    assert adapter.depth_units == "sensor_units"
    assert adapter.depth_scale == 0.001
    assert adapter.intrinsics["width"] == 480
    assert adapter.intrinsics["height"] == 848

    c, r, f, ts = adapter.read_raw_filtered()
    assert c is not None and r is not None and f is not None
    assert r.dtype == np.uint16
    assert f.dtype == np.float32
    assert ts == pytest.approx(1.0)
    assert call_count[0] == 6

    adapter.stop()
    mock_cam.stop.assert_called_once()


def test_adapter_dropped_frames_retried():
    """Verify adapter retries dropped frames (None) up to max_dropped_retries."""
    mock_cam = MagicMock()
    mock_cam.depth_scale = 0.001
    mock_cam.intrinsics_o3d = None

    attempts = [0]
    dummy_c = np.zeros((10, 10, 3), dtype=np.uint8)
    dummy_z16 = np.zeros((10, 10), dtype=np.uint16)

    def mock_read():
        attempts[0] += 1
        if attempts[0] < 3:
            return None, None, None, None, None, None
        return dummy_c, None, None, dummy_z16, None, {"timestamp_ms": 2000.0}

    mock_cam.read_with_raw_and_colorized.side_effect = mock_read

    adapter = RealSenseCaptureAdapter(
        warmup_frames=0,
        max_dropped_retries=5,
        camera_factory=lambda cfg: mock_cam,
    )
    adapter.start()
    c, r, f, ts = adapter.read_raw_filtered()
    assert c is not None
    assert r is not None
    assert attempts[0] == 3
    adapter.stop()


def test_adapter_disconnect_raises_and_stops():
    """Verify hardware failure raises RuntimeError and ensures stop() runs."""
    mock_cam = MagicMock()
    mock_cam.depth_scale = 0.001
    mock_cam.read_with_raw_and_colorized.side_effect = RuntimeError("USB device disconnected")

    adapter = RealSenseCaptureAdapter(
        warmup_frames=0,
        camera_factory=lambda cfg: mock_cam,
    )
    adapter.start()

    with pytest.raises(RuntimeError, match="RealSense camera disconnected"):
        adapter.read_raw_filtered()

    adapter.stop()
    assert mock_cam.stop.called


def test_adapter_refuses_decimation():
    """Verify adapter refuses to start if decimation is active."""
    cfg = CameraConfig(use_decimation=True)
    adapter = RealSenseCaptureAdapter(camera_config=cfg)
    with pytest.raises(RuntimeError, match="decimation is active"):
        adapter.start()


def test_render_frame_shapes_and_instructions():
    """Verify render_frame produces expected dimensions and differing turn text for CW vs CCW."""
    color = np.zeros((480, 270, 3), dtype=np.uint8)
    depth = np.full((480, 270), 1.25, dtype=np.float32)

    for st in ControllerState:
        snap = {
            "state": st.value,
            "stop_index": 2,
            "n_stops": 8,
            "turn_direction": "cw",
            "distance_m": 1.25,
            "valid_depth_pct": 85.0,
            "motion_mm": 3.2,
            "head_visible": True,
            "feet_visible": True,
            "burst_progress": 0.5,
            "last_rejection_reason": "Slight sway",
        }
        out_img = render_frame(snap, color, depth)
        assert out_img.shape == (480, 540, 3)

    # CW instruction vs CCW instruction
    snap_cw = {"state": "TURN_PROMPT", "turn_direction": "cw", "n_stops": 8}
    snap_ccw = {"state": "TURN_PROMPT", "turn_direction": "ccw", "n_stops": 8}

    img_cw = render_frame(snap_cw, color, depth)
    img_ccw = render_frame(snap_ccw, color, depth)

    # Images must not be identical because text says RIGHT vs LEFT
    assert not np.array_equal(img_cw, img_ccw)


def test_cli_import_isolation():
    """Verify importing cli does NOT import pyrealsense2 or open3d."""
    cmd = [
        sys.executable,
        "-c",
        "import sys; from realsense_measure.fullbody import cli; "
        "assert 'pyrealsense2' not in sys.modules, 'pyrealsense2 imported!'; "
        "assert 'open3d' not in sys.modules, 'open3d imported!'; "
        "print('CLEAN_IMPORT')",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"CLI import failed: {proc.stderr}"
    assert "CLEAN_IMPORT" in proc.stdout


def test_cli_capture_fake_end_to_end():
    """Verify python -m realsense_measure.fullbody capture --fake end-to-end creates finalized session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bg_dir = Path(tmpdir) / "bg_calib"
        bg_argv = [
            "capture-background",
            "--out", str(bg_dir),
            "--portrait", "cw",
            "--burst-frames", "2",
            "--fake",
        ]
        assert main(bg_argv) == 0

        out_dir = Path(tmpdir) / "fake_capture"
        argv = [
            "capture",
            "--out", str(out_dir),
            "--subject-id", "SUBJ_CLI01",
            "--stops", "2",
            "--turn", "cw",
            "--portrait", "cw",
            "--fake",
            "--burst-frames", "3",
            "--background", str(bg_dir),
            "--turn-prompt-timeout", "0.1",
            "--hold-still-s", "0.1",
            "--headless",
        ]
        ret = main(argv)
        assert ret == 0
        assert (out_dir / "meta.json").exists()

        cfg = FullBodyConfig()
        stops = load_capture_session(out_dir, cfg)
        assert len(stops) == 2


def test_cli_capture_fake_failure_leaves_unfinalized(monkeypatch):
    """Verify capture failure or interruption leaves session unfinalized with data intact."""
    from realsense_measure.fullbody.capture_controller import CaptureController

    orig_step = CaptureController.step

    def failing_step(self, now):
        self._transition(ControllerState.ABORT, now)
        return ControllerState.ABORT

    monkeypatch.setattr(CaptureController, "step", failing_step)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "failing_capture"
        argv = [
            "capture",
            "--out", str(out_dir),
            "--subject-id", "SUBJ_FAIL",
            "--stops", "2",
            "--fake",
            "--headless",
        ]
        ret = main(argv)
        assert ret == 1
        assert not (out_dir / "meta.json").exists()


def test_cli_check_camera_fake():
    """Verify check-camera --fake creates snapshot images and metrics text without session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "camera_check_out"
        argv = [
            "check-camera",
            "--out", str(out_dir),
            "--portrait", "cw",
            "--fake",
        ]
        ret = main(argv)
        assert ret == 0

        chk_dir = out_dir / "check"
        assert (chk_dir / "check_color.png").exists()
        assert (chk_dir / "check_depth_colorized.png").exists()
        assert (chk_dir / "check_metrics.txt").exists()

        txt = (chk_dir / "check_metrics.txt").read_text(encoding="utf-8")
        assert "portrait_rotation: cw" in txt
        assert "depth_scale:" in txt
        assert "vertical_coverage_m:" in txt
        assert "head_in_frame:" in txt
        assert "feet_in_frame:" in txt
        assert "floor_visible:" in txt
        assert "requested_resolution: 1280x720" in txt
        assert "stream_profile_match: True" in txt
        assert "foreground_blob_found:" in txt
        assert "central_20x20_median_depth_m:" in txt
        assert "automatic_acceptance:" in txt


def test_cli_capture_background_fake():
    """Verify capture-background --fake creates persistent background assets and floor calibration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "background"
        argv = [
            "capture-background",
            "--out", str(out_dir),
            "--burst-frames", "5",
            "--fake",
        ]
        ret = main(argv)
        assert ret == 0
        assert (out_dir / "background_depth.npy").exists()
        assert (out_dir / "background_color.png").exists()
        assert (out_dir / "background_depth_colorized.png").exists()
        assert (out_dir / "background_meta.json").exists()
        assert (out_dir / "raw_burst").is_dir()
        assert (out_dir / "raw_burst" / "color_000.png").exists()
        assert (out_dir / "raw_burst" / "depth_000.npy").exists()

        meta = json.loads((out_dir / "background_meta.json").read_text(encoding="utf-8"))
        assert meta["burst_frames_captured"] == 5


def test_render_frame_turntable_and_countdown():
    """Verify render_frame formats turntable target text and WAIT_STILL countdown correctly."""
    color = np.zeros((720, 1280, 3), dtype=np.uint8)
    depth = np.full((720, 1280), 2.0, dtype=np.float32)

    # Turntable prompt
    snap_tt = {
        "state": "TURN_PROMPT",
        "stop_index": 2,
        "n_stops": 8,
        "turntable": True,
        "turn_direction": "cw",
    }
    img_tt = render_frame(snap_tt, color, depth)
    assert img_tt.shape == (720, 2560, 3)

    # Non-turntable prompt
    snap_turn = {
        "state": "TURN_PROMPT",
        "stop_index": 2,
        "n_stops": 8,
        "turntable": False,
        "turn_direction": "cw",
    }
    img_turn = render_frame(snap_turn, color, depth)
    assert not np.array_equal(img_tt, img_turn)

    # WAIT_STILL countdown
    snap_still = {
        "state": "WAIT_STILL",
        "stop_index": 0,
        "n_stops": 8,
        "stillness_frames_current": 5,
        "stillness_frames_target": 15,
    }
    img_still = render_frame(snap_still, color, depth)
    assert img_still.shape == (720, 2560, 3)


def test_intrinsics_mismatch_fails_on_load():
    """Verify load_capture_session fails if intrinsics differ from expected profile by >0.5 px."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bg_dir = Path(tmpdir) / "bg_calib"
        bg_argv = [
            "capture-background",
            "--out", str(bg_dir),
            "--burst-frames", "2",
            "--resolution", "848x480",
            "--fake",
        ]
        assert main(bg_argv) == 0

        out_dir = Path(tmpdir) / "intr_session"
        argv = [
            "capture",
            "--out", str(out_dir),
            "--subject-id", "SUBJ_INTR01",
            "--stops", "8",
            "--fake",
            "--burst-frames", "2",
            "--background", str(bg_dir),
            "--turn-prompt-timeout", "0.1",
            "--hold-still-s", "0.1",
            "--headless",
            "--resolution", "848x480",
        ]
        assert main(argv) == 0

        cfg = FullBodyConfig()

        # Reading with matching or close expected intrinsics (<=0.5 px diff) must succeed
        stops = load_capture_session(out_dir, cfg, expected_intrinsics={"fx": 425.2, "fy": 425.0, "cx": 424.0, "cy": 240.0})
        assert len(stops) == 8

        # Reading with mismatched intrinsics (>0.5 px diff) must raise ValueError
        with pytest.raises(ValueError, match="Intrinsics mismatch in session"):
            load_capture_session(out_dir, cfg, expected_intrinsics={"fx": 426.0, "fy": 425.0, "cx": 424.0, "cy": 240.0})


def test_background_sha256_verification_on_load():
    """Verify load_capture_session verifies background files.sha256 and detects tampering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bg_dir = Path(tmpdir) / "bg_calib"
        bg_argv = [
            "capture-background",
            "--out", str(bg_dir),
            "--burst-frames", "3",
            "--fake",
        ]
        assert main(bg_argv) == 0

        # Capture session providing background
        sess_dir = Path(tmpdir) / "sess_with_bg"
        argv = [
            "capture",
            "--out", str(sess_dir),
            "--subject-id", "SUBJ_BG01",
            "--stops", "8",
            "--fake",
            "--burst-frames", "2",
            "--background", str(bg_dir),
            "--turn-prompt-timeout", "0.1",
            "--hold-still-s", "0.1",
            "--headless",
        ]
        assert main(argv) == 0

        cfg = FullBodyConfig()
        stops = load_capture_session(sess_dir, cfg)
        assert len(stops) == 8

        # Tamper with background file in session
        tampered_file = sess_dir / "background" / "background_depth.npy"
        assert tampered_file.exists()
        tampered_file.write_bytes(tampered_file.read_bytes() + b"\x00")

        # Loading tampered session must raise ValueError on SHA check
        with pytest.raises(ValueError, match="SHA-256 integrity check failed for background file"):
            load_capture_session(sess_dir, cfg)


def test_modules_compatibility_720p_and_480p():
    """Verify all full-body modules run properly at both 1280x720 and 848x480."""
    import open3d as o3d
    from tests.fullbody.fake_camera import FakeCamera
    from realsense_measure.fullbody.burst import process_stop_burst
    from realsense_measure.fullbody.confidence import compute_confidence_map
    from realsense_measure.fullbody.floor import fit_floor_plane
    from realsense_measure.fullbody.fusion import compute_foreground_mask
    from realsense_measure.fullbody.keypoints import MediaPipePoseDetector

    cfg = FullBodyConfig()
    detector = MediaPipePoseDetector()

    for res in ((1280, 720), (848, 480)):
        w, h = res
        cam = FakeCamera(width=w, height=h)
        cam.start()
        color, raw_d, filt_d, _ = cam.read_raw_filtered()
        cam.stop()

        assert color.shape == (h, w, 3)
        assert raw_d.shape == (h, w)
        d_m = raw_d.astype(np.float32) * cam.depth_scale
        intr = o3d.camera.PinholeCameraIntrinsic(
            w, h, cam.intrinsics["fx"], cam.intrinsics["fy"], cam.intrinsics["cx"], cam.intrinsics["cy"]
        )

        # 1. Foreground mask
        bg_d = np.full((h, w), 2.50, dtype=np.float32)
        fg_mask = compute_foreground_mask(d_m, bg_d, cfg)
        assert fg_mask.shape == (h, w)
        assert np.any(fg_mask)

        # 2. Confidence map
        conf_map, filt_depth = compute_confidence_map(d_m, intr, cfg)
        assert conf_map.shape == (h, w)
        assert np.nanmax(conf_map) > 0.05

        # 3. Floor fit
        # Synthetic floor in lower half
        d_floor = d_m.copy()
        cy = cam.intrinsics["cy"]
        fy = cam.intrinsics["fy"]
        for row in range(int(0.65 * h), h):
            d_floor[row, :] = 0.95 * fy / max(1.0, row - cy)
        floor_fit = fit_floor_plane(d_floor, intr, cfg)
        assert floor_fit is not None
        assert abs(floor_fit.normal[1]) > 0.5  # Up/down normal dominant

        # 4. Burst processing
        rep_c, fused_d, d_std, val_cnt = process_stop_burst([color, color], [d_m, d_m], cfg)
        assert fused_d.shape == (h, w)
        assert d_std.shape == (h, w)

        # 5. Keypoint detector
        pts_2d, confs, vis, p_mask = detector.detect_with_mask(color, d_m)
        assert isinstance(pts_2d, dict)


def test_colorize_depth_background_preservation():
    """Verify colorize_depth_m does not zero out valid depths > 3.0m."""
    from realsense_measure.fullbody.capture_ui import colorize_depth_m

    depths = np.array([[0.0, 0.04, 0.5, 2.0, 3.5, 5.0, np.nan]], dtype=np.float32)
    colored = colorize_depth_m(depths)

    # 0.0, 0.04, and NaN must be black
    assert np.all(colored[0, 0] == [0, 0, 0])
    assert np.all(colored[0, 1] == [0, 0, 0])
    assert np.all(colored[0, 6] == [0, 0, 0])

    # Valid depths at 0.5m, 2.0m, 3.5m, 5.0m must NOT be black
    assert not np.all(colored[0, 2] == [0, 0, 0])
    assert not np.all(colored[0, 3] == [0, 0, 0])
    assert not np.all(colored[0, 4] == [0, 0, 0])
    assert not np.all(colored[0, 5] == [0, 0, 0])


def test_check_camera_preview_fake(monkeypatch, tmp_path):
    """Verify check-camera --preview with fake camera opens WINDOW_NORMAL and handles frames safely."""
    import cv2

    created_windows = []
    shown_frames = []

    def mock_named_window(name, flag):
        created_windows.append((name, flag))

    def mock_imshow(name, img):
        shown_frames.append((name, img.shape))

    # Exit immediately after first waitKey
    monkeypatch.setattr(cv2, "namedWindow", mock_named_window)
    monkeypatch.setattr(cv2, "imshow", mock_imshow)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: ord("q"))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    argv = ["check-camera", "--fake", "--preview", "--out", str(tmp_path)]
    ret = main(argv)
    assert ret == 0
    assert len(created_windows) == 1
    assert created_windows[0][1] == cv2.WINDOW_NORMAL
    assert len(shown_frames) >= 1
    # Frame width must fit <= 1200
    assert shown_frames[0][1][1] <= 1200


def test_check_camera_resolution_negotiation_and_framing(tmp_path):
    """Verify check-camera handles --resolution 848x480 and 1280x720, reporting framing info."""
    for res in ("1280x720", "848x480"):
        res_dir = tmp_path / f"chk_{res}"
        argv = ["check-camera", "--fake", "--resolution", res, "--out", str(res_dir)]
        assert main(argv) == 0

        metrics_file = res_dir / "check" / "check_metrics.txt"
        assert metrics_file.exists()
        txt = metrics_file.read_text(encoding="utf-8")
        assert f"requested_resolution: {res}" in txt
        assert f"negotiated_depth_stream: {res}@30fps" in txt
        assert f"negotiated_color_stream: {res}@30fps" in txt
        assert "stream_profile_match: True" in txt
        assert "foreground_blob_found: True" in txt
        assert "touches_border:" in txt
        assert "central_20x20_median_depth_m:" in txt


def test_preview_colorizer_preserves_raw_capture_bytes(monkeypatch, tmp_path):
    """
    Confirm preview-only changes (WINDOW_NORMAL downscale, depth colorization,
    cam.last_colorized) do not alter any saved raw frames, raw-only filter settings,
    or files written by capture.
    Compares raw PNG and NPY frames byte-for-byte between headless and preview-enabled runs.
    """
    import cv2
    import json

    # Shared background
    bg_dir = tmp_path / "bg_calib"
    bg_argv = [
        "capture-background",
        "--out", str(bg_dir),
        "--burst-frames", "2",
        "--fake",
    ]
    assert main(bg_argv) == 0

    # Run 1: Headless capture (no preview rendering or colorization)
    dir_headless = tmp_path / "cap_headless"
    argv_headless = [
        "capture",
        "--out", str(dir_headless),
        "--subject-id", "SUBJ_01",
        "--stops", "2",
        "--burst-frames", "3",
        "--fake",
        "--background", str(bg_dir),
        "--turn-prompt-timeout", "0.05",
        "--hold-still-s", "0.05",
        "--headless",
        "--no-beep",
    ]
    assert main(argv_headless) == 0

    # Run 2: Preview-enabled capture (drives render_frame, downscaling, colorize_depth_m)
    # Monkeypatch cv2 so GUI displays headlessly during the test
    preview_rendered = []

    def mock_imshow(name, img):
        preview_rendered.append(img.shape)

    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow", mock_imshow)
    monkeypatch.setattr(cv2, "waitKey", lambda *a, **k: 32)  # confirm prompt
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda *a, **k: None)

    dir_preview = tmp_path / "cap_preview"
    argv_preview = [
        "capture",
        "--out", str(dir_preview),
        "--subject-id", "SUBJ_01",
        "--stops", "2",
        "--burst-frames", "3",
        "--fake",
        "--background", str(bg_dir),
        "--turn-prompt-timeout", "0.05",
        "--hold-still-s", "0.05",
        "--no-beep",
    ]
    assert main(argv_preview) == 0
    assert len(preview_rendered) > 0, "Preview rendering should have been triggered"

    # Compare raw files byte-for-byte across all stops
    for stop_idx in (0, 1):
        s_head = dir_headless / f"stop_{stop_idx:02d}"
        s_prev = dir_preview / f"stop_{stop_idx:02d}"
        assert s_head.exists() and s_prev.exists()

        head_files = sorted(f.name for f in s_head.glob("*"))
        prev_files = sorted(f.name for f in s_prev.glob("*"))
        assert head_files == prev_files

        for fname in head_files:
            f1 = s_head / fname
            f2 = s_prev / fname
            if fname.endswith(".png") or fname.endswith(".npy"):
                b1 = f1.read_bytes()
                b2 = f2.read_bytes()
                assert b1 == b2, f"Byte mismatch in raw frame {fname} between headless and preview runs!"
            elif fname == "hashes.json":
                h1 = json.loads(f1.read_text(encoding="utf-8"))
                h2 = json.loads(f2.read_text(encoding="utf-8"))
                assert h1 == h2, f"SHA-256 hash mismatch in hashes.json between headless and preview runs!"

    # Verify camera filter settings in meta.json
    meta = json.loads((dir_preview / "meta.json").read_text(encoding="utf-8"))
    filters = meta.get("device_info", {}).get("filter_settings", {})
    assert filters.get("raw_only") is True
    assert filters.get("spatial") is False
    assert filters.get("temporal") is False
    assert filters.get("hole_filling") is False
    assert filters.get("decimation") is False





