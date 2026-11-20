"""
realsense_measure/fullbody/cli.py

Command-line interface for the model-free full-body scanning package.
Subcommands:
  capture       Live eight-stop capture controller with operator preview.
  check-camera  Verify RealSense orientation, FOV, and depth sanity (no session).
  scan          Run full acquisition/replay, alignment, fusion, and measurement.
  measure       Extract anthropometric measurements from an existing fused scan.
  validate      Compare scan measurements against physical tape measurements.
  sweet-spot    Evaluate capture geometry and report under-covered anatomical regions.
  review        Open 3D interactive viewer for fused mesh and point cloud.
  keypoints-check Audit MediaPipe Pose detection on a single frame.

Strictly lazy imports so importing this module never pulls in Open3D or pyrealsense2.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, List, Optional
import warnings

import cv2
import numpy as np

logger = logging.getLogger("realsense_measure.fullbody")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m realsense_measure.fullbody",
        description="Model-free metric full-body scanning and anthropometry.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # 1. capture (Stage 27 live capture controller)
    p_cap = subparsers.add_parser("capture", help="Live eight- or twelve-stop capture controller with operator preview.")
    p_cap.add_argument("--out", type=str, required=True, help="Destination directory for the capture session.")
    p_cap.add_argument("--subject-id", type=str, required=True, help="Anonymous subject ID (e.g. SUBJ_01, TEST_42).")
    p_cap.add_argument("--stops", type=int, default=8, help="Number of stops: 8 or 12 (default: 8).")
    p_cap.add_argument("--turn", type=str, choices=["cw", "ccw"], default="cw", help="Turn direction ('cw' or 'ccw').")
    p_cap.add_argument("--turntable", action="store_true", default=False, help="Turntable mode (rotates turntable instead of subject turning on foot).")
    p_cap.add_argument("--resolution", type=str, choices=["1280x720", "848x480"], default="1280x720", help="Camera stream resolution: '1280x720' (default) or '848x480'.")
    p_cap.add_argument("--raw-only", action=argparse.BooleanOptionalAction, default=True, help="Save raw z16/color and disable RealSense temporal/hole filters (default: True).")
    p_cap.add_argument("--background", type=str, default="raw_captures/background", help="Path to background calibration directory (default: raw_captures/background).")
    p_cap.add_argument("--portrait", type=str, choices=["cw", "ccw", "none"], default="none", help="Portrait camera orientation: 'cw', 'ccw', or 'none' for horizontal/landscape mount (default: 'none').")
    p_cap.add_argument("--preview-only", action="store_true", help="Launch live operator preview without recording or saving a session.")
    p_cap.add_argument("--fake", action="store_true", help="Use synthetic FakeCamera instead of hardware RealSense.")
    p_cap.add_argument("--burst-frames", type=int, default=60, help="Frames per stop burst (default: 60).")
    p_cap.add_argument("--headless", action="store_true", help="Run without opening GUI preview window.")
    p_cap.add_argument("--dist-min", type=float, default=1.70, help="Minimum subject distance in metres (default: 1.70).")
    p_cap.add_argument("--dist-max", type=float, default=2.40, help="Maximum subject distance in metres (default: 2.40).")
    p_cap.add_argument("--no-head-feet-check", action="store_true", help="Disable strict head and feet unclipped check.")
    p_cap.add_argument("--turn-prompt-timeout", type=float, default=10.0, help="Turn prompt timeout in seconds (default: 10.0).")
    p_cap.add_argument("--hold-still-s", type=float, default=2.0, help="Hold-still countdown duration in seconds (default: 2.0).")
    p_cap.add_argument("--motion-limit", type=float, default=10.0, help="Subject motion limit in mm (default: 10.0).")
    p_cap.add_argument("--max-retakes", type=int, default=3, help="Max retakes per stop (default: 3).")
    p_cap.add_argument("--cam-height-m", type=float, default=None, help="Camera height above floor in metres.")
    p_cap.add_argument("--cam-distance-m", type=float, default=None, help="Camera standoff distance to subject in metres.")
    p_cap.add_argument("--cam-tilt-deg", type=float, default=0.0, help="Camera tilt angle in degrees (default: 0.0).")
    p_cap.add_argument("--clothing", type=str, default=None, help="Subject clothing description (free text).")
    p_cap.add_argument("--pose", type=str, default="A-pose", help="Subject pose description (default: 'A-pose').")
    p_cap.add_argument("--notes", type=str, default=None, help="Session notes (free text).")
    p_cap.add_argument("--subject-type", type=str, choices=["person", "object"], default="person", help="Subject type: 'person' or 'object' (default: 'person').")
    p_cap.add_argument("--object-diameter-mm", type=float, default=None, help="Object diameter in mm for rigid object mode.")
    p_cap.add_argument("--no-beep", action="store_true", help="Disable audio cue beeps.")

    # 2. check-camera (Stage 27 hardware sanity check snapshot)
    p_chk = subparsers.add_parser("check-camera", help="Verify RealSense orientation, FOV, and depth sanity (no session).")
    p_chk.add_argument("--portrait", "--rotation", dest="portrait", type=str, choices=["cw", "ccw", "none"], default="none", help="Camera rotation: 'none' for horizontal/landscape mount, 'cw', or 'ccw' (default: 'none').")
    p_chk.add_argument("--preview", action="store_true", help="Open an interactive live OpenCV preview window for orientation verification (press Q or ESC to exit).")
    p_chk.add_argument("--out", type=str, default="scan_output", help="Output directory to save check snapshot into DIR/check/.")
    p_chk.add_argument("--fake", action="store_true", help="Use synthetic FakeCamera for headless testing.")

    # 3. capture-background
    p_bg = subparsers.add_parser("capture-background", help="Capture empty-scene background burst and calibrate single floor plane.")
    p_bg.add_argument("--out", type=str, default="raw_captures/background", help="Persistent destination directory for background assets (default: raw_captures/background).")
    p_bg.add_argument("--portrait", "--rotation", dest="portrait", type=str, choices=["cw", "ccw", "none"], default="none", help="Camera rotation (default: 'none').")
    p_bg.add_argument("--resolution", type=str, choices=["1280x720", "848x480"], default="1280x720", help="Camera stream resolution: '1280x720' (default) or '848x480'.")
    p_bg.add_argument("--burst-frames", type=int, default=30, help="Number of frames in background burst (default: 30).")
    p_bg.add_argument("--fake", action="store_true", help="Use synthetic FakeCamera for headless testing.")

    # 4. scan
    p_scan = subparsers.add_parser("scan", help="Run full-body alignment, fusion, and measurement.")
    p_scan.add_argument("--capture", type=str, required=True, help="Path to raw capture directory.")
    p_scan.add_argument("--out", type=str, default="scan_output/fullbody", help="Output directory.")
    p_scan.add_argument("--stops", type=int, default=8, help="Number of stops (default: 8).")
    p_scan.add_argument("--voxel", type=float, default=0.005, help="TSDF voxel length in metres.")
    p_scan.add_argument("--background", type=str, default="raw_captures/background", help="Path to background calibration directory.")
    p_scan.add_argument("--no-viz", action="store_true", help="Run without opening GUI windows.")
    p_scan.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    # 5. measure
    p_meas = subparsers.add_parser("measure", help="Extract measurements from an existing fused scan.")
    p_meas.add_argument("--scan", type=str, required=True, help="Directory containing fused.ply or scan_result.json.")

    # 5. validate
    p_val = subparsers.add_parser("validate", help="Compare scan measurements with physical tape values.")
    p_val.add_argument("--scan", type=str, required=True, help="Scan output directory.")
    p_val.add_argument("--tape", type=str, default="data/tape_measurements.json", help="Tape measurements JSON.")

    # 6. sweet-spot
    p_sw = subparsers.add_parser("sweet-spot", help="Evaluate capture geometry information score.")
    p_sw.add_argument("--scan", type=str, required=True, help="Scan directory or capture path.")

    # 7. review
    p_rev = subparsers.add_parser("review", help="Review fused 3D assets in Open3D.")
    p_rev.add_argument("--scan", type=str, required=True, help="Scan output directory.")

    # 8. keypoints-check
    p_kpt = subparsers.add_parser(
        "keypoints-check",
        help="Run MediaPipe Pose on color image, overlay 33 joints, and print confidence.",
    )
    p_kpt.add_argument("color", type=str, help="Path to input color image (PNG/JPG).")
    p_kpt.add_argument("depth", type=str, nargs="?", default=None, help="Optional path to depth array (.npy).")
    p_kpt.add_argument("--out", type=str, default=None, help="Optional output image path for skeleton overlay.")

    # 9. debug-stop
    p_dbg = subparsers.add_parser(
        "debug-stop",
        help="Generate per-stop debug visualisations and quantitative stats.json.",
    )
    p_dbg.add_argument("session_dir", type=str, help="Path to capture session directory.")
    p_dbg.add_argument("--stop", type=int, required=True, help="Stop index to debug (e.g. 0, 1, ...).")
    p_dbg.add_argument("--out", type=str, default=None, help="Optional output directory (default: <session_dir>/debug/stop_NN).")

    # 10. validate-session
    p_vs = subparsers.add_parser(
        "validate-session",
        help="Run comprehensive session validation, leave-one-out cross-validation, and report.",
    )
    p_vs.add_argument("session_dir", type=str, help="Path to capture session directory.")
    p_vs.add_argument("--tape", type=str, default="data/tape_measurements.json", help="Path to ground truth tape measurements JSON.")
    p_vs.add_argument("--out", type=str, default=None, help="Optional path to output validation JSON.")
    p_vs.add_argument("--report", type=str, default=None, help="Optional path to output validation markdown report.")

    # 11. verify-session
    p_vses = subparsers.add_parser(
        "verify-session",
        help="Verify saved capture session without running registration or fusion.",
    )
    p_vses.add_argument("--capture", type=str, required=True, help="Path to raw capture session directory.")

    return parser


def _run_capture(args: argparse.Namespace) -> int:
    """Execute live 8-stop capture controller with operator preview."""
    from realsense_measure.fullbody.capture_controller import CaptureController, CaptureThresholds, ControllerState
    from realsense_measure.fullbody.capture_ui import run_ui_loop
    from realsense_measure.fullbody.session_format import SessionWriter, validate_anonymous_subject_id

    try:
        subject_id = validate_anonymous_subject_id(args.subject_id)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    if not getattr(args, "preview_only", False):
        out_dir.mkdir(parents=True, exist_ok=True)

    res_str = getattr(args, "resolution", "1280x720")
    if getattr(args, "fake", False):
        from tests.fullbody.fake_camera import FakeCamera
        if "x" in res_str:
            w_c, h_c = [int(v) for v in res_str.lower().split("x")]
        else:
            w_c, h_c = 1280, 720
        if getattr(args, "portrait", "none") in ("cw", "ccw"):
            w_c, h_c = h_c, w_c

        # If background exists and has matching resolution, inherit intrinsics for FakeCamera
        fake_fx, fake_fy, fake_cx, fake_cy = None, None, None, None
        bg_cand = Path(getattr(args, "background", "raw_captures/background"))
        bg_cand_meta = bg_cand / "background_meta.json"
        if bg_cand_meta.exists():
            try:
                with open(bg_cand_meta, "r", encoding="utf-8") as f:
                    b_meta = json.load(f)
                b_intr = b_meta.get("intrinsics", {})
                b_w = b_intr.get("width", b_meta.get("width"))
                b_h = b_intr.get("height", b_meta.get("height"))
                if b_w == w_c and b_h == h_c:
                    if "fx" in b_intr and "fy" in b_intr:
                        fake_fx = float(b_intr["fx"])
                        fake_fy = float(b_intr["fy"])
                    if "cx" in b_intr and "cy" in b_intr:
                        fake_cx = float(b_intr["cx"])
                        fake_cy = float(b_intr["cy"])
            except Exception:
                pass

        cam = FakeCamera(
            width=w_c,
            height=h_c,
            depth_scale=0.001,
            depth_units="sensor_units",
            fps=30.0,
            fx=fake_fx,
            fy=fake_fy,
            cx=fake_cx,
            cy=fake_cy,
        )
    else:
        from realsense_measure.fullbody.capture_camera import RealSenseCaptureAdapter, full_body_camera_config
        cam_cfg = full_body_camera_config(
            portrait_rotation=args.portrait,
            resolution=res_str,
            raw_only=getattr(args, "raw_only", True),
        )
        cam = RealSenseCaptureAdapter(
            camera_config=cam_cfg,
            portrait_rotation=args.portrait,
        )

    try:
        cam.start()
    except Exception as exc:
        print(f"Error: Failed to start camera: {exc}", file=sys.stderr)
        return 1

    if getattr(args, "preview_only", False):
        orient_label = "LANDSCAPE (no rotation)" if args.portrait == "none" else f"PORTRAIT {args.portrait.upper()}"
        print("\n" + "=" * 60)
        print("FULL-BODY CAPTURE PREVIEW-ONLY MODE")
        print("=" * 60)
        print(f"Orientation: {orient_label}")
        print("Live preview active. No capture session or files will be written to disk.")
        print("Press 'Q' or ESC in preview window to exit.\n")
        from realsense_measure.fullbody.capture_ui import render_frame
        window_name = f"RealSense Live Capture Preview - {orient_label} [Press Q to Exit]"
        try:
            while True:
                c, r, f, ts = cam.read_raw_filtered()
                if c is not None and r is not None:
                    h_c, w_c = c.shape[:2]
                    scale = cam.depth_scale
                    d_m = r.astype(np.float32) * (scale if cam.depth_units == "sensor_units" else 1.0)
                    roi = d_m[int(0.25 * h_c) : int(0.75 * h_c), int(0.25 * w_c) : int(0.75 * w_c)]
                    valid_roi = roi[(roi > 0.3) & (roi < 3.5)]
                    med_depth_m = float(np.median(valid_roi)) if len(valid_roi) > 0 else None
                    valid_frac = float(len(valid_roi)) / float(roi.size) if roi.size > 0 else 0.0

                    snapshot = {
                        "state": "PREVIEW_ONLY",
                        "stop_index": 0,
                        "n_stops": args.stops,
                        "turn_direction": args.turn,
                        "turntable": getattr(args, "turntable", False),
                        "orientation": args.portrait,
                        "distance_m": med_depth_m,
                        "valid_depth_pct": valid_frac * 100.0,
                        "motion_mm": 0.0,
                        "head_visible": True,
                        "feet_visible": True,
                        "burst_progress": 0.0,
                        "last_rejection_reason": "Preview Only (No Recording)",
                    }
                    display_frame = render_frame(snapshot, c, f if f is not None else d_m)
                    cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            cam.stop()
        return 0

    accepted_stops = 0
    controller: Optional[CaptureController] = None

    try:
        # Load background depth map for foreground segmentation
        bg_dir = Path(getattr(args, "background", "raw_captures/background"))
        stream_w = int(cam.intrinsics.get("width", 1280))
        stream_h = int(cam.intrinsics.get("height", 720))

        allow_missing_bg = False
        bg_depth = None
        if not bg_dir.exists() or not (bg_dir / "background_depth.npy").exists():
            print(
                f"\nWARNING: No background depth map found at '{bg_dir}'.\n"
                f"Capture will proceed with distance-gated foreground segmentation without background subtraction.\n"
                f"For optimal segmentation, run 'python -m realsense_measure.fullbody capture-background' at {stream_w}x{stream_h} first.\n",
                file=sys.stderr,
            )
            allow_missing_bg = True
        else:
            try:
                bg_depth = np.load(str(bg_dir / "background_depth.npy")).astype(np.float32)
            except Exception as e:
                print(f"Capture warning: Failed to load background depth: {e}. Proceeding without background.", file=sys.stderr)
                allow_missing_bg = True
                bg_depth = None

            if bg_depth is not None and bg_depth.shape[:2] != (stream_h, stream_w):
                print(
                    f"Capture error: Background resolution {bg_depth.shape[1]}x{bg_depth.shape[0]} "
                    f"does not match camera stream {stream_w}x{stream_h}. Run capture-background at {stream_w}x{stream_h} first.",
                    file=sys.stderr,
                )
                return 1

            if bg_depth is not None:
                # Check background_meta.json
                meta_file = bg_dir / "background_meta.json"
                bg_meta: Dict[str, Any] = {}
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            bg_meta = json.load(f)
                    except Exception as e:
                        logger.warning(f"Could not read {meta_file}: {e}")

                bg_w = bg_meta.get("width")
                bg_h = bg_meta.get("height")
                if bg_w is not None and bg_h is not None:
                    if (int(bg_w), int(bg_h)) != (stream_w, stream_h):
                        print(
                            f"Capture error: Background resolution {bg_w}x{bg_h} does not match camera stream {stream_w}x{stream_h}. "
                            f"Run capture-background at {stream_w}x{stream_h} first.",
                            file=sys.stderr,
                        )
                        return 1

                bg_intr = bg_meta.get("intrinsics")
                if bg_intr and isinstance(bg_intr, dict):
                    for k in ("fx", "fy", "cx", "cy"):
                        if k in bg_intr and k in cam.intrinsics:
                            diff_val = abs(float(bg_intr[k]) - float(cam.intrinsics[k]))
                            if diff_val > 5.0:
                                print(
                                    f"Capture error: Background intrinsic {k}={bg_intr[k]} does not match camera stream {cam.intrinsics[k]} "
                                    f"(diff={diff_val:.1f}px). Run capture-background at {stream_w}x{stream_h} first.",
                                    file=sys.stderr,
                                )
                                return 1

                # Compare floor-plane normal and camera height to background's and warn
                bg_normal = bg_meta.get("floor_normal")
                bg_cam_h = bg_meta.get("camera_height_m")
                if bg_normal is not None and bg_cam_h is not None:
                    sample_c, sample_r, sample_f, _ = cam.read_raw_filtered()
                    if sample_r is not None:
                        try:
                            import open3d as o3d
                            from realsense_measure.fullbody.config import FullBodyConfig
                            from realsense_measure.fullbody.floor import fit_floor_plane

                            o3d_i = o3d.camera.PinholeCameraIntrinsic(
                                stream_w,
                                stream_h,
                                float(cam.intrinsics["fx"]),
                                float(cam.intrinsics["fy"]),
                                float(cam.intrinsics["cx"]),
                                float(cam.intrinsics["cy"]),
                            )
                            scale = cam.depth_scale if cam.depth_units == "sensor_units" else 1.0
                            d_m_sample = sample_r.astype(np.float32) * scale
                            cur_floor = fit_floor_plane(d_m_sample, o3d_i, FullBodyConfig())
                            if cur_floor is not None:
                                cur_n = cur_floor.normal
                                bg_n = np.array(bg_normal, dtype=np.float32)
                                cur_n_norm = cur_n / max(1e-6, np.linalg.norm(cur_n))
                                bg_n_norm = bg_n / max(1e-6, np.linalg.norm(bg_n))
                                cos_ang = np.clip(abs(np.dot(cur_n_norm, bg_n_norm)), -1.0, 1.0)
                                ang_diff_deg = float(np.rad2deg(np.arccos(cos_ang)))
                                cur_cam_h = abs(float(cur_floor.coefficients[3]))
                                h_diff_m = abs(cur_cam_h - float(bg_cam_h))
                                if ang_diff_deg > 1.0 or h_diff_m > 0.03:
                                    warn_msg = (
                                        f"camera moved since background: normal differs by {ang_diff_deg:.2f} deg (>1.0 deg), "
                                        f"height differs by {h_diff_m * 100:.1f} cm (>3.0 cm)"
                                    )
                                    print(f"\nWARNING: {warn_msg}")
                                    logger.warning(warn_msg)
                        except Exception as floor_exc:
                            logger.debug(f"Floor drift check skipped: {floor_exc}")

        writer = SessionWriter(
            root=out_dir,
            subject_id=subject_id,
            n_stops=args.stops,
            turn_direction=args.turn,
            camera_portrait_rotation=args.portrait,
            intrinsics=cam.intrinsics,
            depth_scale=cam.depth_scale,
            depth_units=cam.depth_units,
            device_info=cam.device_info(),
            turntable=getattr(args, "turntable", False),
            background_dir=bg_dir if (bg_dir.exists() and (bg_dir / "background_depth.npy").exists()) else None,
            cam_height_m=getattr(args, "cam_height_m", None),
            cam_distance_m=getattr(args, "cam_distance_m", None),
            cam_tilt_deg=getattr(args, "cam_tilt_deg", 0.0),
            clothing=getattr(args, "clothing", None),
            pose=getattr(args, "pose", "A-pose"),
            notes=getattr(args, "notes", None),
            subject_type=getattr(args, "subject_type", "person"),
            object_diameter_mm=getattr(args, "object_diameter_mm", None),
            protocol_version="stop-and-go-v1",
            frames_requested=args.burst_frames,
        )

        hold_still_s = float(getattr(args, "hold_still_s", 2.0))
        stillness_frames = int(round(hold_still_s * 30.0))
        thresholds = CaptureThresholds(
            burst_frames=args.burst_frames,
            stillness_frames=stillness_frames,
            motion_limit_mm=getattr(args, "motion_limit", 10.0),
            max_retakes=getattr(args, "max_retakes", 3),
            dist_min_m=getattr(args, "dist_min", 1.70),
            dist_max_m=getattr(args, "dist_max", 2.40),
            require_head_feet=not getattr(args, "no_head_feet_check", False),
            turn_prompt_timeout_s=getattr(args, "turn_prompt_timeout", 10.0),
            burst_timeout_s=8.0,
            subject_type=getattr(args, "subject_type", "person"),
            object_diameter_mm=getattr(args, "object_diameter_mm", None),
            audio_cues=not getattr(args, "no_beep", False),
            allow_missing_background=allow_missing_bg,
        )

        landmark_fn = None
        try:
            from realsense_measure.fullbody.keypoints import MediaPipePoseDetector
            detector = MediaPipePoseDetector()
            def landmark_extractor(color_bgr: np.ndarray, depth_arr: np.ndarray) -> Optional[dict]:
                try:
                    pts_2d, conf, vis = detector.detect(color_bgr)
                    return {
                        "landmarks_2d": {k: [float(v[0]), float(v[1])] for k, v in pts_2d.items()},
                        "confidence": {k: float(v) for k, v in conf.items()},
                        "visibility": {k: float(v) for k, v in vis.items()},
                    }
                except Exception as k_exc:
                    logger.warning(f"Keypoint extraction warning: {k_exc}")
                    return None
            landmark_fn = landmark_extractor
        except Exception:
            landmark_fn = None

        controller = CaptureController(
            camera=cam,
            writer=writer,
            thresholds=thresholds,
            landmark_fn=landmark_fn,
            bg_depth=bg_depth,
            turntable=getattr(args, "turntable", False),
        )

        headless = bool(getattr(args, "headless", False))
        run_ui_loop(controller, headless=headless)

    except KeyboardInterrupt:
        print("\nCapture interrupted by operator (Ctrl-C).")
    except Exception as exc:
        print(f"\nCapture error: {exc}", file=sys.stderr)
    finally:
        cam.stop()

        # Check how many stops were accepted
        for stop_idx in range(args.stops):
            s_json = out_dir / f"stop_{stop_idx:02d}" / "stop.json"
            if s_json.exists():
                try:
                    with open(s_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("accepted_attempt") is not None:
                        accepted_stops += 1
                except Exception:
                    pass

    # Print session summary with accepted stops and per-stop file counts
    print("\n" + "=" * 60)
    print("CAPTURE SESSION SUMMARY")
    print("=" * 60)
    print(f"Session Directory:  {out_dir}")
    print(f"Accepted Stops:     {accepted_stops} of {args.stops}")
    for s_idx in range(args.stops):
        s_folder = out_dir / f"stop_{s_idx:02d}"
        if s_folder.exists():
            n_files = len([f for f in s_folder.rglob("*") if f.is_file()])
            print(f"  - stop_{s_idx:02d}: {n_files} files")
        else:
            print(f"  - stop_{s_idx:02d}: not created")
    print("=" * 60 + "\n")

    if controller is not None and controller.state == ControllerState.DONE:
        print(f"Capture session successfully finalized at: {out_dir}")
        print(f"Accepted all {args.stops} stops.")
        return 0
    else:
        print(f"Session unfinalized at {out_dir}. Accepted stops: {accepted_stops} of {args.stops}.")
        print("Raw attempt data and integrity hashes remain intact on disk.")
        return 1


def _run_check_camera(args: argparse.Namespace) -> int:
    """Verify RealSense orientation, FOV, and depth sanity (no session)."""
    portrait = getattr(args, "portrait", "none")
    orient_label = "LANDSCAPE (no rotation)" if portrait == "none" else f"PORTRAIT {portrait.upper()}"

    if getattr(args, "fake", False):
        from tests.fullbody.fake_camera import FakeCamera
        # Landscape fake camera: native 848x480
        cam = FakeCamera(width=848, height=480, depth_scale=0.001, depth_units="sensor_units")
    else:
        from realsense_measure.fullbody.capture_camera import RealSenseCaptureAdapter, full_body_camera_config
        cam_cfg = full_body_camera_config(portrait_rotation=portrait)
        cam = RealSenseCaptureAdapter(camera_config=cam_cfg, portrait_rotation=portrait)

    try:
        cam.start()
    except Exception as exc:
        print(f"Error starting camera: {exc}", file=sys.stderr)
        return 1

    last_color: Optional[np.ndarray] = None
    last_depth: Optional[np.ndarray] = None

    try:
        if getattr(args, "preview", False):
            from realsense_measure.fullbody.capture_ui import render_frame
            window_name = f"RealSense Camera Check - {orient_label} [Press Q to Exit]"
            print("\n" + "=" * 60)
            print("REALSENSE CAMERA INTERACTIVE PREVIEW MODE")
            print("=" * 60)
            print(f"Orientation: {orient_label}")
            print(f"Window: '{window_name}'")
            print("Press 'Q' or ESC in preview window to exit and capture snapshot.\n")
            while True:
                c, r, f, ts = cam.read_raw_filtered()
                if c is not None and r is not None:
                    last_color = c
                    last_depth = r
                    h_c, w_c = c.shape[:2]
                    scale = cam.depth_scale
                    d_m = r.astype(np.float32) * (scale if cam.depth_units == "sensor_units" else 1.0)
                    roi = d_m[int(0.25 * h_c) : int(0.75 * h_c), int(0.25 * w_c) : int(0.75 * w_c)]
                    valid_roi = roi[(roi > 0.3) & (roi < 3.5)]
                    med_depth_m = float(np.median(valid_roi)) if len(valid_roi) > 0 else None
                    valid_frac = float(len(valid_roi)) / float(roi.size) if roi.size > 0 else 0.0

                    snapshot = {
                        "state": "PREVIEW_ONLY",
                        "stop_index": 0,
                        "n_stops": 1,
                        "turn_direction": "cw",
                        "orientation": portrait,
                        "distance_m": med_depth_m,
                        "valid_depth_pct": valid_frac * 100.0,
                        "motion_mm": 0.0,
                        "head_visible": True,
                        "feet_visible": True,
                        "burst_progress": 0.0,
                        "last_rejection_reason": "Live Camera Check (Preview Mode)",
                    }
                    display_frame = render_frame(snapshot, c, f if f is not None else d_m)
                    cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        else:
            # Stream for approx 5 seconds or until frames acquired
            start_time = time.time()
            while (time.time() - start_time) < 5.0:
                c, r, f, ts = cam.read_raw_filtered()
                if c is not None and r is not None:
                    last_color = c
                    last_depth = r
                time.sleep(0.033)
    finally:
        cam.stop()

    if last_color is None or last_depth is None:
        print("Error: Could not acquire valid frames from camera.", file=sys.stderr)
        return 1

    dev_info = cam.device_info()
    intr = cam.intrinsics
    scale = cam.depth_scale
    h, w = last_depth.shape[:2]

    d_m = last_depth.astype(np.float32) * (scale if cam.depth_units == "sensor_units" else 1.0)
    roi = d_m[int(0.25 * h) : int(0.75 * h), int(0.25 * w) : int(0.75 * w)]
    valid_roi = roi[(roi > 0.3) & (roi < 3.5)]
    med_depth_m = float(np.median(valid_roi)) if len(valid_roi) > 0 else 0.0
    valid_frac = float(len(valid_roi)) / float(roi.size)

    # Automatic acceptance checks:
    import open3d as o3d
    from realsense_measure.fullbody.config import FullBodyConfig
    from realsense_measure.fullbody.coverage import compute_coverage_metrics
    from realsense_measure.fullbody.floor import fit_floor_plane

    cfg_chk = FullBodyConfig()
    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        int(intr["width"]), int(intr["height"]),
        float(intr["fx"]), float(intr["fy"]),
        float(intr["cx"]), float(intr["cy"]),
    )

    # 1. Coverage helper & recommended standoffs
    cov_170 = compute_coverage_metrics(intr, subject_height_m=1.70)
    cov_180 = compute_coverage_metrics(intr, subject_height_m=1.80)
    cov_190 = compute_coverage_metrics(intr, subject_height_m=1.90)

    # 2. Floor plane visible & Camera pitch estimated from floor plane
    floor = fit_floor_plane(d_m, o3d_intr, cfg_chk)
    floor_ok = (floor is not None and floor.inlier_count >= 100)
    pitch_deg: Optional[float] = None
    pitch_ok = False
    if floor is not None:
        ny = float(floor.normal[1])
        nz = float(floor.normal[2])
        pitch_deg = float(np.rad2deg(np.arctan2(abs(nz), max(1e-6, abs(ny)))))
        pitch_ok = bool(pitch_deg <= 15.0)

    # 3. Head apex and sole framing margins (>= 4% at both ends; warn if head margin > 15%)
    head_ok = False
    feet_ok = False
    head_warn = False
    margin_ratio = 0.04
    margin_px = int(margin_ratio * h)

    v_head_top: Optional[float] = None
    v_sole: Optional[float] = None
    framing_source = "heuristic"

    # Check if a background session exists to use the foreground mask directly
    bg_candidates = [
        Path(args.out) / "background",
        Path("raw_captures/background"),
        Path(args.out).parent / "background",
    ]
    bg_depth_file = next((p / "background_depth.npy" for p in bg_candidates if (p / "background_depth.npy").exists()), None)

    if bg_depth_file is not None:
        try:
            bg_depth_map = np.load(str(bg_depth_file)).astype(np.float32)
            from realsense_measure.fullbody.fusion import compute_foreground_mask
            fg_mask = compute_foreground_mask(d_m, bg_depth_map, cfg_chk)
            if np.sum(fg_mask) >= 200:
                fg_rows = np.where(fg_mask)[0]
                v_head_top = float(np.min(fg_rows))
                v_sole = float(np.max(fg_rows))
                framing_source = f"background_foreground_mask ({bg_depth_file.parent.name})"
        except Exception as exc:
            logger.debug(f"Failed foreground mask from background depth: {exc}")

    # If no background session or foreground mask was empty, use keypoints + depth-valid person component
    if v_head_top is None or v_sole is None:
        try:
            from realsense_measure.fullbody.keypoints import MediaPipePoseDetector
            detector = MediaPipePoseDetector()
            pts_2d, conf_2d, vis_2d = detector.detect(last_color, depth_m=d_m)

            # Head cranial landmarks (nose, eyes, ears)
            cranial_keys = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
            cranial_v = [pts_2d[k][1] for k in cranial_keys if k in pts_2d and conf_2d.get(k, 0.0) >= 0.20]

            # Feet/sole: foot_index and heel (strictly exclude ankle landmarks)
            feet_keys = ["left_heel", "right_heel", "left_foot_index", "right_foot_index"]
            feet_v = [pts_2d[k][1] for k in feet_keys if k in pts_2d and conf_2d.get(k, 0.0) >= 0.20]

            # Subject horizontal extent
            if pts_2d:
                all_u = [pts_2d[k][0] for k in pts_2d if conf_2d.get(k, 0.0) >= 0.20]
                u_min = max(0, int(min(all_u) - 30)) if all_u else int(w * 0.2)
                u_max = min(w, int(max(all_u) + 30)) if all_u else int(w * 0.8)
            else:
                u_min, u_max = int(w * 0.2), int(w * 0.8)

            # Depth-valid person component around median standoff
            z_ref = med_depth_m if med_depth_m >= 0.50 else cfg_chk.camera_distance_m
            person_depth = (d_m >= max(0.4, z_ref - 0.45)) & (d_m <= (z_ref + 0.45))
            person_depth[:, :u_min] = False
            person_depth[:, u_max:] = False

            if np.sum(person_depth) >= 100:
                p_rows = np.where(person_depth)[0]
                d_min_row = float(np.min(p_rows))
                d_max_row = float(np.max(p_rows))
                # Head top is minimum row of person depth or cranial landmark minus cranial offset
                if cranial_v:
                    v_head_top = min(d_min_row, min(cranial_v) - 0.05 * h)
                else:
                    v_head_top = d_min_row

                # Sole is maximum row of person depth or foot landmarks
                if feet_v:
                    v_sole = max(d_max_row, max(feet_v))
                else:
                    v_sole = d_max_row
                framing_source = "depth_component_and_keypoints"
            elif cranial_v or feet_v:
                if cranial_v:
                    v_head_top = min(cranial_v) - 0.05 * h
                if feet_v:
                    v_sole = max(feet_v)
                framing_source = "keypoints_only"
        except Exception as exc:
            logger.debug(f"MediaPipe detection in check-camera skipped: {exc}")

    head_margin_pct: Optional[float] = None
    sole_margin_pct: Optional[float] = None

    if v_head_top is not None:
        head_margin_pct = float(max(0.0, v_head_top) / h) * 100.0
        head_ok = bool(head_margin_pct >= 4.0)
        head_warn = bool(head_margin_pct > 15.0)

    if v_sole is not None:
        sole_margin_pct = float(max(0.0, h - 1.0 - v_sole) / h) * 100.0
        feet_ok = bool(sole_margin_pct >= 4.0)

    # 4. Standoff inside config range [capture_dist_min_m, capture_dist_max_m]
    standoff_ok = bool(cfg_chk.capture_dist_min_m <= med_depth_m <= cfg_chk.capture_dist_max_m)

    # 5. Vertical coverage in metres
    z_eval = med_depth_m if med_depth_m >= 0.50 else cfg_chk.camera_distance_m
    fy = float(intr.get("fy", 427.0))
    vertical_coverage_m = float(h * z_eval / fy) if fy > 0 else 0.0

    is_accepted = bool(head_ok and feet_ok and floor_ok and standoff_ok and pitch_ok)

    print("\n" + "=" * 64)
    print("REALSENSE CAMERA CHECK REPORT")
    print("=" * 64)
    print(f"Device Name:        {dev_info.get('name', 'Intel RealSense')}")
    print(f"Serial Number:      {dev_info.get('serial_number', 'UNKNOWN')}")
    print(f"Firmware Version:   {dev_info.get('firmware_version', 'UNKNOWN')}")
    print(f"Depth Scale:        {scale}")
    print(f"Stream Resolution:  {intr.get('width')}x{intr.get('height')}")
    print(f"Intrinsics:         fx={intr.get('fx'):.2f}, fy={intr.get('fy'):.2f}, cx={intr.get('cx'):.2f}, cy={intr.get('cy'):.2f}")
    print(f"Orientation:        {orient_label}")
    print(f"Framing Source:     {framing_source}")
    print(f"ROI Median Depth:   {med_depth_m:.3f} m")
    print(f"ROI Valid Fraction: {valid_frac * 100.0:.1f} %")
    print(f"Vertical Coverage:  {vertical_coverage_m:.2f} m (at {z_eval:.2f} m standoff)")
    print("-" * 64)
    print("COVERAGE & RECOMMENDED STANDOFF (5% MARGIN):")
    print(f"  Vertical coverage rate:   {cov_180.vertical_rate_m_per_m:.2f} m/m")
    print(f"  Horizontal coverage rate: {cov_180.horizontal_rate_m_per_m:.2f} m/m")
    print(f"  Recommended standoff for H=1.70m: {cov_170.recommended_standoff_m:.2f} m")
    print(f"  Recommended standoff for H=1.80m: {cov_180.recommended_standoff_m:.2f} m")
    print(f"  Recommended standoff for H=1.90m: {cov_190.recommended_standoff_m:.2f} m")
    print("-" * 64)
    print("AUTOMATIC ACCEPTANCE CHECKS:")
    warn_str = " [WARN: >15% unused vertical space]" if head_warn else ""
    if head_margin_pct is not None:
        head_desc = f"PASS (margin={head_margin_pct:.1f}%, row={v_head_top:.1f}px >= {margin_px}px){warn_str}" if head_ok else f"FAIL (margin={head_margin_pct:.1f}%, row={v_head_top:.1f}px < {margin_px}px)"
    else:
        head_desc = "FAIL (no cranial landmark or foreground detected)"

    if sole_margin_pct is not None:
        feet_desc = f"PASS (margin={sole_margin_pct:.1f}%, sole row={v_sole:.1f}px <= {h - margin_px}px)" if feet_ok else f"FAIL (margin={sole_margin_pct:.1f}%, sole row={v_sole:.1f}px > {h - margin_px}px)"
    else:
        feet_desc = "FAIL (no foot landmark or foreground detected)"

    floor_desc = f"PASS (inliers={floor.inlier_count}, normal=[{floor.normal[0]:.2f}, {floor.normal[1]:.2f}, {floor.normal[2]:.2f}])" if floor_ok else "FAIL (floor plane not found)"
    standoff_desc = f"PASS ({med_depth_m:.2f}m in [{cfg_chk.capture_dist_min_m:.2f}, {cfg_chk.capture_dist_max_m:.2f}]m)" if standoff_ok else f"FAIL ({med_depth_m:.2f}m out of [{cfg_chk.capture_dist_min_m:.2f}, {cfg_chk.capture_dist_max_m:.2f}]m)"
    pitch_desc = f"PASS ({pitch_deg:.1f} deg <= 15.0 deg)" if pitch_ok else (f"FAIL/WARN ({pitch_deg:.1f} deg > 15.0 deg)" if pitch_deg is not None else "FAIL (no floor plane)")
    print(f"  1. Head in frame (>=4%):   {head_desc}")
    print(f"  2. Sole in frame (>=4%):   {feet_desc}")
    print(f"  3. Floor pixels visible:   {floor_desc}")
    print(f"  4. Standoff in config:     {standoff_desc}")
    print(f"  5. Camera pitch <= 15 deg: {pitch_desc}")
    print("-" * 64)
    print(f"OVERALL ACCEPTANCE:          {'ACCEPTED' if is_accepted else 'REJECTED'}")
    print("=" * 64 + "\n")

    check_dir = Path(args.out) / "check"
    check_dir.mkdir(parents=True, exist_ok=True)

    from realsense_measure.fullbody.capture_ui import colorize_depth_m
    depth_vis = colorize_depth_m(d_m)

    cv2.imwrite(str(check_dir / "check_color.png"), last_color)
    cv2.imwrite(str(check_dir / "check_depth_colorized.png"), depth_vis)
    np.save(str(check_dir / "check_depth.npy"), d_m)

    metrics_txt = check_dir / "check_metrics.txt"
    with open(metrics_txt, "w", encoding="utf-8") as f:
        f.write(f"device_name: {dev_info.get('name')}\n")
        f.write(f"serial_number: {dev_info.get('serial_number')}\n")
        f.write(f"firmware_version: {dev_info.get('firmware_version')}\n")
        f.write(f"depth_scale: {scale}\n")
        f.write(f"resolution: {intr.get('width')}x{intr.get('height')}\n")
        f.write(f"intrinsics: fx={intr.get('fx')}, fy={intr.get('fy')}, cx={intr.get('cx')}, cy={intr.get('cy')}\n")
        f.write(f"portrait_rotation: {args.portrait}\n")
        f.write(f"roi_median_depth_m: {med_depth_m:.3f}\n")
        f.write(f"roi_valid_fraction: {valid_frac:.4f}\n")
        f.write(f"vertical_coverage_m: {vertical_coverage_m:.3f}\n")
        f.write(f"vertical_coverage_rate: {cov_180.vertical_rate_m_per_m:.4f}\n")
        f.write(f"horizontal_coverage_rate: {cov_180.horizontal_rate_m_per_m:.4f}\n")
        f.write(f"recommended_standoff_h170_m: {cov_170.recommended_standoff_m:.3f}\n")
        f.write(f"recommended_standoff_h180_m: {cov_180.recommended_standoff_m:.3f}\n")
        f.write(f"recommended_standoff_h190_m: {cov_190.recommended_standoff_m:.3f}\n")
        f.write(f"camera_pitch_deg: {pitch_deg if pitch_deg is not None else 0.0:.2f}\n")
        f.write(f"camera_pitch_ok: {pitch_ok}\n")
        f.write(f"standoff_in_range: {standoff_ok}\n")
        f.write(f"head_in_frame: {head_ok}\n")
        f.write(f"head_margin_pct: {head_margin_pct if head_margin_pct is not None else 0.0:.2f}\n")
        f.write(f"head_margin_warn: {head_warn}\n")
        f.write(f"feet_in_frame: {feet_ok}\n")
        f.write(f"sole_margin_pct: {sole_margin_pct if sole_margin_pct is not None else 0.0:.2f}\n")
        f.write(f"floor_visible: {floor_ok}\n")
        f.write(f"framing_source: {framing_source}\n")
        f.write(f"automatic_acceptance: {'ACCEPTED' if is_accepted else 'REJECTED'}\n")

    print(f"Snapshot saved to: {check_dir}")
    print("  - check_color.png")
    print("  - check_depth_colorized.png")
    print("  - check_depth.npy")
    print("  - check_metrics.txt\n")
    return 0


def _run_capture_background(args: argparse.Namespace) -> int:
    """Capture empty-scene background burst, save raw depth/color, and calibrate single floor plane."""
    import open3d as o3d
    from realsense_measure.fullbody.config import FullBodyConfig
    from realsense_measure.fullbody.floor import fit_floor_plane, save_floor_plane

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = FullBodyConfig()

    portrait = getattr(args, "portrait", "none")
    res_str = getattr(args, "resolution", "1280x720")
    if getattr(args, "fake", False):
        from tests.fullbody.fake_camera import FakeCamera
        if "x" in res_str:
            w_c, h_c = [int(v) for v in res_str.lower().split("x")]
        else:
            w_c, h_c = 1280, 720
        if portrait in ("cw", "ccw"):
            w_c, h_c = h_c, w_c
        cam = FakeCamera(
            width=w_c,
            height=h_c,
            depth_scale=0.001,
            depth_units="sensor_units",
            fps=30.0,
            empty_scene=True,
        )
    else:
        from realsense_measure.fullbody.capture_camera import RealSenseCaptureAdapter, full_body_camera_config
        cam_cfg = full_body_camera_config(portrait_rotation=portrait, resolution=res_str)
        cam = RealSenseCaptureAdapter(camera_config=cam_cfg, portrait_rotation=portrait)

    try:
        cam.start()
    except Exception as exc:
        print(f"Error starting camera: {exc}", file=sys.stderr)
        return 1

    burst_frames = int(getattr(args, "burst_frames", 30))
    print(f"\nCapturing {burst_frames}-frame empty-scene background burst ...")

    raw_colors = []
    raw_depths = []
    try:
        start_t = time.time()
        while len(raw_depths) < burst_frames and (time.time() - start_t) < 15.0:
            c, r, f, ts = cam.read_raw_filtered()
            if c is not None and r is not None:
                raw_colors.append(c)
                raw_depths.append(r)
            time.sleep(0.033)
    finally:
        cam.stop()

    min_required = min(burst_frames, max(1, burst_frames // 2))
    if len(raw_depths) < min_required:
        print(f"Error: Acquired only {len(raw_depths)} frames, expected at least {min_required} (target: {burst_frames}).", file=sys.stderr)
        return 1

    print(f"Acquired {len(raw_depths)} frames. Computing temporal median background depth ...")
    scale = cam.depth_scale if cam.depth_units == "sensor_units" else 1.0
    depth_m_stack = np.stack([d.astype(np.float32) * scale for d in raw_depths], axis=0)
    valid_mask = (depth_m_stack > 0.1) & (depth_m_stack < 6.0)
    masked_stack = np.where(valid_mask, depth_m_stack, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bg_median_depth = np.nanmedian(masked_stack, axis=0)
    bg_median_depth = np.nan_to_num(bg_median_depth, nan=0.0).astype(np.float32)

    rep_color = raw_colors[len(raw_colors) // 2]

    # Save raw burst frames to persistent subfolder
    raw_dir = out_dir / "raw_burst"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for idx, (c, d) in enumerate(zip(raw_colors, raw_depths)):
        cv2.imwrite(str(raw_dir / f"color_{idx:03d}.png"), c)
        np.save(str(raw_dir / f"depth_{idx:03d}.npy"), d)

    # Save processed background assets
    np.save(str(out_dir / "background_depth.npy"), bg_median_depth)
    cv2.imwrite(str(out_dir / "background_color.png"), rep_color)

    from realsense_measure.fullbody.capture_ui import colorize_depth_m
    depth_vis = colorize_depth_m(bg_median_depth)
    cv2.imwrite(str(out_dir / "background_depth_colorized.png"), depth_vis)

    intr = cam.intrinsics
    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        int(intr["width"]), int(intr["height"]),
        float(intr["fx"]), float(intr["fy"]),
        float(intr["cx"]), float(intr["cy"]),
    )

    print("Fitting floor plane from background depth map ...")
    floor = fit_floor_plane(bg_median_depth, o3d_intr, cfg)

    # Query IMU gravity if available via pyrealsense2
    imu_gravity = getattr(cam, "get_imu_gravity", lambda: None)()
    imu_angle_deg: Optional[float] = None
    if imu_gravity is not None and floor is not None:
        try:
            dot_prod = abs(float(np.dot(imu_gravity, floor.normal)))
            cos_val = np.clip(dot_prod / (np.linalg.norm(imu_gravity) * np.linalg.norm(floor.normal)), -1.0, 1.0)
            imu_angle_deg = float(np.rad2deg(np.arccos(cos_val)))
        except Exception:
            imu_angle_deg = None

    cam_h_m = float(abs(floor.coefficients[3])) if floor is not None else None
    meta = {
        "width": int(intr["width"]),
        "height": int(intr["height"]),
        "intrinsics": intr,
        "device_info": cam.device_info(),
        "burst_frames_requested": burst_frames,
        "burst_frames_captured": len(raw_depths),
        "portrait_rotation": portrait,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "floor_detected": floor is not None,
        "camera_height_m": cam_h_m,
        "floor_normal": floor.normal.tolist() if floor is not None else None,
        "imu_gravity": imu_gravity.tolist() if imu_gravity is not None else None,
        "imu_floor_angle_deg": imu_angle_deg,
    }
    with open(out_dir / "background_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if floor is not None:
        save_floor_plane(floor, out_dir / "background_floor.json")
        print("\n" + "=" * 60)
        print("BACKGROUND CAPTURE & FLOOR CALIBRATION REPORT")
        print("=" * 60)
        print(f"Destination:        {out_dir}")
        print(f"Frames Captured:    {len(raw_depths)}")
        print(f"Floor Detected:     YES (inliers={floor.inlier_count}, ratio={floor.inlier_ratio:.2f}, rmse={floor.rmse_m*1000:.1f} mm)")
        print(f"Floor Normal:       [{floor.normal[0]:.4f}, {floor.normal[1]:.4f}, {floor.normal[2]:.4f}]")
        print(f"Camera Height:      {cam_h_m:.3f} m (perpendicular distance above floor)")
        if imu_gravity is not None:
            print(f"IMU Gravity:        [{imu_gravity[0]:.4f}, {imu_gravity[1]:.4f}, {imu_gravity[2]:.4f}]")
            if imu_angle_deg is not None:
                print(f"IMU vs Floor Angle: {imu_angle_deg:.2f} deg")
        else:
            print("IMU Sensor:         Not available or unstreamable")
        print(f"Floor Assets:       {out_dir / 'background_floor.json'}")
        print("=" * 60 + "\n")
    else:
        print("\n" + "=" * 60)
        print("BACKGROUND CAPTURE REPORT (FLOOR NOT DETECTED)")
        print("=" * 60)
        print(f"Destination:        {out_dir}")
        print(f"Frames Captured:    {len(raw_depths)}")
        print("Floor Detected:     NO (RANSAC found insufficient lower-region inliers)")
        print("=" * 60 + "\n")

    return 0


def run_pipeline(capture_dir: Path, out_dir: Path, cfg: Any) -> Any:
    """Execute the complete full-body reconstruction and measurement pipeline."""
    from realsense_measure.fullbody.alignment import align_stops
    from realsense_measure.fullbody.confidence import compute_confidence_map
    from realsense_measure.fullbody.floor import fit_floor_plane, load_floor_plane
    from realsense_measure.fullbody.fusion import compute_foreground_mask, fuse_aligned_stops
    from realsense_measure.fullbody.io import load_capture_session, save_scan_result
    from realsense_measure.fullbody.keypoints import MediaPipePoseDetector, lift_keypoints_to_3d
    from realsense_measure.fullbody.measure_scan import measure_fused_scan
    from realsense_measure.fullbody.types import ScanResult
    import open3d as o3d

    logger.info(f"Loading capture session from {capture_dir} ...")
    stops = load_capture_session(capture_dir, cfg)

    # 1. Resolve single background floor plane & background depth
    bg_dir = getattr(cfg, "background_dir", None)
    if bg_dir is None:
        bg_dir = Path("raw_captures/background")
    else:
        bg_dir = Path(bg_dir)

    session_bg = Path(capture_dir) / "background"
    active_bg_dir = session_bg if session_bg.exists() else bg_dir

    bg_depth: Optional[np.ndarray] = None
    rep_floor: Optional[FloorPlane] = None
    floor_observed = False

    bg_depth_p = active_bg_dir / "background_depth.npy"
    if bg_depth_p.exists():
        try:
            bg_depth = np.load(str(bg_depth_p)).astype(np.float32)
            logger.info(f"Loaded background depth map from {bg_depth_p} (shape {bg_depth.shape})")
        except Exception as exc:
            logger.warning(f"Failed to load background depth: {exc}")

    bg_floor_p = active_bg_dir / "background_floor.json"
    if bg_floor_p.exists():
        rep_floor = load_floor_plane(bg_floor_p)
        if rep_floor is not None:
            floor_observed = True
            logger.info(
                f"Using calibrated single floor plane from {bg_floor_p}: "
                f"normal={rep_floor.normal}, height={rep_floor.point_on_plane[1]:.3f} m"
            )

    if rep_floor is None and stops and stops[0].depth is not None:
        # Fit floor plane ONCE from the first stop (no per-stop consensus)
        rep_floor = fit_floor_plane(stops[0].depth, stops[0].intrinsics, cfg)
        if rep_floor is not None:
            floor_observed = True
            logger.info(
                f"Fitted single floor plane from stop 0: "
                f"normal={rep_floor.normal}, height={rep_floor.point_on_plane[1]:.3f} m"
            )

    # If there is no background and floor could not be fitted, fail loudly with instructions
    if rep_floor is None:
        raise RuntimeError(
            f"No background calibration found in '{active_bg_dir}'.\n"
            "Full-body scanning requires a pre-calibrated empty scene background and floor plane.\n"
            "Please run: python -m realsense_measure.fullbody capture-background --out <SESSION_OR_BG_DIR>\n"
            "to capture background depth and calibrate the floor plane before running scan."
        )

    camera_height_m = float(abs(rep_floor.coefficients[3]))
    session_meta = {"camera_height_m": camera_height_m}
    logger.info(f"Floor plane established (observed={floor_observed}, normal={rep_floor.normal}, camera_height={camera_height_m:.3f} m).")

    detector = MediaPipePoseDetector()
    logger.info(f"Processing {len(stops)} stops (keypoint detection, confidence map, background subtraction) ...")
    for s in stops:
        # 1. 2D keypoints and 3D lifting (MediaPipe kept ONLY for keypoints)
        pts_2d, conf_2d, vis_2d = detector.detect(s.color, depth_m=s.depth)
        s.keypoints_camera = lift_keypoints_to_3d(pts_2d, conf_2d, vis_2d, s.depth, s.intrinsics, cfg)

        # 2. Assign the single floor plane to each stop
        s.floor_plane = rep_floor

        # 3. Confidence map
        conf_map, filtered_depth = compute_confidence_map(
            s.depth, s.intrinsics, cfg, depth_std_m=s.depth_std, valid_counts=s.valid_counts
        )
        s.confidence_map = conf_map

        # 4. Point cloud in camera frame with shared background-depth subtraction
        color_rgb = np.ascontiguousarray(s.color[:, :, ::-1])
        fg_mask = compute_foreground_mask(s.depth, bg_depth, cfg)
        depth_m = s.depth.copy()
        depth_m[~fg_mask] = 0.0
        if s.confidence_map is not None:
            depth_m[s.confidence_map < cfg.min_confidence_threshold] = 0.0

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_rgb),
            o3d.geometry.Image(depth_m),
            depth_scale=1.0,
            depth_trunc=cfg.depth_max_m,
            convert_rgb_to_intensity=False,
        )
        s.pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, s.intrinsics)

    # Alignment
    logger.info("Aligning multi-stop point clouds with joint prior and 4-DoF MAP-ICP ...")
    aligned = align_stops(stops, rep_floor, cfg)

    # Fusion
    logger.info("Fusing aligned point clouds via Open3D TSDF Integration ...")
    mesh, fused_pcd, hole_map = fuse_aligned_stops(stops, aligned, cfg, bg_depth=bg_depth)

    # Anthropometric measurements
    logger.info("Extracting tape-style anthropometric measurements ...")
    from realsense_measure.fullbody.keypoints import aggregate_canonical_keypoints
    ref_kpts = aggregate_canonical_keypoints(aligned, cfg)
    if ref_kpts is None and aligned:
        ref_kpts = aligned[0].keypoints_canonical

    measurements = measure_fused_scan(
        fused_pcd,
        ref_kpts,
        cfg,
        hole_map=hole_map,
        floor_observed=floor_observed,
    )

    subj_id = "subject_01"
    meta_p = Path(capture_dir) / "meta.json"
    if meta_p.exists():
        try:
            with open(meta_p, "r", encoding="utf-8") as mf:
                subj_id = json.load(mf).get("subject_id", subj_id)
        except Exception:
            pass

    scan_res = ScanResult(
        subject_id=subj_id,
        mesh=mesh,
        pcd=fused_pcd,
        aligned_stops=aligned,
        measurements=measurements,
        hole_map=hole_map,
    )

    save_scan_result(scan_res, out_dir)
    return scan_res


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.subcommand == "capture":
        return _run_capture(args)

    elif args.subcommand == "check-camera":
        return _run_check_camera(args)

    elif args.subcommand == "capture-background":
        return _run_capture_background(args)

    elif args.subcommand == "scan":
        from realsense_measure.fullbody.config import FullBodyConfig
        cfg = FullBodyConfig()
        cfg.output_dir = Path(args.out)
        cfg.num_stops = args.stops
        cfg.tsdf_voxel_length_m = args.voxel
        cfg.verbose = args.verbose
        if getattr(args, "background", None):
            cfg.background_dir = Path(args.background)

        result = run_pipeline(Path(args.capture), Path(args.out), cfg)
        print("\n" + result.summary_table() + "\n")
        return 0

    elif args.subcommand == "measure":
        from realsense_measure.fullbody.io import load_scan_result
        p = Path(args.scan)
        scan = load_scan_result(p)
        print("\n" + scan.summary_table() + "\n")
        return 0

    elif args.subcommand == "validate":
        from realsense_measure.fullbody.io import load_scan_result
        from realsense_measure.fullbody.validate import compare_with_tape, format_validation_table, load_tape_measurements
        scan = load_scan_result(Path(args.scan))
        tape_vals, rep_floor, subj = load_tape_measurements(Path(args.tape))
        report = compare_with_tape(scan.measurements, tape_vals, rep_floor)
        print("\n" + format_validation_table(report) + "\n")
        return 0

    elif args.subcommand == "sweet-spot":
        from realsense_measure.fullbody.config import FullBodyConfig
        from realsense_measure.fullbody.io import load_scan_result
        from realsense_measure.fullbody.sweet_spot import evaluate_capture_configuration
        scan = load_scan_result(Path(args.scan))
        cfg = FullBodyConfig()
        if scan.pcd is not None:
            sw_rep = evaluate_capture_configuration(scan.pcd, None, cfg)
            print(f"\nConfiguration Score: {sw_rep.total_score:.1f}")
            print(f"Per-part scores: {sw_rep.per_part_scores}")
            if sw_rep.under_covered_parts:
                print(f"Under-covered parts: {sw_rep.under_covered_parts}")
            if sw_rep.recommendations:
                print(f"Recommendations: {sw_rep.recommendations}\n")
        return 0

    elif args.subcommand == "review":
        import open3d as o3d
        from realsense_measure.fullbody.io import load_scan_result
        scan = load_scan_result(Path(args.scan))
        geoms = []
        if scan.mesh is not None and len(scan.mesh.vertices) > 0:
            geoms.append(scan.mesh)
        elif scan.pcd is not None and len(scan.pcd.points) > 0:
            geoms.append(scan.pcd)
        if geoms:
            o3d.visualization.draw_geometries(geoms, window_name=f"Full-Body Scan: {args.scan}")
        else:
            print("No 3D mesh or point cloud found to review.")
        return 0

    elif args.subcommand == "keypoints-check":
        c_path = Path(args.color)
        d_path = Path(args.depth) if args.depth else None
        o_path = Path(args.out) if args.out else None
        return _run_keypoints_check(c_path, d_path, o_path)

    elif args.subcommand == "debug-stop":
        from realsense_measure.fullbody.debug import debug_stop
        debug_stop(args.session_dir, args.stop, out_dir=args.out)
        return 0

    elif args.subcommand == "validate-session":
        from realsense_measure.fullbody.validate import validate_session
        validate_session(
            args.session_dir,
            tape_path=args.tape,
            out_path=args.out,
            report_path=args.report,
        )
        return 0

    elif args.subcommand == "verify-session":
        from realsense_measure.fullbody.verify_session import verify_session
        return verify_session(Path(args.capture))

    return 0


def _run_keypoints_check(color_path: Path, depth_path: Optional[Path], out_path: Optional[Path]) -> int:
    """Execute MediaPipe Pose detection on a single frame, print confidence table, and write overlay."""
    color = cv2.imread(str(color_path))
    if color is None:
        print(f"Error: Could not read color image from {color_path}", file=sys.stderr)
        return 1
    h, w = color.shape[:2]

    try:
        import mediapipe as mp
    except ImportError:
        print("Error: mediapipe is not installed.", file=sys.stderr)
        return 1

    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "pose"):
        print("Error: MediaPipe solutions.pose is not available in installed mediapipe build.", file=sys.stderr)
        return 1

    mp_pose_names = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_thumb", "right_thumb",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]

    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

    if not res.pose_landmarks:
        print("No human pose detected by MediaPipe in image.", file=sys.stderr)
        return 1

    depth = np.load(str(depth_path)) if depth_path and depth_path.exists() else None

    print("\n" + "=" * 82)
    print(f"MEDIAPIPE POSE 33-LANDMARK AUDIT: {color_path.name}")
    print("=" * 82)
    if depth is not None:
        print(f"{'Idx':<4} | {'Landmark':<20} | {'u (px)':>7} | {'v (px)':>7} | {'Conf':>6} | {'Vis':>6} | {'Depth (m)':>9}")
    else:
        print(f"{'Idx':<4} | {'Landmark':<20} | {'u (px)':>7} | {'v (px)':>7} | {'Conf':>6} | {'Vis':>6}")
    print("-" * 82)

    vis_img = color.copy()
    for idx, lm in enumerate(res.pose_landmarks.landmark):
        name = mp_pose_names[idx] if idx < len(mp_pose_names) else f"joint_{idx}"
        u = float(lm.x * (w - 1))
        v = float(lm.y * (h - 1))
        conf = float(lm.presence if hasattr(lm, "presence") and lm.presence > 0 else (1.0 if lm.visibility > 0.5 else lm.visibility))
        vis = float(lm.visibility)

        if depth is not None:
            iu = int(np.clip(round(u), 0, w - 1))
            iv = int(np.clip(round(v), 0, h - 1))
            zv = float(depth[iv, iu])
            d_str = f"{zv:.3f}" if zv > 0 else "0.000"
            print(f"{idx:<4} | {name:<20} | {u:>7.1f} | {v:>7.1f} | {conf:>6.2f} | {vis:>6.2f} | {d_str:>9}")
        else:
            print(f"{idx:<4} | {name:<20} | {u:>7.1f} | {v:>7.1f} | {conf:>6.2f} | {vis:>6.2f}")

        cv2.circle(vis_img, (int(round(u)), int(round(v))), 4, (0, 255, 0), -1)
        cv2.putText(vis_img, str(idx), (int(round(u)) + 5, int(round(v)) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    if hasattr(mp.solutions.pose, "POSE_CONNECTIONS"):
        for a, b in mp.solutions.pose.POSE_CONNECTIONS:
            lm_a = res.pose_landmarks.landmark[a]
            lm_b = res.pose_landmarks.landmark[b]
            pt_a = (int(round(lm_a.x * (w - 1))), int(round(lm_a.y * (h - 1))))
            pt_b = (int(round(lm_b.x * (w - 1))), int(round(lm_b.y * (h - 1))))
            cv2.line(vis_img, pt_a, pt_b, (0, 200, 255), 2)

    print("=" * 82 + "\n")

    out_file = out_path if out_path else color_path.parent / f"{color_path.stem}_keypoints.png"
    cv2.imwrite(str(out_file), vis_img)
    print(f"Saved annotated keypoint overlay to: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
