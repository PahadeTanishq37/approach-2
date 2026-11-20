# Stop-and-Go Protocol V1 Implementation Report

This report documents the implementation and verification of the `stop-and-go-v1` capture protocol and session verifier for full-body scanning in `realsense_measure`.

Branch: `capture/stop-and-go-v1`

---

## 1. Files Changed and Added

| File | Status | Summary of Changes |
|---|---|---|
| `realsense_measure/body/capture_session.py` | Modified | Updated `validate_anonymous_subject_id` regex to accept rigid object and test identifiers (`BUCKET_01`, `OBJ_99`, `OBJECT_01`, `CYLINDER_A`). |
| `realsense_measure/fullbody/session_format.py` | Modified | Extended subject-ID validation regex; updated `SessionWriter` to accept and serialize rig metadata, protocol metadata (`protocol_version="stop-and-go-v1"`, nominal stop angles, requested frames); added graceful warning for missing background calibration. |
| `realsense_measure/fullbody/capture_controller.py` | Modified | Set default `burst_frames=60`, `burst_timeout_s=8.0`; added `play_audio_cue` with `winsound.Beep` (terminal bell fallback); implemented object-mode gating (`fg_fraction >= 0.002`, `bbox_h_ratio >= 0.10`, zero border clipping on all 4 borders, bypassing head/sole margin and person height checks); enforced proportional $\ge 90\%$ burst frame validation; added audio cues on hold-start and burst-end. |
| `realsense_measure/fullbody/capture_ui.py` | Modified | Updated on-screen turn prompt HUD overlay to: `"Stop {k}/{n}: turn to tick {k} ({target_angle:.0f} deg)"`. |
| `realsense_measure/fullbody/cli.py` | Modified | Added rig metadata CLI options (`--cam-height-m`, `--cam-distance-m`, `--cam-tilt-deg`, `--clothing`, `--pose`, `--notes`), object mode options (`--subject-type`, `--object-diameter-mm`), `--no-beep`; updated `--burst-frames` default to 60; added background intrinsics inheritance in `--fake` mode; implemented `verify-session` subcommand; updated session summary to report recursive file count. |
| `realsense_measure/fullbody/verify_session.py` | Added | Session integrity and sanity verifier (`python -m realsense_measure.fullbody verify-session --capture <dir>`): verifies file counts (60/60 color PNG & depth NPY), SHA256 checksums, median distance, border clipping, valid depth fraction, sway (mm), and floor-plane fit residual (mm). Generates 8-tile contact sheet (`contact_sheet.png`) and `verify_report.json`. |
| `tests/fullbody/fake_camera.py` | Modified | Added optional `cx` and `cy` keyword arguments and properties to match background profile when inherited. |
| `tests/fullbody/test_stop_and_go_v1.py` | Added | Test suite covering object subject ID validation, 60-frame burst saving on FakeCamera, object mode gating, rig metadata persistence round-trip, and `verify-session` PASS/FAIL behaviors. |

---

## 2. New CLI Flags and Defaults

The `capture` subcommand now supports the following options:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--burst-frames` | int | `60` | Number of raw frames recorded per stop (~2 s at 30 fps) |
| `--subject-type` | str (`person`\|`object`) | `person` | Subject type (`object` bypasses person height/sole gating) |
| `--object-diameter-mm` | float | `None` | Known physical diameter of object (e.g., bucket/cylinder) |
| `--cam-height-m` | float | `None` | Measured camera lens optical center height above floor (m) |
| `--cam-distance-m` | float | `None` | Measured distance from camera to center of turntable (m) |
| `--cam-tilt-deg` | float | `0.0` | Camera tilt angle relative to horizontal (deg) |
| `--clothing` | str | `None` | Clothing description (e.g., "fitted athletic wear") |
| `--pose` | str | `"A-pose"` | Captured subject pose |
| `--notes` | str | `None` | Free-text session notes |
| `--no-beep` | flag | `False` | Disable audio cues at hold start and burst end |

The new `verify-session` subcommand:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--capture` | Path | (Required) | Path to full-body session capture directory to inspect |

---

## 3. Empirical Write Performance & Live Loop Timing

### Benchmark Results
- **Frames per stop**: 60 Color PNGs (1280x720, 8-bit BGR, compressed via OpenCV) + 60 Depth NPYs (1280x720, `uint16`, binary numpy format)
- **Total write time per stop**: `0.405 s` (measured on FakeCamera test run)
- **Average time per frame pair**: `6.75 ms`

### Live Loop Integrity Confirmation
During real-time acquisition in `CAPTURE_BURST`, incoming camera frames are pushed directly into in-memory FIFO buffers (`self._burst_colors`, `self._burst_depths`, `self._burst_timestamps`) at 30 fps without blocking disk I/O. 

Disk writes execute entirely within the `_step_evaluate` state *after* the 60 frames are captured and prior to entering `TURN_PROMPT`. Because writing completes in ~0.4 s while the subject transitions / turns to the next tick over 2–10 s, disk serialization has **zero impact** on live frame acquisition and causes no dropped frames.

---

## 4. Test Results vs Baseline

| Metric | Baseline (`cleanup/fullbody-reset`) | Current (`capture/stop-and-go-v1`) | Difference |
|---|---|---|---|
| **Passed Tests** | 399 | **404** | +5 (new stop-and-go-v1 test suite) |
| **Skipped Tests** | 3 | 3 | 0 |
| **Deselected Tests** | 39 | 39 | 0 |
| **Failed Tests** | 0 | 0 | 0 |

All 5 new tests in `tests/fullbody/test_stop_and_go_v1.py` pass:
- `test_subject_id_accepts_bucket_and_rigid_objects`: PASS
- `test_60_frame_burst_saving_fake_camera`: PASS
- `test_object_mode_passes_short_object`: PASS
- `test_rig_and_protocol_metadata_roundtrip`: PASS
- `test_verify_session_good_missing_and_clipped`: PASS

---

## 5. End-to-End Terminal Output

### Capture Run (`--fake --headless --subject-type object --subject-id BUCKET_01`)
```
> python -m realsense_measure.fullbody capture --fake --headless --subject-type object --subject-id BUCKET_01 --turn-prompt-timeout 0.2 --hold-still-s 0.1 --out raw_captures/test_bucket

[WAIT_FOR_SUBJECT] Dist: 2.00m | Head: -- | Sole: -- | Clip: -- | BBox: (359, 51, 921, 611) | Reason: None
Stop 1/8: turn to tick 1 (0 deg)
Stop 0 accepted: 60 frames saved, sway 0.0 mm, foreground 229573 px
[TURN_PROMPT     ] Dist: 2.00m | Head: 7.1% | Sole: 15.0% | Clip: 0.0% | BBox: (359, 51, 921, 611) | Reason: None
Stop 2/8: turn to tick 2 (45 deg)
Stop 1 accepted: 60 frames saved, sway 0.0 mm, foreground 229573 px
[TURN_PROMPT     ] Dist: 2.00m | Head: 7.1% | Sole: 15.0% | Clip: 0.0% | BBox: (359, 51, 921, 611) | Reason: None
Stop 3/8: turn to tick 3 (90 deg)
Stop 2 accepted: 60 frames saved, sway 0.0 mm, foreground 229573 px
[TURN_PROMPT     ] Dist: 2.00m | Head: 7.1% | Sole: 15.0% | Clip: 0.0% | BBox: (359, 51, 921, 611) | Reason: None
Stop 4/8: turn to tick 4 (135 deg)
Stop 3 accepted: 60 frames saved, sway 0.0 mm, foreground 229573 px
[TURN_PROMPT     ] Dist: 2.00m | Head: 7.1% | Sole: 15.0% | Clip: 0.0% | BBox: (359, 51, 921, 611) | Reason: None
Stop 5/8: turn to tick 5 (180 deg)
Stop 4 accepted: 60 frames saved, sway 0.0 mm, foreground 229573 px
[TURN_PROMPT     ] Dist: 2.00m | Head: 7.1% | Sole: 15.0% | Clip: 0.0% | BBox: (359, 51, 921, 611) | Reason: None
Stop 6/8: turn to tick 6 (225 deg)
Stop 5 accepted: 60 frames saved, sway 0.0 mm, foreground 229573 px
[TURN_PROMPT     ] Dist: 2.00m | Head: 7.1% | Sole: 15.0% | Clip: 0.0% | BBox: (359, 51, 921, 611) | Reason: None
Stop 7/8: turn to tick 7 (270 deg)
Stop 6 accepted: 60 frames saved, sway 0.0 mm, foreground 229573 px
[TURN_PROMPT     ] Dist: 2.00m | Head: 7.1% | Sole: 15.0% | Clip: 0.0% | BBox: (359, 51, 921, 611) | Reason: None
Stop 8/8: turn to tick 8 (315 deg)
Stop 7 accepted: 60 frames saved, sway 0.0 mm, foreground 229573 px

============================================================
CAPTURE SESSION SUMMARY
============================================================
Session Directory:  raw_captures\test_bucket
Accepted Stops:     8 of 8
  - stop_00: 126 files
  - stop_01: 126 files
  - stop_02: 126 files
  - stop_03: 126 files
  - stop_04: 126 files
  - stop_05: 126 files
  - stop_06: 126 files
  - stop_07: 126 files
============================================================

Capture session successfully finalized at: raw_captures\test_bucket
Accepted all 8 stops.
```

### Session Verification Run (`verify-session`)
```
> python -m realsense_measure.fullbody verify-session --capture raw_captures/test_bucket

==================================================================================================
SESSION VERIFICATION REPORT: raw_captures\test_bucket
==================================================================================================
Stop  | Status | Frames (Saved/Req) | SHA256  | Dist (m) | Clipped | Valid % | Sway (mm) | Floor Res (mm)
--------------------------------------------------------------------------------------------------
0     | PASS   | 60 / 60            | OK      | 2.00     | No      | 24.9%   | 0.0       | --            
1     | PASS   | 60 / 60            | OK      | 2.00     | No      | 24.9%   | 0.0       | --            
2     | PASS   | 60 / 60            | OK      | 2.00     | No      | 24.9%   | 0.0       | --            
3     | PASS   | 60 / 60            | OK      | 2.00     | No      | 24.9%   | 0.0       | --            
4     | PASS   | 60 / 60            | OK      | 2.00     | No      | 24.9%   | 0.0       | --            
5     | PASS   | 60 / 60            | OK      | 2.00     | No      | 24.9%   | 0.0       | --            
6     | PASS   | 60 / 60            | OK      | 2.00     | No      | 24.9%   | 0.0       | --            
7     | PASS   | 60 / 60            | OK      | 2.00     | No      | 24.9%   | 0.0       | --            
--------------------------------------------------------------------------------------------------
SESSION CHECKS:
  - All 8 stops present and accepted: PASS (8/8)
  - Background calibration:           PASS (present)
  - Camera level:                     N/A (no floor plane detected)
  - Camera orientation & resolution:  PASS (landscape, 1280x720)
--------------------------------------------------------------------------------------------------
OVERALL VERDICT: PASS
Contact sheet saved to: raw_captures\test_bucket\contact_sheet.png
Verification report saved to: raw_captures\test_bucket\verify_report.json
==================================================================================================
```

---

## 6. Scope Boundaries & Untouched Components

In accordance with task constraints, the following modules were strictly preserved and **not modified**:
- `realsense_measure/fullbody/alignment.py`
- `realsense_measure/fullbody/fusion.py`
- All head measurement and head reconstruction modules (`realsense_measure/head/`, `realsense_measure/pipeline/`, etc.)
- Existing core utility modules not designated for capture updates
