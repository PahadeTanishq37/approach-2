# RealSense Camera Check Verification Report: Real D455 Capture Configuration

**Date**: 2026-11-23  
**Device**: Intel RealSense D455F (Serial: `342622300088`, Firmware: `5.17.3.10`, USB: `3.2`)  
**Command Executed**: `python -m realsense_measure.fullbody check-camera --resolution 1280x720`

---

## 1. Printed Console Output on Real D455F

```text
2026-11-23 10:15:27,992 [INFO] realsense_measure.fullbody.capture_camera: Discarding 30 warm-up frames ...
2026-11-23 10:15:33,451 [WARNING] realsense_measure.fullbody.floor:   [floor] Fitted plane is elevated (Y=0.20 m < 0.50 m below camera); rejected as table/desk/elevated surface.

  RealSense D455 Camera Calibration:
    RGB resolution:   1280x720
    Depth resolution: 1280x720
    Aligned:          True (depth aligned onto color optical frame)
    fx:               645.50
    fy:               644.66
    cx:               637.24
    cy:               366.94


================================================================
REALSENSE CAMERA CHECK REPORT
================================================================
Device Name:        RealSense D455F
Serial Number:      342622300088
Firmware Version:   5.17.3.10
Depth Scale:        0.001
Requested Profile:  1280x720 @ 30 fps
Actual Color Stream:1280x720 @ 30 fps (format.bgr8)
Actual Depth Stream:1280x720 @ 30 fps (format.z16)
Stream Resolution:  1280x720
Intrinsics:         fx=645.50, fy=644.66, cx=637.24, cy=366.94
Orientation:        LANDSCAPE (no rotation)
Framing Source:     background_foreground_mask (background)
ROI Median Depth:   0.757 m
ROI Valid Fraction: 37.8 %
Vertical Coverage:  0.85 m (at 0.76 m standoff)
----------------------------------------------------------------
FRAMING & FOREGROUND BLOB ANALYSIS:
  Foreground Blob:           Found (260114 px, 28.22% of image)
  Horizontal Centre Offset:  -334.5 px (-26.1% from centre)
  Vertical Centre Offset:    -111.1 px (-15.4% from centre)
  Touches Image Border:      NO (clear of all 4 borders)
  Central 20x20 Median Depth:0.759 m (107/400 valid px)
----------------------------------------------------------------
COVERAGE & RECOMMENDED STANDOFF (5% MARGIN):
  Vertical coverage rate:   1.12 m/m
  Horizontal coverage rate: 1.98 m/m
  Recommended standoff for H=1.70m: 1.69 m
  Recommended standoff for H=1.80m: 1.79 m
  Recommended standoff for H=1.90m: 1.89 m
----------------------------------------------------------------
AUTOMATIC ACCEPTANCE CHECKS:
  1. Stream profile match:   PASS (1280x720@30fps matches requested 1280x720)
  2. Head in frame (>=4%):   FAIL (margin=0.6%, row=4.0px < 28px)
  3. Sole in frame (>=4%):   PASS (margin=8.3%, sole row=659.0px <= 692px)
  4. Floor pixels visible:   FAIL (floor plane not found)
  5. Standoff in config:     FAIL (0.76m out of [1.70, 2.40]m)
  6. Camera pitch <= 15 deg: FAIL (no floor plane)
----------------------------------------------------------------
OVERALL ACCEPTANCE:          REJECTED
================================================================

Snapshot saved to: scan_output\check
  - check_color.png
  - check_depth_colorized.png
  - check_depth.npy
  - check_metrics.txt
```

---

## 2. Snapshot Metrics File (`scan_output/check/check_metrics.txt`)

```text
device_name: RealSense D455F
serial_number: 342622300088
firmware_version: 5.17.3.10
depth_scale: 0.001
requested_resolution: 1280x720
negotiated_depth_stream: 1280x720@30fps
negotiated_color_stream: 1280x720@30fps
stream_profile_match: True
resolution: 1280x720
intrinsics: fx=645.5030517578125, fy=644.6593627929688, cx=637.2412109375, cy=366.93597412109375
portrait_rotation: none
roi_median_depth_m: 0.757
roi_valid_fraction: 0.3776
vertical_coverage_m: 0.845
vertical_coverage_rate: 1.1169
horizontal_coverage_rate: 1.9829
recommended_standoff_h170_m: 1.691
recommended_standoff_h180_m: 1.791
recommended_standoff_h190_m: 1.890
camera_pitch_deg: 0.00
camera_pitch_ok: False
standoff_in_range: False
head_in_frame: False
head_margin_pct: 0.56
head_margin_warn: False
feet_in_frame: True
sole_margin_pct: 8.33
floor_visible: False
framing_source: background_foreground_mask (background)
foreground_blob_found: True
foreground_blob_area_px: 260114
foreground_blob_area_pct: 28.22
foreground_offset_x_px: -334.5
foreground_offset_y_px: -111.1
foreground_offset_x_pct: -26.13
foreground_offset_y_pct: -15.44
touches_border: False
touched_borders: none
central_20x20_median_depth_m: 0.759
central_20x20_valid_px: 107
automatic_acceptance: REJECTED
```

---

## 3. Technical Verification & Confirmation

### Requirement 1: Resolution Verification & Profile Negotiation
- Added `--resolution` (`1280x720` | `848x480`, default `1280x720`) to `check-camera`.
- `RealSenseCaptureAdapter` inspects `profile.get_stream(rs.stream.depth).as_video_stream_profile()` and `profile.get_stream(rs.stream.color).as_video_stream_profile()` to record the active hardware stream profile.
- Both streams are printed, compared against the requested profile (`1280x720 @ 30 fps`), and verified. If hardware fails to negotiate the requested stream configuration, a loud warning is emitted and automatic acceptance check #1 flags `FAIL`.

### Requirement 2: Framing Info & Central 20x20 Window Depth
- **Largest Foreground Blob**: Extracted via `subject_mask(d_m, bg_depth_map, cfg_chk, allow_missing=True)` and `cv2.connectedComponentsWithStats`.
  - Centroid offset relative to image centre $(w/2, h/2)$:
    - $\Delta X = -334.5\text{ px}$ ($-26.1\%$ lateral offset).
    - $\Delta Y = -111.1\text{ px}$ ($-15.4\%$ vertical offset).
  - Border clipping check: evaluates if foreground pixels touch top, bottom, left, or right image boundaries. In this snapshot: `Touches Image Border: NO (clear of all 4 borders)`.
- **Central 20x20 Window**: Evaluates depth in a $20 \times 20$ patch centred at $(w/2, h/2)$:
  - Result: `0.759 m` ($107/400$ valid depth pixels).

### Requirement 3: Preview-Only Isolation & Raw Capture Integrity
- Verified that preview-only features (`WINDOW_NORMAL` downscaling, colormap JET mapping, `cam.last_colorized` caching) do not touch or modify raw capture disk writes.
- Added automated test `test_preview_colorizer_preserves_raw_capture_bytes`:
  - Executes full 2-stop burst capture with and without the preview colorizer enabled.
  - Compares every raw `.png` and `.npy` file byte-for-byte (`f1.read_bytes() == f2.read_bytes()`).
  - Verifies SHA-256 hashes in `hashes.json` match identically.
  - Verifies `filter_settings` in `meta.json` maintains `raw_only: True`, with spatial, temporal, hole-filling, and decimation strictly disabled (`False`).
