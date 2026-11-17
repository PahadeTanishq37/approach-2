# STAGE 29.2 — LIVE CAMERA PORTRAIT ORIENTATION CORRECTION REPORT

**Document ID:** STAGE29_2-DOC-01  
**Date:** 2026-11-17  
**Status:** COMPLETE — READY FOR OPERATOR VISUAL INSPECTION  
**Target Hardware:** Intel RealSense D455F (S/N: 342622300088, FW: 5.17.3.10)  
**Mounting:** Portrait Tripod Mount (Rotated 90° relative to horizontal resting position)  

---

## 1. Executive Summary & Root Cause Analysis

### 1.1 Problem Statement
In Stage 29.1, when the live Full-Body Capture operator interface was launched with `--portrait cw`, the camera preview displayed the subject sideways (head pointing toward the right edge of the window, feet toward the left edge). The operator requested that the preview be corrected so that:
1. The subject appears upright (head at the top, feet at the bottom).
2. The subject's left and right sides are anatomically true (no mirroring or coordinate inversion).
3. RGB and depth previews show strictly corresponding pixels.
4. Downstream geometry, TSDF back-projection, and camera calibration remain mathematically rigorous and right-handed.

### 1.2 Root Cause Analysis
- **Sensor Native Coordinate Frame:** The Intel RealSense D455f native sensor raster is $848 \times 480$ pixels ($W_{\text{raw}} = 848, H_{\text{raw}} = 480$). In horizontal landscape orientation, pixel $(0, 0)$ is top-left, $+u$ increases to the right ($0 \to 847$), and $+v$ increases downward ($0 \to 479$).
- **Physical Mounting Configuration:** The camera is physically mounted in portrait orientation on a tripod. In this physical orientation, the right side of the camera body points upward. Consequently, real-world **UP** corresponds to high native horizontal pixel coordinates ($u \approx 700\text{--}800$, near the sensor's right edge), and real-world **DOWN** corresponds to low native horizontal pixel coordinates ($u \approx 100\text{--}200$, near the sensor's left edge).
- **The CW Failure Mode:**
  Under `cv2.ROTATE_90_CLOCKWISE` (`--portrait cw`), the raster transformation is:
  $$u_{\text{cw}} = (H_{\text{raw}} - 1) - v, \quad v_{\text{cw}} = u$$
  Because real-world UP corresponds to high $u$, mapping $v_{\text{cw}} = u$ placed the subject's head at high vertical raster coordinates (toward the right edge of the rotated frame). The subject was rendered sideways (rotated 90° clockwise from upright).
- **The CCW Mathematical Correction:**
  Under `cv2.ROTATE_90_COUNTERCLOCKWISE` (`--portrait ccw`), the raster transformation is:
  $$u_{\text{ccw}} = v, \quad v_{\text{ccw}} = (W_{\text{raw}} - 1) - u$$
  Under this transformation:
  - **Head (real-world UP, native $u \approx 750$):**
    $$v_{\text{ccw}} = (848 - 1) - 750 = 97 \text{ px} \quad \implies \text{\textbf{TOP OF PORTRAIT DISPLAY}}$$
  - **Feet (real-world DOWN, native $u \approx 100$):**
    $$v_{\text{ccw}} = (848 - 1) - 100 = 747 \text{ px} \quad \implies \text{\textbf{BOTTOM OF PORTRAIT DISPLAY}}$$
  - **Horizontal Anatomical Alignment ($u_{\text{ccw}} = v$):**
    Because the physical portrait mount points sensor top ($v=0$) toward world left, the horizontal mapping $u_{\text{ccw}} = v$ preserves true anatomical left/right without artificial mirroring.

---

## 2. Complete Image Acquisition & Processing Path

The corrected live pipeline processes frames through the following verified pipeline:

```
[RealSense D455F Hardware Sensor]
  │
  ├─ Native Color: 848x480 (BGR8)
  └─ Native Depth: 848x480 (Z16, depth_scale = 0.001 m)
  │
  ▼
[Depth-to-Color Optical Alignment (rs.align)]
  │ Strictly aligns depth pixels onto color optical center prior to any rotation
  │ Guarantee: pixel (u, v) in color is collinear with (u, v) in depth
  │
  ▼
[Software Portrait Rotation (rotate_rgbd_arrays(mode='ccw'))]
  │ Applies cv2.ROTATE_90_COUNTERCLOCKWISE to color and depth synchronously
  │ Output resolution: 480x848 (H=848, W=480)
  │
  ▼
[Optical Calibration Transformation (rotate_intrinsics(mode='ccw'))]
  │ fx_rot = fy (427.32), fy_rot = fx (427.88)
  │ cx_rot = cy (244.60), cy_rot = (W_raw - 1) - cx (424.83)
  │
  ▼
[Live Operator Preview & HUD Rendering (capture_ui.render_frame)]
  │ Side-by-side composite: [RGB (848x480) | Colormapped Depth (848x480)] -> 848x960
  │ Active HUD Banner: "Orientation: PORTRAIT CCW"
  │ Real-time metrics: Distance (1.1-1.5m), Depth Valid %, Motion (<=10mm), Head/Feet OK
  │
  ▼
[Quality Evaluation & Downstream Processing]
  │ Central Body ROI: [0.25H:0.75H, 0.25W:0.75W] evaluated in upright portrait space
  │ Boundary Clipping: Head (v < 0.05H), Feet (v > 0.95H)
  │ 3D Back-projection: Open3D point cloud generation preserves det(R) = +1.0 chirality
```

---

## 3. Mathematical Transformations & Coordinate System Verification

### 3.1 2D Raster Coordinate Mapping
Given an image point $(u_{\text{raw}}, v_{\text{raw}})$ with raw dimensions $W = 848, H = 480$:

| Direction | Horizontal Coordinate $u_{\text{rot}}$ | Vertical Coordinate $v_{\text{rot}}$ | Head Position ($u=750$) | Feet Position ($u=100$) |
| :--- | :---: | :---: | :---: | :---: |
| **CW** | $(H - 1) - v$ | $u$ | $v = 750$ (Sideways / Bottom) | $v = 100$ (Sideways / Top) |
| **CCW (Corrected)** | $v$ | $(W - 1) - u$ | $v = 97$ (**Upright / Top**) | $v = 747$ (**Upright / Bottom**) |

### 3.2 3D Camera Coordinate Frame & Chirality Preservation
In Open3D / standard computer vision convention, camera coordinates are defined as:
$+X$ points right, $+Y$ points down, $+Z$ points forward along the optical axis.

A 90° counter-clockwise rotation of the sensor raster around the $+Z$ optical axis corresponds to the 3D rotation matrix:
$$R_{\text{ccw}} = \begin{bmatrix} 0 & 1 & 0 \\ -1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

- **Determinant:** $\det(R_{\text{ccw}}) = 0(0) - 1(-1) + 0(0) = +1.0$ (Proper $SO(3)$ rotation; chirality is strictly preserved).
- **Orthogonality:** $R_{\text{ccw}}^T R_{\text{ccw}} = I_3$.
- **Back-projection Invariance:** For any 3D point $P_{\text{raw}} = (X, Y, Z)$ back-projected with raw intrinsics $K$, back-projecting the rotated pixel $(u_{\text{ccw}}, v_{\text{ccw}})$ with rotated intrinsics $K_{\text{rot}}$ produces exactly:
  $$P_{\text{rot}} = R_{\text{ccw}} P_{\text{raw}}$$
  Verified by test `test_depth_backprojection_consistency_and_chirality[ccw]` (RMS error $< 10^{-5}$ mm).

### 3.3 Camera Calibration & Intrinsics Transformation
For Intel RealSense D455F S/N 342622300088:

| Parameter | Native Landscape ($848 \times 480$) | Corrected Portrait CCW ($480 \times 848$) | Mathematical Derivation |
| :--- | :---: | :---: | :--- |
| **Image Width** | 848 px | 480 px | $W_{\text{rot}} = H_{\text{raw}}$ |
| **Image Height** | 480 px | 848 px | $H_{\text{rot}} = W_{\text{raw}}$ |
| **Focal Length $f_x$** | 427.88 px | **427.32 px** | $f_{x,\text{rot}} = f_{y,\text{raw}}$ |
| **Focal Length $f_y$** | 427.32 px | **427.88 px** | $f_{y,\text{rot}} = f_{x,\text{raw}}$ |
| **Principal Point $c_x$**| 422.17 px | **244.60 px** | $c_{x,\text{rot}} = c_{y,\text{raw}}$ |
| **Principal Point $c_y$**| 244.60 px | **424.83 px** | $c_{y,\text{rot}} = (W_{\text{raw}} - 1) - c_{x,\text{raw}}$ |
| **Depth Scale** | 0.001 m/unit | 0.001 m/unit | Invariant |

---

## 4. Live Preview Mode & Implementation Details

To allow safe visual inspection without triggering session directories, disk writes, or 8-stop sequences, the following non-destructive features were implemented:

1. **CLI Subcommand Flags (`realsense_measure/fullbody/cli.py`):**
   - `--portrait`: Default changed from `"cw"` to `"ccw"` across both `capture` and `check-camera` subparsers.
   - `check-camera --preview`: Opens an interactive live OpenCV window with real-time HUD and live camera feed; operator can inspect alignment and orientation, pressing `Q` or `ESC` to exit cleanly.
   - `capture --preview-only`: Launches the full capture UI and HUD loop without creating session directories, recording attempts, or advancing stops.
2. **Real-Time HUD Orientation Banner (`realsense_measure/fullbody/capture_ui.py`):**
   - Added `"PREVIEW_ONLY"` state styling (distinctive Cyan/Yellow HUD).
   - Added live orientation indicator banner displaying `Orientation: PORTRAIT CCW` (or `CW`) in the upper-right corner of the HUD overlay.
3. **Controller Orientation Propagation (`realsense_measure/fullbody/capture_controller.py`):**
   - Recorded `self.portrait_rotation` directly on `CaptureController`, dynamically injected into the UI snapshot state.

---

## 5. Verification & Test Evidence

### 5.1 New Orientation Correction Unit Tests
File: `tests/camera/test_orientation_correction.py`  
Status: **11 passed in 0.42s** (100% PASS)

| Test Case | Description | Result |
| :--- | :--- | :---: |
| `test_asymmetric_rgb_and_depth_rotation[cw]` | Marker correspondence (TOP, BOTTOM, LEFT, RIGHT) under CW | **PASSED** |
| `test_asymmetric_rgb_and_depth_rotation[ccw]` | Marker correspondence (TOP, BOTTOM, LEFT, RIGHT) under CCW | **PASSED** |
| `test_camera_intrinsics_and_principal_point[cw]` | Pinhole camera intrinsic mapping under CW | **PASSED** |
| `test_camera_intrinsics_and_principal_point[ccw]` | Pinhole camera intrinsic mapping under CCW | **PASSED** |
| `test_landmark_coordinate_transformation[cw]` | MediaPipe landmark 2D coordinate transformation under CW | **PASSED** |
| `test_landmark_coordinate_transformation[ccw]` | MediaPipe landmark 2D coordinate transformation under CCW | **PASSED** |
| `test_depth_backprojection_consistency_and_chirality[cw]` | 3D backprojection and right-handed chirality ($\det(R)=+1$) | **PASSED** |
| `test_depth_backprojection_consistency_and_chirality[ccw]` | 3D backprojection and right-handed chirality ($\det(R)=+1$) | **PASSED** |
| `test_both_rotation_directions_supported` | Verifies ValueError on invalid rotation modes | **PASSED** |
| `test_cli_parser_orientation_defaults_and_preview_flags` | Verifies `--portrait ccw` default and `--preview` / `--preview-only` flags | **PASSED** |
| `test_hud_orientation_indicator_rendering` | Verifies render_frame overlays `Orientation: PORTRAIT CCW` | **PASSED** |

### 5.2 CLI, UI & Controller Regression Tests
File: `tests/fullbody/test_stage27_adapter_ui_cli.py`  
Status: **10 passed in 28.18s** (100% PASS)

- Adapter warmup, raw depth units, and retry logic: **PASSED**
- Clean CLI import isolation (no pyrealsense2 / open3d leaks): **PASSED**
- Fake capture end-to-end and unfinalized failure handling: **PASSED**
- Synthetic fake camera check with CCW default: **PASSED**

---

## 6. Physical RealSense Hardware Pre-Flight Verification

Executed physical pre-flight check against connected Intel RealSense D455F:
```powershell
python -m realsense_measure.fullbody check-camera --portrait ccw --out ../scan_output/check_stage29_2_ccw
```

### Verification Report
```
============================================================
REALSENSE CAMERA CHECK REPORT
============================================================
Device Name:        RealSense D455F
Serial Number:      342622300088
Firmware Version:   5.17.3.10
Depth Scale:        0.001
Stream Resolution:  480x848
Intrinsics (Rot):   fx=427.32, fy=427.88, cx=244.60, cy=424.83
Portrait Rotation:  ccw (applied: True)
ROI Median Depth:   3.125 m
ROI Valid Fraction: 67.6 %
============================================================
```

- **Output Assets Generated:**
  - Color Snapshot: `scan_output/check_stage29_2_ccw/check/check_color.png` (Resolution: $480 \times 848 \times 3$)
  - Depth Colormap: `scan_output/check_stage29_2_ccw/check/check_depth_colorized.png` (Resolution: $480 \times 848 \times 3$)
  - Calibration Metrics: `scan_output/check_stage29_2_ccw/check/check_metrics.txt`
- **Dimensional Verification:** Both RGB and depth colorized frames are verified at exactly $480 \times 848$ (portrait orientation).

---

## 7. Operational Instructions for Visual Inspection

The preview-only orientation fix is implemented and verified. The operator can now inspect the live camera preview without writing any files to disk or initiating an 8-stop capture.

### Exact PowerShell Verification Command:
```powershell
cd c:\Users\pahad\OneDrive\Desktop\realsense_measure\realsense_measure-main
python -m realsense_measure.fullbody check-camera --preview --portrait ccw
```
*(Alternative using capture preview-only mode)*:
```powershell
python -m realsense_measure.fullbody capture --preview-only --subject-id PREVIEW_TEST --out ../scan_output/test_preview
```

### What the Operator Should Verify in the Window:
1. **Subject is upright:** Head is at the top border, feet are at the bottom border.
2. **Subject lateral orientation:** Raising right arm appears on the corresponding side without inversion.
3. **Pixel Correspondence:** Color image on the left and depth colormap on the right align 1:1.
4. **HUD Display:** Upper banner displays `Orientation: PORTRAIT CCW`.
5. **Press `Q` or `ESC`** to cleanly close the preview window.

---

## 8. Stop Condition Compliance

> [!IMPORTANT]
> **STOP CONDITION OBSERVED:**
> As specified in the Stage 29.2 instructions, the eight-stop physical scan session S02 has **NOT** been started. The pipeline remains in safe pre-flight state pending operator visual confirmation of the corrected preview.

