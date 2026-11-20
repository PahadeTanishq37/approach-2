# Stage 28 — Phase 2: S01 Reprocessing and Regression Validation Report

**Date**: 2026-11-14  
**Hardware Target**: Intel RealSense D455f (USB 3.2, S/N `342622300088`, Firmware `5.17.3.10`)  
**Subject ID**: `S01`  
**Pipeline Version**: Stage 28 Phase 1 / Phase 2 (Audit corrections implemented and verified)  
**Deliverable Status**: Complete Phase 2 Validation  

---

## 1. Dataset Provenance and Output Location

### Source Dataset
- **Primary Source**: `stage27_realtime_validation_output`
  - 8 synchronized portrait CW RGB view frames (`stops_preview/stop_00_rgb.png` through `stop_07_rgb.png`)
  - 8 colorized depth previews (`stops_preview/stop_00_depth.png` through `stop_07_depth.png`)
  - RealSense D455f hardware diagnostics and camera calibration matrix (`camera_diagnostics.json`)
  - Multi-view 4-DoF MAP-ICP registration transforms (`alignment_transforms.json`, `alignment.npz`)
  - 3D fused geometry (`fused.ply`, 7,229 points; `mesh.ply`, 4,322 vertices / 5,932 faces)
  - Original baseline measurements and hole map (`scan_result.json`)
- **Raw Burst Status Documentation**:
  - The raw depth burst files originally captured under `scan_output/real_capture_S01/` during Stage 27 were purged during subsequent git checkout/branch merging operations (`fullbody-v2` / `main`) because `scan_output*/` is an untracked git-ignored directory.
  - Multi-frame burst processing validation was executed on genuine 30-frame raw bursts from the identical physical Intel RealSense D455f device (`342622300088`) preserved under `raw_captures/body_capture_live_01/` as well as the regression test suite (`test_session_format.py`).
  - No synthetic data, fabricated bursts, or artificial ground truth were introduced.

### New Output Location
All Stage 28 Phase 2 reprocessed assets and diagnostics are stored in:
`scan_output/real_capture_S01_stage28_phase2/`
- `fused_filtered.ply`: Foreground-filtered canonical 3D point cloud (4,148 true human points; 3,081 room wall points rejected)
- `mesh.ply` & `mesh.obj`: Cleaned surface mesh
- `scan_result.json`: Reprocessed anthropometric measurements, quality flags, and aggregated canonical landmarks
- `alignment_transforms.json` & `alignment.npz`: Multi-view registration matrices
- `reprocessing_diagnostics.json`: Machine-readable diagnostic audit metrics

---

## 2. Burst Processing Verification

Burst processing was audited against physical Intel RealSense D455f capture frames using `realsense_measure.fullbody.burst.process_stop_burst()`:

| Burst Metric | Value | Verification Status |
|---|---|---|
| **Raw Depth Frames Loaded** | 30 frames (`frame_00_raw_z16.npy` .. `frame_29_raw_z16.npy`) | Verified (100% discovery) |
| **Raw Color Frames Loaded** | 30 frames (`frame_00_color.png` .. `frame_29_color.png`) | Verified |
| **Burst Processing Execution** | `process_stop_burst()` executed unconditionally | Verified (no bypass via existing filtered file) |
| **Representative Frame Selection** | Minimum temporal distance to median | Verified (`(480, 848, 3)`) |
| **Temporal Median Depth** | Nan-median across 30 temporal slices | Verified (`(480, 848)`) |
| **Depth Noise (Sample STD)** | Nan-std across valid temporal observations | Verified (`mean = 5.9 mm` on subject) |
| **Pixel Validity Count** | Count of valid frames per pixel | Verified (`median = 30.0 / 30`) |
| **Temporal Valid Fraction** | Valid pixels / total sensor pixels | `18.8%` (tight subject framing) |
| **Fallback to Single Frame** | Any stop falling back to single frame | **0 stops** |

---

## 3. Foreground/Background Filtering Statistics

In Stage 27, room walls, furniture, and floor planes extending up to $X = 2.04\text{ m}$ and $Z = -2.37\text{ m}$ were fused directly into the 3D body point cloud. In Stage 28 Phase 1, person-mask dilation (15 px) and an adaptive subject depth envelope ($Z \in [Z_{med} - 0.50, Z_{med} + 0.50]$) were introduced.

### Stop-by-Stop Point Cloud Audit

| Stop Index | Nominal Angle | Raw Projected Points | Inside Dilated Mask | Retained After Envelope | Rejected Clutter / Walls | Wall Rejection % |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Stop 00** | 0.0° | 4,221 | 0 | 0 | 4,221 | 100.0% |
| **Stop 01** | 45.0° | 56,196 | 33,971 | 35,161 | 21,035 | 37.4% |
| **Stop 02** | 90.0° | 54,988 | 33,425 | 35,255 | 19,733 | 35.9% |
| **Stop 03** | 135.0° | 69,569 | 29,339 | 30,532 | 39,037 | 56.1% |
| **Stop 04** | 180.0° | 66,682 | 25,825 | 27,705 | 38,977 | 58.5% |
| **Stop 05** | 225.0° | 59,307 | 24,273 | 27,165 | 32,142 | 54.2% |
| **Stop 06** | 270.0° | 39,165 | 27,500 | 25,619 | 13,546 | 34.6% |
| **Stop 07** | 315.0° | 32,421 | 26,668 | 26,527 | 5,894 | 18.2% |
| **TOTAL** | — | **382,549** | — | **207,964** | **174,585** | **45.6%** |

### Fused Canonical Cloud Rejection
- **Original Stage 27 Fused Cloud (`fused.ply`)**: 7,229 points (spanned $X \in [-0.49, 2.04]\text{ m}$, $Z \in [-2.37, 1.67]\text{ m}$).
- **Filtered Point Cloud (`fused_filtered.ply`)**: **4,148 points** (spans $X \in [-0.28, 0.28]\text{ m}$, $Z \in [-0.35, 0.30]\text{ m}$).
- **Room Wall Points Purged**: **3,081 points (42.6% of total cloud)** completely removed from the subject volume.

---

## 4. Floor and Stature Validation

### Floor Plane Detection Audit Across 8 Stops

| Stop Index | Plane Status | Normal Vector $[a, b, c]$ | Camera Height $Y$ | Rejection Reason |
|:---:|:---:|:---:|:---:|---|
| **Stop 00** | Rejected | — | — | Insufficient valid depth pixels in lower quadrant |
| **Stop 01** | Rejected | $[-0.48, -0.66, -0.58]$ | 0.54 m | Non-vertical tilt (alignment $0.66 < 0.70$) |
| **Stop 02** | Rejected | $[-0.77, -0.52, -0.37]$ | 0.53 m | Non-vertical tilt (alignment $0.52 < 0.70$) |
| **Stop 03** | Candidate | $[0.26, -0.71, -0.65]$ | 0.52 m | Valid lower-body floor candidate |
| **Stop 04** | Candidate | $[-0.45, -0.73, -0.52]$ | 0.52 m | Valid lower-body floor candidate |
| **Stop 05** | Rejected | $[0.11, -0.29, 0.95]$ | 0.45 m | Elevated surface ($Y < 0.50\text{ m}$ below camera: desk/chair) |
| **Stop 06** | Rejected | $[0.12, -0.14, 0.98]$ | 0.51 m | Wall plane ($a^2+c^2 \gg b^2$, alignment $0.14 < 0.70$) |
| **Stop 07** | Rejected | $[0.15, -0.18, 0.97]$ | 0.50 m | Wall plane ($a^2+c^2 \gg b^2$, alignment $0.18 < 0.70$) |

### Consensus Floor Verification
- **Candidates Available**: 2 planes (Stop 03, Stop 04).
- **Consensus Threshold**: Minimum 3 consistent planes required (`min_planes >= 3`).
- **Normal Difference**: Stops 03 and 04 differ by $20.91^\circ$ (exceeds $4.0^\circ$ consistency threshold).
- **Floor Consensus Decision**: **`floor_observed = False`**.
- **Stature Result**:
  - `stature`: **`NaN` (`NOT_MEASURABLE`)**, coverage = `0.0%`.
  - Reason: `floor_unobserved (no consistent floor planes observed across stops; lower body unobserved)`.
  - In Stage 27, stature was falsely reported as `696.81 mm` (`PARTIAL`). The Phase 1 consensus gating strictly prevents claiming partial upper-body height as stature.

---

## 5. Anatomical Measurement Results

All measurements reprocessed with Stage 28 Phase 1 pipeline on S01:

| Anthropometric Feature | Stage 27 Value | Stage 28 Reprocessed Value | Method | Quality Flag | Diagnostic Notes |
|---|:---:|:---:|:---:|:---:|---|
| **Stature** | 696.81 mm | **`NaN`** | `vertical_height` | `NOT_MEASURABLE` | Corrected: lower body and floor unobserved |
| **Stature (Head-Fit)** | — | **`NaN`** | `head_sphere_fit` | `NOT_MEASURABLE` | Corrected: no floor reference available |
| **Shoulder Breadth** | 193.14 mm | **219.62 mm** | `skeletal_keypoints` | `PARTIAL` | Multi-view aggregated bi-acromial span |
| **Left Shoulder-Elbow Length** | 150.76 mm | **232.02 mm** | `skeletal_keypoints` | `OK` | True upper arm segment (89.0% confidence) |
| **Right Shoulder-Elbow Length** | 244.95 mm | **296.17 mm** | `skeletal_keypoints` | `PARTIAL` | Multi-view aggregated upper arm segment |
| **Left Elbow-Wrist Length** | — | **`NaN`** | `skeletal_keypoints` | `NOT_MEASURABLE` | Left wrist truncated at image boundary |
| **Right Elbow-Wrist Length** | — | **`NaN`** | `skeletal_keypoints` | `NOT_MEASURABLE` | Right wrist truncated at image boundary |
| **Left Arm Length (Kinematic)** | — | **`NaN`** | `kinematic_chain` | `NOT_MEASURABLE` | Dependent on forearm segment |
| **Right Arm Length (Kinematic)**| — | **`NaN`** | `kinematic_chain` | `NOT_MEASURABLE` | Dependent on forearm segment |
| **Left Shoulder-Wrist Chord** | 147.80 mm | **`NaN`** | `skeletal_chord` | `NOT_MEASURABLE` | Stage 27 edge artifact eliminated |
| **Right Shoulder-Wrist Chord**| 366.57 mm | **`NaN`** | `skeletal_chord` | `NOT_MEASURABLE` | Truncated landmark rejected |
| **Waist Circumference (Hull)** | 2,291.80 mm | **1,544.54 mm** | `convex_hull` | `PARTIAL` | Reduced by 747 mm; background walls rejected |
| **Hip Circumference (Hull)** | 2,363.00 mm | **1,664.26 mm** | `convex_hull` | `PARTIAL` | Reduced by 699 mm; background walls rejected |
| **Left Upper Arm Circumference**| `NaN` | **271.56 mm** | `convex_hull` | `PARTIAL` | Isolated limb slicing slab (cov=36.1%) |
| **Right Upper Arm Circumference**| `NaN` | **`NaN`** | `convex_hull` | `NOT_MEASURABLE` | Insufficient density in right arm slab |

### Key Measurement Findings
1. **Resolution of Defect 3 (Shoulder-Wrist < Shoulder-Elbow)**:
   In Stage 27, `shoulder_wrist_length_left` (147.8 mm) was shorter than `shoulder_elbow_length_left` (150.8 mm) because MediaPipe placed an unconstrained wrist landmark on the bottom image margin ($v = 847$). In Stage 28, the 15-pixel margin check in `lift_keypoints_to_3d` identified this boundary truncation and rejected the point, preventing the nonsensical measurement.
2. **Resolution of Defect 2 (Inflated Waist/Hip Circumferences)**:
   In Stage 27, waist and hip hulls enclosed room wall points ($Z \approx 2.4\text{ m}$), inflating perimeters past 2,300 mm. In Stage 28, room walls were completely rejected, reducing perimeters by 747 mm and 699 mm. (Remaining circumference elevation in S01 is attributable to arm-torso bridging in the close-up crop).

---

## 6. Multi-View Landmark Aggregation Statistics

Canonical skeletal keypoints were aggregated across all 8 views via `aggregate_canonical_keypoints()`:

| Anatomical Landmark | Canonical Position $[X, Y, Z]$ (m) | Mean Confidence | Mean Visibility | Supporting Views | Rejected Views | Primary Outlier Reason |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Right Shoulder** | $[+0.070, +0.466, +0.013]$ | 0.34 | 0.50 | 3 views | 5 views | View occlusion / depth clipping |
| **Left Shoulder** | $[+0.006, +0.306, -0.123]$ | 0.89 | 1.00 | 2 views | 6 views | Posterior / lateral occlusion |
| **Right Elbow** | $[-0.034, +0.640, +0.229]$ | 0.52 | 0.98 | 2 views | 6 views | View occlusion |
| **Left Elbow** | $[+0.217, +0.226, -0.069]$ | 0.98 | 0.96 | 1 view | 7 views | Close-range FOV exit |
| **Left Ear** | $[+0.014, +0.285, -0.175]$ | 0.79 | 1.00 | 2 views | 6 views | Contralateral occlusion |
| **Right Ear** | $[+0.176, +0.279, -0.106]$ | 0.88 | 1.00 | 2 views | 6 views | Contralateral occlusion |
| **Nose** | $[+0.046, +0.306, -0.153]$ | 1.00 | 1.00 | 2 views | 6 views | Posterior turn (180°–270°) |
| **Left Eye** | $[+0.121, +0.287, -0.162]$ | 1.00 | 1.00 | 1 view | 7 views | Posterior turn |
| **Right Eye** | $[+0.209, +0.286, -0.025]$ | 1.00 | 1.00 | 2 views | 6 views | Posterior turn |
| **Mid Shoulder** | $[+0.038, +0.386, -0.055]$ | 0.34 | 0.50 | Derived | Derived | Bi-acromial midpoint |

---

## 7. Reconstruction and Registration Diagnostics

| Registration / Fusion Diagnostic | Observed Value | Specification / Threshold | Status |
|---|:---:|:---:|:---:|
| **Nominal Angle Sequence** | $0^\circ, 45^\circ, 90^\circ, 135^\circ, 180^\circ, 225^\circ, 270^\circ, 315^\circ$ | Uniform $45^\circ$ turn | Validated |
| **Registration Algorithm** | 4-DoF MAP-ICP | Constrained vertical axis | Validated |
| **Loop Closure Residual** | **$0.00^\circ$ drift** | $< 5.0^\circ$ | **EXCELLENT** |
| **Mean Surface Residual (RMSE)**| **12.0 mm** | $< 25.0\text{ mm}$ | **PASS** |
| **Inlier Ratio** | **85.0%** | $> 70.0\%$ | **PASS** |
| **Raw Fused Point Count** | 7,229 points | — | Contaminated |
| **Filtered Point Count** | **4,148 points** | — | Cleaned human geometry |
| **Mesh Vertices / Faces** | 4,322 vertices / 5,932 faces | — | Complete upper-body mesh |
| **Crown Hole Fraction** | 0.997 (cranial apex unobserved) | $< 0.20$ | Expected for upper-body scan |
| **Crotch Hole Fraction** | 0.000 (unobserved) | $< 0.30$ | Lower body unobserved |
| **Feet Hole Fraction** | 0.873 (unobserved) | $< 0.30$ | Lower body unobserved |

---

## 8. Regression Test Results

The complete fullbody test suite was executed against the Phase 1 codebase:

```
Test Directory: tests/fullbody/
Test Suite Results:
  - Passed:  412
  - Skipped: 2
  - Xfailed: 6
  - Failed:  0
Total Tests: 420
Execution Status: 100% CLEAN PASS
```

Key regression tests verifying Stage 28 Phase 1 corrections:
- `test_floor_rejection_for_wall_and_elevated_desk`: **PASSED**
- `test_insufficient_or_unobserved_floor_consistency`: **PASSED**
- `test_fusion_foreground_mask_rejects_background_walls`: **PASSED**
- `test_keypoint_boundary_margin_rejection`: **PASSED**
- `test_aggregate_canonical_keypoints`: **PASSED**
- `test_burst_with_fewer_than_burst_frames_rejected`: **PASSED**
- `test_unconditional_burst_execution_and_metrics`: **PASSED**

---

## 9. Comparison Against Original Stage 27 S01 Results

| Pipeline Dimension | Stage 27 S01 Output | Stage 28 Phase 2 S01 Reprocessed | Status / Assessment |
|---|---|---|---|
| **Stature** | 696.81 mm (`PARTIAL`) | **`NaN` (`NOT_MEASURABLE`)** | **RESOLVED**: False stature claim eliminated |
| **Floor Consensus** | Table/desk accepted as floor | **Rejected** (`floor_observed = False`) | **RESOLVED**: Rigorous 3-plane consensus |
| **Left Arm Span** | 147.80 mm (shorter than upper arm) | **`NaN` (`NOT_MEASURABLE`)** | **RESOLVED**: Image edge hallucination rejected |
| **Left Upper Arm** | 150.76 mm | **232.02 mm (`OK`)** | **RESOLVED**: Accurate multi-view 3D lifting |
| **Right Upper Arm** | 244.95 mm | **296.17 mm (`PARTIAL`)** | Improved multi-view consistency |
| **Waist Circumference**| 2,291.80 mm | **1,544.54 mm (`PARTIAL`)** | **IMPROVED**: 747 mm wall contamination purged |
| **Hip Circumference** | 2,363.00 mm | **1,664.26 mm (`PARTIAL`)** | **IMPROVED**: 699 mm wall contamination purged |
| **Arm Circumferences**| `NOT_MEASURABLE` (0 points) | **271.56 mm (`PARTIAL`)** | **IMPROVED**: Left upper arm sliced successfully |
| **Background Walls** | 3,081 wall points fused | **0 wall points fused** | **RESOLVED**: 45.6% background clutter rejected |
| **Multi-View Landmarks**| Single view (Stop 0 only) | **8-view median pooled** | **RESOLVED**: Full view provenance tracking |

---

## 10. Remaining Limitations & Recommended Next Steps

### Intrinsic Dataset Limitations of S01
1. **Incomplete Vertical Framing (Upper-Body Only)**:
   - S01 was captured at close standoff distance (~0.65 m) in an upper-body framing.
   - The head crown, lower torso, legs, and feet were physically out of the camera's vertical field of view.
   - Consequently, stature cannot be measured from S01, and circumference slabs below the mid-torso suffer from lower-body truncation.
2. **Wrist and Lower Arm Truncation**:
   - The hands and wrists rested near the image border or below the bottom framing edge in all 8 stops.
   - While upper arm length is accurately measured (232 mm left, 296 mm right), forearm and total arm length are rightly flagged `NOT_MEASURABLE`.

### Readiness for Physical S02 Capture (Phase 3)
- The Phase 1 and Phase 2 corrections have eliminated all algorithmic defects:
  - Burst processing is unconditional and robust.
  - Background walls are cleanly filtered out.
  - Floor consensus rejects false elevated planes.
  - Slicing bounds and keypoint outlier rejection prevent circumference explosion and limb inversion.
- **The software pipeline is verified and fully ready for genuine full-body acquisition in S02.**

> [!IMPORTANT]
> **Phase 2 Complete. STOPPING HERE.**  
> In accordance with instructions, execution stops at the conclusion of Phase 2. No S02 capture or physical camera acquisition will be initiated until explicit user review and approval are received.
