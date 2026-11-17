# Stage 28 — Phase 2.1: S01 Evidence Reconciliation and Measurement Sanity Audit Report

**Date**: 2026-11-14  
**Audit Target**: Physical Dataset `S01` (Acquired 2026-11-13 via Intel RealSense D455f S/N `342622300088`)  
**Pipeline Codebase**: `realsense_measure-main` (Stage 28 Phase 1 corrections)  
**Audit Purpose**: Complete evidence reconciliation, mathematical verification of filtering statistics, coordinate sanity check, circumference geometry audit, and registration metric review.  
**Execution Mode**: Audit Only. No camera activation, no S02 capture, no code changes.  

---

## 1. Exact S01 Data Provenance

An exhaustive audit of all filesystem assets was performed to establish exactly which data files exist, their true provenance, and which files were used during the Stage 28 Phase 2 reprocessing.

### Provenance Audit Matrix

| File Path | Original Origin | Asset Classification | Capture Metadata Available? | Actually Used in Phase 2? | Role in Phase 2 Reprocessing |
|---|---|---|:---:|:---:|---|
| `stage27_realtime_validation_output/stops_preview/stop_00_rgb.png` .. `stop_07_rgb.png` | S01 Capture (2026-11-13 22:45) | RGB view previews (8-bit PNG, 848×480) | Yes (`camera_diagnostics.json`) | **YES** | Input to MediaPipe Tasks API for 2D keypoint and person mask detection. |
| `stage27_realtime_validation_output/stops_preview/stop_00_depth.png` .. `stop_07_depth.png` | S01 Capture (2026-11-13 22:45) | Colorized depth previews (8-bit uint8 RGB) | Yes | **NO** | Visual colorized preview only; contains no metric floating-point depth. |
| `stage27_realtime_validation_output/camera_diagnostics.json` | S01 Capture (2026-11-13 22:45) | Sensor metadata & camera intrinsics | Yes | **YES** | Defined pinhole intrinsics ($f_x=428.83, f_y=429.39, c_x=234.40, c_y=422.17$) and depth scale ($0.001$). |
| `stage27_realtime_validation_output/alignment_transforms.json` & `alignment.npz` | Stage 27 Output (2026-11-13 22:45) | 4-DoF MAP-ICP registration transforms | Yes | **YES** | Positioned pinhole cameras and transformed lifted keypoints to canonical frame. |
| `stage27_realtime_validation_output/mesh.ply` & `mesh.obj` | Stage 27 TSDF Volume Output | Reconstructed 3D triangle mesh (4,322 verts) | Yes | **YES** | **Target for Open3D raycasting** used to regenerate per-view depth maps for 3D keypoint lifting. |
| `stage27_realtime_validation_output/fused.ply` | Stage 27 TSDF Volume Output | Fused 3D point cloud (7,229 points) | Yes | **YES** | Input cloud for foreground filtering (`fused_filtered.ply`, 4,148 pts) and circumference slicing. |
| `stage27_realtime_validation_output/scan_result.json` | Stage 27 Measurement Output | Baseline measurement JSON | Yes | **YES** | Reference baseline for hole map and measurement comparison. |
| `raw_captures/body_capture_live_01/frame_*_raw_z16.npy` & `frame_*_depth.npy` | **Stage 11 Hardware Smoke Test** (2026-11-04 14:32) | Raw sensor depth arrays (uint16 / float32) | Yes (`meta.json`) | **NO (SURROGATE ONLY)** | **Used strictly as an independent surrogate** to verify `process_stop_burst()` algorithm logic; **NOT S01 data**. |
| `scan_output/real_capture_S01/stop_*/attempt_*/raw_depth_*.npy` | S01 Physical Acquisition (2026-11-13 22:31) | Raw 30-frame depth bursts (sensor units) | Yes (sha256 manifests) | **UNAVAILABLE** | **Purged from disk** at 23:01 on 2026-11-13 during git branch checkout/merge (`scan_output*/` is gitignored). |

### Explicit Technical Distinctions

1. **Reprocessing of Original S01 Raw RGB-D Bursts**:
   - **DID NOT OCCUR.** The raw 30-frame depth burst files (`raw_depth_000.npy` .. `raw_depth_029.npy`) were removed during git operations prior to Phase 1. Any claim that the raw S01 depth bursts were loaded and reprocessed is factually inaccurate.
2. **Reprocessing of Surviving S01 Intermediate & Reconstructed Assets**:
   - **THIS IS WHAT ACTUALLY OCCURRED.** Phase 2 reprocessed surviving S01 assets from `stage27_realtime_validation_output`:
     - MediaPipe was executed on the 8 surviving physical RGB frames (`stops_preview/stop_00_rgb.png` .. `stop_07_rgb.png`).
     - Depth maps for 3D lifting were generated via **Open3D raycasting (`RaycastingScene.cast_rays`)** from the Stage 27 fused mesh (`mesh.ply`) through the calibrated camera intrinsics positioned at the Stage 27 alignment poses ($T_{stop}^{-1}$).
     - Circumferences were re-sliced from the Stage 27 canonical point cloud (`fused.ply`) using the Phase 1 foreground filter.
3. **Independent Burst-Processing Verification Using Another Physical Capture**:
   - Multi-frame burst processing (`process_stop_burst()`) was verified on a genuine 30-frame capture from the same physical camera (`342622300088`), but that capture was `raw_captures/body_capture_live_01` (recorded 2026-11-04), not S01.

---

## 2. Reproducibility Limitations

Given the surviving assets, the following reproducibility boundaries are documented:

1. **Independently Reproducible**:
   - 2D MediaPipe Pose detection and person mask segmentation on all 8 RGB views.
   - Open3D raycast depth reconstruction from `mesh.ply` and `alignment_transforms.json`.
   - 3D keypoint lifting using reconstructed depth and camera intrinsics.
   - Multi-view canonical keypoint aggregation with outlier rejection.
   - Foreground/background filtering on `fused.ply` and stop-by-stop raycast depth.
   - Torso and limb cross-section slicing and circumference computation.
   - Consensus floor plane evaluation and stature gating (`NOT_MEASURABLE`).
2. **NOT Independently Reproducible (Irrecoverable Without New Capture)**:
   - Temporal median depth calculation across the original 30 S01 burst frames.
   - Frame-by-frame depth noise (standard deviation) of S01.
   - Original multi-frame sensor validity counts for S01.
   - Re-registration from scratch using raw sensor point clouds.

---

## 3. Reconciled Point-Filtering Statistics

In the initial Phase 2 report, an apparent contradiction was present: "Points inside dilated person mask: 201,001" vs "Points retained after depth-envelope filtering: 207,964". Sequential filtering cannot yield more retained points than initial candidate points.

### Root Cause of the Discrepancy
Code audit of the diagnostic script revealed that:
1. Counter 1 (`mask_pts_cnt = np.sum(p_mask & (depth_m > 0))`) counted points inside the **raw (undilated)** MediaPipe segmentation mask ($N = \mathbf{201,001}$).
2. The pipeline then applies morphological dilation with a $15 \times 15$ ellipse kernel (`dilated_m`), expanding the mask boundary to preserve extremities and outer contours ($N = \mathbf{212,892}$).
3. The adaptive subject depth envelope ($Z_{med} \pm 0.50\text{ m}$) was applied **within the dilated mask**, rejecting 4,928 out-of-envelope background wall points.
4. Final retained count: $212,892 - 4,928 = \mathbf{207,964}$.
5. The initial report mislabeled the raw undilated mask count (201,001) as "inside dilated mask", creating the apparent paradox.

### Reconciled Mathematical Audit Table

All counts below are evaluated on the exact 8 camera views and reconcile with 100% mathematical precision:
$$\text{Total Retained} + \text{Total Rejected} = \text{Total Raw Projected Points}$$

| Stop Index | Nominal Angle | Total Raw Projected Points ($N_{raw}$) | Raw (Undilated) Mask Points | Dilated Mask Points ($N_{dil}$) | Retained After Depth Envelope ($N_{ret}$) | Rejected by Mask ($\sim N_{dil}$) | Rejected by Depth Envelope | Total Points Rejected ($N_{rej}$) | Mathematical Check ($N_{ret} + N_{rej} = N_{raw}$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Stop 00** | 180.0° | 4,221 | 0 | 0 | 0 | 4,221 | 0 | 4,221 | **EXACT MATCH** |
| **Stop 01** | 111.7° | 56,196 | 33,971 | 35,161 | 35,161 | 21,035 | 0 | 21,035 | **EXACT MATCH** |
| **Stop 02** | 90.0° | 54,988 | 33,425 | 35,255 | 35,255 | 19,733 | 0 | 19,733 | **EXACT MATCH** |
| **Stop 03** | 45.0° | 69,569 | 29,339 | 31,414 | 30,532 | 38,155 | 882 | 39,037 | **EXACT MATCH** |
| **Stop 04** | 0.0° | 66,682 | 25,825 | 27,705 | 27,705 | 38,977 | 0 | 38,977 | **EXACT MATCH** |
| **Stop 05** | 315.0° | 59,307 | 24,273 | 27,165 | 27,165 | 32,142 | 0 | 32,142 | **EXACT MATCH** |
| **Stop 06** | 270.0° | 39,165 | 27,500 | 28,714 | 25,619 | 10,451 | 3,095 | 13,546 | **EXACT MATCH** |
| **Stop 07** | 225.0° | 32,421 | 26,668 | 27,478 | 26,527 | 4,943 | 951 | 5,894 | **EXACT MATCH** |
| **TOTAL** | — | **382,549** | **201,001** | **212,892** | **207,964** | **169,657** | **4,928** | **174,585** | **EXACT MATCH** |

- **Filtering Architecture**: Sequential. Raw Points $\rightarrow$ Dilated Person Mask $\rightarrow$ Subject Depth Envelope.
- **Wall & Clutter Rejection Ratio**: $174,585 / 382,549 = \mathbf{45.64\%}$.
- **Body Preservation Ratio**: $207,964 / 382,549 = \mathbf{54.36\%}$.

---

## 4. Stop 00 Investigation

Stop 00 exhibited anomalous behavior: 4,221 raw projected points, zero points inside the person mask, and zero retained points.

### Root Cause Analysis
Inspection of `stops_preview/stop_00_rgb.png` and raycast geometry revealed:
1. **MediaPipe Person Mask**: MediaPipe PoseLandmarker successfully detected a person in `stop_00_rgb.png`! The raw mask covered **131,571 pixels** spanning $Y \in [136, 651]$ and $X \in [0, 471]$ (middle and lower frame).
2. **Raycast Depth Frustum**: Raycasting from `mesh.ply` with the Stop 00 camera transform $T_{0}$ generated only 4,221 hits spanning $Y \in [0, 50]$ and $X \in [106, 292]$ (extreme top edge of image).
3. **Geometric Disconnect**:
   - In `alignment.npz`, the translation vector for Stop 00 is $t = [-0.063, -0.093, +0.014]\text{ m}$.
   - The $Z$ translation is only **$14\text{ mm}$**!
   - In Stage 27, Stop 00 was assigned as the origin of the canonical coordinate frame. The subject's torso geometry in `mesh.ply` sits directly centered at $(X=0, Y=0, Z=0)$.
   - When raycasting with the pinhole at $(0, 0, 0)$, the virtual camera was positioned **inside the subject's body geometry**!
   - Consequently, the raycaster hit only a few ceiling/cranial vertices at the upper border ($Y \le 50$) and zero body points in the central frame ($Y \in [136, 651]$).
   - Because the raycast depth and the RGB person mask had zero spatial overlap, the person mask intersected zero depth pixels, resulting in 0 retained points.

---

## 5. Canonical Coordinate and Landmark Sanity Audit

### Canonical Coordinate Frame Definition
Inspection of `compute_gravity_transform()` in `alignment.py` and `floor.py` establishes the canonical system:
- **$+Y$ Axis**: **VERTICAL UPWARDS** (aligned with gravity normal; floor at $Y=0$, body extends $+Y$).
- **$+Z$ Axis**: **DEPTH / ANTERIOR-POSTERIOR** (looking forward from canonical origin).
- **$+X$ Axis**: **LATERAL** (subject's left/right).
- **Origin**: Ground level below the subject coordinate center.

### Anatomical Landmark Inspection

| Landmark | Canonical Coordinates $[X, Y, Z]$ (m) | Expected Relationship | Observed Discrepancy |
|---|:---:|---|---|
| **Left Shoulder** | $[+0.006, \mathbf{+0.306}, -0.123]$ | Shoulder heights should be approximately equal | **$16.0\text{ cm}$ vertical height difference** |
| **Right Shoulder** | $[+0.070, \mathbf{+0.466}, +0.013]$ | $Y_{left} \approx Y_{right} \pm 2\text{ cm}$ | Right shoulder is 16 cm higher than left |
| **Right Elbow** | $[-0.034, \mathbf{+0.640}, +0.229]$ | In standing pose, elbow must be BELOW shoulder ($Y_{elbow} < Y_{shoulder}$) | **Right elbow is $17.4\text{ cm}$ HIGHER than right shoulder** |
| **Left Elbow** | $[+0.217, +0.226, -0.069]$ | $Y_{elbow} < Y_{shoulder}$ | Left elbow ($Y=0.226$) is 8.0 cm below left shoulder ($Y=0.306$) — plausible |

### Root Cause: Physical Camera Orientation Error in S01
A forensic examination of the 2D keypoints across all 8 stops uncovered the root cause:
- In `stop_01_rgb.png`:
  - `nose`: $(u = 276.8, v = 399.6)$
  - `left_eye`: $(u = 294.7, v = 417.0)$
  - `right_eye`: $(u = 303.3, v = 381.5)$
  - `left_shoulder`: $(u = 156.7, v = 438.6)$
  - `right_shoulder`: $(u = 214.4, v = 246.9)$
- **Key Diagnostic**:
  - The vector from shoulder midpoint ($u=185.5$) to nose ($u=276.8$) points horizontally along $+u$ ($\Delta u = +91.3\text{ px}$).
  - The vector from right shoulder ($v=246.9$) to left shoulder ($v=438.6$) points vertically along $+v$ ($\Delta v = +191.7\text{ px}$).
  - **THE HUMAN SUBJECT IS ORIENTED HORIZONTALLY (SIDEWAYS) IN THE PORTRAIT IMAGE!**
- **Impact on 3D Reconstruction**:
  - When the camera is sideways, the subject's lateral axis (shoulder-to-shoulder) was captured along the camera's optical $Y$ axis ($v$).
  - When backprojected to 3D, the physical shoulder width ($~40\text{ cm}$) was encoded as a difference along $y_{cam}$.
  - The alignment rotation matrix $T_{stop}$ mapped $y_{cam}$ directly into canonical vertical $Y$.
  - Consequently, **the subject's anatomical shoulder width was erroneously mapped as a vertical height difference ($16.0\text{ cm}$)**!
  - Similarly, the right arm extending to the side in camera coordinates was transformed as pointing upwards into canonical $+Y$, placing the right elbow at $Y = 0.640\text{ m}$ (above the shoulder).

**Audit Conclusion**: The S01 dataset suffers from a severe physical camera orientation mismatch. The physical camera mount was rotated 90° relative to the software's orientation expectation. No landmark aggregation algorithm can correct an inverted optical axis without correcting the rotation convention.

---

## 6. Waist, Hip, and Arm Circumference Geometry Audit

### Slicing Slab Analysis

| Measurement Slice | Slice Canonical Height ($Y$) | Point Count | In-Plane Span $U$ (Lateral) | In-Plane Span $V$ (Depth) | Convex Hull Perimeter | Diagnostic Findings |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Waist** | $Y = 0.429\text{ m}$ | 343 | $0.467\text{ m}$ ($[-0.21, +0.25]$) | $0.489\text{ m}$ ($[-0.27, +0.22]$) | **$1,544.54\text{ mm}$** | Slices through upper chest/shoulders due to stature scaling; arms included |
| **Hip** | $Y = 0.393\text{ m}$ | 347 | $0.523\text{ m}$ ($[-0.25, +0.28]$) | $0.502\text{ m}$ ($[-0.27, +0.24]$) | **$1,664.26\text{ mm}$** | Slices through mid-torso and adjoining upper arms |
| **Left Upper Arm** | Origin: $[0.11, 0.27, -0.10]$ | 333 | $0.399\text{ m}$ | $0.838\text{ m}$ | **$271.56\text{ mm}$** | Partially isolated limb slab; coverage = $36.1\%$ |
| **Right Upper Arm**| Origin: $[0.02, 0.55, +0.12]$ | 179 | $0.524\text{ m}$ | $0.431\text{ m}$ | **`NaN`** | Insufficient inlier density; rejected |

### Why Waist and Hip Circumferences Remain Inflated (~1,550–1,660 mm)
1. **Anatomical Level Misplacement**:
   - `extract_anatomical_levels()` calculates waist height as $Y_{waist} = 0.56\text{--}0.67 \times \text{stature}$.
   - Because the lower body is unobserved, `stature` defaulted to the upper-body cloud height ($0.697\text{ m}$).
   - $0.60 \times 0.697\text{ m} = \mathbf{0.418\text{ m}}$.
   - As audited in Section 5, the subject's shoulders sit at $Y = 0.306\text{ m}$ and $Y = 0.466\text{ m}$!
   - **The "waist" slice was placed directly across the upper chest and shoulders!**
2. **Arm-Torso Bridging**:
   - At $Y = 0.429\text{ m}$, the cross-section intersects both the torso and the adjacent upper arms.
   - The in-plane span is $46.7\text{ cm}$ wide by $48.9\text{ cm}$ deep.
   - The convex hull wraps around the torso and both arms together, yielding a perimeter of $1,544.5\text{ mm}$.
3. **Geometry Completeness**:
   - The cross-section represents only partial surface geometry from an upper-body crop. It does NOT represent an anatomical waist or hip circumference.
   - Retaining the quality flag **`PARTIAL`** is mandatory; these values must not be cited as valid anthropometric circumferences.

---

## 7. Registration Evidence Audit

In the Phase 2 report, three registration claims were presented:
1. "0.00° loop-closure drift"
2. "12.0 mm mean surface RMSE"
3. "85.0% inlier ratio"

### Audit Findings on Registration Metrics
1. **Source of 12.0 mm RMSE and 85.0% Inlier Ratio**:
   - In `stage27_realtime_validation_output/alignment.npz`, **NO residual arrays, RMSE values, or inlier ratios were saved**. The archive contains only yaw angles and $4 \times 4$ matrices.
   - In the Phase 2 reprocessing script, `residual_after_mm=12.0` and `inlier_fraction=0.85` were passed as **hardcoded placeholder defaults** to the `AlignedStop` constructor!
   - **These numbers were NOT computed from point cloud registration residuals.** Presenting them as observed metrics was an error.
2. **Source of 0.00° Loop Closure Drift**:
   - In `alignment_transforms.json`, Stop 00 is pinned at yaw $180.0^\circ$.
   - The calculated drift was $\text{yaw}_{refined} - \text{yaw}_{nominal} = 180.0^\circ - 180.0^\circ = \mathbf{0.00^\circ}$.
   - **What this actually establishes**: It proves that Stop 00 was assigned as the fixed reference frame.
   - **What it does NOT establish**: It does NOT prove that a 360° closed-loop physical registration occurred without angular drift. The intermediate yaws show that only Stop 01 was refined by ICP (to $111.7^\circ$), while Stops 02 through 07 remained at their exact prior angles ($90^\circ, 45^\circ, 0^\circ, 315^\circ, 270^\circ, 225^\circ$).

---

## 8. Corrected Measurement Claims

To ensure scientific integrity, all unsupported claims in the Phase 2 report are formally retracted and corrected:

| Initial Report Claim | Corrected Factual Statement | Reason for Correction |
|---|---|---|
| *"Accurate multi-view 3D lifting"* | **Multi-view 3D lifting executed on reconstructed mesh raycasts; subject orientation was sideways.** | Camera orientation was rotated 90°, producing 16 cm shoulder disparity. |
| *"Loop closure: 0.00° drift"* | **Stop 00 reference yaw pinned at nominal 180.0°; no global loop closure optimization was computed.** | Single-view pinning does not establish physical loop closure. |
| *"Mean surface residual: 12.0 mm; Inlier ratio: 85.0%"* | **Values were placeholder arguments in the reprocessing script; original ICP residuals were not saved in S01.** | Metrics were not computed from point clouds. |
| *"Software pipeline is verified and fully ready for genuine full-body acquisition in S02"* | **Algorithmic modules pass unit tests, but camera mounting and framing must be physically corrected before S02.** | Sideways orientation and close standoff would ruin S02 if repeated. |
| *"100% CLEAN PASS on anthropometry"* | **Anthropometric measurements on S01 are invalid or partial due to upper-body truncation and camera tilt.** | Stature is `NaN`; waist and hip circumferences enclose shoulders and arms. |

---

## 9. Remaining Blockers Before S02

The following physical and procedural blockers MUST be resolved before initiating capture S02:

1. **Physical Portrait Orientation Verification (CRITICAL)**:
   - S01 was captured with the camera rotated 90° relative to the software expectation, causing the subject to appear sideways.
   - **Required Action**: Run `check-camera --portrait cw` (or `--portrait ccw`) and visually inspect `check_color.png`. The subject MUST appear upright (head at top $v \approx 100$, feet at bottom $v \approx 800$, shoulders horizontal along $u$). Do not proceed to scanning until upright orientation is visually confirmed.
2. **Standoff Distance & Vertical FOV Framing**:
   - S01 was captured at ~0.65 m standoff, clipping the head crown, hands, lower torso, legs, and floor.
   - **Required Action**: Position the subject at **2.4 m to 2.8 m standoff distance**. The entire body from head apex to floor/feet must be visible in the preview window with at least 5% vertical margin.
3. **Consensus Floor Plane Visibility**:
   - The lower quadrant of the camera FOV must observe at least $1.0\text{ m}^2$ of uninterrupted floor surrounding the subject's feet in at least 4 of the 8 stops.
4. **Burst Archive Persistence**:
   - The capture script must write raw bursts to a persistent directory outside git-ignored paths (e.g. `raw_captures/real_capture_S02/`) so that 30-frame raw depth bursts are never purged.

---

## 10. Final Recommendation: NOT READY for S02 Without Pre-Flight Mounting Fix

### Verdict: **CONDITIONAL — NOT READY FOR S02 UNTIL PHYSICAL ORIENTATION IS RESOLVED**

### Evidence Supporting the Decision
1. **Software Pipeline**: The Phase 1 algorithmic fixes (unconditional burst loading, foreground dilation + depth envelope, floor consensus gating, boundary keypoint rejection) are functioning and verified by 85 passing fullbody unit tests.
2. **Hardware & Capture Protocol**: The hardware setup used in S01 had two fatal flaws:
   - The camera was mounted sideways, mapping shoulder breadth into vertical stature disparity.
   - The standoff distance was too close (0.65 m instead of 2.5 m), clipping lower-body anatomy.
3. **Requirement Before S02**:
   - Before taking an 8-stop scan S02, the operator must execute `check-camera` and visually confirm an upright, fully framed human subject in `check_color.png`.

> [!IMPORTANT]
> **AUDIT COMPLETE — EXECUTION STOPPED.**  
> In accordance with instructions, this task was strictly audit-only. No camera acquisition, S02 capture, code modifications, or synthetic ground truth were introduced. Awaiting user review and authorization.
