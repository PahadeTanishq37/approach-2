# Stage 28 — Phase 1 Implementation Report: Stage 27 Audit Fixes & Verification

## Executive Summary

Stage 28 Phase 1 addresses the critical software, geometric, and measurement defects discovered during the Stage 27 eight-stop physical scan audit. All five defect categories identified in the audit were verified directly against the code, corrected at their root locations, and validated through extensive automated unit and integration tests.

The entire test suite across the repository passes cleanly: **412 passed, 2 skipped, 6 expected failures (xfailed), 0 failures (100% pass rate)**.

In accordance with strict protocol:
- **Phase 1 is complete.**
- **No Phase 2 reprocessing or physical capture has been initiated.**
- **Awaiting user approval before proceeding to Phase 2.**

---

## Section A: Initial Code Audit vs. Verified Defects

| # | Finding | Actual Code Location | Verified Defect? | Proposed & Implemented Correction |
|---|---------|----------------------|------------------|------------------------------------|
| 1 | **Burst Processing Bypass** | `realsense_measure/fullbody/io.py:220-230` | **Yes.** Multi-frame burst processing was conditionally skipped if `filtered_depth.npy` existed on disk, bypassing temporal median, std dev, flying-pixel filtering, and noise rejection (`s.depth_std` set to 0). | Removed the bypass condition. When `burst_depths` contains $>1$ frames, `process_stop_burst()` is executed unconditionally, producing true temporal medians, noise maps, and valid count masks while preserving raw frames. |
| 2 | **Wall Intrusion in Torso Slicing** | `realsense_measure/fullbody/measure_scan.py:260-290`<br>`realsense_measure/fullbody/fusion.py:55-65`<br>`realsense_measure/fullbody/cli.py:350-365` | **Yes.** Torso slicing slabs lacked lateral and radial gating, allowing room walls at $Z = 1.8\text{ m}$ to be enveloped into the convex hull, inflating waist to 2291.8 mm and hip to 2363.0 mm. | Added dilated MediaPipe 2D `person_mask` coupled with body-envelope depth consistency ($Z \in [Z_{med} - 0.50, Z_{med} + 0.50]$) at point-cloud generation and TSDF integration. Gated torso ring slicing to lateral $\|X\| < 0.28\text{ m}$, depth $\|Z\| < 0.35\text{ m}$, and radial distance $r \le 0.35\text{ m}$ from trunk centroid. |
| 3 | **Floor Tilt & Stature Truncation** | `realsense_measure/fullbody/floor.py:180-215`<br>`realsense_measure/fullbody/measure_scan.py:165-200`<br>`realsense_measure/fullbody/cli.py:365-385` | **Yes.** A single rogue floor plane (or desk/table top) was accepted as floor consensus, tilting coordinates by 29.8°. With clipped lower legs, stature measured 696.8 mm from an unobserved floor. | Enforced $\ge 3$ agreeing floor planes (`min_planes >= 3`) across multi-stop scans. Added verticality gating (normal up alignment $\ge 0.70$) and elevated surface rejection ($Y_{cam} \ge 0.50\text{ m}$ below camera). When floor is unobserved (`floor_observed = False`), stature and `stature_head_fit` return `NaN`, flagged `NOT_MEASURABLE` with notes. |
| 4 | **Arm Measurement Asymmetry & Chord vs Kinematic Chain** | `realsense_measure/fullbody/measure_scan.py:210-235` | **Yes.** Arm length was computed as the direct Euclidean chord $\|P_{wrist} - P_{shoulder}\|$. On bent arm postures, this chord was shorter than the upper arm segment ($147.8\text{ mm} < 150.8\text{ mm}$). | Implemented anatomical kinematic chain summation: $\text{arm\_length} = \text{shoulder\_elbow} + \text{elbow\_wrist}$. The pipeline now reports separate upper arm, forearm, total kinematic arm length (`method="kinematic_chain"`), and straight chord (`method="skeletal_chord"`). |
| 5 | **Multi-View Keypoint Aggregation Locked to Stop 0** | `realsense_measure/fullbody/cli.py:389`<br>`realsense_measure/fullbody/keypoints.py:195, 328-420` | **Yes.** Reconstruction locked to `ref_kpts = aligned[0].keypoints_canonical`, completely discarding keypoints from the remaining 7 stops and leaving boundary-clipped landmarks at image edges. | Added 15 px boundary margin rejection to `lift_keypoints_to_3d` to eliminate boundary-clipped edge landmarks. Implemented `aggregate_canonical_keypoints()` using multi-view consensus: frame verification (`canonical`), confidence $\ge 0.20$, robust median pooling, and $\le 0.15\text{ m}$ outlier rejection with provenance tracking. |

---

## Section B: Implementation Details

### 1. Burst Processing (`fullbody/io.py`)
- Unconditional multi-frame burst processing: `_load_stops_v2` checks `len(burst_depths) > 1`. If present, it executes `process_stop_burst(burst_depths, cfg)`.
- Temporal median depth, depth standard deviation, valid count maps, and noise rejection are computed cleanly on every load.
- Original raw burst files (`depth_*.npy`) are preserved completely intact and never modified.
- Added regression test `test_stops_v2_burst_processing_not_bypassed` in `tests/fullbody/test_session_format.py`.

### 2. Foreground / Background Rejection (`fullbody/fusion.py`, `fullbody/cli.py`, `fullbody/measure_scan.py`)
- **2D Person Mask Detection**: `MediaPipePoseDetector.detect_with_mask` enables MediaPipe segmentation masks (`output_segmentation_masks=True`), returning a binary foreground mask.
- **Silhouette Preservation with Depth Gating**: An 11–15 px morphological dilation ensures hair, fingers, and garment contours are never clipped. Within the dilated mask, pixels are strictly gated to the subject depth envelope: $Z \in [Z_{med} - 0.50\text{ m}, Z_{med} + 0.50\text{ m}]$. Background walls ($Z \ge 2.0\text{ m}$) are completely rejected.
- **Torso Slicing Envelope**: In `extract_ring_circumference`, torso slicing bounds points to lateral $\|X\| < 0.28\text{ m}$, depth $\|Z\| < 0.35\text{ m}$, and radial distance $r \le 0.35\text{ m}$ from trunk centroid, preventing any stray background points from inflating convex hulls.
- **Diagnostics**: `test_fusion_foreground_mask_rejects_background_walls` verifies:
  - Original pixels: 76,800 (7,529 person, 69,271 wall).
  - Retained points in fusion: 6,598 person points.
  - Rejected wall points in fusion: 69,271 (0 wall points entered fusion).

### 3. Floor Plane & Gravity Alignment (`fullbody/floor.py`, `fullbody/measure_scan.py`, `fullbody/cli.py`)
- **Multi-Stop Consensus**: `verify_floor_consistency` requires at least 3 agreeing floor plane detections (`min_planes >= 3`) before declaring `floor_observed = True`.
- **False Plane Rejection**:
  - Vertical walls are rejected via `alignment = dot(normal, expected_up) >= cfg.floor_normal_min_up_alignment` (0.70).
  - Elevated surfaces (tables, desks, boxes) are rejected if $Y_{cam} < 0.50\text{ m}$ below camera height.
- **Stature Invalidation**: When `floor_observed = False`, `stature` and `stature_head_fit` return `value_mm = NaN`, `quality_flag = NOT_MEASURABLE`, and notes `floor_unobserved (no verified floor plane in scan)`.

### 4. Anatomical Arm Measurements (`fullbody/measure_scan.py`)
- **Kinematic Chain Sum**:
  $$\text{arm\_length} = \|\mathbf{p}_{\text{shoulder}} - \mathbf{p}_{\text{elbow}}\| + \|\mathbf{p}_{\text{elbow}} - \mathbf{p}_{\text{wrist}}\|$$
- Reports four separate, unambiguous arm metrics:
  - `shoulder_elbow_length_left` / `shoulder_elbow_length_right`
  - `elbow_wrist_length_left` / `elbow_wrist_length_right`
  - `arm_length_left` / `arm_length_right` (`method="kinematic_chain"`)
  - `shoulder_wrist_chord_left` / `shoulder_wrist_chord_right` (`method="skeletal_chord"`)
- Verified that on bent postures, kinematic arm length is strictly greater than chord length.

### 5. Multi-View Keypoint Consensus (`fullbody/keypoints.py`, `fullbody/cli.py`)
- **Edge Landmark Rejection**: In `lift_keypoints_to_3d`, 2D landmarks within 15 px of image boundaries are assigned zero confidence and zeroed out, preventing edge-stuck MediaPipe detections on truncated limbs.
- **Robust Multi-View Aggregation**: `aggregate_canonical_keypoints` pools landmarks across all accepted stops transformed to canonical space:
  - Validates `frame == "canonical"`.
  - Discards landmarks with confidence $< 0.20$ or visibility $< 0.20$.
  - Computes spatial median $\mathbf{m}_j$ and discards outliers with $\|\mathbf{p}_i - \mathbf{m}_j\| > 0.15\text{ m}$.
  - Computes confidence-weighted average across inlier views.
  - Attaches per-landmark view provenance metadata: `supporting_views` (with weights and residuals) and `rejected_views` (with rejection reasons).

---

## Section C: Verification & Test Results

### 1. Targeted Unit Tests
| Test File | Tests Run | Result | Notes |
|-----------|-----------|--------|-------|
| `tests/fullbody/test_session_format.py` | 8 | **PASS** | Validates burst processing unconditional execution & multi-frame median |
| `tests/fullbody/test_floor.py` | 4 | **PASS** | Validates floor RANSAC, wall rejection, desk rejection, $\ge 3$ stop requirement |
| `tests/fullbody/test_fusion.py` | 1 | **PASS** | Validates foreground mask dilation, depth consistency, 0 wall points in TSDF |
| `tests/fullbody/test_keypoints_and_parts.py` | 6 | **PASS (5 pass, 1 skip)** | Validates 15px boundary check, multi-view median, $>15\text{cm}$ outlier rejection |
| `tests/fullbody/test_measure_scan.py` | 8 | **PASS** | Validates kinematic arm chain, unobserved floor stature rejection, torso wall gating |
| `tests/fullbody/test_e2e.py` | 3 | **PASS** | Full synthetic pipeline end-to-end integration |
| `tests/fullbody/test_ladder.py` | 2 | **PASS** | Multi-seed 5-level degradation ladder diagnostic |

### 2. Full Repository Test Suite
Command: `pytest`
- **Total collected items**: 420
- **Passed**: 412
- **Skipped**: 2
- **Expected failures (xfailed)**: 6 (pre-existing known issues)
- **Failed**: **0**
- **Execution duration**: 975.09s (16m 15s)

---

## Section D: Remaining Limitations & Phase Completion Classification

### 1. Limitations Resolved in Phase 1 (Software / Pipeline)
- [x] Multi-frame burst data is now fully utilized.
- [x] Room walls and background objects can no longer enter TSDF fusion or distort torso circumferences.
- [x] Floor planes are rigorously validated and require cross-stop agreement.
- [x] Arm lengths reflect anatomical kinematic chains instead of chords.
- [x] Keypoints are robustly aggregated across all 8 views with outlier rejection.

### 2. Limitations Requiring Physical Acquisition (Phase 2 & Phase 3)
- **Lower-Body Geometric Truncation in S01**: The physical dataset S01 was captured with the tripod positioned too close ($1.3\text{ m}$) at normal height without full-height framing, clipping the subject's feet, ankles, and floor. Consequently, stature cannot be measured from S01.
- **True Stature Acquisition**: Acquiring genuine physical full-body stature requires physical scan S02 with camera distance $1.8\text{–}2.2\text{ m}$ to capture head-to-floor geometry.

---

## Section E: Phase Completion Decision

> [!IMPORTANT]
> **Phase 1 is officially COMPLETE.**
> In accordance with our execution protocol, work is **STOPPED** at the end of Phase 1.
> No reprocessing of S01 or physical acquisition of S02 will occur until explicit user authorization is received.

