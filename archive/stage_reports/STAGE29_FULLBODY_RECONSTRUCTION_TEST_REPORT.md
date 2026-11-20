# STAGE 29 — END-TO-END FULL-BODY 3D RECONSTRUCTION TEST REPORT

**Date**: 2026-11-17  
**Repository Branch**: `main`  
**Git Commit SHA**: `907dad896c85f0d938cfaea8f4f9abd0a4d15a06`  
**Pipeline Entry Point**: `realsense_measure.fullbody scan`  
**Target Output Directory**: `scan_output/stage29_reconstruction_test/`  
**Audit Purpose**: End-to-end evaluation of the existing full-body 3D reconstruction pipeline on physical data up to the generation of a 3D human body mesh, without modifying codebase algorithms.  

---

## 1. Executive Summary

This report documents the end-to-end testing and technical evaluation of the full-body 3D reconstruction pipeline in `realsense_measure` as executed on the current `main` branch. 

The pipeline CLI command `python -m realsense_measure.fullbody scan` was executed end-to-end on the physical dataset `body_03` (Intel RealSense D455f) and written to the newly created target directory `scan_output/stage29_reconstruction_test/`. 

### Key Findings
1. **Pipeline Execution**: The entire pipeline executed without unhandled exceptions or crashes, generating aligned per-stop camera transforms (`alignment.npz`), a fused metric point cloud (`fused.ply`), a reconstructed 3D triangle surface mesh (`mesh.ply` and `mesh.obj`), and anthropometric measurement results (`scan_result.json`).
2. **Surface Generation**: The reconstruction produced a 3D surface mesh containing **32,431 vertices and 57,204 triangles** via the bounded `UniformTSDFVolume` integration engine, and a fused point cloud of **35,408 metric 3D points**.
3. **Multi-View Alignment**: The 4-DoF MAP-ICP registration converged with bounded recovery clamping on all stops, yielding a loop closure drift of **-3.08°** (well within the ±20.0° tolerance threshold).
4. **Physical Full-Body Completeness**: The resulting mesh **does not constitute a verified, coherent head-to-feet human body reconstruction**. The dataset is a 4-stop sequence (0°, 90°, 180°, 270°) where 2 stops (Stop 1 and Stop 3) failed MediaPipe pose detection. The floor was unobserved across stops (normal deviation > 0.70 verticality gate), resulting in `stature` being correctly flagged `NOT_MEASURABLE` (`floor_unobserved`).
5. **Mesh Topology**: The reconstructed mesh contains **13 disjoint connected components**, with the dominant component comprising only **28.5%** of the triangles. It exhibits 9,254 non-manifold edges and large unobserved gaps across the lower torso, pelvic girdle, lower legs, and feet.

---

## 2. Repository Version & Architecture Audit

- **Active Commit**: `907dad896c85f0d938cfaea8f4f9abd0a4d15a06`
- **Execution Path Analysis**:
  ```
  load_capture_session() [io.py]
          │
          ▼
  MediaPipePoseDetector.detect_with_mask() [keypoints.py]
          │
          ▼
  lift_keypoints_to_3d() [keypoints.py]
          │
          ▼
  fit_floor_plane() & verify_floor_consistency() [floor.py]
          │
          ▼
  compute_confidence_map() [confidence.py]
          │
          ▼
  align_stops() [alignment.py] (4-DoF MAP-ICP + Loop Closure)
          │
          ▼
  fuse_aligned_stops() [fusion.py] (ScalableTSDFVolume -> UniformTSDFVolume fallback)
          │
          ▼
  aggregate_canonical_keypoints() [keypoints.py]
          │
          ▼
  measure_fused_scan() [measure_scan.py]
          │
          ▼
  save_scan_result() [io.py]
  ```
- **Module Roles in Normal Pipeline**:
  - `cli.py`: Command-line dispatch and `run_pipeline()` orchestrator.
  - `io.py`: Ingestion of `stops_v2` or legacy `meta.json` formats.
  - `burst.py`: Multi-frame temporal median, standard deviation, and flying-pixel noise rejection.
  - `keypoints.py`: MediaPipe Pose detection, 3D metric lifting, boundary margin rejection, and canonical aggregation.
  - `floor.py`: Consensus floor plane RANSAC, verticality gating ($|b| \ge 0.70$), and desk rejection ($Y \ge 0.50\text{ m}$).
  - `alignment.py`: 4-DoF MAP-ICP registration regularized around consensus body axis.
  - `fusion.py`: TSDF volumetric integration and point-union fallback.
  - `parts.py`: Cylindrical capsule labelling for rigid torso vs articulated limbs.
  - `measure_scan.py`: Slicing slabs, polar contour circumference extraction, and kinematic chain segment lengths.
  - `template.py`: *Optional template deformation fitter* — confirmed **not** invoked in the standard reconstruction pipeline.

---

## 3. Dataset Discovery and Provenance

An audit of all candidate scan and capture datasets across the workspace was performed:

| Dataset Identifier | Filesystem Location | Classification | Camera / Hardware | Stops / Frames | Usability for End-to-End Test |
|---|---|---|---|---|---|
| `body_03` | `realsense_measure-main/realsense_measure/raw_captures/body_03` | **1. Original physical camera capture** | Intel RealSense D455f (`848x480`) | 4 stops (4 frames, metric `.npy` depth) | **USED FOR PRIMARY TEST** — Ingested and reconstructed end-to-end into `scan_output/stage29_reconstruction_test/`. |
| `body_01` | `realsense_measure-main/realsense_measure/raw_captures/body_01` | **1. Original physical camera capture** | Intel RealSense D455f (`848x480`) | 4 stops (4 frames, metric `.npy` depth) | Tested: High inter-stop ICP residual (27 mm > 15 mm) caused stops to be rejected from TSDF, yielding point-cloud union only (0 mesh vertices). |
| `body_02` | `realsense_measure-main/realsense_measure/raw_captures/body_02` | **1. Original physical camera capture** | Intel RealSense D455f (`848x480`) | 4 stops (4 frames, metric `.npy` depth) | Tested: Similar inter-stop ICP residual rejection; TSDF yielded 0 mesh vertices. |
| `body_capture_01` | `raw_captures/body_capture_01` | **1. Original physical camera capture** | Intel RealSense D455f (`1280x720`) | 4 stops (room-scale, 65m depth) | Tested: Only head/shoulders detected (7 landmarks); mesh yielded 2,856 vertices. |
| `body_capture_02` | `raw_captures/body_capture_02` | **1. Original physical camera capture** | Intel RealSense D455f (`1280x720`, SUBJ_02) | 4 stops (room-scale, 65m depth) | Tested: Upper-body only (11 landmarks); mesh yielded 2,225 vertices. |
| `body_capture_live_01` | `raw_captures/body_capture_live_01` | **1. Original physical camera capture** | Intel RealSense D455f (`848x480`) | 30 frames (Single view smoke test) | Inapplicable: Single static perspective (0° rotation), not a multi-view scan. |
| `S01 (Stage 27)` | `scan_output/real_capture_S01` | **1. Original physical camera capture (Purged)** | Intel RealSense D455f (`342622300088`) | 8 stops (30 frames/burst) | **UNAVAILABLE**: Raw bursts purged on 2026-11-13 during git branch operations. |
| `stage27_realtime_validation_output` | `stage27_realtime_validation_output` | **2. Derived data from physical capture** | Intel RealSense D455f | 8 PNG color previews, derived PLY/OBJ | Inapplicable: Contains no metric depth arrays; `load_capture_session()` correctly rejects layout. |
| `real_capture_S01_stage28_phase2` | `scan_output/real_capture_S01_stage28_phase2` | **4. Reused / reconstructed output** | Intel RealSense D455f | Reprocessed assets from Stage 28 Phase 2 | Historical benchmark reference only; not an independent raw capture. |
| `synth_humanoid` | `tests/fullbody/synth_humanoid.py` | **3. Synthetic procedural data** | Procedural 8-stop cylinder/capsule model | 8 stops | Used strictly in automated unit and regression tests. |

---

## 4. Exact Commands Executed

The official test was executed using the current CLI interface without code modifications:

```powershell
cd C:\Users\pahad\OneDrive\Desktop\realsense_measure\realsense_measure-main
python -m realsense_measure.fullbody scan `
  --capture realsense_measure/raw_captures/body_03 `
  --out ../scan_output/stage29_reconstruction_test `
  --verbose
```

### Exit Code and Log Summary
- **Exit Code**: `0`
- **Elapsed Time**: 3.65 seconds
- **Output Artifacts Created**:
  - `scan_output/stage29_reconstruction_test/alignment.npz` (5.8 KB)
  - `scan_output/stage29_reconstruction_test/fused.ply` (1.3 MB)
  - `scan_output/stage29_reconstruction_test/mesh.ply` (2.4 MB)
  - `scan_output/stage29_reconstruction_test/mesh.obj` (3.7 MB, exported)
  - `scan_output/stage29_reconstruction_test/scan_result.json` (7.6 KB)

---

## 5. Input Data Quality Evaluation

The input dataset `body_03` comprises 4 stops captured with an Intel RealSense D455f at 848×480 native resolution:

| Stop ID | Nominal Angle | RGB Dim | Depth Dim | Depth Scale / Units | Valid Depth % | Median Depth | 10th %ile Depth | 90th %ile Depth | Detected Keypoints | Person Mask % | Valid 3D Pts |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Stop 0** | 0.0° | 480×848 | 480×848 | 1.0 m (float32) | 91.6% | 2.926 m | 0.875 m | 7.161 m | 13 (Head, Shoulders, Arms, Hips) | 20.9% | 85,098 |
| **Stop 1** | 90.0° | 480×848 | 480×848 | 1.0 m (float32) | 89.1% | 3.304 m | 0.904 m | 7.398 m | 0 (No pose detected) | 16.4% | 66,748 |
| **Stop 2** | 180.0° | 480×848 | 480×848 | 1.0 m (float32) | 90.8% | 2.881 m | 0.845 m | 7.258 m | 21 (Full skeleton: Head to Feet) | 24.3% | 98,892 |
| **Stop 3** | 270.0° | 480×848 | 480×848 | 1.0 m (float32) | 87.7% | 3.408 m | 1.030 m | 7.516 m | 0 (No pose detected) | 14.8% | 60,236 |

### Input Observations & Data Defects
1. **Pose Detection Gaps**: MediaPipe Pose detection succeeded on Stop 0 (13 joints) and Stop 2 (21 joints), but completely failed on Stop 1 (lateral right) and Stop 3 (lateral left) due to subject distance (~3.3m) and limb self-occlusion.
2. **Floor Gating Failure**: Across all 4 stops, the fitted plane normals deviated significantly from verticality ($|b| = 0.63, 0.21, 0.26, 0.11 < 0.70$). As a result, the floor was rejected as unobserved.
3. **Subject Truncation**: The subject was positioned at ~2.9–3.4 m distance, causing the lower limbs to be partially clipped by the camera frame in Stop 0 and Stop 1.

---

## 6. Multi-View Alignment Diagnostics

Alignment was performed using keypoint priors combined with 4-DoF MAP-ICP registration regularized to the vertical body axis.

| Stop ID | Nominal Angle | Refined Yaw | Yaw Correction | Translation ($t_x, t_y, t_z$) | Residual Before | Residual After | Inlier Fraction | Bounded Recovery Clamped? | Stop Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Stop 0** | 0.0° | -176.92° | +3.08° | `[0.010, -1.000, 1.047]` m | 29.9 mm | 29.9 mm | 0.88 | **YES** | ACCEPTED (Clamped) |
| **Stop 1** | 90.0° | 90.77° | +0.77° | `[1.525, -1.000, -0.046]` m | 28.5 mm | 28.5 mm | 0.85 | **YES** | ACCEPTED (Clamped) |
| **Stop 2** | 180.0° | 1.54° | +1.54° | `[-0.046, -1.000, -1.525]` m | 26.0 mm | 26.0 mm | 0.91 | **YES** | ACCEPTED (Clamped) |
| **Stop 3** | 270.0° | 272.31° | +2.31° | `[-1.525, -1.000, 0.046]` m | 27.0 mm | 27.0 mm | 0.84 | **YES** | ACCEPTED (Clamped) |

### Alignment Analysis
- **ICP Clamping**: The 4-DoF ICP refinement exceeded the divergence clamp threshold and fell back to the bounded prior on all 4 stops (`Bounded recovery clamp triggered for 4 stop adjustment(s)`).
- **Loop-Closure Residual**: Enforced loop closure over 360° span resulted in **-3.08°** residual angular drift, which is within the 20.0° acceptance threshold.
- **Ghosting / Double Surfaces**: The registered point clouds exhibit moderate ghosting (~27 mm mean surface discrepancy) due to small posture changes between the 4 independent stops.

---

## 7. Fusion Diagnostics & 3D Geometry

- **Integration Mode**: Open3D `ScalableTSDFVolume` returned 0 vertices due to bounding volume initialization over sparse stops. The pipeline cleanly triggered the built-in fallback to bounded `UniformTSDFVolume` (`length=2.23m`, `resolution=384`, `voxel_length=0.005m`, `sdf_trunc=0.02m`).
- **Surface Extraction**: The uniform TSDF volume successfully extracted the 3D surface mesh.

### Quantitative Mesh Statistics

| Metric | Measured Value | Standard Target | Assessment |
|---|---|---|---|
| **Vertex Count** | **32,431** | > 10,000 | Good geometric resolution |
| **Triangle Count** | **57,204** | > 20,000 | High-density triangulation |
| **Connected Components** | **13** | 1 (Single manifold) | **Fragmented** (13 disjoint shells) |
| **Largest Component Triangles** | 16,304 | > 90% | **28.5%** (Torso shell only) |
| **Edge Manifoldness** | `True` | `True` | Clean edge topology |
| **Vertex Manifoldness** | `False` | `True` | Pinch points at thin limb interfaces |
| **Non-Manifold Edges** | 9,254 | 0 | Present along component boundaries |
| **Degenerate Triangles** | 149 (0.26%) | < 0.5% | Acceptable numerical noise |
| **Mesh Bounding Box** | `[-0.88, -2.12, -1.05]` to `[0.94, -0.00, 1.05]` m | Full body envelope | Height: 2.12 m (contaminated by ground) |
| **PCD Point Count** | **35,408** | > 20,000 | Dense spatial coverage |
| **PCD Extent** | `X=1.87 m, Y=2.35 m, Z=3.25 m` | — | Extends to room boundaries |

---

## 8. Anatomical Region-by-Region Assessment

Each anatomical region was evaluated on the reconstructed 3D mesh and fused point cloud:

| Anatomical Region | Reconstruction Status | Observation Support | Evidence & Notes |
|---|:---:|:---:|---|
| **Head and Neck** | **PARTIALLY RECONSTRUCTED** | Directly observed in Stops 0 & 2 | Upper cranial vault and neck collar reconstructed; crown surface has unobserved open hole (`hole_map['crown'] = 1.0`). |
| **Shoulders** | **PARTIALLY RECONSTRUCTED** | Directly observed in Stops 0 & 2 | Both acromion surfaces reconstructed; shoulder breadth measured at 216.3 mm (partial coverage 50%). |
| **Chest and Torso** | **PARTIALLY RECONSTRUCTED** | Observed in Stops 0, 1, 2, 3 | Dominant component (16,304 triangles) captures chest front and back; chest circumference returned NaN (coverage 25% < 35% threshold). |
| **Pelvis and Hips** | **MISSING** | Partially observed in Stop 0 & 2 | Disjointed lower torso boundary; insufficient points in slicing slab (`waist` and `hip` returned NaN with 0% coverage). |
| **Left Upper Arm** | **PARTIALLY RECONSTRUCTED** | Observed in Stop 0 & 2 | Skeletal segment length: 262.8 mm; surface shell is fragmented. |
| **Left Forearm** | **PARTIALLY RECONSTRUCTED** | Observed in Stop 0 & 2 | Skeletal segment length: 190.2 mm; kinematic arm length: 453.0 mm. |
| **Right Upper Arm** | **PARTIALLY RECONSTRUCTED** | Observed in Stop 0 & 2 | Skeletal segment length: 257.8 mm; surface shell is fragmented. |
| **Right Forearm** | **PARTIALLY RECONSTRUCTED** | Observed in Stop 0 & 2 | Skeletal segment length: 202.1 mm; kinematic arm length: 459.9 mm. |
| **Hands** | **MISSING** | Unobserved | Wrists observed but digits and palmar surfaces unresolvable at 3m distance. |
| **Left & Right Thighs** | **MISSING** | Severely clipped in Stops 0 & 1 | Points exist sparsely in Stop 2, but TSDF failed to generate a closed tubular surface. |
| **Knees** | **MISSING** | Observed only in Stop 2 | Disjoint cloud points without connecting surface triangles. |
| **Lower Legs** | **MISSING** | Unobserved in Stops 0, 1, 3 | Clipped by bottom of camera view; TSDF volume extracted no leg geometry. |
| **Feet** | **MISSING** | Unobserved | Floor unobserved; feet unobserved (`hole_map['feet'] = 1.0`). |

---

## 9. Visual Evidence File Paths

All visual evidence has been rendered, verified, and saved to `scan_output/stage29_reconstruction_test/visualizations/`:

1. **Individual Stop Point Clouds**:  
   `scan_output/stage29_reconstruction_test/visualizations/01_individual_stop_point_clouds.png`
2. **Registered Multi-View Point Cloud**:  
   `scan_output/stage29_reconstruction_test/visualizations/02_registered_multiview_point_cloud.png`
3. **Final Fused Point Cloud**:  
   `scan_output/stage29_reconstruction_test/visualizations/03_final_fused_point_cloud.png`
4. **Final Triangle Mesh**:  
   `scan_output/stage29_reconstruction_test/visualizations/04_final_triangle_mesh.png`
5. **Front View (0°)**:  
   `scan_output/stage29_reconstruction_test/visualizations/05_front_view.png`
6. **Back View (180°)**:  
   `scan_output/stage29_reconstruction_test/visualizations/06_back_view.png`
7. **Left-Side View (-90°)**:  
   `scan_output/stage29_reconstruction_test/visualizations/07_left_side_view.png`
8. **Right-Side View (+90°)**:  
   `scan_output/stage29_reconstruction_test/visualizations/08_right_side_view.png`
9. **Front Three-Quarter View (-45°)**:  
   `scan_output/stage29_reconstruction_test/visualizations/09_front_three_quarter_view.png`
10. **Back Three-Quarter View (+135°)**:  
    `scan_output/stage29_reconstruction_test/visualizations/10_back_three_quarter_view.png`
11. **Anatomical Coverage & Vertical Region Distribution**:  
    `scan_output/stage29_reconstruction_test/visualizations/11_anatomical_coverage_visualization.png`
12. **Exported 3D Assets**:  
    - PLY Mesh: `scan_output/stage29_reconstruction_test/mesh.ply`
    - OBJ Mesh: `scan_output/stage29_reconstruction_test/mesh.obj`
    - Fused PCD: `scan_output/stage29_reconstruction_test/fused.ply`
    - Per-Stop PCDs: `scan_output/stage29_reconstruction_test/visualizations/stop_00_pcd.ply` .. `stop_03_pcd.ply`

---

## 10. Test Suite Results

The relevant full-body test suite was executed against the repository:

```powershell
pytest tests/fullbody/test_e2e.py `
       tests/fullbody/test_fusion.py `
       tests/fullbody/test_alignment.py `
       tests/fullbody/test_floor.py `
       tests/fullbody/test_keypoints_and_parts.py `
       tests/fullbody/test_burst.py `
       tests/fullbody/test_missing_coverage.py `
       tests/fullbody/test_session_format.py `
       tests/fullbody/test_cli_smoke.py -v
```

### Test Outcome
- **Collected Tests**: 50 items
- **Passed**: **47 tests** (94.0%)
- **Skipped**: 2 tests (`test_handedness_real_photo`, `test_cli_keypoints_check_with_sample_image` — requiring external image assets)
- **XFailed**: 1 test (`test_tsdf_scalable_volume_minimal_reproducer` — documented upstream Open3D bug where ScalableTSDF fails on non-contiguous bounding boxes)
- **Failed**: **0 tests** (100% pass rate among runnable tests)
- **Execution Duration**: 30.01 seconds

*Note on Test vs. Physical Reconstruction Distinction*: The automated test suite verifies algorithmic components (synthetic humanoid end-to-end, RANSAC floor fitting, ICP clamping, mask dilation, and burst noise rejection) using synthetic humanoid geometry. Synthetic test passing confirms code stability, but does not substitute for empirical physical reconstruction validation.

---

## 11. Failures, Verified Defects, and Suspected Defects

### Verified Defects
1. **Physical Full-Body Capture Scarcity**: All existing physical datasets with raw metric depth (`body_01`, `body_02`, `body_03`, `body_capture_01`, `body_capture_02`) are **4-stop captures** acquired at room distances (2.9–3.4 m) with clipped lower extremities. None of the available raw datasets is an authentic 8-stop full-body capture with unclipped feet and head.
2. **TSDF Volume Component Fragmentation**: TSDF integration on 4 stops produces 13 disjoint connected components with 9,254 non-manifold boundary edges. 4 views at 90° intervals leave 90° blind spots that cannot be bridged by TSDF zero-crossing isosurface extraction without intermediate 45° views.
3. **Floor Invisibility at > 2.5 m**: When the camera is pointed horizontally at a standing subject 3 metres away, the floor plane is either completely outside the vertical FOV (480 px landscape height) or observed at such grazing angles that RANSAC plane fitting fails the $|b| \ge 0.70$ verticality test.

### Suspected Defects Requiring Further Investigation
1. **MAP-ICP Inter-Stop Residual Threshold Gating**: In `body_01` and `body_02`, stops were rejected because inter-stop residual exceeded 15.0 mm (measuring ~27 mm). Setting `accepted = False` completely suppressed those stops from TSDF integration, causing 0 mesh vertices. A softer weighting or fallback to initial priors would prevent complete mesh failure on slightly misaligned scans.

---

## 12. Recommended Next Steps

1. **Acquire New 8-Stop Physical Capture S02 (Priority 1)**:  
   Now that Stage 28 camera pre-flight is verified (portrait mounting, upright orientation, and calibrated intrinsics), capture a genuine 8-stop dataset S02 at the recommended sweet-spot distance of 1.30–1.45 m with the subject standing on a non-reflective floor.
2. **TSDF Soft-Acceptance Fallback (Priority 2)**:  
   In `fusion.py`, if all stops are rejected due to residual > 15 mm, allow stops with clamped priors to integrate into the TSDF volume rather than dropping to 0 vertices.
3. **Interactive 3D Review**:  
   Review the generated `scan_output/stage29_reconstruction_test/mesh.ply` using `python -m realsense_measure.fullbody review --scan scan_output/stage29_reconstruction_test`.

---

## 13. Stage 29 Milestone Verdict

```
PARTIAL — MESH GENERATED, BUT FULL-BODY RECONSTRUCTION NOT VERIFIED
```

**Justification**:  
The reconstruction pipeline ran end-to-end without software failure and successfully produced an exportable 3D triangle mesh (32,431 vertices, 57,204 triangles in both PLY and OBJ formats) and a 35,408-point fused point cloud. However, the available raw physical capture (`body_03`) contains only 4 stops, has clipped lower extremities, lacks floor plane visibility, and yields 13 fragmented surface components with unobserved lower legs, feet, and pelvic circumference. Therefore, a coherent head-to-feet full-body reconstruction is not verified on physical data.

