# Rigid Baseline Reconstruction (Layer 1) & Cylinder Calibration Report

This report documents the implementation and verification of the Layer 1 rigid baseline reconstruction pipeline (`rigid_baseline.py`), the rigid object calibration tool (`tools/calib_object.py`), associated CLI subcommands, and synthetic test suites.

Branch: `rigid/baseline-v1` (branched from `capture/stop-and-go-v1`)

---

## 1. Files Added & Changed

| File | Status | Description |
|---|---|---|
| `realsense_measure/fullbody/rigid_baseline.py` | Added | Layer 1 rigid baseline pipeline: robust burst temporal median, background subtraction, floor gravity alignment, vertical rotation axis estimation, nominal alignment, bounded 3-DOF ICP refinement (yaw in $\pm 5^\circ$, translation in $\pm 3\text{ cm}$), ScalableTSDFVolume integration, mesh component cleanup, colored overlay generation, and JSON reporting. |
| `tools/calib_object.py` | Added | Cylinder/bucket diameter calibration tool: 20 horizontal slices across middle 60% of height, RANSAC + least-squares circle fitting, diameter, circle RMS, angular coverage, roundness, error vs tape, and 360-degree surface residual plot (`residual_map.png`). |
| `realsense_measure/fullbody/cli.py` | Modified | Added `scan-rigid` and `calib-object` subcommands with argument parsing and execution dispatchers. Existing `scan` subcommand was untouched. |
| `realsense_measure/fullbody/verify_session.py` | Modified | Updated frame-count check to read `frames_requested` from `meta.json` (PASS at $\ge 100\%$, WARN at $90\% \le \cdot < 100\%$, FAIL at $< 90\%$). Updated camera tilt check to flag $> 2.0^\circ$ as WARN. |
| `tests/fullbody/fake_camera.py` | Modified | Added optional planar floor simulation with scriptable pitch tilt (`floor=True`, `floor_tilt_deg=0.0/3.0`, `floor_height_m=0.95`). |
| `tests/fullbody/test_stop_and_go_v1.py` | Modified | Added Step 0 regression tests: `test_verify_session_56_frames_warn` and `test_verify_session_floor_residual_and_3deg_tilt_warn`. |
| `tests/fullbody/test_rigid_baseline.py` | Added | Synthetic validation suite covering rigid cylinder reconstruction, 5 cm axis offset recovery, yaw perturbation with/without refinement, and rigid vs drifting humanoid. |

### Step 0c: Verification of Untouched Core Pipeline
Executing:
```bash
git diff --stat cleanup/fullbody-reset..HEAD -- realsense_measure/pipeline.py realsense_measure/registration.py realsense_measure/reconstruction.py realsense_measure/targets/head.py realsense_measure/fullbody/alignment.py realsense_measure/fullbody/fusion.py
```
**Output**:
*(Empty — 0 files changed, 0 insertions, 0 deletions)*

---

## 2. Synthetic Cylinder Results

All tests evaluate synthetic 8-stop turntable sessions rendered with analytic geometry and camera pinhole models.

| Test Case | Metric | Ground Truth / Target | Measured / Recovered | Status |
|---|---|---|---|---|
| **Clean Cylinder** ($r=150$ mm, 4 mm noise, known axis) | Diameter | 300.00 mm | **300.21 mm** (error: $+0.21$ mm) | PASS (< 3.0 mm) |
| | Circle-Fit RMS | Reported | **1.44 mm** | PASS |
| | Connected Components | 1 component | **1** | PASS |
| **5 cm Axis Offset** ($x=0.05$ m, $z=2.00$ m) | Rotation Axis $(x_0, z_0)$ | (0.0500, 2.0000) m | **(0.0499, 1.9976) m** | PASS (< 10.0 mm) |
| | Axis Error ($\Delta x, \Delta z$) | 0.0 mm | $\Delta x = 0.10$ mm, $\Delta z = 2.40$ mm | PASS |
| | Total Axis Error | < 10.0 mm | **2.40 mm** | PASS |
| | Axis Spread across slices | Minimal | **0.07 mm** | PASS |
| **Yaw Perturbation** ($\pm 3^\circ$, 1.5 cm eccentric turntable) | Diameter Error (**WITH refine**) | < 5.0 mm | **1.13 mm** | PASS |
| | Circle RMS (**WITH refine**) | Reported | **1.16 mm** | PASS |
| | Diameter Error (**WITHOUT refine**) | Baseline comparison | **22.22 mm** | Measured |
| | Circle RMS (**WITHOUT refine**) | Baseline comparison | **2.41 mm** | Measured |

### Per-Stop Refinement Corrections on $\pm 3^\circ$ Perturbation
| Stop | Nominal Yaw | Injected Error | Refined Yaw Correction | Refined Translation $(\Delta x, \Delta z)$ | Residual Before $\to$ After | Bound Exceeded |
|---|---|---|---|---|---|---|
| 0 | $0.0^\circ$ | $-2.5^\circ$ | $+2.41^\circ$ | $(-3.4\text{ mm}, +1.2\text{ mm})$ | $3.8\text{ mm} \to 1.1\text{ mm}$ | False |
| 1 | $45.0^\circ$ | $+3.0^\circ$ | $-2.88^\circ$ | $(+4.1\text{ mm}, -2.0\text{ mm})$ | $4.2\text{ mm} \to 1.2\text{ mm}$ | False |
| 2 | $90.0^\circ$ | $-1.8^\circ$ | $+1.74^\circ$ | $(-2.1\text{ mm}, +0.8\text{ mm})$ | $2.9\text{ mm} \to 1.0\text{ mm}$ | False |
| 3 | $135.0^\circ$ | $+2.2^\circ$ | $-2.11^\circ$ | $(+2.9\text{ mm}, -1.5\text{ mm})$ | $3.4\text{ mm} \to 1.1\text{ mm}$ | False |
| 4 | $180.0^\circ$ | $-3.0^\circ$ | $+2.92^\circ$ | $(-4.0\text{ mm}, +1.9\text{ mm})$ | $4.5\text{ mm} \to 1.2\text{ mm}$ | False |
| 5 | $225.0^\circ$ | $+1.5^\circ$ | $-1.44^\circ$ | $(+1.8\text{ mm}, -0.7\text{ mm})$ | $2.6\text{ mm} \to 1.1\text{ mm}$ | False |
| 6 | $270.0^\circ$ | $-2.0^\circ$ | $+1.94^\circ$ | $(-2.6\text{ mm}, +1.1\text{ mm})$ | $3.1\text{ mm} \to 1.0\text{ mm}$ | False |
| 7 | $315.0^\circ$ | $+2.8^\circ$ | $-2.69^\circ$ | $(+3.7\text{ mm}, -1.8\text{ mm})$ | $3.9\text{ mm} \to 1.2\text{ mm}$ | False |

---

## 3. Humanoid Results: Rigid vs 2 cm Arm Drift

Synthetic humanoid captures generated via `SyntheticHumanoidCaptureRenderer` at $480 \times 848$ resolution:

| Scenario | Connected Components (Final Mesh) | TSDF Components Before Cleanup | Arm Ghosting (Surface Distance) | Observations |
|---|---|---|---|---|
| **Rigid Humanoid** (0 cm drift, 0 sway) | **1** | 4,175 | **0.0 mm** | Perfectly integrated manifold mesh; all 8 stops align consistently onto canonical body surface. |
| **Drifting Humanoid** (2 cm arm drift) | **1** (largest torso cluster kept) | **5,010** | **34.9 mm** | Severe ghosting on both upper and lower arms; TSDF volume extracts duplicate surface shells and thin disconnected flakes around wrists and forearms. |

**Key Takeaway**: The rigid baseline assumes a globally rigid transformation. While it cleanly extracts the rigid torso and pelvis, inter-stop subject motion (such as 2 cm arm drift) causes visible ghosting and surface doubling. This demonstrates the necessity of the Layer 2 parametric body model prior.

---

## 4. Test Suite Summary vs Baseline

| Stage / Test Suite | Tests Passed | Tests Skipped | Tests Deselected | Failures | Notes |
|---|---|---|---|---|---|
| Baseline (`cleanup/fullbody-reset`) | 399 | 3 | 39 | 0 | Baseline commit |
| Post Stop-and-Go V1 (`capture/stop-and-go-v1`) | 404 | 3 | 39 | 0 | +5 tests in `test_stop_and_go_v1.py` |
| **Current (`rigid/baseline-v1`)** | **411** | **2** | **39** | **0** | +2 Step 0 regression tests, +4 `test_rigid_baseline.py` tests; 1 previously skipped camera test unblocked |

All 411 tests passed in 175.35 seconds.

---

## 5. Limitations & Vulnerabilities on Real Data

Honest engineering assessment of where Layer 1 rigid baseline will degrade or fail on real human subjects:

1. **Non-Rigid Postural Sway**:
   - Humans breathing or shifting balance undergo dynamic 10–30 mm sway and axial spine curvature variations.
   - Constrained 3-DOF refinement operates globally per stop; if the chest sways forward while hips stay stationary, rigid alignment cannot simultaneously align both.
2. **Horizontal Rotation Axis Estimation with Subject Sway**:
   - For an object or stationary cylinder, horizontal slice arc fitting recovers the axis to < 2.5 mm.
   - For a person who sways between stops, the per-stop foreground centroids jitter. If a person leans forward at stop 0 and backward at stop 4, the horizontal slice center will skew, introducing an eccentric wobble into nominal alignment.
3. **Limb Articulation & Arm Position Inconsistency**:
   - As proven in Test 4, a mere 2 cm difference in arm abduction angle between stops produces ~35 mm ghosting in the fused mesh. Real humans rarely hold arm abduction to within $< 10$ mm over 8 stops without physical hand guides.
4. **Self-Occlusion in A-Pose**:
   - When arms hang at $35^\circ$ abduction, the medial surfaces of the arms and lateral torso cast depth shadows when viewed from oblique angles. Rigid nominal rotation without deformation cannot close these crevices smoothly.
5. **RealSense Depth Noise at Grazing Angles**:
   - On human skin and dark fabric at steep grazing angles (silhouette edges), RealSense D455 stereo matching produces depth inflation ("flying pixels") that even temporal median filtering cannot fully eliminate without aggressive background subtraction.

These issues confirm why Layer 1 is designed as a rigid verification baseline (and rigid object calibrator), and motivate Layer 2 (SMPL/parametric body prior non-rigid registration).

---

## 6. Untouched Scope & Preserved Boundaries

The following files and components were strictly preserved with **zero modifications**:
- `realsense_measure/fullbody/alignment.py`
- `realsense_measure/fullbody/fusion.py`
- `realsense_measure/pipeline.py`
- `realsense_measure/registration.py`
- `realsense_measure/reconstruction.py`
- `realsense_measure/targets/head.py`
- All head measurement and head scanning modules
- Existing `python -m realsense_measure.fullbody scan` subcommand
