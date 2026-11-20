# Full-Body Repository Cleanup Report

**Branch**: `cleanup/fullbody-reset`  
**Date**: 2026-11-20  
**Scope**: Repository cleanup of legacy Anny-era body modules, relocation of historical reports/docs into `archive/`, documentation alignment, and dependency auditing. Zero changes were made to the operational head pipeline.

---

## 1. Files Moved to Archive

The `archive/` directory was created to store historical artifacts from Stages 27–29. `archive/README.md` was added noting that these files are historical and unmaintained.

| Original Path | New Archive Path |
|---|---|
| `STAGE27_REALTIME_VALIDATION_REPORT.md` | `archive/stage_reports/STAGE27_REALTIME_VALIDATION_REPORT.md` |
| `STAGE28_PHASE1_IMPLEMENTATION_REPORT.md` | `archive/stage_reports/STAGE28_PHASE1_IMPLEMENTATION_REPORT.md` |
| `STAGE28_PHASE2_1_EVIDENCE_RECONCILIATION_REPORT.md` | `archive/stage_reports/STAGE28_PHASE2_1_EVIDENCE_RECONCILIATION_REPORT.md` |
| `STAGE28_PHASE2_2_CAMERA_PREFLIGHT_REPORT.md` | `archive/stage_reports/STAGE28_PHASE2_2_CAMERA_PREFLIGHT_REPORT.md` |
| `STAGE28_PHASE2_3_PHYSICAL_MOUNT_VALIDATION_REPORT.md` | `archive/stage_reports/STAGE28_PHASE2_3_PHYSICAL_MOUNT_VALIDATION_REPORT.md` |
| `STAGE28_PHASE2_S01_REPROCESSING_REPORT.md` | `archive/stage_reports/STAGE28_PHASE2_S01_REPROCESSING_REPORT.md` |
| `STAGE29_2_ORIENTATION_CORRECTION_REPORT.md` | `archive/stage_reports/STAGE29_2_ORIENTATION_CORRECTION_REPORT.md` |
| `STAGE29_FULLBODY_RECONSTRUCTION_TEST_REPORT.md` | `archive/stage_reports/STAGE29_FULLBODY_RECONSTRUCTION_TEST_REPORT.md` |
| `docs/STAGE27_CAPTURE_DESIGN.md` | `archive/docs/STAGE27_CAPTURE_DESIGN.md` |
| `docs/presentation.html` | `archive/docs/presentation.html` |

*Retained in `docs/`*:
- `docs/LEGACY_NOTES.md`
- `docs/FULLBODY.md` (updated with top status note)

---

## 2. Files Deleted & Grep Evidence

Before deletion, exhaustive greps across `realsense_measure/`, `tests/`, `tools/`, and `scripts/` verified that nothing outside the target test file imported or referenced the target modules.

### A. `realsense_measure/body/measure.py`
- **Grep Pattern**: `body\.measure|from realsense_measure\.body\.measure`
- **Findings**:
  - `tests/body/test_measure_geometry.py:31`: `from realsense_measure.body.measure import ...`
  - `realsense_measure/body/__init__.py:8`: `from .measure import Measurement, measure_body, is_contour_closed` (removed)
  - Zero imports or usages anywhere in head pipeline, fullbody pipeline, tools, or scripts.
- **Action**: Safely deleted via `git rm`.

### B. `realsense_measure/body/model.py`
- **Grep Pattern**: `body\.model|from realsense_measure\.body\.model`
- **Findings**:
  - `tests/body/test_measure_geometry.py:478`: `from realsense_measure.body.model import BodyModel, Joint`
  - `realsense_measure/body/__init__.py:7`: `from .model import BodyModel, BodyOutput` (removed)
  - Zero imports or usages in head pipeline, fullbody pipeline, tools, or scripts.
- **Action**: Safely deleted via `git rm`.

### C. `tests/body/test_measure_geometry.py`
- **Grep Pattern**: `test_measure_geometry|measure_geometry`
- **Findings**:
  - Only tested `measure.py` and `model.py` (17 unit test functions).
  - No external references.
- **Action**: Safely deleted via `git rm`.

### D. Preserved Modules in `realsense_measure/body/`
The following files were explicitly preserved as required:
- `realsense_measure/body/capture_session.py`: Imported by `camera/recording.py`, `fullbody/session_format.py`, `pipeline.py`, `tests/body/test_body_burst_capture.py`, `tools/run_sweet_spot_experiment.py`.
- `realsense_measure/body/capture_motion.py`: Imported by `pipeline.py`, `tests/body/test_body_burst_capture.py`.
- `realsense_measure/body/hardware_smoke_test.py`: Imported by `main.py` (`cli_body_smoke_test`).
- `tests/body/test_body_burst_capture.py`: Active test suite (7 passing tests).

> **Proposal regarding `body/` package name**:  
> Having both `realsense_measure/body/` and `realsense_measure/fullbody/` can cause cognitive ambiguity. The remaining modules in `body/` are shared session manifests, stillness motion evaluators, and sensor smoke tests. In a future refactoring prompt, we propose migrating these three utilities into `realsense_measure/camera/` (e.g. `camera/burst_motion.py`, `camera/session_manifest.py`, `camera/hardware_check.py`) and deprecating `realsense_measure/body/` entirely.

---

## 3. Candidate Full-Body Modules (Step 3 Audit)

The following candidates in `realsense_measure/fullbody/` were audited and kept intact (neither deleted nor modified):

| Candidate Module | Who Imports It | Which Tests Cover It | Current Purpose & Architectural Notes |
|---|---|---|---|
| `alignment.py` | `fullbody.cli`<br>`fullbody.debug`<br>`fullbody.validate` | `test_alignment.py`<br>`test_e2e.py`<br>`test_floor.py`<br>`test_keypoints_and_parts.py`<br>`test_ladder.py`<br>`test_missing_coverage.py` | Multi-view pose graph alignment, multi-scale colored ICP edge solving, Levenberg-Marquardt optimization, and loop closure. Scheduled for redesign. |
| `fusion.py` | `fullbody.capture_controller`<br>`fullbody.cli`<br>`fullbody.debug`<br>`fullbody.validate` | `test_e2e.py`<br>`test_fusion.py`<br>`test_ladder.py`<br>`test_missing_coverage.py`<br>`test_stage27_adapter_ui_cli.py`<br>`test_synthetic_background.py` | Foreground mask calculation, volumetric TSDF voxel integration, Poisson surface reconstruction, and watertight mesh cleaning. Scheduled for redesign. |
| `keypoints.py` | `fullbody.cli`<br>`fullbody.validate` | `test_keypoints_and_parts.py`<br>`test_stage27_adapter_ui_cli.py` | MediaPipe Pose detector, 2D to 3D camera coordinate projection, MAD outlier rejection, and canonical keypoint aggregation. |
| `landmarks.py` | `fullbody.measure_scan` | `test_measure_scan.py`<br>`test_missing_coverage.py` | Slicing plane geometric definitions (crown, neck, chest, waist, hip, crotch, limbs) in ISB canonical coordinates. |
| `template.py` | *(None in production code)* | `test_missing_coverage.py` | Generic rigged template interface, skinning weights, and unobserved region clamping fallback. |
| `sweet_spot.py` | `fullbody.cli` (`sweet-spot`) | `test_validate_and_sweet_spot.py` | Standoff distance information scoring and sensor sweet spot optimization across camera heights and distances. |
| `coverage.py` | `fullbody.cli` (`coverage-report`) | `test_coverage.py` | Vertical and angular coverage metric calculations from depth standoffs. |
| `debug.py` | `fullbody.cli` (`debug-stop`) | `test_debug_and_validation.py` | Per-stop diagnostic exporter: mask overlay, depth diff, masked cloud PLY, and data-only `stats.json`. |

---

## 4. Documentation & Data Verification

1. **`README.md`**:
   - Removed references to `body/model.py`, `body/anny_backend.py`, `body/measure.py`, `body/prior.py`.
   - Updated directory tree to match existing structure.
2. **`docs/FULLBODY.md`**:
   - Added Status notice at the top:
     > **Status:** Full-body pipeline is being rebuilt as stop-and-go + body-model prior. See plan.
3. **`data/tape_measurements.json`**:
   - Checked loader in `realsense_measure/fullbody/validate.py`. The file contains valid JSON with null fields for unmeasured data and instructions; loaded cleanly without modifications.
4. **`raw_captures/` in `.gitignore`**:
   - Verified present in `.gitignore` (`raw_captures/`, `!raw_captures/**/background/**`).
   - Grepped codebase for any code deleting `raw_captures/`: zero auto-delete or purge logic found.

---

## 5. Test Result Diff vs Baseline

Baseline test output was recorded to `C:\tmp\baseline_tests.txt` (and repository root `baseline_tests_raw.txt`).

| Metric | Safety Baseline (Step 0) | Post-Cleanup (Step 6) | Diff |
|---|---|---|---|
| Total Tests Collected | 458 | 441 | **-17** (exact count of removed `test_measure_geometry.py` tests) |
| Tests in `tests/body/` | 24 (7 burst + 17 measure) | 7 (`test_body_burst_capture.py`) | -17 dead Anny-era geometry tests |
| Non-Slow Test Pass Rate | 399 passed, 3 skipped | 399 passed, 3 skipped, 39 deselected | **0 new failures** |
| CLI `--help` Checks | Pass (`fullbody` & `main`) | Pass (`fullbody` & `main`) | Unchanged |

---

## 6. Items Not Touched / Left for Subsequent Tasks

1. **`alignment.py` and `fusion.py`**: Left intact per Step 3 instructions to be addressed in subsequent dedicated prompts.
2. **`template.py` and rigging tests**: Left intact in `realsense_measure/fullbody/` and `tests/fullbody/`.
3. **`data/tape_measurements.json`**: Left comments/instructions intact as the schema is already valid JSON and consumed without error by `load_tape_measurements()`.
4. **Temporary test outputs**: Legacy run directories in the repo root (`scan_output*`, `stage_wise_output`, `scan_seed*`) were left untouched as they are gitignored and not part of the core source tree.
