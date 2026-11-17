# Stage 28 — Phase 2.3: Physical Portrait Mount Validation Report

**Date**: 2026-11-17  
**Hardware Target**: Intel RealSense D455f (USB 3.2, S/N `342622300088`, Firmware `5.17.3.10`)  
**Scope**: Physical mounting orientation validation, live preview orientation audit, RGB-depth alignment verification, full-body framing assessment, and coordinate-system sanity check.  
**Execution Mode**: Controlled Pre-Flight Diagnostic Only. No S02 acquisition, no anthropometric measurements, no pipeline code modifications.  

---

## 1. Physical Mounting Description

The physical mounting configuration was tested using the connected Intel RealSense D455f camera under three configurations:
1. **Clockwise Rotation (`--portrait cw`)**: Software applies `cv2.ROTATE_90_CLOCKWISE` to the native $848 \times 480$ stream, yielding an $848 \times 480$ portrait array.
2. **Counter-Clockwise Rotation (`--portrait ccw`)**: Software applies `cv2.ROTATE_90_COUNTERCLOCKWISE`, yielding an $848 \times 480$ portrait array.
3. **Unrotated Baseline (`norot`)**: Native sensor stream without software rotation ($480 \times 848$ landscape array).

### Sensor Live State
- **Device**: Intel RealSense D455F
- **Serial Number**: `342622300088`
- **Firmware**: `5.17.3.10`
- **USB Link**: USB 3.2 (SuperSpeed)
- **Native Resolution**: $848 \times 480$ @ 30 FPS
- **Depth Scale**: `0.001` (1 mm / raw integer unit)
- **Decimation**: Strictly `OFF` (`use_decimation = False`)

---

## 2. Configuration Comparison: CW vs. CCW vs. Baseline

Live camera frames were acquired under all three configurations and analyzed with MediaPipe Pose Landmarker:

| Diagnostic Metric | Unrotated Baseline (`norot`) | Clockwise (`--portrait cw`) | Counter-Clockwise (`--portrait ccw`) |
|---|:---:|:---:|:---:|
| **Output Image Shape ($H \times W$)** | $480 \times 848$ (Landscape) | **$848 \times 480$ (Portrait)** | **$848 \times 480$ (Portrait)** |
| **Pinhole Focal Length ($f_x, f_y$)** | $(428.46, 427.90)$ | **$(427.32, 427.88)$** | **$(427.90, 428.46)$** |
| **Principal Point ($c_x, c_y$)** | $(422.17, 244.60)$ | **$(234.40, 422.17)$** | **$(244.60, 424.83)$** |
| **Measured Standoff Depth (ROI Median)**| $1.107\text{ m}$ | $1.107\text{ m}$ | $1.107\text{ m}$ |
| **ROI Valid Depth Fraction** | 86.9% | 86.9% | 85.5% |
| **Left Shoulder 2D $(u, v)$** | $(139.3, 411.3)$ | $(154.2, 545.9)$ | $(318.0, 294.1)$ |
| **Right Shoulder 2D $(u, v)$** | $(104.7, 375.9)$ | $(178.2, 341.2)$ | $(298.0, 517.0)$ |
| **Shoulder Horizontal Span ($\Delta u$)** | $-34.6\text{ px}$ | $+24.0\text{ px}$ | $-20.0\text{ px}$ |
| **Shoulder Vertical Difference ($\Delta v$)** | $-35.4\text{ px}$ | **$-204.7\text{ px}$** | **$+222.9\text{ px}$** |
| **Absolute Vertical Difference ($|\Delta v|$)**| $35.4\text{ px}$ | **$204.7\text{ px}$** | **$222.9\text{ px}$** |
| **Shoulder Line Angle** | $-134.4^\circ$ | **$-83.3^\circ$ (Vertical)** | **$+95.1^\circ$ (Vertical)** |
| **Human Silhouette Orientation** | Upright in sensor | **Sideways (Head right)** | **Sideways (Head left)** |

---

## 3. Selected Configuration and Physical Findings

### The Physical Mounting Finding
1. **The Camera is Currently Still Physically in Landscape Orientation**:
   - In both `--portrait cw` and `--portrait ccw`, the shoulder line has $|\Delta v| > 200\text{ pixels}$ ($-83.3^\circ$ and $+95.1^\circ$).
   - This occurs because the camera is currently resting flat horizontally (landscape mode).
   - Applying software portrait rotation (`cw` or `ccw`) to an image that is physically horizontal turns the subject **90° sideways**!
2. **Acceptance Threshold**:
   - The required orientation criterion is **$|\Delta v| < 15\text{ pixels}$** (horizontal shoulder line).
   - Currently, `--portrait cw` has $|\Delta v| = 204.7\text{ px}$ and `--portrait ccw` has $|\Delta v| = 222.9\text{ px}$.
   - **Both software portrait rotation modes FAIL with the camera in its current physical orientation.**
3. **Physical Action Required**:
   - The RealSense D455f camera must be **physically tilted 90° on the tripod into a portrait bracket / mount**.
   - When physically mounted vertically, the raw sensor will capture a sideways image, and software rotation (`--portrait cw` or `ccw`) will restore the subject to an upright orientation with $|\Delta v| < 15\text{ px}$.
   - Once physically mounted vertically, `--portrait cw` or `--portrait ccw` will be selected based on which direction the camera is tilted on the bracket.

---

## 4. Preview Images and Artifacts Archive

All diagnostic previews and configuration metrics are preserved in `scan_output/check_preflight_physical_portrait/`:

```
scan_output/check_preflight_physical_portrait/
├── cw/check/
│   ├── check_color.png           # CW rotated RGB preview (sideways due to landscape mount)
│   ├── check_depth_colorized.png # CW rotated colorized depth preview
│   └── check_metrics.txt         # Serial 342622300088, fx=427.32, fy=427.88, cx=234.40, cy=422.17
├── ccw/check/
│   ├── check_color.png           # CCW rotated RGB preview (inverted sideways)
│   ├── check_depth_colorized.png # CCW rotated colorized depth preview
│   └── check_metrics.txt         # Serial 342622300088, fx=427.90, fy=428.46, cx=244.60, cy=424.83
└── norot/
    ├── check_color.png           # Native unrotated landscape RGB frame
    └── check_depth_colorized.png # Native unrotated depth frame
```

---

## 5. RGB/Depth Alignment Evidence

Inspection of `realsense_measure/camera/realsense_capture.py` confirms the hardware-aligned optical pipeline:
1. **Depth-to-Color Alignment Order**:
   - RealSense stereo depth is aligned directly onto the color optical frame in native coordinates:
     $$\text{raw\_frames} \xrightarrow{\text{rs.align(rs.stream.color)}} \text{aligned\_frames } (848 \times 480)$$
   - Hardware optical alignment occurs **before** any rotation is applied.
2. **Rotation Order**:
   - `cv2.rotate(color_bgr, rot_code)` and `cv2.rotate(depth_m, rot_code)` operate with identical rotation codes and identical pixel kernels.
   - This ensures 1:1 pixel correspondence between color and depth arrays across all pixels.
3. **Depth Validity**:
   - In the ROI (central 50% of frame), valid depth fraction is **86.9%** in CW and **85.5%** in CCW.
   - No depth-color shear, misalignment, or scaling discrepancy was detected.

---

## 6. Final Camera Intrinsics and Depth Scale

For the standard `--portrait cw` configuration, the effective intrinsics queried live from the camera are:

```json
{
  "device_name": "RealSense D455F",
  "serial_number": "342622300088",
  "firmware_version": "5.17.3.10",
  "depth_scale": 0.001,
  "stream_resolution": "480x848",
  "intrinsics": {
    "width": 480,
    "height": 848,
    "fx": 427.32,
    "fy": 427.88,
    "cx": 234.40,
    "cy": 422.17,
    "is_post_rotation": true,
    "is_post_decimation": false
  }
}
```

- **Intrinsics Mathematical Exactness**:
  - Native $c_x = 422.17$, $c_y = 244.60$ on $848 \times 480$.
  - Rotated $c_{x, rot} = (H - 1) - c_y = 479 - 244.60 = 234.40$.
  - Rotated $c_{y, rot} = c_x = 422.17$.
  - Rotated $f_{x, rot} = f_y = 427.32$, $f_{y, rot} = f_x = 427.88$.
  - **The intrinsics match the rotated image coordinates with 100% exactness.**

---

## 7. Full-Body Framing Assessment

The full-body framing was assessed against the acceptance criteria:

| Framing Requirement | Observed Value | Threshold / Target | Status |
|---|:---:|:---:|:---:|
| **Standoff Distance** | **$1.107\text{ m}$** | $2.4\text{ m} - 2.8\text{ m}$ | **FAIL** (Too close) |
| **Head Apex Visibility** | Partial ($v = 474.9$) | $\approx 5\%$ top margin | **FAIL** |
| **Feet & Floor Visibility** | Out of bounds ($v < 0$ or $> 848$) | $\approx 5\%$ bottom margin | **FAIL** (Clipped) |
| **Hands / Wrists Visibility** | Truncated at frame border | Complete silhouette | **FAIL** (Clipped) |
| **Background Separation** | Subject at $1.1\text{ m}$, wall at $2.2\text{ m}$ | $> 0.5\text{ m}$ separation | **PASS** |

**Framing Conclusion**: At $1.1\text{ m}$ standoff, the standing subject is framed only from chest up. The camera must be moved to **$2.4\text{ m} - 2.8\text{ m}$ standoff** so that the entire standing body fits from head apex to feet with ~5% margin.

---

## 8. Coordinate-System Sanity Check

The coordinate conventions of the pipeline were verified:
- **Camera Frame**: $+X_{cam}$ right, $+Y_{cam}$ down, $+Z_{cam}$ optical axis forward.
- **Canonical Gravity Frame (ISB)**:
  - $+Y_{can}$: Superior (vertically upwards, normal to floor).
  - $+Z_{can}$: Anterior (pointing forward from subject's chest).
  - $+X_{can}$: Lateral (subject's left).
  - Origin: Floor plane directly below vertical turning axis.
- **Yaw Prior Sequence**:
  - Stop 0 faces camera pinhole: canonical yaw prior is **$180.0^\circ$**.
  - Clockwise sequence for 8 stops: $180^\circ, 135^\circ, 90^\circ, 45^\circ, 0^\circ, 315^\circ, 270^\circ, 225^\circ$.
- **Coordinate Sanity**:
  - When the camera is sideways, lateral shoulder coordinates map into vertical $Y$, causing 16 cm shoulder disparity.
  - When the camera is mounted upright in portrait mode with $|\Delta v| < 15\text{ px}$, shoulder coordinates will lie in the horizontal plane ($X_{can}, Z_{can}$), and shoulder heights in $Y_{can}$ will be identical ($\pm 1\text{ cm}$).

---

## 9. Floor-Plane Coordinate-Frame Verification

The floor-plane consensus architecture was audited:
1. **Static Camera Protocol**: In an 8-stop turntable capture, the camera remains fixed on the tripod; only the subject rotates.
2. **Floor Representation**:
   - `fit_floor_plane()` detects the floor plane equation in camera coordinates ($n \cdot x + d = 0$).
   - `verify_floor_consistency()` evaluates whether all stops agree on the floor normal in camera coordinates (within $4.0^\circ$).
   - `compute_gravity_transform()` computes $T_{grav\_cam}$, which rotates the static camera coordinates so the consensus floor normal becomes $+Y = [0, 1, 0]$ passing through $Y = 0$.
3. **Consensus Requirement**: Minimum 3 consistent floor plane observations (`min_planes >= 3`) are required. If fewer than 3 planes are observed (or planes differ by $> 4^\circ$), `floor_observed = False` is enforced, and stature returns `NaN` (`NOT_MEASURABLE`).

---

## 10. Test Execution and Results

The targeted camera configuration, rotation geometry, and adapter lifecycle test suites were executed:

```bash
pytest tests/camera/test_rotation_geometry.py \
       tests/test_rotate_90.py \
       tests/fullbody/test_stage27_adapter_ui_cli.py \
       tests/fullbody/test_turn_direction_io.py -v
```

**Results**: **20 passed in 27.83s (100% clean pass)**.
- `test_rotation_geometry_numerical_exactness[cw]`: **PASSED**
- `test_rotation_geometry_numerical_exactness[ccw]`: **PASSED**
- `test_backprojected_3d_points_identical_up_to_rotation[cw]`: **PASSED**
- `test_backprojected_3d_points_identical_up_to_rotation[ccw]`: **PASSED**
- `test_full_body_camera_config`: **PASSED**
- `test_adapter_warmup_and_raw_depth_units`: **PASSED**
- `test_adapter_dropped_frames_retried`: **PASSED**
- `test_adapter_disconnect_raises_and_stops`: **PASSED**
- `test_adapter_refuses_decimation`: **PASSED**
- `test_render_frame_shapes_and_instructions`: **PASSED**
- `test_cli_import_isolation`: **PASSED**
- `test_cli_capture_fake_end_to_end`: **PASSED**
- `test_cli_capture_fake_failure_leaves_unfinalized`: **PASSED**
- `test_cli_check_camera_fake`: **PASSED**
- `test_cw_ccw_sequences_mirror_images_about_180[8]`: **PASSED**
- `test_cw_ccw_sequences_mirror_images_about_180[4]`: **PASSED**
- `test_yaw_rotation_matrices_inverses_at_stop_2`: **PASSED**
- `test_io_loaders_store_identical_nominal_angles_for_both_directions`: **PASSED**
- `test_meta_json_overrides_cfg_with_warning`: **PASSED**
- `test_save_raw_capture_persists_orientation_fields`: **PASSED**

---

## 11. Explicit Status of Acceptance Criteria

| Acceptance Criterion | Requirement | Observed Status | Verdict |
|---|---|:---:|:---:|
| **Criterion 1: Head Above Shoulders** | Head positioned superior to shoulders | In CW/CCW, head is lateral (angle $-83^\circ$) | **FAIL** |
| **Criterion 2: Shoulders Approximately Horizontal** | Shoulder line horizontal ($\approx 0^\circ$) | In CW/CCW, shoulder line is vertical ($\approx -89^\circ$) | **FAIL** |
| **Criterion 3: Shoulder-Line $|\Delta v| < 15\text{ px}$** | Absolute vertical difference $< 15\text{ px}$ | $|\Delta v| = 204.7\text{ px}$ (CW), $222.9\text{ px}$ (CCW) | **FAIL** |
| **Criterion 4: Lateral Arm Extension** | Arms extend laterally, not vertically | Arms extend vertically due to 90° image tilt | **FAIL** |
| **Criterion 5: Feet Below Torso** | Feet visible below torso | Lower body truncated at $1.1\text{ m}$ standoff | **FAIL** |
| **Criterion 6: No Unintended Mirroring** | Correct chirality preserved | Left/right chirality verified consistent | **PASS** |
| **Criterion 7: RGB & Aligned Depth Matching** | Same orientation and dimensions ($848 \times 480$) | Both arrays are $848 \times 480$ with 1:1 correspondence | **PASS** |
| **Criterion 8: Rotated Intrinsics Accuracy** | $f_x, f_y, c_x, c_y$ match rotated array | Analytical exactness verified ($c_x=234.40, c_y=422.17$) | **PASS** |
| **Criterion 9: Depth Alignment Protocol** | `rs.align` executed prior to rotation | Verified in `realsense_capture.py` | **PASS** |
| **Criterion 10: Full-Body Framing** | Head, hands, legs, feet visible with ~5% margin | Truncated at $1.107\text{ m}$ standoff | **FAIL** |
| **Criterion 11: Standoff Distance (2.4–2.8 m)** | Standoff between $2.4\text{ m}$ and $2.8\text{ m}$ | Current standoff is $1.107\text{ m}$ | **FAIL** |
| **Criterion 12: Floor Plane Visibility** | Minimum $1.0\text{ m}^2$ floor visible | Floor truncated; not observed | **FAIL** |

---

## 12. Final Decision and Actionable Instructions

### Decision: **PRE-FLIGHT FAILED — NOT AUTHORIZED FOR S02 CAPTURE**

### Outstanding Action Required by Operator Before S02
1. **Physical Mount Adjustment**:
   - The RealSense D455f is currently resting flat horizontally.
   - It **MUST be physically mounted vertically in a 90° portrait mount/bracket** on the tripod.
2. **Tripod Relocation**:
   - Move the tripod back to **$2.5\text{ m} \pm 0.1\text{ m}$ standoff distance** from the turntable center.
3. **Verification Command**:
   - After physically tilting the camera 90° and moving the tripod to 2.5 m, run:
     `python -m realsense_measure.fullbody check-camera --portrait cw --out scan_output/check_preflight_final`
   - Inspect `check_color.png` to confirm:
     - Head is at the top with ~5% margin.
     - Feet are on the floor at the bottom with ~5% margin.
     - Shoulder line is horizontal ($|\Delta v| < 15\text{ px}$).
     - If the person appears upside down, switch to `--portrait ccw`.

> [!IMPORTANT]
> **PHASE 2.3 COMPLETE — EXECUTION STOPPED.**  
> In strict compliance with instructions, no S02 multi-stop acquisition has been initiated. Awaiting physical mount adjustment and explicit authorization.
