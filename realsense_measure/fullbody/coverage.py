"""
realsense_measure/fullbody/coverage.py

Geometric coverage and standoff calculation for fullbody capture.
Calculates vertical/horizontal coverage per metre of standoff, recommended standoff
for a subject of height H, and head/feet frame guides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class CoverageMetrics:
    """
    Geometric FOV coverage and standoff recommendations.
    """
    vertical_rate_m_per_m: float          # metres of vertical coverage per metre of standoff
    horizontal_rate_m_per_m: float        # metres of horizontal coverage per metre of standoff
    recommended_standoff_m: float         # standoff fitting subject height H with margin (e.g. 5%)
    standoff_min_m: float                 # minimum standoff fitting height H
    standoff_max_m: float                 # maximum recommended standoff
    head_guide_row_px: int                # pixel row (v) for top margin guide
    feet_guide_row_px: int                # pixel row (v) for bottom margin guide
    subject_height_m: float
    arm_span_m: float
    margin_ratio: float


def compute_coverage_metrics(
    intrinsics: Any,
    subject_height_m: float = 1.80,
    arm_span_m: float = 1.80,
    margin_ratio: float = 0.05,
    max_standoff_m: float = 2.40,
) -> CoverageMetrics:
    """
    Compute vertical/horizontal coverage in metres per metre of standoff,
    and the standoff range that fits a subject of height H with top and bottom margin.

    Parameters
    ----------
    intrinsics:
        Camera pinhole intrinsics (dict, Open3D PinholeCameraIntrinsic, or object with fx, fy, width, height).
    subject_height_m:
        Target subject height in metres (default 1.80 m).
    arm_span_m:
        Target subject arm span in metres (default 1.80 m).
    margin_ratio:
        Top and bottom margin ratio (default 0.05 = 5%).
    max_standoff_m:
        Upper bound on recommended capture standoff (default 2.40 m).

    Returns
    -------
    CoverageMetrics
    """
    if hasattr(intrinsics, "get_focal_length"):
        fx, fy = intrinsics.get_focal_length()
        w = float(intrinsics.width)
        h = float(intrinsics.height)
    elif isinstance(intrinsics, dict):
        fx = float(intrinsics.get("fx", 428.0))
        fy = float(intrinsics.get("fy", 428.0))
        w = float(intrinsics.get("width", 848))
        h = float(intrinsics.get("height", 480))
    else:
        fx = float(getattr(intrinsics, "fx", 428.0))
        fy = float(getattr(intrinsics, "fy", 428.0))
        w = float(getattr(intrinsics, "width", 848))
        h = float(getattr(intrinsics, "height", 480))

    vert_rate = float(h / fy) if fy > 0 else 0.0
    horiz_rate = float(w / fx) if fx > 0 else 0.0

    usable_fraction = max(0.10, 1.0 - 2.0 * margin_ratio)
    v_req = subject_height_m / usable_fraction
    z_min_height = v_req / vert_rate if vert_rate > 0 else 2.0

    # Arm span check
    h_req = arm_span_m / usable_fraction
    z_min_arm = h_req / horiz_rate if horiz_rate > 0 else 1.0

    standoff_min = max(z_min_height, z_min_arm)
    standoff_rec = standoff_min
    standoff_max = max(standoff_rec, max_standoff_m)

    head_row = int(round(margin_ratio * h))
    feet_row = int(round((1.0 - margin_ratio) * h))

    return CoverageMetrics(
        vertical_rate_m_per_m=vert_rate,
        horizontal_rate_m_per_m=horiz_rate,
        recommended_standoff_m=standoff_rec,
        standoff_min_m=standoff_min,
        standoff_max_m=standoff_max,
        head_guide_row_px=head_row,
        feet_guide_row_px=feet_row,
        subject_height_m=subject_height_m,
        arm_span_m=arm_span_m,
        margin_ratio=margin_ratio,
    )

