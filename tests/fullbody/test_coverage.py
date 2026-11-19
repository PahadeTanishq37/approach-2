"""
tests/fullbody/test_coverage.py

Unit tests for coverage helper and recommended standoff calculations:
- Coverage helper gives ~1.12 m/m vertical for fy~428 at 480 rows.
- Recommended standoff ranges for H=1.70, 1.80, 1.90 with 5% margins.
- Head/feet guide row calculation.
"""

from __future__ import annotations

import pytest

from realsense_measure.fullbody.coverage import compute_coverage_metrics


def test_coverage_helper_vertical_rate_and_standoffs():
    """Verify coverage helper gives ~1.12 m/m vertical for fy~428 at 480 rows and computes standoffs."""
    intr = {"fx": 428.11, "fy": 427.55, "width": 848, "height": 480}

    # Coverage for standard 1.80 m subject
    cov_180 = compute_coverage_metrics(intr, subject_height_m=1.80)

    # 1. Vertical coverage rate ~ 1.12 m/m
    assert cov_180.vertical_rate_m_per_m == pytest.approx(1.1227, abs=0.01)

    # 2. Horizontal coverage rate ~ 1.98 m/m
    assert cov_180.horizontal_rate_m_per_m == pytest.approx(1.9808, abs=0.01)

    # 3. Recommended standoff for H = 1.80 with 5% top and bottom margin (90% usable height)
    # V_req = 1.80 / 0.90 = 2.00 m. Standoff = 2.00 / 1.1227 = 1.78 m
    assert cov_180.recommended_standoff_m == pytest.approx(1.78, abs=0.02)

    # 4. Recommended standoff for H = 1.70
    cov_170 = compute_coverage_metrics(intr, subject_height_m=1.70)
    # V_req = 1.70 / 0.90 = 1.8889 m. Standoff = 1.8889 / 1.1227 = 1.68 m
    assert cov_170.recommended_standoff_m == pytest.approx(1.68, abs=0.02)

    # 5. Recommended standoff for H = 1.90
    cov_190 = compute_coverage_metrics(intr, subject_height_m=1.90)
    # V_req = 1.90 / 0.90 = 2.1111 m. Standoff = 2.1111 / 1.1227 = 1.88 m
    assert cov_190.recommended_standoff_m == pytest.approx(1.88, abs=0.02)

    # 6. Guide rows at 5% margins
    assert cov_180.head_guide_row_px == int(round(0.05 * 480))  # 24 px
    assert cov_180.feet_guide_row_px == int(round(0.95 * 480))  # 456 px

