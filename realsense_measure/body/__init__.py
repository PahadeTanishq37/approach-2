"""
realsense_measure/body/__init__.py

Core body capture and session support sub-package.
"""

from .capture_session import (
    validate_anonymous_subject_id,
    CaptureQualityChecker,
    CaptureSessionManifest,
)
from .capture_motion import (
    evaluate_burst_stillness,
    BurstMotionResult,
)
from .hardware_smoke_test import (
    CollisionError,
    check_capture_collision,
)

__all__ = [
    "validate_anonymous_subject_id",
    "CaptureQualityChecker",
    "CaptureSessionManifest",
    "evaluate_burst_stillness",
    "BurstMotionResult",
    "CollisionError",
    "check_capture_collision",
]
