"""
realsense_measure/fullbody/session_format.py

Extended session format ("stops_v2") writer and verification utilities.
Strictly decoupled from Open3D, MediaPipe, and pyrealsense2 so it runs
in any environment with only standard scientific libraries (numpy, cv2).

Design rules:
- Raw burst files are write-once. Never overwrite or delete a raw file.
- Never delete a rejected attempt; keep complete auditable attempt history.
- Atomic finalization via temporary file and atomic replace.
- Pre-save and load-time SHA-256 integrity verification.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from realsense_measure.fullbody.orientation import nominal_angle_deg

logger = logging.getLogger(__name__)

# Validate subject ID using existing validator if importable, otherwise duplicate rule
try:
    from realsense_measure.body.capture_session import validate_anonymous_subject_id
except Exception:
    import re

    def validate_anonymous_subject_id(subject_id: Optional[str]) -> str:
        """Validate anonymous subject ID without leaking PII."""
        if not subject_id or not isinstance(subject_id, str):
            raise ValueError("subject_id must be a non-empty string.")
        if "@" in subject_id:
            raise ValueError(f"subject_id '{subject_id}' contains '@' (email rejected for PII protection).")
        if " " in subject_id:
            raise ValueError(f"subject_id '{subject_id}' contains spaces (names rejected for PII protection).")
        pattern = r"^(SUBJ|ANON|PARTICIPANT|SUBJECT|P|TEST|S|BUCKET|OBJ|OBJECT|CYLINDER)(_?[A-Za-z0-9]+)$"
        if not re.match(pattern, subject_id, re.IGNORECASE):
            raise ValueError(
                f"subject_id '{subject_id}' does not match authorized anonymous format (e.g. 'SUBJ_01', 'S01', 'ANON_42')."
            )
        return subject_id


def warn_if_git_ignored(path: Union[str, Path]) -> bool:
    """Check if the given path is git-ignored and log a warning if so."""
    p = Path(path)
    try:
        import subprocess
        res = subprocess.run(
            ["git", "check-ignore", "-q", str(p)],
            capture_output=True,
        )
        if res.returncode == 0:
            logger.warning(
                f"Session destination '{p}' is git-ignored. "
                "Full-body session captures should be tracked and persistent under raw_captures/<subject>/."
            )
            return True
    except Exception:
        pass
    return False


class AttemptHandle:
    """Handle to an open stop attempt directory."""

    def __init__(self, stop_index: int, attempt_index: int, path: Path):
        self.stop_index = stop_index
        self.attempt_index = attempt_index
        self.path = path
        self.is_finished = False


class SessionWriter:
    """
    Writer for the 'stops_v2' extended session format.
    
    Structure:
      root/
        meta.json
        stop_00/
          stop.json
          attempt_00/
            raw_color_000.png
            raw_depth_000.npy
            timestamps.json
            quality.json
            landmarks.json (optional)
            filtered_depth.npy (optional)
            files.sha256
          attempt_01/ (if retried)
        stop_01/
        ...
    """

    def __init__(
        self,
        root: Union[str, Path],
        subject_id: str,
        n_stops: int,
        turn_direction: str,
        camera_portrait_rotation: str,
        intrinsics: Dict[str, Any],
        depth_scale: float,
        camera_config_snapshot: Optional[Dict[str, Any]] = None,
        device_info: Optional[Dict[str, Any]] = None,
        software_version: str = "1.0.0",
        depth_units: Optional[str] = None,
        turntable: bool = False,
        background_dir: Optional[Union[str, Path]] = None,
        cam_height_m: Optional[float] = None,
        cam_distance_m: Optional[float] = None,
        cam_tilt_deg: float = 0.0,
        clothing: Optional[str] = None,
        pose: str = "A-pose",
        notes: Optional[str] = None,
        subject_type: str = "person",
        object_diameter_mm: Optional[float] = None,
        protocol_version: str = "stop-and-go-v1",
        frames_requested: int = 60,
    ):
        self.root = Path(root)
        warn_if_git_ignored(self.root)
        self.subject_id = validate_anonymous_subject_id(subject_id)
        self.n_stops = int(n_stops)
        self.turntable = bool(turntable)
        self.turn_direction = str(turn_direction).lower()
        if self.turn_direction not in ("cw", "ccw"):
            raise ValueError(f"Invalid turn_direction '{turn_direction}'. Must be 'cw' or 'ccw'.")
        self.camera_portrait_rotation = str(camera_portrait_rotation).lower()
        if self.camera_portrait_rotation not in ("cw", "ccw", "none"):
            raise ValueError(f"Invalid camera_portrait_rotation '{camera_portrait_rotation}'. Must be 'cw', 'ccw', or 'none'.")
        if depth_units is not None and depth_units not in ("sensor_units", "m"):
            raise ValueError(f"Invalid depth_units '{depth_units}'. Must be 'sensor_units' or 'm'.")
        self.depth_units = depth_units

        # Rig and protocol metadata
        self.cam_height_m = float(cam_height_m) if cam_height_m is not None else None
        self.cam_distance_m = float(cam_distance_m) if cam_distance_m is not None else None
        self.cam_tilt_deg = float(cam_tilt_deg)
        self.clothing = str(clothing) if clothing is not None else None
        self.pose = str(pose)
        self.notes = str(notes) if notes is not None else None
        self.subject_type = str(subject_type).lower()
        self.object_diameter_mm = float(object_diameter_mm) if object_diameter_mm is not None else None
        self.protocol_version = str(protocol_version)
        self.frames_requested = int(frames_requested)

        # Validate intrinsics dictionary
        req_intr = ("width", "height", "fx", "fy", "cx", "cy")
        for k in req_intr:
            if k not in intrinsics:
                raise ValueError(f"Missing required intrinsic key '{k}'")

        self.intrinsics = {
            "width": int(intrinsics["width"]),
            "height": int(intrinsics["height"]),
            "fx": float(intrinsics["fx"]),
            "fy": float(intrinsics["fy"]),
            "cx": float(intrinsics["cx"]),
            "cy": float(intrinsics["cy"]),
            "is_post_rotation": bool(intrinsics.get("is_post_rotation", True)),
            "is_post_decimation": bool(intrinsics.get("is_post_decimation", False)),
        }
        self.depth_scale = float(depth_scale)
        self.camera_config_snapshot = dict(camera_config_snapshot or {})
        self.device_info = dict(device_info or {})
        self.software_version = software_version
        self.created_utc = datetime.now(timezone.utc).isoformat()

        self.root.mkdir(parents=True, exist_ok=True)

        # Copy background session into session directory if provided
        self.background_dir = Path(background_dir) if background_dir is not None else None
        self.background_copied = False
        if self.background_dir is not None:
            if self.background_dir.exists() and self.background_dir.is_dir():
                target_bg = self.root / "background"
                target_bg.mkdir(parents=True, exist_ok=True)
                import shutil
                for item in self.background_dir.iterdir():
                    if item.is_file():
                        shutil.copy2(item, target_bg / item.name)
                # Ensure files.sha256 manifest exists for background
                sha_file = target_bg / "files.sha256"
                if not sha_file.exists():
                    sha_map: Dict[str, str] = {}
                    for f in sorted(target_bg.iterdir()):
                        if f.is_file() and f.name != "files.sha256":
                            sha_map[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
                    with open(sha_file, "w", encoding="utf-8") as sf:
                        for fname, h in sorted(sha_map.items()):
                            sf.write(f"{h}  {fname}\n")
                self.background_copied = True
            else:
                logger.warning(
                    f"WARNING: Background directory '{self.background_dir}' is missing. "
                    "Session will not contain local background calibration copy."
                )

        self._finalized = False
        self._init_stops()

    def _init_stops(self) -> None:
        """Initialize stop directories and initial stop.json if not present."""
        for stop_idx in range(self.n_stops):
            stop_dir = self.root / f"stop_{stop_idx:02d}"
            stop_dir.mkdir(parents=True, exist_ok=True)
            stop_json_path = stop_dir / "stop.json"
            if not stop_json_path.exists():
                stop_meta = {
                    "stop_index": stop_idx,
                    "nominal_angle_deg": nominal_angle_deg(stop_idx, self.n_stops),
                    "accepted_attempt": None,
                    "attempts": [],
                }
                with open(stop_json_path, "w", encoding="utf-8") as f:
                    json.dump(stop_meta, f, indent=2)

    def begin_attempt(self, stop_index: int) -> AttemptHandle:
        """
        Open a new attempt directory for a stop.
        Does NOT delete or overwrite previous attempts.
        """
        if self._finalized:
            raise RuntimeError("Cannot begin attempt on finalized session.")
        if stop_index < 0 or stop_index >= self.n_stops:
            raise IndexError(f"Stop index {stop_index} out of range [0, {self.n_stops - 1}]")

        stop_dir = self.root / f"stop_{stop_index:02d}"
        existing_attempts = sorted([d for d in stop_dir.iterdir() if d.is_dir() and d.name.startswith("attempt_")])
        attempt_idx = len(existing_attempts)
        attempt_dir = stop_dir / f"attempt_{attempt_idx:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)

        return AttemptHandle(stop_index=stop_index, attempt_index=attempt_idx, path=attempt_dir)

    def write_burst(
        self,
        handle: AttemptHandle,
        colors: List[np.ndarray],
        depths: List[np.ndarray],
        timestamps: Optional[List[float]] = None,
    ) -> None:
        """
        Write raw burst color frames and depth arrays.
        Enforces write-once: fails if files already exist.
        """
        if handle.is_finished:
            raise RuntimeError("Attempt is already finished.")
        if len(colors) != len(depths):
            raise ValueError(f"Color count ({len(colors)}) does not match depth count ({len(depths)})")
        if len(depths) == 0:
            raise ValueError("Cannot write empty burst.")

        n_frames = len(depths)
        if not hasattr(self, "_recorded_depth_dtype"):
            self._recorded_depth_dtype = str(depths[0].dtype)
            if self.depth_units is None:
                if np.issubdtype(depths[0].dtype, np.floating):
                    self.depth_units = "m"
                else:
                    self.depth_units = "sensor_units"

        ts_list = list(timestamps) if timestamps is not None else [float(i) for i in range(n_frames)]
        if len(ts_list) != n_frames:
            raise ValueError(f"Timestamp count ({len(ts_list)}) does not match frame count ({n_frames})")

        for i in range(n_frames):
            color_path = handle.path / f"raw_color_{i:03d}.png"
            depth_path = handle.path / f"raw_depth_{i:03d}.npy"

            if color_path.exists():
                raise FileExistsError(f"Raw file already exists: {color_path}")
            if depth_path.exists():
                raise FileExistsError(f"Raw file already exists: {depth_path}")

            # Lossless write
            cv2.imwrite(str(color_path), colors[i])
            # Raw depth in native sensor units / camera output
            np.save(str(depth_path), depths[i])

        ts_path = handle.path / "timestamps.json"
        if ts_path.exists():
            raise FileExistsError(f"Timestamps file already exists: {ts_path}")
        with open(ts_path, "w", encoding="utf-8") as f:
            json.dump({"frame_count": n_frames, "timestamps": ts_list}, f, indent=2)

    def write_quality(self, handle: AttemptHandle, report: Dict[str, Any]) -> None:
        """Write quality assessment report for attempt."""
        if handle.is_finished:
            raise RuntimeError("Attempt is already finished.")
        p = handle.path / "quality.json"
        if p.exists():
            raise FileExistsError(f"Quality file already exists: {p}")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    def write_landmarks(self, handle: AttemptHandle, landmarks: Dict[str, Any]) -> None:
        """Write optional landmark coordinates for attempt."""
        if handle.is_finished:
            raise RuntimeError("Attempt is already finished.")
        p = handle.path / "landmarks.json"
        if p.exists():
            raise FileExistsError(f"Landmarks file already exists: {p}")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(landmarks, f, indent=2)

    def write_filtered_depth(self, handle: AttemptHandle, filtered_depth: np.ndarray) -> None:
        """
        Write optional pre-filtered depth array for attempt.
        Refuses if the file already exists.
        """
        if handle.is_finished:
            raise RuntimeError("Attempt is already finished.")
        p = handle.path / "filtered_depth.npy"
        if p.exists():
            raise FileExistsError(f"Filtered depth already exists: {p}")
        np.save(str(p), filtered_depth)

    def finish_attempt(
        self,
        handle: AttemptHandle,
        accepted: bool,
        reason: str = "",
        sway_mm: Optional[float] = None,
    ) -> None:
        """
        Mark an attempt as finished, compute SHA-256 for all files in attempt,
        and update stop.json with attempt status, failure reason, and sway_mm.
        """
        if handle.is_finished:
            raise RuntimeError("Attempt is already finished.")

        # Compute SHA-256 for all attempt files
        sha_map: Dict[str, str] = {}
        for f in sorted(handle.path.iterdir()):
            if f.is_file() and f.name != "files.sha256":
                sha = hashlib.sha256(f.read_bytes()).hexdigest()
                sha_map[f.name] = sha

        sha_path = handle.path / "files.sha256"
        with open(sha_path, "w", encoding="utf-8") as f:
            for fname, h in sorted(sha_map.items()):
                f.write(f"{h}  {fname}\n")

        # Update stop.json
        stop_json_path = handle.path.parent / "stop.json"
        with open(stop_json_path, "r", encoding="utf-8") as f:
            stop_meta = json.load(f)

        attempt_entry = {
            "attempt": handle.attempt_index,
            "status": "ACCEPTED" if accepted else "REJECTED",
            "reason": reason,
            "sway_mm": round(float(sway_mm), 2) if sway_mm is not None else None,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        stop_meta["attempts"].append(attempt_entry)
        if accepted:
            stop_meta["accepted_attempt"] = handle.attempt_index
            stop_meta["accepted_sway_mm"] = round(float(sway_mm), 2) if sway_mm is not None else None

        with open(stop_json_path, "w", encoding="utf-8") as f:
            json.dump(stop_meta, f, indent=2)

        handle.is_finished = True

    def finalize(self) -> Path:
        """
        Finalize session by writing root meta.json atomically.
        Fails if any stop lacks an accepted attempt.
        """
        if self._finalized:
            return self.root / "meta.json"

        # Check all stops have an accepted attempt
        for stop_idx in range(self.n_stops):
            stop_json_path = self.root / f"stop_{stop_idx:02d}" / "stop.json"
            if not stop_json_path.exists():
                raise RuntimeError(f"Missing stop metadata at stop {stop_idx}")
            with open(stop_json_path, "r", encoding="utf-8") as f:
                s_meta = json.load(f)
            if s_meta.get("accepted_attempt") is None:
                raise RuntimeError(f"Stop {stop_idx} has no accepted attempt; cannot finalize session.")

        depth_storage_dtype = getattr(self, "_recorded_depth_dtype", "uint16" if self.depth_units == "sensor_units" else "float32")
        resolved_units = self.depth_units or "sensor_units"
        storage_scale = self.depth_scale if resolved_units == "sensor_units" else 1.0

        # Build nominal stop angles in turn direction
        nom_angles = [
            nominal_angle_deg(i, self.n_stops) for i in range(self.n_stops)
        ]

        # Background metadata record
        bg_meta_record: Optional[Dict[str, Any]] = None
        bg_target = self.root / "background"
        if bg_target.exists() and (bg_target / "background_depth.npy").exists():
            bg_meta_record = {
                "present": True,
                "relative_path": "background",
            }
            sha_f = bg_target / "files.sha256"
            if sha_f.exists():
                bg_meta_record["manifest_sha256"] = hashlib.sha256(sha_f.read_bytes()).hexdigest()
        elif self.background_dir is not None:
            bg_meta_record = {
                "present": False,
                "source_path": str(self.background_dir),
            }

        meta_doc = {
            "schema_version": 2,
            "layout": "stops_v2",
            "protocol_version": self.protocol_version,
            "subject_id": self.subject_id,
            "subject_type": self.subject_type,
            "object_diameter_mm": self.object_diameter_mm,
            "n_stops": self.n_stops,
            "stop_count": self.n_stops,
            "turn_direction": self.turn_direction,
            "turntable": getattr(self, "turntable", False),
            "camera_portrait_rotation": self.camera_portrait_rotation,
            "camera_rotation": self.camera_portrait_rotation,
            "depth_scale": self.depth_scale,
            "depth_storage": {
                "dtype": depth_storage_dtype,
                "units": resolved_units,
                "depth_scale": float(storage_scale),
            },
            "intrinsics": self.intrinsics,
            "camera_config": self.camera_config_snapshot,
            "device_info": self.device_info,
            "rig_metadata": {
                "cam_height_m": self.cam_height_m,
                "cam_distance_m": self.cam_distance_m,
                "cam_tilt_deg": self.cam_tilt_deg,
                "clothing": self.clothing,
                "pose": self.pose,
                "notes": self.notes,
            },
            "nominal_stop_angles": nom_angles,
            "frames_requested": self.frames_requested,
            "background": bg_meta_record,
            "created_utc": self.created_utc,
            "software_version": self.software_version,
            "finalized_utc": datetime.now(timezone.utc).isoformat(),
        }

        meta_path = self.root / "meta.json"
        tmp_meta_path = self.root / "meta.json.tmp"

        with open(tmp_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_doc, f, indent=2)

        # Atomic replace
        os.replace(tmp_meta_path, meta_path)
        self._finalized = True
        logger.info(f"Finalized session {self.root} with {self.n_stops} accepted stops.")
        return meta_path

