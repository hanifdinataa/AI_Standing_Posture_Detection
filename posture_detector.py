"""
╔══════════════════════════════════════════════════════════════════╗
║              AI Posture Detection — Posture Detector            ║
║  Classifies sitting posture using MediaPipe Pose landmarks.     ║
╚══════════════════════════════════════════════════════════════════╝

Posture classification relies on three signals:
    1. Neck angle  — angle at the shoulder (ear → shoulder → hip)
    2. Shoulder tilt — vertical asymmetry between left/right shoulder
    3. Forward head — horizontal offset of nose relative to shoulder midpoint

The detector applies a moving-average filter to smooth frame-to-frame jitter
before classifying into GOOD / WARNING / BAD.
"""

from typing import Optional, Tuple, Dict

import config as cfg
from utils import calculate_angle, midpoint, MovingAverageFilter, logger


class PostureDetector:
    """Analyse upper-body landmarks to classify sitting posture."""

    def __init__(self):
        self._neck_angle_filter = MovingAverageFilter(cfg.SMOOTHING_WINDOW_SIZE)
        self._tilt_filter = MovingAverageFilter(cfg.SMOOTHING_WINDOW_SIZE)
        self._fwd_filter = MovingAverageFilter(cfg.SMOOTHING_WINDOW_SIZE)

        self.status: str = cfg.POSTURE_GOOD
        self.neck_angle: float = 180.0
        self.shoulder_tilt: float = 0.0
        self.forward_offset: float = 0.0
        self.confidence: float = 0.0

    # ── public API ──────────────────────────────────────────────
    def update(self, landmarks: list, frame_w: int, frame_h: int) -> str:
        """
        Analyse pose landmarks and return posture status.

        Parameters
        ----------
        landmarks : list of mediapipe NormalizedLandmark
        frame_w, frame_h : frame dimensions for de-normalisation

        Returns
        -------
        Status string: GOOD / WARNING / BAD
        """
        points = self._extract_points(landmarks)
        if points is None:
            return self.status  # keep last known status

        ear, shoulder, hip, nose, l_shoulder, r_shoulder = points

        # 1. Neck angle (ear → shoulder → hip)
        raw_neck = calculate_angle(ear, shoulder, hip)
        self.neck_angle = self._neck_angle_filter.update(raw_neck)

        # 2. Shoulder tilt (normalised y-difference)
        raw_tilt = abs(l_shoulder[1] - r_shoulder[1]) / frame_h
        self.shoulder_tilt = self._tilt_filter.update(raw_tilt)

        # 3. Forward head posture (nose x vs. shoulder midpoint x)
        mid_sh = midpoint(l_shoulder, r_shoulder)
        raw_fwd = (nose[1] - mid_sh[1]) / frame_h  # y-axis: lower = more forward
        self.forward_offset = self._fwd_filter.update(raw_fwd)

        # Classification ─────────────────────────────
        bad_signals = 0

        if self.neck_angle < cfg.NECK_ANGLE_WARNING:
            bad_signals += 2  # strong signal
        elif self.neck_angle < cfg.NECK_ANGLE_GOOD:
            bad_signals += 1

        if self.shoulder_tilt > cfg.SHOULDER_TILT_THRESHOLD:
            bad_signals += 1

        if self.forward_offset > cfg.FORWARD_HEAD_THRESHOLD:
            bad_signals += 1

        # Decide status
        if bad_signals >= 3:
            self.status = cfg.POSTURE_BAD
            self.confidence = min(1.0, bad_signals / 4.0)
            logger.warning("Bad posture detected  (neck=%.1f°  tilt=%.3f  fwd=%.3f)",
                           self.neck_angle, self.shoulder_tilt, self.forward_offset)
        elif bad_signals >= 1:
            self.status = cfg.POSTURE_WARNING
            self.confidence = 0.5
        else:
            self.status = cfg.POSTURE_GOOD
            self.confidence = 1.0
            logger.info("Good posture detected  (neck=%.1f°)", self.neck_angle)

        return self.status

    def get_info(self) -> Dict:
        """Return a snapshot of current posture metrics."""
        return {
            "status": self.status,
            "neck_angle": self.neck_angle,
            "shoulder_tilt": self.shoulder_tilt,
            "forward_offset": self.forward_offset,
            "confidence": self.confidence,
        }

    def reset(self):
        """Reset smoothing filters and status."""
        self._neck_angle_filter.reset()
        self._tilt_filter.reset()
        self._fwd_filter.reset()
        self.status = cfg.POSTURE_GOOD
        self.neck_angle = 180.0

    # ── internals ───────────────────────────────────────────────
    @staticmethod
    def _extract_points(landmarks) -> Optional[Tuple]:
        """
        Pull relevant pixel-coordinates from MediaPipe landmarks.
        Uses the more visible side (left or right) for ear/shoulder/hip.

        Returns (ear, shoulder, hip, nose, l_shoulder, r_shoulder) or None.
        """
        try:
            nose = (landmarks[0].x, landmarks[0].y)

            l_ear = landmarks[7]   # left ear
            r_ear = landmarks[8]   # right ear
            l_sh = landmarks[11]   # left shoulder
            r_sh = landmarks[12]   # right shoulder
            l_hip = landmarks[23]  # left hip
            r_hip = landmarks[24]  # right hip

            # Choose the side with higher visibility
            if l_ear.visibility >= r_ear.visibility:
                ear = (l_ear.x, l_ear.y)
                shoulder = (l_sh.x, l_sh.y)
                hip = (l_hip.x, l_hip.y)
            else:
                ear = (r_ear.x, r_ear.y)
                shoulder = (r_sh.x, r_sh.y)
                hip = (r_hip.x, r_hip.y)

            l_shoulder_pt = (l_sh.x, l_sh.y)
            r_shoulder_pt = (r_sh.x, r_sh.y)

            # Visibility gate
            min_vis = min(l_sh.visibility, r_sh.visibility)
            if min_vis < 0.3:
                return None

            return ear, shoulder, hip, nose, l_shoulder_pt, r_shoulder_pt

        except (IndexError, AttributeError):
            return None
