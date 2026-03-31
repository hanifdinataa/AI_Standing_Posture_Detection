"""
╔══════════════════════════════════════════════════════════════════╗
║           AI Posture Detection — Distance Estimator             ║
║  Estimates user-to-screen distance from shoulder width in       ║
║  pixels using a pinhole camera model.                           ║
╚══════════════════════════════════════════════════════════════════╝

Distance estimation uses the relationship:
    distance = (real_width × focal_length) / pixel_width

- On the first valid frame the focal length is auto-calibrated
  assuming the user is sitting at ~60 cm (a comfortable default).
- Subsequent frames use this focal length to estimate real distance.
"""

from typing import Optional, Dict

import config as cfg
from utils import calculate_distance, MovingAverageFilter, logger


class DistanceEstimator:
    """Estimate face-to-screen distance using shoulder-width heuristic."""

    def __init__(self):
        self._focal_length: Optional[float] = None
        self._calibrated: bool = False
        self._distance_filter = MovingAverageFilter(cfg.SMOOTHING_WINDOW_SIZE)

        self.status: str = cfg.DISTANCE_GOOD_STR
        self.distance_cm: float = 60.0  # initial assumption
        self.shoulder_px: float = 0.0

    # ── public API ──────────────────────────────────────────────
    def update(self, landmarks: list, frame_w: int, frame_h: int) -> str:
        """
        Estimate distance and return status string.

        Parameters
        ----------
        landmarks : MediaPipe NormalizedLandmarks
        frame_w, frame_h : pixel dimensions

        Returns
        -------
        Status string: TOO CLOSE / GOOD / TOO FAR
        """
        shoulder_px = self._get_shoulder_width_px(landmarks, frame_w, frame_h)
        if shoulder_px is None or shoulder_px < 10:
            return self.status

        self.shoulder_px = shoulder_px

        # Auto-calibrate focal length once
        if not self._calibrated:
            self._calibrate(shoulder_px)

        # Pinhole model: d = (W_real × f) / W_pixel
        raw_distance = (cfg.REFERENCE_SHOULDER_WIDTH_CM * self._focal_length) / shoulder_px
        self.distance_cm = self._distance_filter.update(raw_distance)

        # Classify
        if self.distance_cm < cfg.DISTANCE_TOO_CLOSE:
            self.status = cfg.DISTANCE_TOO_CLOSE_STR
            logger.warning("Too close! Distance ≈ %.0f cm", self.distance_cm)
        elif self.distance_cm > cfg.DISTANCE_TOO_FAR:
            self.status = cfg.DISTANCE_TOO_FAR_STR
            logger.info("Too far — distance ≈ %.0f cm", self.distance_cm)
        else:
            self.status = cfg.DISTANCE_GOOD_STR
            logger.info("Good distance ≈ %.0f cm", self.distance_cm)

        return self.status

    def get_info(self) -> Dict:
        return {
            "status": self.status,
            "distance_cm": self.distance_cm,
            "shoulder_px": self.shoulder_px,
            "calibrated": self._calibrated,
        }

    def reset(self):
        """Reset filter but keep calibration."""
        self._distance_filter.reset()
        self.status = cfg.DISTANCE_GOOD_STR
        self.distance_cm = 60.0

    def recalibrate(self):
        """Force re-calibration on next frame."""
        self._calibrated = False
        self._focal_length = None
        logger.info("Distance estimator recalibration scheduled")

    # ── internals ───────────────────────────────────────────────
    def _calibrate(self, shoulder_px: float):
        """
        Calibrate focal length assuming user is at ~60 cm (comfortable default).
        focal_length = (pixel_width × assumed_distance) / real_width
        """
        assumed_distance = 60.0  # cm
        self._focal_length = (shoulder_px * assumed_distance) / cfg.REFERENCE_SHOULDER_WIDTH_CM
        self._calibrated = True
        logger.info(
            "Distance calibrated — focal_length=%.1f  shoulder_px=%.1f",
            self._focal_length, shoulder_px,
        )

    @staticmethod
    def _get_shoulder_width_px(landmarks, frame_w: int, frame_h: int) -> Optional[float]:
        """Return pixel distance between left and right shoulder."""
        try:
            l_sh = landmarks[11]
            r_sh = landmarks[12]

            if l_sh.visibility < 0.4 or r_sh.visibility < 0.4:
                return None

            pt1 = (l_sh.x * frame_w, l_sh.y * frame_h)
            pt2 = (r_sh.x * frame_w, r_sh.y * frame_h)
            return calculate_distance(pt1, pt2)

        except (IndexError, AttributeError):
            return None
