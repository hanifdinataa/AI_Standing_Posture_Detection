"""
╔══════════════════════════════════════════════════════════════════╗
║           AI Posture Detection — Focus Detector                 ║
║  Detects user attention state: FOCUSED / DISTRACTED / AWAY.     ║
╚══════════════════════════════════════════════════════════════════╝

Focus detection uses:
    1. Head yaw estimation  —  ratio of nose-to-ear distances
       to infer whether the user is looking straight ahead.
    2. Face absence timer   —  if no landmarks are detected for
       a configurable duration, the user is classified as AWAY.
    3. Blink detection (optional) — Eye Aspect Ratio (EAR) to
       detect blinks and potential fatigue.
"""

import time
from typing import Optional, Dict

import config as cfg
from utils import calculate_distance, MovingAverageFilter, logger


class FocusDetector:
    """Detect whether the user is focused on the screen."""

    def __init__(self):
        self._yaw_filter = MovingAverageFilter(cfg.SMOOTHING_WINDOW_SIZE)

        # State tracking
        self.status: str = cfg.FOCUS_FOCUSED
        self.head_yaw: float = 0.0       # estimated yaw in degrees
        self.confidence: float = 1.0

        # Timing
        self._last_face_seen: float = time.time()
        self._distraction_start: Optional[float] = None
        self._total_distracted_sec: float = 0.0
        self._total_away_sec: float = 0.0
        self._last_update: float = time.time()

        # Blink detection
        self._blink_count: int = 0
        self._ear_consec: int = 0
        self._blink_filter = MovingAverageFilter(cfg.SMOOTHING_WINDOW_SIZE)

    # ── public API ──────────────────────────────────────────────
    def update(
        self,
        landmarks: Optional[list],
        frame_w: int,
        frame_h: int,
    ) -> str:
        """
        Update focus state and return status string.

        Parameters
        ----------
        landmarks : MediaPipe landmarks or None if no pose detected
        frame_w, frame_h : frame dimensions
        """
        now = time.time()
        dt = now - self._last_update
        self._last_update = now

        # ── No face detected ──
        if landmarks is None or not self._has_face(landmarks):
            absent_sec = now - self._last_face_seen

            if absent_sec >= cfg.FACE_ABSENT_AWAY_SEC:
                self.status = cfg.FOCUS_AWAY
                self._total_away_sec += dt
                logger.warning("User AWAY — no face for %.0fs", absent_sec)
            else:
                # Short absence: keep previous status or mark distracted
                if self.status == cfg.FOCUS_FOCUSED:
                    self.status = cfg.FOCUS_DISTRACTED
            return self.status

        # Face is visible — reset absence timer
        self._last_face_seen = now

        # ── Estimate head yaw ──
        raw_yaw = self._estimate_yaw(landmarks, frame_w, frame_h)
        if raw_yaw is not None:
            self.head_yaw = self._yaw_filter.update(raw_yaw)

        # ── Blink detection (optional) ──
        self._detect_blinks(landmarks, frame_w, frame_h)

        # ── Classify ──
        if abs(self.head_yaw) > cfg.HEAD_TURN_THRESHOLD:
            self.status = cfg.FOCUS_DISTRACTED
            self.confidence = max(0.0, 1.0 - abs(self.head_yaw) / 90.0)
            self._total_distracted_sec += dt

            # Track distraction start
            if self._distraction_start is None:
                self._distraction_start = now

            away_dur = now - self._distraction_start
            if away_dur > cfg.DISTRACTION_WARNING_SEC:
                logger.warning("User distracted for %.0fs", away_dur)
        else:
            self.status = cfg.FOCUS_FOCUSED
            self.confidence = 1.0
            self._distraction_start = None

        return self.status

    def get_distraction_duration(self) -> float:
        """Seconds of continuous current distraction (0 if focused)."""
        if self._distraction_start is None:
            return 0.0
        return time.time() - self._distraction_start

    def get_info(self) -> Dict:
        return {
            "status": self.status,
            "head_yaw": self.head_yaw,
            "confidence": self.confidence,
            "distraction_duration": self.get_distraction_duration(),
            "total_distracted_sec": self._total_distracted_sec,
            "total_away_sec": self._total_away_sec,
            "blink_count": self._blink_count,
        }

    def reset(self):
        """Reset all state for a new session."""
        self._yaw_filter.reset()
        self.status = cfg.FOCUS_FOCUSED
        self.head_yaw = 0.0
        self._last_face_seen = time.time()
        self._distraction_start = None
        self._total_distracted_sec = 0.0
        self._total_away_sec = 0.0
        self._blink_count = 0
        self._ear_consec = 0

    # ── internals ───────────────────────────────────────────────
    @staticmethod
    def _has_face(landmarks) -> bool:
        """Check if key face landmarks are visible."""
        try:
            nose = landmarks[0]
            l_ear = landmarks[7]
            r_ear = landmarks[8]
            return (
                nose.visibility > 0.4
                and (l_ear.visibility > 0.3 or r_ear.visibility > 0.3)
            )
        except (IndexError, AttributeError):
            return False

    def _estimate_yaw(
        self, landmarks, frame_w: int, frame_h: int
    ) -> Optional[float]:
        """
        Estimate head yaw (horizontal rotation) in degrees.

        Heuristic: compare distance from nose to left-ear vs nose to right-ear.
        When looking straight ahead these distances are roughly equal.
        A large asymmetry indicates the head is turned.

        Returns approximate yaw in degrees: negative = left, positive = right.
        """
        try:
            nose = landmarks[0]
            l_ear = landmarks[7]
            r_ear = landmarks[8]

            if nose.visibility < 0.4:
                return None

            nose_pt = (nose.x * frame_w, nose.y * frame_h)

            d_left, d_right = None, None
            if l_ear.visibility > 0.3:
                l_pt = (l_ear.x * frame_w, l_ear.y * frame_h)
                d_left = calculate_distance(nose_pt, l_pt)
            if r_ear.visibility > 0.3:
                r_pt = (r_ear.x * frame_w, r_ear.y * frame_h)
                d_right = calculate_distance(nose_pt, r_pt)

            if d_left is not None and d_right is not None:
                # Ratio-based yaw estimation
                total = d_left + d_right
                if total < 1:
                    return 0.0
                ratio = (d_right - d_left) / total  # -1..+1
                yaw = ratio * 90.0  # scale to degrees
                return yaw

            # Only one ear visible → assume turned heavily that way
            if d_left is None and d_right is not None:
                return -35.0  # turned heavily left
            if d_right is None and d_left is not None:
                return 35.0   # turned heavily right

            return None

        except (IndexError, AttributeError):
            return None

    def _detect_blinks(self, landmarks, frame_w: int, frame_h: int):
        """
        Simple blink detection using Eye Aspect Ratio (EAR).
        Uses MediaPipe face mesh indices mapped through pose landmarks.
        """
        try:
            # MediaPipe Pose eye landmarks:
            # 2 = left eye inner,  5 = right eye inner
            # 1 = left eye,        4 = right eye
            # 3 = left eye outer,  6 = right eye outer
            l_eye_top = landmarks[2]
            l_eye_bot = landmarks[3]
            r_eye_top = landmarks[5]
            r_eye_bot = landmarks[6]

            if (l_eye_top.visibility < 0.4 or l_eye_bot.visibility < 0.4 or
                    r_eye_top.visibility < 0.4 or r_eye_bot.visibility < 0.4):
                return

            # Approximate EAR using vertical distance between eye landmarks
            l_dist = abs(l_eye_top.y - l_eye_bot.y) * frame_h
            r_dist = abs(r_eye_top.y - r_eye_bot.y) * frame_h
            avg_ear = (l_dist + r_dist) / 2.0

            # Normalise by face size (nose to ear distance)
            nose = landmarks[0]
            ear = landmarks[7] if landmarks[7].visibility > landmarks[8].visibility else landmarks[8]
            face_scale = calculate_distance(
                (nose.x * frame_w, nose.y * frame_h),
                (ear.x * frame_w, ear.y * frame_h),
            )
            if face_scale < 1:
                return

            ear_ratio = avg_ear / face_scale

            smoothed_ear = self._blink_filter.update(ear_ratio)

            if smoothed_ear < cfg.EAR_BLINK_THRESHOLD:
                self._ear_consec += 1
            else:
                if self._ear_consec >= cfg.EAR_CONSEC_FRAMES:
                    self._blink_count += 1
                self._ear_consec = 0

        except (IndexError, AttributeError):
            pass
