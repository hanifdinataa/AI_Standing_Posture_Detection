"""
╔══════════════════════════════════════════════════════════════════╗
║           AI Posture Detection — Scoring System                 ║
║  Manages productivity score, health score, break reminders,     ║
║  and fatigue indicators.                                        ║
╚══════════════════════════════════════════════════════════════════╝

Scoring Model
─────────────
Productivity Score (0–100):
    • Starts at 100
    • Bad posture:  −0.2 per second
    • Looking away: −0.5 per second
    • Too close:    −0.3 per second
    • Good posture: +0.1 per second (recovery)

Health Score:
    health = (good_posture_time / total_time) × 100
"""

import time
from typing import Dict, List

import config as cfg
from utils import logger


class ScoreManager:
    """Track productivity and health scores throughout a session."""

    def __init__(self):
        self._start_time: float = time.time()
        self._last_update: float = time.time()

        # Scores
        self.productivity_score: float = cfg.INITIAL_PRODUCTIVITY_SCORE
        self.health_score: float = 100.0

        # Cumulative timers (seconds)
        self.good_posture_time: float = 0.0
        self.bad_posture_time: float = 0.0
        self.focused_time: float = 0.0
        self.distracted_time: float = 0.0
        self.away_time: float = 0.0
        self.too_close_time: float = 0.0

        # Break reminder
        self._last_break_reminder: float = time.time()
        self._break_due: bool = False

        # Warnings buffer
        self.active_warnings: List[str] = []

    # ── public API ──────────────────────────────────────────────
    def update(
        self,
        posture_status: str,
        focus_status: str,
        distance_status: str,
    ) -> Dict:
        """
        Update scores based on current detector states.

        Returns a dict of current scores and warnings.
        """
        now = time.time()
        dt = now - self._last_update
        self._last_update = now

        self.active_warnings = []

        # ── Accumulate time buckets ──
        if posture_status == cfg.POSTURE_GOOD:
            self.good_posture_time += dt
        else:
            self.bad_posture_time += dt

        if focus_status == cfg.FOCUS_FOCUSED:
            self.focused_time += dt
        elif focus_status == cfg.FOCUS_DISTRACTED:
            self.distracted_time += dt
        else:
            self.away_time += dt

        if distance_status == cfg.DISTANCE_TOO_CLOSE_STR:
            self.too_close_time += dt

        # ── Productivity score adjustments ──
        delta = 0.0

        # Bad posture penalty
        if posture_status in (cfg.POSTURE_BAD, cfg.POSTURE_WARNING):
            penalty = cfg.SCORE_DEDUCT_BAD_POSTURE * dt
            if posture_status == cfg.POSTURE_BAD:
                penalty *= 1.5  # stronger penalty for BAD vs WARNING
            delta -= penalty
            self.active_warnings.append("Bad posture detected")

        # Looking away / distracted penalty
        if focus_status == cfg.FOCUS_DISTRACTED:
            delta -= cfg.SCORE_DEDUCT_LOOKING_AWAY * dt * 0.5
            self.active_warnings.append("Focus lost")
        elif focus_status == cfg.FOCUS_AWAY:
            delta -= cfg.SCORE_DEDUCT_LOOKING_AWAY * dt
            self.active_warnings.append("User away")

        # Too close penalty
        if distance_status == cfg.DISTANCE_TOO_CLOSE_STR:
            delta -= cfg.SCORE_DEDUCT_TOO_CLOSE * dt
            self.active_warnings.append("Move away from screen")

        # Good state recovery
        if (posture_status == cfg.POSTURE_GOOD and
                focus_status == cfg.FOCUS_FOCUSED and
                distance_status == cfg.DISTANCE_GOOD_STR):
            delta += cfg.SCORE_RECOVERY_GOOD * dt

        self.productivity_score = max(
            cfg.SCORE_MIN,
            min(cfg.SCORE_MAX, self.productivity_score + delta),
        )

        # ── Health score ──
        total_time = self.get_session_duration()
        if total_time > 0:
            self.health_score = (self.good_posture_time / total_time) * 100.0
        else:
            self.health_score = 100.0

        # ── Break reminder ──
        if now - self._last_break_reminder >= cfg.BREAK_REMINDER_INTERVAL_SEC:
            self._break_due = True
            self._last_break_reminder = now
            self.active_warnings.append("Time for a break!")
            logger.info("Break reminder triggered — take a 5-minute break")

        return self.get_info()

    def get_session_duration(self) -> float:
        """Seconds since session start."""
        return time.time() - self._start_time

    def is_break_due(self) -> bool:
        """Check and consume break reminder flag."""
        if self._break_due:
            self._break_due = False
            return True
        return False

    def get_fatigue_level(self) -> str:
        """Simple fatigue estimator based on bad posture accumulation."""
        ratio = self.bad_posture_time / max(1.0, self.get_session_duration())
        if ratio > 0.5:
            return "HIGH"
        elif ratio > 0.25:
            return "MODERATE"
        return "LOW"

    def get_info(self) -> Dict:
        """Return a snapshot of all scoring data."""
        return {
            "productivity_score": round(self.productivity_score, 1),
            "health_score": round(self.health_score, 1),
            "session_time": self.get_session_duration(),
            "good_posture_time": round(self.good_posture_time, 1),
            "bad_posture_time": round(self.bad_posture_time, 1),
            "focused_time": round(self.focused_time, 1),
            "distracted_time": round(self.distracted_time, 1),
            "away_time": round(self.away_time, 1),
            "too_close_time": round(self.too_close_time, 1),
            "fatigue_level": self.get_fatigue_level(),
            "warnings": self.active_warnings,
        }

    def reset(self):
        """Reset all scores for a new session."""
        self._start_time = time.time()
        self._last_update = time.time()
        self._last_break_reminder = time.time()
        self.productivity_score = cfg.INITIAL_PRODUCTIVITY_SCORE
        self.health_score = 100.0
        self.good_posture_time = 0.0
        self.bad_posture_time = 0.0
        self.focused_time = 0.0
        self.distracted_time = 0.0
        self.away_time = 0.0
        self.too_close_time = 0.0
        self._break_due = False
        self.active_warnings = []
        logger.info("Scores reset — new session started")
