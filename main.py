"""
╔══════════════════════════════════════════════════════════════════╗
║      AI Posture Detection — Ergonomic Assistant (Main App)      ║
║  Real-time webcam posture monitoring with ML-powered analysis.  ║
╚══════════════════════════════════════════════════════════════════╝

Controls
────────
    [S]  Start / Stop session
    [R]  Reset session
    [C]  Re-calibrate distance
    [Q]  Quit application
"""

import sys
import time
import platform

import cv2
import numpy as np
import mediapipe as mp

import config as cfg
from utils import logger, draw_status_panel, draw_warning_banner, draw_skeleton
from posture_detector import PostureDetector
from distance_detector import DistanceEstimator
from focus_detector import FocusDetector
from scoring_system import ScoreManager
from analytics import SessionAnalytics


# ──────────────────────────────────────────────────────────────────
# OPTIONAL: sound alert (Windows only)
# ──────────────────────────────────────────────────────────────────
def _play_alert():
    """Play a short beep on Windows; silently skip on other platforms."""
    try:
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(800, 200)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────
# APPLICATION CLASS
# ──────────────────────────────────────────────────────────────────
class PostureApp:
    """Main application orchestrating all detectors and the render loop."""

    def __init__(self):
        # ── Detectors ──
        self.posture = PostureDetector()
        self.distance = DistanceEstimator()
        self.focus = FocusDetector()
        self.score = ScoreManager()
        self.analytics = SessionAnalytics()

        # ── MediaPipe ──
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=cfg.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=cfg.MIN_TRACKING_CONFIDENCE,
            model_complexity=cfg.MODEL_COMPLEXITY,
        )

        # ── State ──
        self.session_active: bool = False
        self._alert_cooldown: float = 0.0
        self._fps_counter: int = 0
        self._fps_time: float = time.time()
        self._current_fps: float = 0.0

    # ── Main entry point ────────────────────────────────────────
    def run(self):
        """Open webcam and start the processing loop."""
        self._print_banner()

        cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
        if not cap.isOpened():
            logger.error("Cannot open webcam (index=%d)", cfg.CAMERA_INDEX)
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)

        logger.info("Webcam opened — press [S] to start session, [Q] to quit")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break

                frame = cv2.resize(frame, (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT))
                frame = cv2.flip(frame, 1)  # mirror

                # ── Process frame ──
                stats = self._process_frame(frame)

                # ── Draw overlay ──
                frame = self._render(frame, stats)

                # ── FPS counter ──
                self._update_fps()

                cv2.imshow("-", frame)

                # ── Key handling ──
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    self._toggle_session()
                elif key == ord("r"):
                    self._reset_session()
                elif key == ord("c"):
                    self.distance.recalibrate()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self._shutdown(cap)

    # ── Frame processing ────────────────────────────────────────
    def _process_frame(self, frame: np.ndarray) -> dict:
        """Run all detectors on a single frame and return stats dict."""
        h, w = frame.shape[:2]

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        landmarks = None
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

        # ── Run detectors ──
        if landmarks and self.session_active:
            posture_status = self.posture.update(landmarks, w, h)
            distance_status = self.distance.update(landmarks, w, h)
            focus_status = self.focus.update(landmarks, w, h)

            # Update scores
            score_info = self.score.update(posture_status, focus_status, distance_status)

            # Record analytics
            posture_info = self.posture.get_info()
            distance_info = self.distance.get_info()
            focus_info = self.focus.get_info()

            self.analytics.record(
                posture_status=posture_status,
                focus_status=focus_status,
                distance_status=distance_status,
                neck_angle=posture_info["neck_angle"],
                distance_cm=distance_info["distance_cm"],
                head_yaw=focus_info["head_yaw"],
                productivity_score=score_info["productivity_score"],
                health_score=score_info["health_score"],
            )

            # Sound alert logic
            now = time.time()
            if now > self._alert_cooldown:
                if posture_status == cfg.POSTURE_BAD:
                    _play_alert()
                    self._alert_cooldown = now + 10  # don't spam every frame
                elif focus_status == cfg.FOCUS_AWAY:
                    _play_alert()
                    self._alert_cooldown = now + 15

            return {
                "posture_status": posture_status,
                "distance_status": distance_status,
                "focus_status": focus_status,
                "productivity_score": score_info["productivity_score"],
                "health_score": score_info["health_score"],
                "session_time": score_info["session_time"],
                "neck_angle": posture_info["neck_angle"],
                "warnings": score_info["warnings"],
                "landmarks": landmarks,
                "fps": self._current_fps,
            }

        elif landmarks and not self.session_active:
            # Show preview without scoring
            self.posture.update(landmarks, w, h)
            self.distance.update(landmarks, w, h)
            self.focus.update(landmarks, w, h)

            return {
                "posture_status": self.posture.status,
                "distance_status": self.distance.status,
                "focus_status": self.focus.status,
                "productivity_score": 0,
                "health_score": 0,
                "session_time": 0,
                "neck_angle": self.posture.neck_angle,
                "warnings": ["Press [S] to start session"],
                "landmarks": landmarks,
                "fps": self._current_fps,
            }
        else:
            # No landmarks
            self.focus.update(None, w, h)
            return {
                "posture_status": "—",
                "distance_status": "—",
                "focus_status": self.focus.status,
                "productivity_score": self.score.productivity_score if self.session_active else 0,
                "health_score": self.score.health_score if self.session_active else 0,
                "session_time": self.score.get_session_duration() if self.session_active else 0,
                "neck_angle": None,
                "warnings": ["No pose detected — face the camera"],
                "landmarks": None,
                "fps": self._current_fps,
            }

    # ── Rendering ───────────────────────────────────────────────
    def _render(self, frame: np.ndarray, stats: dict) -> np.ndarray:
        """Draw skeleton, status panel, and warning banners."""
        h, w = frame.shape[:2]

        # Draw skeleton if landmarks available
        landmarks = stats.get("landmarks")
        if landmarks:
            draw_skeleton(
                frame, landmarks,
                stats.get("posture_status", cfg.POSTURE_GOOD),
                w, h,
            )

        # Draw status panel
        frame = draw_status_panel(frame, stats)

        # Draw warning banners
        warnings = stats.get("warnings", [])
        posture_status = stats.get("posture_status", "")
        focus_status = stats.get("focus_status", "")
        distance_status = stats.get("distance_status", "")

        if posture_status == cfg.POSTURE_BAD:
            draw_warning_banner(frame, "BAD POSTURE DETECTED — Straighten up!", cfg.COLOR_RED)
        elif distance_status == cfg.DISTANCE_TOO_CLOSE_STR:
            draw_warning_banner(frame, "TOO CLOSE — Move away from screen!", cfg.COLOR_YELLOW)
        elif focus_status == cfg.FOCUS_AWAY:
            draw_warning_banner(frame, "USER AWAY — Return to screen", cfg.COLOR_RED)
        elif focus_status == cfg.FOCUS_DISTRACTED:
            dur = self.focus.get_distraction_duration()
            if dur > cfg.DISTRACTION_WARNING_SEC:
                draw_warning_banner(frame, f"DISTRACTED for {dur:.0f}s — Focus!", cfg.COLOR_YELLOW)

        # Session indicator
        if self.session_active:
            cv2.circle(frame, (20, h - 20), 8, cfg.COLOR_GREEN, -1)
            cv2.putText(frame, "REC", (33, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, cfg.COLOR_GREEN, 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (20, h - 20), 8, cfg.COLOR_RED, -1)
            cv2.putText(frame, "IDLE", (33, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, cfg.COLOR_RED, 1, cv2.LINE_AA)

        # FPS
        cv2.putText(frame, f"FPS: {self._current_fps:.0f}", (w - cfg.PANEL_WIDTH - 80, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, cfg.COLOR_WHITE, 1, cv2.LINE_AA)

        return frame

    # ── Session controls ────────────────────────────────────────
    def _toggle_session(self):
        """Start or stop the tracking session."""
        if self.session_active:
            self.session_active = False
            score_info = self.score.get_info()
            self.analytics.print_summary(score_info)
            logger.info("Session stopped")
        else:
            self.session_active = True
            self.score.reset()
            self.posture.reset()
            self.focus.reset()
            self.distance.reset()
            self.analytics.reset()
            logger.info("Session started — monitoring active")

    def _reset_session(self):
        """Reset all detectors and scores."""
        self.score.reset()
        self.posture.reset()
        self.focus.reset()
        self.distance.reset()
        self.analytics.reset()
        logger.info("Session reset")

    # ── Utilities ───────────────────────────────────────────────
    def _update_fps(self):
        """Simple FPS counter."""
        self._fps_counter += 1
        now = time.time()
        elapsed = now - self._fps_time
        if elapsed >= 1.0:
            self._current_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_time = now

    def _shutdown(self, cap):
        """Clean up resources."""
        if self.session_active:
            score_info = self.score.get_info()
            self.analytics.print_summary(score_info)

        cap.release()
        cv2.destroyAllWindows()
        self.pose.close()
        logger.info("Application shut down cleanly")

    @staticmethod
    def _print_banner():
        """Print startup banner to console."""
        print("\n" + "═" * 55)
        print("  ╔═══════════════════════════════════════════════╗")
        print("  ║   POSTURE AI — Ergonomic Assistant v1.0.0    ║")
        print("  ║   Real-Time Posture & Focus Monitoring       ║")
        print("  ╚═══════════════════════════════════════════════╝")
        print("═" * 55)
        print("  Controls:")
        print("    [S] Start / Stop session")
        print("    [R] Reset session")
        print("    [C] Re-calibrate distance")
        print("    [Q] Quit")
        print("═" * 55 + "\n")


# ──────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = PostureApp()
    app.run()
