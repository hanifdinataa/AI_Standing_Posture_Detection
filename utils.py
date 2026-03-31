"""
╔══════════════════════════════════════════════════════════════════╗
║              AI Posture Detection — Utilities                   ║
║  Shared math, drawing helpers, and logging configuration.       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import math
import logging
import time
from collections import deque
from typing import Tuple, Optional, List

import cv2
import numpy as np

import config as cfg


# ──────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ──────────────────────────────────────────────────────────────────
def setup_logger(name: str = "PostureAI", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger with coloured-prefix formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(levelname)s] %(asctime)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = setup_logger()


# ──────────────────────────────────────────────────────────────────
# MATH UTILITIES
# ──────────────────────────────────────────────────────────────────
def calculate_angle(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
) -> float:
    """
    Calculate the angle ∠ABC formed at point B using the dot-product formula.

    Parameters
    ----------
    a : (x, y) — first point  (e.g. ear)
    b : (x, y) — vertex point (e.g. shoulder)
    c : (x, y) — third point  (e.g. hip)

    Returns
    -------
    Angle in degrees [0, 180].

    Math
    ----
        v1 = A - B
        v2 = C - B
        angle = arccos( dot(v1, v2) / (|v1| * |v2|) )
    """
    v1 = np.array([a[0] - b[0], a[1] - b[1]])
    v2 = np.array([c[0] - b[0], c[1] - b[1]])

    dot = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)

    if mag1 == 0 or mag2 == 0:
        return 0.0

    cos_angle = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)


def calculate_distance(
    p1: Tuple[float, float], p2: Tuple[float, float]
) -> float:
    """Euclidean distance between two 2D points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def midpoint(
    p1: Tuple[float, float], p2: Tuple[float, float]
) -> Tuple[float, float]:
    """Return the midpoint of two 2D points."""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


# ──────────────────────────────────────────────────────────────────
# SMOOTHING FILTER
# ──────────────────────────────────────────────────────────────────
class MovingAverageFilter:
    """Simple moving-average smoother for scalar values."""

    def __init__(self, window_size: int = cfg.SMOOTHING_WINDOW_SIZE):
        self.window_size = window_size
        self._buffer: deque = deque(maxlen=window_size)

    def update(self, value: float) -> float:
        """Add a value and return the smoothed result."""
        self._buffer.append(value)
        return sum(self._buffer) / len(self._buffer)

    def reset(self):
        self._buffer.clear()


# ──────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ──────────────────────────────────────────────────────────────────
def draw_text_with_bg(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = cfg.FONT_SCALE_BODY,
    color: Tuple[int, int, int] = cfg.COLOR_WHITE,
    bg_color: Optional[Tuple[int, int, int]] = cfg.COLOR_DARK_BG,
    thickness: int = cfg.FONT_THICKNESS,
    padding: int = 6,
) -> int:
    """
    Draw text with a semi-transparent background rectangle.
    Returns the y-position after the text for stacking.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = position
    x1, y1 = x - padding, y - th - padding
    x2, y2 = x + tw + padding, y + baseline + padding

    if bg_color is not None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
        cv2.addWeighted(overlay, cfg.PANEL_ALPHA, frame, 1 - cfg.PANEL_ALPHA, 0, frame)

    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return y2 + 4


def draw_status_panel(
    frame: np.ndarray,
    stats: dict,
) -> np.ndarray:
    """
    Draw a semi-transparent status panel on the right side of the frame.

    Parameters
    ----------
    stats : dict with keys
        posture_status, distance_status, focus_status,
        productivity_score, health_score, session_time,
        neck_angle (optional), warnings (list[str])
    """
    h, w = frame.shape[:2]
    panel_x = w - cfg.PANEL_WIDTH

    # Draw panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, 0), (w, h), cfg.COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, cfg.PANEL_ALPHA, frame, 1 - cfg.PANEL_ALPHA, 0, frame)

    # Decorative header bar
    cv2.rectangle(frame, (panel_x, 0), (w, 40), cfg.COLOR_ACCENT_BLUE, -1)
    cv2.putText(
        frame, "POSTURE AI MONITOR", (panel_x + 12, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, cfg.COLOR_WHITE, 2, cv2.LINE_AA,
    )

    y = 70
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── Status indicators ──
    status_items = [
        ("POSTURE", stats.get("posture_status", "—")),
        ("DISTANCE", stats.get("distance_status", "—")),
        ("FOCUS", stats.get("focus_status", "—")),
    ]

    for label, value in status_items:
        color = _status_color(value)
        cv2.putText(frame, label + ":", (panel_x + 15, y),
                    font, cfg.FONT_SCALE_BODY, cfg.COLOR_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, str(value), (panel_x + 130, y),
                    font, cfg.FONT_SCALE_BODY, color, 2, cv2.LINE_AA)
        y += cfg.LINE_HEIGHT

    # Separator
    y += 5
    cv2.line(frame, (panel_x + 10, y), (w - 10, y), cfg.COLOR_ACCENT_BLUE, 1)
    y += 20

    # ── Scores ──
    prod = stats.get("productivity_score", 0)
    health = stats.get("health_score", 0)

    cv2.putText(frame, "PRODUCTIVITY", (panel_x + 15, y),
                font, cfg.FONT_SCALE_SMALL, cfg.COLOR_ACCENT_CYAN, 1, cv2.LINE_AA)
    y += 24
    _draw_progress_bar(frame, panel_x + 15, y, cfg.PANEL_WIDTH - 40, 14, prod / 100.0)
    cv2.putText(frame, f"{prod:.0f}", (panel_x + cfg.PANEL_WIDTH - 35, y + 12),
                font, cfg.FONT_SCALE_BODY, cfg.COLOR_WHITE, 1, cv2.LINE_AA)
    y += 30

    cv2.putText(frame, "HEALTH", (panel_x + 15, y),
                font, cfg.FONT_SCALE_SMALL, cfg.COLOR_ACCENT_CYAN, 1, cv2.LINE_AA)
    y += 24
    _draw_progress_bar(frame, panel_x + 15, y, cfg.PANEL_WIDTH - 40, 14, health / 100.0)
    cv2.putText(frame, f"{health:.0f}%", (panel_x + cfg.PANEL_WIDTH - 45, y + 12),
                font, cfg.FONT_SCALE_BODY, cfg.COLOR_WHITE, 1, cv2.LINE_AA)
    y += 30

    # Separator
    cv2.line(frame, (panel_x + 10, y), (w - 10, y), cfg.COLOR_ACCENT_BLUE, 1)
    y += 20

    # ── Session timer ──
    session_sec = stats.get("session_time", 0)
    mins, secs = divmod(int(session_sec), 60)
    hrs, mins = divmod(mins, 60)
    time_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"
    cv2.putText(frame, "SESSION", (panel_x + 15, y),
                font, cfg.FONT_SCALE_SMALL, cfg.COLOR_ACCENT_CYAN, 1, cv2.LINE_AA)
    cv2.putText(frame, time_str, (panel_x + 100, y),
                font, cfg.FONT_SCALE_BODY, cfg.COLOR_WHITE, 1, cv2.LINE_AA)
    y += cfg.LINE_HEIGHT

    # ── Neck angle (debug) ──
    neck = stats.get("neck_angle")
    if neck is not None:
        cv2.putText(frame, f"NECK ANGLE: {neck:.1f} deg", (panel_x + 15, y),
                    font, cfg.FONT_SCALE_SMALL, cfg.COLOR_WHITE, 1, cv2.LINE_AA)
        y += cfg.LINE_HEIGHT

    # ── Warnings ──
    warnings = stats.get("warnings", [])
    if warnings:
        y += 5
        cv2.line(frame, (panel_x + 10, y), (w - 10, y), cfg.COLOR_RED, 1)
        y += 18
        for warn_text in warnings:
            cv2.putText(frame, f"! {warn_text}", (panel_x + 15, y),
                        font, cfg.FONT_SCALE_SMALL, cfg.COLOR_RED, 1, cv2.LINE_AA)
            y += 22

    return frame


def draw_warning_banner(
    frame: np.ndarray,
    text: str,
    color: Tuple[int, int, int] = cfg.COLOR_RED,
):
    """Draw a full-width warning banner at the top of the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w - cfg.PANEL_WIDTH, 45), color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(
        frame, text, (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, cfg.COLOR_WHITE, 2, cv2.LINE_AA,
    )


def draw_skeleton(
    frame: np.ndarray,
    landmarks: list,
    status: str,
    frame_w: int,
    frame_h: int,
):
    """
    Draw pose skeleton connections and landmarks.
    Color-coded: green for good posture, red for bad.
    """
    color = cfg.COLOR_GREEN if status == cfg.POSTURE_GOOD else cfg.COLOR_RED

    # Key connections: shoulder↔shoulder, ear↔shoulder, shoulder↔hip
    connections = [
        (11, 12),  # left shoulder → right shoulder
        (7, 11),   # left ear → left shoulder
        (8, 12),   # right ear → right shoulder
        (11, 23),  # left shoulder → left hip
        (12, 24),  # right shoulder → right hip
        (0, 7),    # nose → left ear (approx)
        (0, 8),    # nose → right ear (approx)
    ]

    for i, j in connections:
        if i < len(landmarks) and j < len(landmarks):
            lm1 = landmarks[i]
            lm2 = landmarks[j]
            if lm1.visibility > 0.5 and lm2.visibility > 0.5:
                pt1 = (int(lm1.x * frame_w), int(lm1.y * frame_h))
                pt2 = (int(lm2.x * frame_w), int(lm2.y * frame_h))
                cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

    # Draw key landmarks as circles
    key_indices = [0, 2, 5, 7, 8, 11, 12, 23, 24]  # nose, eyes, ears, shoulders, hips
    for idx in key_indices:
        if idx < len(landmarks):
            lm = landmarks[idx]
            if lm.visibility > 0.5:
                pt = (int(lm.x * frame_w), int(lm.y * frame_h))
                cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 7, cfg.COLOR_WHITE, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────────
def _status_color(status: str) -> Tuple[int, int, int]:
    """Map a status string to a BGR colour."""
    status_upper = str(status).upper()
    if status_upper in ("GOOD", "FOCUSED"):
        return cfg.COLOR_GREEN
    elif status_upper in ("WARNING", "DISTRACTED", "TOO FAR"):
        return cfg.COLOR_YELLOW
    else:
        return cfg.COLOR_RED


def _draw_progress_bar(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    progress: float,
):
    """Draw a progress bar with gradient fill."""
    progress = max(0.0, min(1.0, progress))

    # Background
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)

    # Fill
    fill_w = int(w * progress)
    if progress > 0.7:
        bar_color = cfg.COLOR_GREEN
    elif progress > 0.4:
        bar_color = cfg.COLOR_YELLOW
    else:
        bar_color = cfg.COLOR_RED

    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + h), bar_color, -1)

    # Border
    cv2.rectangle(frame, (x, y), (x + w, y + h), cfg.COLOR_WHITE, 1)


def format_time(seconds: float) -> str:
    """Format seconds into HH:MM:SS string."""
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"
