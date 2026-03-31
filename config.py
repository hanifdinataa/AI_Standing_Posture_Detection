"""
╔══════════════════════════════════════════════════════════════════╗
║              AI Posture Detection — Configuration               ║
║  Central configuration for all thresholds, scoring rates,       ║
║  and application settings.                                      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os

# ──────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ──────────────────────────────────────────────────────────────────
# CAMERA SETTINGS
# ──────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# ──────────────────────────────────────────────────────────────────
# MEDIAPIPE SETTINGS
# ──────────────────────────────────────────────────────────────────
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
MODEL_COMPLEXITY = 1  # 0=lite, 1=full, 2=heavy

# ──────────────────────────────────────────────────────────────────
# POSTURE THRESHOLDS
# ──────────────────────────────────────────────────────────────────
NECK_ANGLE_GOOD = 160        # degrees — above this is good posture
NECK_ANGLE_WARNING = 150     # degrees — below this is bad posture
SHOULDER_TILT_THRESHOLD = 0.04  # normalised y-difference for tilt detection
FORWARD_HEAD_THRESHOLD = 0.03  # normalised x-offset (nose vs shoulder midpoint)

# ──────────────────────────────────────────────────────────────────
# DISTANCE THRESHOLDS (centimeters)
# ──────────────────────────────────────────────────────────────────
DISTANCE_TOO_CLOSE = 40     # cm
DISTANCE_GOOD_MIN = 40      # cm
DISTANCE_GOOD_MAX = 80      # cm
DISTANCE_TOO_FAR = 90       # cm

# Reference shoulder width for distance calibration (average adult)
REFERENCE_SHOULDER_WIDTH_CM = 40.0
# Default focal length estimate (will be calibrated on first frame)
DEFAULT_FOCAL_LENGTH = 600.0

# ──────────────────────────────────────────────────────────────────
# FOCUS / ATTENTION THRESHOLDS
# ──────────────────────────────────────────────────────────────────
HEAD_TURN_THRESHOLD = 25     # degrees — beyond this = distracted
FACE_ABSENT_AWAY_SEC = 10   # seconds with no face → AWAY status
DISTRACTION_WARNING_SEC = 15 # seconds looking away → warning
DISTRACTION_PENALTY_SEC = 30 # seconds looking away → score penalty

# Eye Aspect Ratio for blink detection (optional)
EAR_BLINK_THRESHOLD = 0.21
EAR_CONSEC_FRAMES = 3

# ──────────────────────────────────────────────────────────────────
# SCORING RATES (per second)
# ──────────────────────────────────────────────────────────────────
SCORE_DEDUCT_BAD_POSTURE = 0.2   # points/s
SCORE_DEDUCT_LOOKING_AWAY = 0.5  # points/s
SCORE_DEDUCT_TOO_CLOSE = 0.3    # points/s
SCORE_RECOVERY_GOOD = 0.1       # points/s
INITIAL_PRODUCTIVITY_SCORE = 100.0
SCORE_MIN = 0.0
SCORE_MAX = 100.0

# ──────────────────────────────────────────────────────────────────
# BREAK REMINDER
# ──────────────────────────────────────────────────────────────────
BREAK_REMINDER_INTERVAL_SEC = 30 * 60  # 30 minutes

# ──────────────────────────────────────────────────────────────────
# SMOOTHING
# ──────────────────────────────────────────────────────────────────
SMOOTHING_WINDOW_SIZE = 5   # frames for moving average filter

# ──────────────────────────────────────────────────────────────────
# COLOURS (BGR for OpenCV)
# ──────────────────────────────────────────────────────────────────
COLOR_GREEN = (0, 200, 0)
COLOR_YELLOW = (0, 220, 255)
COLOR_RED = (0, 0, 230)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_DARK_BG = (30, 30, 30)
COLOR_PANEL_BG = (40, 40, 40)
COLOR_ACCENT_BLUE = (230, 160, 50)
COLOR_ACCENT_CYAN = (210, 200, 0)

# ──────────────────────────────────────────────────────────────────
# UI LAYOUT
# ──────────────────────────────────────────────────────────────────
PANEL_WIDTH = 280            # pixels — right-side status panel
PANEL_ALPHA = 0.75           # overlay transparency
FONT_SCALE_TITLE = 0.6
FONT_SCALE_BODY = 0.5
FONT_SCALE_SMALL = 0.4
FONT_THICKNESS = 1
LINE_HEIGHT = 28

# ──────────────────────────────────────────────────────────────────
# STATUS ENUMS (string constants)
# ──────────────────────────────────────────────────────────────────
POSTURE_GOOD = "GOOD"
POSTURE_WARNING = "WARNING"
POSTURE_BAD = "BAD"

DISTANCE_TOO_CLOSE_STR = "TOO CLOSE"
DISTANCE_GOOD_STR = "GOOD"
DISTANCE_TOO_FAR_STR = "TOO FAR"

FOCUS_FOCUSED = "FOCUSED"
FOCUS_DISTRACTED = "DISTRACTED"
FOCUS_AWAY = "AWAY"
