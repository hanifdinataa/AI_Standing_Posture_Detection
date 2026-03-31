<![CDATA[<div align="center">

# 🎯 Posture AI — Ergonomic Assistant

### Real-Time Posture Detection & Productivity Monitoring

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-00897B?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)

*An AI-powered desktop application that uses computer vision to monitor your sitting posture, screen distance, and focus in real-time — helping you stay healthy and productive.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Installation](#-installation)
- [Usage](#-usage)
- [Scoring System](#-scoring-system)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Demo Mode](#-demo-mode)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 🔍 Overview

**Posture AI** is a portfolio-grade computer vision project that demonstrates real-time human pose estimation, behavioural analysis, and productivity scoring. It uses **Google MediaPipe Pose** to track 33 body landmarks and applies geometric analysis to classify sitting posture, estimate screen distance, and detect user attention.

### Key Highlights

- 🦴 **Real-time skeleton tracking** with colour-coded posture feedback
- 📏 **Distance estimation** using a pinhole camera model
- 👁️ **Focus detection** with head yaw estimation and absence tracking
- 📊 **Dual scoring** — Productivity Score + Health Score
- 📈 **Session analytics** exported to CSV for post-session review
- ⏰ **Break reminders** every 30 minutes
- 🔊 **Sound alerts** for bad posture (Windows)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     main.py                             │
│              (Application Orchestrator)                 │
├─────────────┬────────────┬──────────────┬───────────────┤
│   Posture   │  Distance  │    Focus     │   Scoring     │
│  Detector   │ Estimator  │  Detector    │   System      │
│             │            │             │               │
│ Neck angle  │ Pinhole    │ Head yaw    │ Productivity  │
│ Shoulder    │ camera     │ Face        │ Health        │
│ tilt        │ model      │ absence     │ Break         │
│ Forward     │            │ Blink       │ reminders     │
│ head        │            │ detection   │               │
├─────────────┴────────────┴──────────────┴───────────────┤
│                    Analytics                            │
│              (CSV session logging)                      │
├─────────────────────────────────────────────────────────┤
│              config.py  +  utils.py                     │
│         (Thresholds, math, drawing helpers)              │
├─────────────────────────────────────────────────────────┤
│           MediaPipe Pose  +  OpenCV                     │
│         (Landmark detection + video I/O)                │
└─────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 1. Real-Time Posture Detection
Tracks upper-body landmarks (nose, ears, shoulders, hips) and classifies posture using three signals:
- **Neck angle** — angle at the shoulder joint (ear → shoulder → hip)
- **Shoulder tilt** — vertical asymmetry between left and right shoulder
- **Forward head** — horizontal offset of nose relative to shoulder midpoint

| Status | Condition |
|--------|-----------|
| 🟢 **GOOD** | Neck angle > 160°, no tilt, head aligned |
| 🟡 **WARNING** | Minor deviations detected |
| 🔴 **BAD** | Neck angle < 150°, significant slouching |

### 2. Screen Distance Estimation
Uses shoulder width in pixels with a **pinhole camera model** to estimate real-world distance:

```
distance = (shoulder_width_cm × focal_length) / shoulder_width_px
```

| Status | Range |
|--------|-------|
| 🔴 **TOO CLOSE** | < 40 cm |
| 🟢 **GOOD** | 40 – 80 cm |
| 🟡 **TOO FAR** | > 90 cm |

### 3. Focus Detection
Estimates head yaw angle using nose-to-ear distance ratios:

| Status | Condition |
|--------|-----------|
| 🟢 **FOCUSED** | Head facing forward |
| 🟡 **DISTRACTED** | Head turned > 25° |
| 🔴 **AWAY** | No face detected > 10 seconds |

### 4. Looking-Away Timer
Tracks continuous distraction duration with escalating warnings:
- **> 15 sec** → "User distracted" warning
- **> 30 sec** → Productivity score penalty

### 5. Visual Warning System
Colour-coded warning banners overlaid on the video feed:
- 🟢 Green = Good state
- 🟡 Yellow = Warning
- 🔴 Red = Action needed

### 6. Dual Scoring System
See [Scoring System](#-scoring-system) for full details.

### 7. Session Analytics
Per-second CSV logging with columns: `timestamp`, `posture_status`, `focus_status`, `distance_status`, `neck_angle`, `distance_cm`, `head_yaw`, `productivity_score`, `health_score`.

### 8. Real-Time Overlay UI
Professional HUD panel with:
- Status indicators with colour coding
- Progress bars for Productivity and Health scores
- Session timer
- Neck angle readout
- Warning messages
- FPS counter

### 9. Skeleton Visualisation
MediaPipe skeleton drawn with colour-coded connections:
- 🟢 Green skeleton = good posture
- 🔴 Red skeleton = bad posture

### 10. Advanced Features
- 👁️ Blink detection via Eye Aspect Ratio (EAR)
- 😴 Fatigue level estimation
- ⏰ 30-minute break reminders
- 🔊 Sound alerts (Windows)

---

## 🧮 How It Works

### Neck Angle Calculation

The neck angle is measured at the **shoulder joint** using the three-point angle formula:

```
Vectors:
    v1 = Ear - Shoulder
    v2 = Hip - Shoulder

Angle:
    θ = arccos( dot(v1, v2) / (|v1| × |v2|) )
```

A straight, upright posture yields an angle close to **180°**. As the user slouches forward, the ear moves ahead of the shoulder line, decreasing the angle below **150°**.

### Distance Estimation (Pinhole Camera Model)

```
focal_length = (shoulder_px × assumed_distance) / shoulder_cm
distance     = (shoulder_cm × focal_length) / shoulder_px
```

The system auto-calibrates on the first frame, assuming a comfortable starting distance of 60 cm. Subsequent frames use this calibrated focal length for continuous estimation.

### Head Yaw Estimation

```
ratio = (dist_nose_to_right_ear - dist_nose_to_left_ear) /
        (dist_nose_to_right_ear + dist_nose_to_left_ear)

yaw ≈ ratio × 90°
```

When looking straight ahead, both ear distances are equal (ratio ≈ 0). Turning the head creates asymmetry.

### Smoothing

All measurements pass through a **moving-average filter** (configurable window size, default 5 frames) to reduce landmark jitter and produce stable classifications.

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- Webcam

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/AI_Posture_Detection.git
cd AI_Posture_Detection

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🎮 Usage

```bash
python main.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `S` | Start / Stop session |
| `R` | Reset session |
| `C` | Re-calibrate distance |
| `Q` | Quit application |

### Workflow
1. Launch the app — webcam preview activates
2. Press **S** to start a session
3. The system monitors posture, distance, and focus in real-time
4. Press **S** again to stop — a session summary prints to console
5. CSV log saved automatically in `logs/`

---

## 📊 Scoring System

### Productivity Score (0–100)

| Condition | Rate |
|-----------|------|
| Bad posture | **−0.3/sec** (BAD) or **−0.2/sec** (WARNING) |
| Looking away | **−0.5/sec** (AWAY) or **−0.25/sec** (DISTRACTED) |
| Too close to screen | **−0.3/sec** |
| All good (recovery) | **+0.1/sec** |

### Health Score

```
Health Score = (Good Posture Time / Total Session Time) × 100%
```

### Fatigue Level

| Level | Condition |
|-------|-----------|
| LOW | Bad posture < 25% of session |
| MODERATE | Bad posture 25–50% of session |
| HIGH | Bad posture > 50% of session |

---

## 📁 Project Structure

```
AI_Posture_Detection/
│
├── main.py                 # Application entry point & orchestrator
├── posture_detector.py     # PostureDetector class
├── distance_detector.py    # DistanceEstimator class
├── focus_detector.py       # FocusDetector class
├── scoring_system.py       # ScoreManager class
├── analytics.py            # SessionAnalytics (CSV logging)
├── utils.py                # Math utilities & drawing helpers
├── config.py               # Central configuration & thresholds
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── models/                 # Reserved for ML model files
├── data/                   # Reserved for training data
└── logs/                   # Session CSV logs (auto-created)
```

---

## ⚙️ Configuration

All thresholds and settings are centralised in `config.py`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NECK_ANGLE_GOOD` | 160° | Above = good posture |
| `NECK_ANGLE_WARNING` | 150° | Below = bad posture |
| `DISTANCE_TOO_CLOSE` | 40 cm | Too close threshold |
| `DISTANCE_TOO_FAR` | 90 cm | Too far threshold |
| `HEAD_TURN_THRESHOLD` | 25° | Distraction threshold |
| `FACE_ABSENT_AWAY_SEC` | 10s | No face = AWAY |
| `SMOOTHING_WINDOW_SIZE` | 5 | Moving average window |
| `BREAK_REMINDER_INTERVAL_SEC` | 1800s | Break reminder (30 min) |

---

## 🎬 Demo Mode

To test without a session (preview mode):
1. Run `python main.py`
2. The webcam feed shows skeleton overlay and basic status
3. Press **S** to start scoring and analytics

To test specific features:
- **Bad posture**: Lean forward / slouch — watch the skeleton turn red
- **Distance**: Move closer / further from the camera
- **Focus**: Turn your head left/right past 25° or look away for 10+ seconds

---

## 🔮 Future Improvements

- [ ] **GUI Dashboard** — PyQt5/Tkinter panel with graphs and controls
- [ ] **ML Posture Classifier** — Train a scikit-learn model on labelled posture data
- [ ] **Multi-person tracking** — Support multiple users
- [ ] **Historical analytics** — Dashboard with session-over-session trends
- [ ] **REST API** — FastAPI backend for remote monitoring
- [ ] **Notification integrations** — Desktop notifications, Slack webhooks
- [ ] **Calibration wizard** — Guided setup for accurate distance measurement
- [ ] **Eye tracking** — Gaze direction estimation using iris landmarks
- [ ] **Exercise suggestions** — Recommend stretches based on detected issues

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

**Built with ❤️ using MediaPipe, OpenCV, and Python**

*A portfolio project demonstrating Computer Vision, Human Pose Estimation, Real-Time AI Analysis, and Behavioural Scoring.*

</div>
]]>
