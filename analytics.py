"""
╔══════════════════════════════════════════════════════════════════╗
║           AI Posture Detection — Session Analytics              ║
║  Records per-frame metrics and exports session data to CSV.     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import csv
import time
from datetime import datetime
from typing import Dict, Optional

import config as cfg
from utils import logger


class SessionAnalytics:
    """Log and persist session telemetry to CSV files."""

    CSV_HEADERS = [
        "timestamp",
        "elapsed_sec",
        "posture_status",
        "focus_status",
        "distance_status",
        "neck_angle",
        "distance_cm",
        "head_yaw",
        "productivity_score",
        "health_score",
    ]

    def __init__(self, log_dir: str = cfg.LOG_DIR):
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path: str = os.path.join(
            self._log_dir, f"session_{self._session_id}.csv"
        )
        self._start_time: float = time.time()
        self._row_count: int = 0
        self._last_log_time: float = 0.0
        self._log_interval: float = 1.0  # log once per second

        # Cumulative snapshot for summaries
        self._summary: Dict = {}

        # Create CSV with headers
        self._init_csv()
        logger.info("Session analytics initialised — %s", self._csv_path)

    # ── public API ──────────────────────────────────────────────
    def record(
        self,
        posture_status: str,
        focus_status: str,
        distance_status: str,
        neck_angle: float,
        distance_cm: float,
        head_yaw: float,
        productivity_score: float,
        health_score: float,
    ):
        """
        Record a data point. Writes to CSV at most once per second
        to avoid gigantic log files.
        """
        now = time.time()
        if now - self._last_log_time < self._log_interval:
            return

        self._last_log_time = now
        elapsed = now - self._start_time

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_sec": f"{elapsed:.1f}",
            "posture_status": posture_status,
            "focus_status": focus_status,
            "distance_status": distance_status,
            "neck_angle": f"{neck_angle:.1f}",
            "distance_cm": f"{distance_cm:.1f}",
            "head_yaw": f"{head_yaw:.1f}",
            "productivity_score": f"{productivity_score:.1f}",
            "health_score": f"{health_score:.1f}",
        }

        self._write_row(row)
        self._row_count += 1

    def get_summary(self, score_info: Dict) -> Dict:
        """
        Build an end-of-session summary.
        """
        duration = time.time() - self._start_time
        return {
            "session_id": self._session_id,
            "duration_sec": round(duration, 1),
            "csv_path": self._csv_path,
            "total_records": self._row_count,
            **score_info,
        }

    def print_summary(self, score_info: Dict):
        """Pretty-print session summary to console."""
        summary = self.get_summary(score_info)
        print("\n" + "═" * 55)
        print("  SESSION SUMMARY")
        print("═" * 55)
        print(f"  Session ID     : {summary['session_id']}")
        print(f"  Duration       : {summary['duration_sec']:.0f} seconds")
        print(f"  Productivity   : {summary.get('productivity_score', '—')}")
        print(f"  Health Score   : {summary.get('health_score', '—')}%")
        print(f"  Good Posture   : {summary.get('good_posture_time', 0):.0f}s")
        print(f"  Bad Posture    : {summary.get('bad_posture_time', 0):.0f}s")
        print(f"  Focused        : {summary.get('focused_time', 0):.0f}s")
        print(f"  Distracted     : {summary.get('distracted_time', 0):.0f}s")
        print(f"  Away           : {summary.get('away_time', 0):.0f}s")
        print(f"  Fatigue Level  : {summary.get('fatigue_level', '—')}")
        print(f"  CSV Log        : {summary['csv_path']}")
        print(f"  Total Records  : {summary['total_records']}")
        print("═" * 55 + "\n")

    def reset(self):
        """Start a new session file."""
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = os.path.join(
            self._log_dir, f"session_{self._session_id}.csv"
        )
        self._start_time = time.time()
        self._row_count = 0
        self._last_log_time = 0.0
        self._init_csv()
        logger.info("Analytics reset — new session %s", self._session_id)

    @property
    def csv_path(self) -> str:
        return self._csv_path

    # ── internals ───────────────────────────────────────────────
    def _init_csv(self):
        """Create the CSV file with headers."""
        with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
            writer.writeheader()

    def _write_row(self, row: Dict):
        """Append a single row to the CSV."""
        try:
            with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
                writer.writerow(row)
        except IOError as e:
            logger.error("Failed to write analytics row: %s", e)
