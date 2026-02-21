"""
Module 4: PyQt6 GUI — AI Interviewer Interface
================================================
Three screens:
  1. Landing       — Resume upload + job role selector
  2. Interview     — Live waveform + transcript + phase tracker
  3. Report        — Radar chart, behavioral scores, recommendations

Aesthetic: Dark glassmorphic terminal-meets-dashboard
Colors: Deep navy (#0a0e1a), electric cyan (#00d4ff), 
        amber accent (#f59e0b), glass panels (#ffffff10)
Fonts: JetBrains Mono (monospace), Outfit (sans)
"""

from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import (
    Qt, QThread, QTimer, pyqtSignal, QObject, QPropertyAnimation,
    QEasingCurve, QRect, QSize, pyqtProperty
)
from PyQt6.QtGui import (
    QColor, QFont, QPalette, QLinearGradient, QPainter, QPen,
    QBrush, QPixmap, QDragEnterEvent, QDropEvent, QPainterPath,
    QFontDatabase, QIcon
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QFileDialog, QScrollArea,
    QFrame, QStackedWidget, QProgressBar, QSizePolicy, QSplitter,
    QTextEdit, QGraphicsDropShadowEffect
)

logger = logging.getLogger(__name__)

# ─── Color Palette ────────────────────────────────────────────────────────────

COLORS = {
    "bg_dark": "#040812",
    "bg_panel": "#0c1220",
    "bg_glass": "rgba(255, 255, 255, 12)",
    "border": "rgba(0, 212, 255, 20)",
    "border_active": "#00d4ff",
    "cyan": "#00d4ff",
    "cyan_dim": "#0099bb",
    "amber": "#f59e0b",
    "green": "#10b981",
    "red": "#ef4444",
    "purple": "#8b5cf6",
    "text_primary": "#e2e8f0",
    "text_secondary": "#64748b",
    "text_accent": "#00d4ff",
}

STYLESHEET = """
QMainWindow, QWidget {
    background-color: #040812;
    color: #e2e8f0;
    font-family: 'Outfit', 'Segoe UI', sans-serif;
}

QLabel { color: #e2e8f0; }
QLabel#heading { 
    font-size: 28px; font-weight: 700; 
    color: #e2e8f0; letter-spacing: 1px;
}
QLabel#subheading { 
    font-size: 14px; color: #64748b; letter-spacing: 0.5px;
}
QLabel#accent { color: #00d4ff; font-weight: 600; }
QLabel#mono {
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
    font-size: 12px; color: #00d4ff;
}

QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #00d4ff, stop:1 #0099bb);
    color: #040812; font-weight: 700; font-size: 14px;
    border: none; border-radius: 8px; padding: 12px 28px;
    letter-spacing: 0.5px;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #33ddff, stop:1 #00aacc);
}
QPushButton:pressed { background: #007799; }
QPushButton:disabled { background: #1e293b; color: #475569; }

QPushButton#danger {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #ef4444, stop:1 #dc2626);
    color: white;
}
QPushButton#ghost {
    background: transparent; color: #00d4ff;
    border: 1px solid rgba(0, 212, 255, 40);
}
QPushButton#ghost:hover { border-color: #00d4ff; background: rgba(0, 212, 255, 10); }

QComboBox {
    background: #0c1220; border: 1px solid rgba(0, 212, 255, 30);
    border-radius: 8px; padding: 10px 16px; color: #e2e8f0;
    font-size: 13px; min-width: 200px;
}
QComboBox:hover { border-color: rgba(0, 212, 255, 80); }
QComboBox::drop-down { border: none; width: 30px; }
QComboBox::down-arrow { 
    image: none; border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid #00d4ff; margin-right: 10px;
}
QComboBox QAbstractItemView {
    background: #0c1220; border: 1px solid rgba(0, 212, 255, 40);
    selection-background-color: rgba(0, 212, 255, 20);
    color: #e2e8f0;
}

QScrollBar:vertical {
    background: transparent; width: 6px; margin: 0;
}
QScrollBar::handle:vertical {
    background: rgba(0, 212, 255, 30); border-radius: 3px; min-height: 20px;
}
QScrollBar::handle:vertical:hover { background: rgba(0, 212, 255, 60); }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QTextEdit {
    background: #060d1a; border: 1px solid rgba(0, 212, 255, 15);
    border-radius: 8px; color: #cbd5e1; font-size: 13px;
    padding: 12px; line-height: 1.6;
    font-family: 'JetBrains Mono', monospace;
}

QProgressBar {
    background: #0c1220; border: none; border-radius: 4px; height: 6px;
}
QProgressBar::chunk { background: #00d4ff; border-radius: 4px; }

QFrame#glass {
    background: rgba(12, 18, 32, 180);
    border: 1px solid rgba(0, 212, 255, 15);
    border-radius: 12px;
}
QFrame#glass_active {
    background: rgba(12, 18, 32, 200);
    border: 1px solid rgba(0, 212, 255, 60);
    border-radius: 12px;
}
"""


# ─── Reusable Glass Card Widget ───────────────────────────────────────────────

class GlassCard(QFrame):
    def __init__(self, parent=None, active=False):
        super().__init__(parent)
        self.setObjectName("glass_active" if active else "glass")
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 212, 255, 30))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

    def set_active(self, active: bool):
        self.setObjectName("glass_active" if active else "glass")
        self.style().unpolish(self)
        self.style().polish(self)


class MetricBadge(QWidget):
    """Compact metric display: value + label."""
    def __init__(self, label: str, value: str = "—", color: str = COLORS["cyan"], parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)

        self.value_label = QLabel(value)
        self.value_label.setFont(QFont("JetBrains Mono", 22, QFont.Weight.Bold))
        self.value_label.setStyleSheet(f"color: {color};")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_text = QLabel(label.upper())
        self.label_text.setStyleSheet("color: #475569; font-size: 10px; letter-spacing: 1px;")
        self.label_text.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.value_label)
        layout.addWidget(self.label_text)

    def update_value(self, value: str, color: str = None):
        self.value_label.setText(value)
        if color:
            self.value_label.setStyleSheet(f"color: {color};")


# ─── Live Waveform Widget ─────────────────────────────────────────────────────

class WaveformWidget(QWidget):
    """Animated waveform display for microphone input level."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._samples = np.zeros(120)
        self._is_active = False
        self._phase = 0.0

        self._timer = QTimer()
        self._timer.timeout.connect(self._animate)
        self._timer.start(40)  # 25fps

    def push_level(self, rms: float):
        """Update with current audio energy (0–1)."""
        self._samples = np.roll(self._samples, -1)
        self._samples[-1] = rms
        self._is_active = rms > 0.01
        self.update()

    def _animate(self):
        if not self._is_active:
            self._phase += 0.05  # Idle animation
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        mid = h // 2

        # Background
        painter.fillRect(0, 0, w, h, QColor(6, 13, 26))

        if not self._is_active:
            # Idle pulse (sinusoidal)
            self._draw_idle_wave(painter, w, h, mid)
        else:
            # Live audio waveform
            self._draw_audio_wave(painter, w, h, mid)

    def _draw_idle_wave(self, painter, w, h, mid):
        pen = QPen(QColor(0, 212, 255, 40))
        pen.setWidth(1)
        painter.setPen(pen)
        step = w / 60
        for i in range(60):
            x = int(i * step)
            y = mid + int(np.sin(i * 0.2 + self._phase) * 6)
            painter.drawEllipse(x, y, 2, 2)

    def _draw_audio_wave(self, painter, w, h, mid):
        n = len(self._samples)
        step = w / n

        # Glow effect — draw 3 layers
        for glow_alpha, glow_width, glow_mult in [(15, 6, 1.2), (40, 3, 1.0), (200, 1, 0.9)]:
            pen = QPen(QColor(0, 212, 255, glow_alpha))
            pen.setWidth(glow_width)
            painter.setPen(pen)

            path = QPainterPath()
            for i, sample in enumerate(self._samples):
                x = i * step
                amp = sample * mid * 0.85 * glow_mult
                y_top = mid - amp
                y_bot = mid + amp
                if i == 0:
                    path.moveTo(x, y_top)
                else:
                    path.lineTo(x, y_top)

            # Mirror bottom
            for i in range(n - 1, -1, -1):
                sample = self._samples[i]
                x = i * step
                amp = sample * mid * 0.85 * glow_mult
                path.lineTo(x, mid + amp)

            path.closeSubpath()
            painter.drawPath(path)


# ─── Phase Progress Bar ───────────────────────────────────────────────────────

class PhaseTrackerWidget(QWidget):
    """Horizontal phase progress indicator."""

    PHASES = ["INTRO", "WARM UP", "TECHNICAL", "BEHAVIORAL", "DEEP DIVE", "WRAP UP"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self._current = 0

    def set_phase(self, phase_name: str):
        name_map = {
            "INTRO": 0, "WARM_UP": 1, "TECHNICAL_CORE": 2,
            "BEHAVIORAL": 3, "SKILL_PROBE": 4, "WRAP_UP": 5
        }
        self._current = name_map.get(phase_name.upper(), 0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        n = len(self.PHASES)
        seg_w = w / n

        for i, phase in enumerate(self.PHASES):
            x = int(i * seg_w)
            is_done = i < self._current
            is_active = i == self._current

            # Connector line
            if i < n - 1:
                color = QColor(0, 212, 255, 120 if is_done else 25)
                painter.setPen(QPen(color, 1))
                painter.drawLine(int(x + seg_w * 0.55), h // 2,
                                 int(x + seg_w * 0.95), h // 2)

            # Circle
            cx = int(x + seg_w * 0.5)
            cy = h // 2

            if is_active:
                painter.setBrush(QBrush(QColor(0, 212, 255)))
                painter.setPen(QPen(QColor(0, 212, 255), 2))
                painter.drawEllipse(cx - 7, cy - 7, 14, 14)
            elif is_done:
                painter.setBrush(QBrush(QColor(16, 185, 129)))
                painter.setPen(QPen(QColor(16, 185, 129), 1))
                painter.drawEllipse(cx - 5, cy - 5, 10, 10)
            else:
                painter.setBrush(QBrush(QColor(30, 41, 59)))
                painter.setPen(QPen(QColor(71, 85, 105), 1))
                painter.drawEllipse(cx - 5, cy - 5, 10, 10)

            # Label
            font = QFont("Outfit", 8)
            painter.setFont(font)
            painter.setPen(QColor(0, 212, 255) if is_active else
                           QColor(100, 116, 139) if not is_done else QColor(16, 185, 129))
            painter.drawText(int(x), h - 2, int(seg_w), 12,
                             Qt.AlignmentFlag.AlignHCenter, phase)


# ─── Landing Screen ───────────────────────────────────────────────────────────

class LandingScreen(QWidget):
    start_interview = pyqtSignal(str, str)  # file_path, job_role

    def __init__(self, parent=None):
        super().__init__(parent)
        self._file_path: Optional[str] = None
        self._setup_ui()
        self.setAcceptDrops(True)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(80, 60, 80, 60)
        layout.setSpacing(0)

        # ── Header ──────────────────────────────────────────────────────
        header = QHBoxLayout()
        logo_label = QLabel("◈ AI INTERVIEWER")
        logo_label.setStyleSheet(
            "font-family: 'JetBrains Mono', monospace; font-size: 16px;"
            "color: #00d4ff; font-weight: 700; letter-spacing: 3px;"
        )
        version_label = QLabel("v2.0 · BETA")
        version_label.setStyleSheet("color: #1e3a5f; font-size: 11px; letter-spacing: 2px;")
        header.addWidget(logo_label)
        header.addStretch()
        header.addWidget(version_label)
        layout.addLayout(header)
        layout.addSpacing(60)

        # ── Hero text ────────────────────────────────────────────────────
        hero_card = GlassCard()
        hero_layout = QVBoxLayout(hero_card)
        hero_layout.setContentsMargins(60, 50, 60, 50)
        hero_layout.setSpacing(16)

        title = QLabel("Your AI Interview\nSession Begins Here")
        title.setStyleSheet(
            "font-size: 42px; font-weight: 800; color: #e2e8f0; line-height: 1.2;"
        )
        title.setWordWrap(True)

        subtitle = QLabel(
            "Upload your resume · Select your target role · Begin your adaptive voice viva"
        )
        subtitle.setStyleSheet("font-size: 15px; color: #475569; letter-spacing: 0.3px;")

        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)
        hero_layout.addSpacing(32)

        # ── File Drop Zone ────────────────────────────────────────────
        self.drop_zone = DropZone()
        self.drop_zone.file_selected.connect(self._on_file_selected)
        hero_layout.addWidget(self.drop_zone)
        hero_layout.addSpacing(20)

        # ── Job Role Row ──────────────────────────────────────────────
        role_row = QHBoxLayout()
        role_label = QLabel("Target Role:")
        role_label.setStyleSheet("color: #94a3b8; font-size: 13px;")

        self.role_combo = QComboBox()
        roles = [
            "Software Engineer", "ML Engineer", "Data Scientist",
            "Frontend Engineer", "Backend Engineer", "DevOps Engineer",
            "Full Stack Engineer"
        ]
        for r in roles:
            self.role_combo.addItem(r)

        role_row.addWidget(role_label)
        role_row.addWidget(self.role_combo)
        role_row.addStretch()
        hero_layout.addLayout(role_row)
        hero_layout.addSpacing(28)

        # ── Start Button ──────────────────────────────────────────────
        self.start_btn = QPushButton("BEGIN INTERVIEW  →")
        self.start_btn.setFixedHeight(52)
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._on_start)
        hero_layout.addWidget(self.start_btn)

        layout.addWidget(hero_card)
        layout.addStretch()

        # ── Status bar ─────────────────────────────────────────────────
        self.status_label = QLabel("Upload a PDF or DOCX resume to get started")
        self.status_label.setStyleSheet("color: #334155; font-size: 12px; font-family: monospace;")
        layout.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignCenter)

    def _on_file_selected(self, path: str):
        self._file_path = path
        self.start_btn.setEnabled(True)
        fname = Path(path).name
        self.status_label.setText(f"✓ Loaded: {fname}")
        self.status_label.setStyleSheet("color: #10b981; font-size: 12px; font-family: monospace;")

    def _on_start(self):
        if self._file_path:
            self.start_interview.emit(self._file_path, self.role_combo.currentText())

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith((".pdf", ".docx")):
                self.drop_zone.file_selected.emit(path)
                break


class DropZone(QWidget):
    file_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(130)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAcceptDrops(True)
        self._hover = False

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.icon_label = QLabel("⬆")
        self.icon_label.setStyleSheet("font-size: 32px; color: #1e3a5f;")
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.text_label = QLabel("Drop resume here or click to browse")
        self.text_label.setStyleSheet("color: #475569; font-size: 13px;")
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.type_label = QLabel("PDF · DOCX")
        self.type_label.setStyleSheet(
            "color: #1e3a5f; font-size: 10px; letter-spacing: 2px; "
            "font-family: monospace;"
        )
        self.type_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label)
        layout.addWidget(self.type_label)

    def mousePressEvent(self, event):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Resume", "", "Documents (*.pdf *.docx *.doc)"
        )
        if path:
            self.file_selected.emit(path)
            self.icon_label.setText("✓")
            self.icon_label.setStyleSheet("font-size: 32px; color: #10b981;")
            self.text_label.setText(Path(path).name)
            self.text_label.setStyleSheet("color: #10b981; font-size: 13px;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        border_color = QColor(0, 212, 255, 60 if self._hover else 25)
        bg_color = QColor(0, 212, 255, 8 if self._hover else 3)

        painter.setBrush(QBrush(bg_color))
        painter.setPen(QPen(border_color, 1, Qt.PenStyle.DashLine))
        painter.drawRoundedRect(1, 1, w - 2, h - 2, 10, 10)

    def enterEvent(self, event):
        self._hover = True
        self.update()

    def leaveEvent(self, event):
        self._hover = False
        self.update()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            self._hover = True
            self.update()
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self._hover = False
        self.update()

    def dropEvent(self, event):
        self._hover = False
        self.update()
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith((".pdf", ".docx")):
                self.file_selected.emit(path)
                break


# ─── Interview Screen ─────────────────────────────────────────────────────────

class InterviewScreen(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # ── Top Bar ───────────────────────────────────────────────────
        top_bar = QHBoxLayout()

        self.candidate_label = QLabel("CANDIDATE: —")
        self.candidate_label.setStyleSheet(
            "font-family: 'JetBrains Mono', monospace; font-size: 12px; "
            "color: #00d4ff; letter-spacing: 2px;"
        )

        self.timer_label = QLabel("00:00")
        self.timer_label.setStyleSheet(
            "font-family: 'JetBrains Mono', monospace; font-size: 20px; "
            "color: #f59e0b; font-weight: 700;"
        )

        self.phase_badge = QLabel("● INTRO")
        self.phase_badge.setStyleSheet(
            "font-size: 11px; color: #00d4ff; letter-spacing: 2px; "
            "background: rgba(0,212,255,12); padding: 4px 10px; border-radius: 4px;"
        )

        top_bar.addWidget(self.candidate_label)
        top_bar.addStretch()
        top_bar.addWidget(self.phase_badge)
        top_bar.addSpacing(16)
        top_bar.addWidget(self.timer_label)
        layout.addLayout(top_bar)

        # ── Phase Tracker ──────────────────────────────────────────────
        self.phase_tracker = PhaseTrackerWidget()
        layout.addWidget(self.phase_tracker)

        # ── Main Content Row ───────────────────────────────────────────
        content_row = QHBoxLayout()
        content_row.setSpacing(16)

        # Left: AI Question + Waveform
        left_col = QVBoxLayout()
        left_col.setSpacing(12)

        ai_card = GlassCard()
        ai_layout = QVBoxLayout(ai_card)
        ai_layout.setContentsMargins(20, 16, 20, 16)
        ai_label = QLabel("AI INTERVIEWER")
        ai_label.setStyleSheet(
            "font-size: 10px; color: #1e3a5f; letter-spacing: 3px; font-family: monospace;"
        )
        self.ai_text = QLabel("Initializing session...")
        self.ai_text.setWordWrap(True)
        self.ai_text.setStyleSheet(
            "font-size: 16px; color: #e2e8f0; line-height: 1.7; padding: 8px 0;"
        )
        self.ai_text.setMinimumHeight(100)
        ai_layout.addWidget(ai_label)
        ai_layout.addWidget(self.ai_text)
        left_col.addWidget(ai_card)

        # Waveform card
        wave_card = GlassCard()
        wave_layout = QVBoxLayout(wave_card)
        wave_layout.setContentsMargins(16, 12, 16, 12)
        wave_header = QHBoxLayout()
        wave_title = QLabel("MICROPHONE")
        wave_title.setStyleSheet("font-size: 10px; color: #1e3a5f; letter-spacing: 3px; font-family: monospace;")
        self.mic_status = QLabel("● LISTENING")
        self.mic_status.setStyleSheet("font-size: 10px; color: #10b981; letter-spacing: 1px;")
        wave_header.addWidget(wave_title)
        wave_header.addStretch()
        wave_header.addWidget(self.mic_status)
        self.waveform = WaveformWidget()
        wave_layout.addLayout(wave_header)
        wave_layout.addWidget(self.waveform)
        left_col.addWidget(wave_card)

        # Metrics row
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(8)
        self.metric_confidence = MetricBadge("Confidence", "—%", COLORS["cyan"])
        self.metric_wpm = MetricBadge("WPM", "—", COLORS["amber"])
        self.metric_stutter = MetricBadge("Disfluency", "—/min", COLORS["green"])
        self.metric_phase = MetricBadge("Q Count", "0", COLORS["purple"])

        for m in [self.metric_confidence, self.metric_wpm, self.metric_stutter, self.metric_phase]:
            card = GlassCard()
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(0, 0, 0, 0)
            card_layout.addWidget(m)
            metrics_row.addWidget(card)

        left_col.addLayout(metrics_row)
        content_row.addLayout(left_col, 3)

        # Right: Live Transcript
        right_col = QVBoxLayout()
        right_col.setSpacing(12)

        transcript_card = GlassCard()
        t_layout = QVBoxLayout(transcript_card)
        t_layout.setContentsMargins(16, 12, 16, 16)
        t_header = QHBoxLayout()
        t_title = QLabel("LIVE TRANSCRIPT")
        t_title.setStyleSheet("font-size: 10px; color: #1e3a5f; letter-spacing: 3px; font-family: monospace;")
        self.transcript_count = QLabel("0 exchanges")
        self.transcript_count.setStyleSheet("font-size: 10px; color: #334155;")
        t_header.addWidget(t_title)
        t_header.addStretch()
        t_header.addWidget(self.transcript_count)

        self.transcript_feed = QTextEdit()
        self.transcript_feed.setReadOnly(True)
        self.transcript_feed.setPlaceholderText("Transcript will appear here...")
        self.transcript_feed.setMinimumHeight(280)

        t_layout.addLayout(t_header)
        t_layout.addWidget(self.transcript_feed)
        right_col.addWidget(transcript_card, 1)

        # End Interview Button
        self.end_btn = QPushButton("END INTERVIEW")
        self.end_btn.setObjectName("danger")
        self.end_btn.setFixedHeight(44)
        right_col.addWidget(self.end_btn)

        content_row.addLayout(right_col, 2)
        layout.addLayout(content_row)

        # ── Timer ──────────────────────────────────────────────────────
        self._start_time = time.time()
        self._exchange_count = 0
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_timer)
        self._timer.start(1000)

    def _update_timer(self):
        elapsed = int(time.time() - self._start_time)
        m, s = divmod(elapsed, 60)
        self.timer_label.setText(f"{m:02d}:{s:02d}")

    def set_candidate(self, name: str, role: str):
        self.candidate_label.setText(f"CANDIDATE: {name.upper()}  ·  {role.upper()}")
        self._start_time = time.time()

    def set_phase(self, phase_name: str):
        self.phase_badge.setText(f"● {phase_name.replace('_', ' ')}")
        self.phase_tracker.set_phase(phase_name)

    def append_ai_text(self, chunk: str):
        """Append streamed text chunk to AI display."""
        current = self.ai_text.text()
        if current == "Initializing session...":
            current = ""
        self.ai_text.setText(current + chunk)

    def set_ai_text_complete(self, full_text: str):
        self.ai_text.setText(full_text)

    def append_transcript(self, speaker: str, text: str, is_interim: bool = False):
        if is_interim:
            cursor = self.transcript_feed.textCursor()
            # Replace last line if it's an interim result
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.select(cursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
            color = "#475569"
            self.transcript_feed.append(
                f'<span style="color:{color}; font-style:italic;">{text}</span>'
            )
        else:
            if speaker == "AI":
                color = COLORS["cyan"]
                prefix = "◈ AI"
            else:
                color = "#94a3b8"
                prefix = "▸ YOU"
            self.transcript_feed.append(
                f'<br><b style="color:{color}; font-size:10px; '
                f'letter-spacing:1px;">{prefix}</b><br>'
                f'<span style="color:#cbd5e1;">{text}</span>'
            )
            self._exchange_count += 1
            self.transcript_feed.verticalScrollBar().setValue(
                self.transcript_feed.verticalScrollBar().maximum()
            )
            self.transcript_count.setText(f"{self._exchange_count} exchanges")

    def update_behavioral(self, snapshot):
        """Update live metrics from behavioral analyzer snapshot."""
        conf = snapshot.confidence_index
        color = (COLORS["green"] if conf >= 70 else
                 COLORS["amber"] if conf >= 45 else COLORS["red"])
        self.metric_confidence.update_value(f"{conf:.0f}%", color)
        self.metric_wpm.update_value(f"{snapshot.wpm:.0f}")
        self.metric_stutter.update_value(f"{snapshot.stutter_events_per_min:.1f}/m")
        self.waveform.push_level(min(1.0, snapshot.voice_energy / 100))

    def update_question_count(self, count: int):
        self.metric_phase.update_value(str(count))


# ─── Report Screen ────────────────────────────────────────────────────────────

class ReportScreen(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(16)

        # Header
        header = QHBoxLayout()
        title = QLabel("◈ INTERVIEW REPORT")
        title.setStyleSheet(
            "font-family: 'JetBrains Mono', monospace; font-size: 18px; "
            "color: #00d4ff; font-weight: 700; letter-spacing: 3px;"
        )
        self.export_btn = QPushButton("EXPORT PDF")
        self.export_btn.setObjectName("ghost")
        self.export_btn.setFixedWidth(130)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.export_btn)
        outer.addLayout(header)

        # Scroll area for report content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        content = QWidget()
        scroll.setWidget(content)
        self.report_layout = QVBoxLayout(content)
        self.report_layout.setSpacing(16)
        outer.addWidget(scroll)

    def populate_report(self, behavior_report, session_data: dict, skill_matches: list):
        """Render the full report with charts and metrics."""
        layout = self.report_layout

        # Clear existing
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # ── Score Summary Row ─────────────────────────────────────────
        scores_card = GlassCard()
        scores_layout = QHBoxLayout(scores_card)
        scores_layout.setContentsMargins(24, 20, 24, 20)

        score_metrics = [
            ("Overall\nReadiness", f"{session_data.get('readiness_score', 0):.0f}",
             COLORS["cyan"], "%"),
            ("Communication\nScore", f"{behavior_report.communication_score:.0f}",
             COLORS["green"], "%"),
            ("Fluency\nScore", f"{behavior_report.fluency_score:.0f}",
             COLORS["amber"], "%"),
            ("Confidence\nIndex", f"{behavior_report.final_confidence_score:.0f}",
             COLORS["purple"], "%"),
            ("Avg Speed",
             f"{behavior_report.average_wpm:.0f}", COLORS["text_primary"], " wpm"),
        ]

        for label, val, color, unit in score_metrics:
            badge_widget = QWidget()
            badge_layout = QVBoxLayout(badge_widget)
            badge_layout.setContentsMargins(12, 0, 12, 0)

            val_label = QLabel(f"{val}{unit}")
            val_label.setStyleSheet(
                f"font-family: 'JetBrains Mono', monospace; font-size: 28px; "
                f"font-weight: 800; color: {color};"
            )
            val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            lbl = QLabel(label)
            lbl.setStyleSheet("font-size: 11px; color: #475569; letter-spacing: 0.5px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setWordWrap(True)

            badge_layout.addWidget(val_label)
            badge_layout.addWidget(lbl)
            scores_layout.addWidget(badge_widget)

        layout.addWidget(scores_card)

        # ── Skill Readiness Chart ─────────────────────────────────────
        if skill_matches:
            self._add_skill_chart(layout, skill_matches)

        # ── Behavioral Timeline ────────────────────────────────────────
        if behavior_report.snapshots:
            self._add_confidence_chart(layout, behavior_report)

        # ── Recommendations ───────────────────────────────────────────
        self._add_recommendations(layout, behavior_report.recommendations)

    def _add_skill_chart(self, layout, skill_matches):
        """Add horizontal bar chart of skill readiness scores."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

            fig, ax = plt.subplots(figsize=(9, 0.55 * min(len(skill_matches), 10) + 1.5))
            fig.patch.set_facecolor("#0c1220")
            ax.set_facecolor("#060d1a")

            skills = [m.skill for m in skill_matches[:10]][::-1]
            scores = [m.score * 100 for m in skill_matches[:10]][::-1]
            colors = ["#10b981" if s >= 60 else "#f59e0b" if s >= 35 else "#ef4444"
                      for s in scores]

            bars = ax.barh(skills, scores, color=colors, height=0.6, alpha=0.9)
            ax.set_xlim(0, 100)
            ax.axvline(x=60, color="rgba(0,212,255,0.3)", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.set_xlabel("Match Score (%)", color="#64748b", fontsize=10)
            ax.tick_params(colors="#64748b", labelsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_color("#1e293b")
            ax.set_title("SKILL READINESS", color="#00d4ff", fontsize=11,
                         fontfamily="monospace", pad=12, loc="left", fontweight="bold")

            for bar, score in zip(bars, scores):
                ax.text(min(score + 2, 95), bar.get_y() + bar.get_height() / 2,
                        f"{score:.0f}%", va="center", color="#94a3b8", fontsize=8)

            plt.tight_layout()

            card = GlassCard()
            card_layout = QVBoxLayout(card)
            canvas = FigureCanvasQTAgg(fig)
            card_layout.addWidget(canvas)
            layout.addWidget(card)
            plt.close(fig)

        except Exception as e:
            logger.warning(f"Skill chart render error: {e}")
            # Fallback text display
            card = GlassCard()
            cl = QVBoxLayout(card)
            lbl = QLabel("SKILL ANALYSIS")
            lbl.setStyleSheet("font-size: 11px; color: #1e3a5f; letter-spacing: 3px; font-family: monospace;")
            cl.addWidget(lbl)
            for m in skill_matches[:8]:
                row = QHBoxLayout()
                skill_lbl = QLabel(m.skill)
                skill_lbl.setStyleSheet("color: #94a3b8; font-size: 12px;")
                score_lbl = QLabel(f"{m.score*100:.0f}%")
                color = "#10b981" if m.score >= 0.6 else "#f59e0b" if m.score >= 0.35 else "#ef4444"
                score_lbl.setStyleSheet(f"color: {color}; font-family: monospace; font-size: 12px;")
                row.addWidget(skill_lbl)
                row.addStretch()
                row.addWidget(score_lbl)
                cl.addLayout(row)
            layout.addWidget(card)

    def _add_confidence_chart(self, layout, behavior_report):
        """Add confidence timeline chart."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

            snaps = behavior_report.snapshots
            times = [(s.timestamp - snaps[0].timestamp) for s in snaps]
            confidence = [s.confidence_index for s in snaps]
            wpm = [s.wpm for s in snaps]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
            fig.patch.set_facecolor("#0c1220")

            for ax in [ax1, ax2]:
                ax.set_facecolor("#060d1a")
                for spine in ax.spines.values():
                    spine.set_color("#1e293b")
                ax.tick_params(colors="#64748b", labelsize=9)

            # Confidence
            ax1.fill_between(times, confidence, alpha=0.15, color="#00d4ff")
            ax1.plot(times, confidence, color="#00d4ff", linewidth=1.5)
            ax1.set_ylabel("Confidence %", color="#64748b", fontsize=9)
            ax1.set_ylim(0, 100)
            ax1.axhline(60, color="#1e293b", linewidth=0.8, linestyle="--")
            ax1.set_title("BEHAVIORAL TIMELINE", color="#00d4ff", fontsize=11,
                          fontfamily="monospace", pad=10, loc="left", fontweight="bold")

            # WPM
            ax2.fill_between(times, wpm, alpha=0.15, color="#f59e0b")
            ax2.plot(times, wpm, color="#f59e0b", linewidth=1.5)
            ax2.set_ylabel("WPM", color="#64748b", fontsize=9)
            ax2.set_xlabel("Time (seconds)", color="#64748b", fontsize=9)
            ax2.axhline(140, color="#1e293b", linewidth=0.8, linestyle="--")

            plt.tight_layout()

            card = GlassCard()
            card_layout = QVBoxLayout(card)
            canvas = FigureCanvasQTAgg(fig)
            card_layout.addWidget(canvas)
            layout.addWidget(card)
            plt.close(fig)

        except Exception as e:
            logger.warning(f"Confidence chart error: {e}")

    def _add_recommendations(self, layout, recommendations: list[str]):
        card = GlassCard()
        cl = QVBoxLayout(card)
        cl.setContentsMargins(24, 20, 24, 20)
        cl.setSpacing(12)

        header = QLabel("RECOMMENDATIONS")
        header.setStyleSheet("font-size: 11px; color: #1e3a5f; letter-spacing: 3px; font-family: monospace;")
        cl.addWidget(header)

        for i, rec in enumerate(recommendations):
            icon = "●" if i == 0 else "○"
            color = COLORS["cyan"] if i == 0 else "#475569"
            lbl = QLabel(f"{icon}  {rec}")
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"font-size: 13px; color: {color}; padding: 4px 0; line-height: 1.6;")
            cl.addWidget(lbl)

        layout.addWidget(card)
        layout.addStretch()


# ─── Main Application Window ──────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Interviewer")
        self.setMinimumSize(1200, 750)
        self.resize(1360, 820)

        # Dark title bar on Windows
        self._setup_dark_titlebar()

        self.setStyleSheet(STYLESHEET)

        # Central stacked widget (screens)
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        # Create screens
        self.landing = LandingScreen()
        self.interview = InterviewScreen()
        self.report = ReportScreen()

        self._stack.addWidget(self.landing)    # index 0
        self._stack.addWidget(self.interview)  # index 1
        self._stack.addWidget(self.report)     # index 2

        # Connect navigation signals
        self.landing.start_interview.connect(self.on_start_interview)
        self.interview.end_btn.clicked.connect(self.on_end_interview)
        self.report.export_btn.clicked.connect(self.on_export_pdf)

        # App state
        self._session_worker = None
        self._behavioral_analyzer = None
        self._audio_manager = None
        self._current_ai_response = ""
        self._resume_profile = None

    def _setup_dark_titlebar(self):
        """Enable dark title bar on Windows 10+."""
        try:
            if sys.platform == "win32":
                import ctypes
                HWND_OFFSET = 0
                DWMWA_USE_IMMERSIVE_DARK_MODE = 20
                hwnd = int(self.winId())
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
                    ctypes.byref(ctypes.c_int(1)), ctypes.sizeof(ctypes.c_int)
                )
        except Exception:
            pass

    # ── Screen Navigation ──────────────────────────────────────────────────────

    def show_landing(self):
        self._stack.setCurrentIndex(0)

    def show_interview(self):
        self._stack.setCurrentIndex(1)

    def show_report(self):
        self._stack.setCurrentIndex(2)

    # ── Session Lifecycle ──────────────────────────────────────────────────────

    def on_start_interview(self, file_path: str, job_role: str):
        """
        Called when user clicks BEGIN INTERVIEW.
        Runs resume processing + spins up the session worker.
        """
        from PyQt6.QtWidgets import QProgressDialog
        import threading

        # Show progress dialog during resume processing
        progress = QProgressDialog("Analyzing resume...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setWindowTitle("Processing")
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()

        def _process():
            try:
                from core.resume_engine import ResumeIntelligenceEngine
                engine = ResumeIntelligenceEngine()
                self._resume_profile = engine.process(file_path, job_role)
                return True
            except Exception as e:
                return str(e)

        result_holder = [None]

        def _worker():
            result_holder[0] = _process()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        while t.is_alive():
            QApplication.processEvents()
            time.sleep(0.05)

        progress.close()

        if result_holder[0] is not True:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Resume Error", str(result_holder[0]))
            return

        self._launch_session()

    def _launch_session(self):
        from core.voice_pipeline import (
            InterviewSession, SessionWorker, build_system_prompt, PHASE_SEQUENCE
        )
        from core.behavioral_analyzer import BehavioralAnalyzer
        from core.audio_manager import AudioManager

        profile = self._resume_profile
        weak_skills = [
            m.skill for m in profile.skill_matches
            if m.score < 0.4
        ][:5]

        # Build interview session
        session = InterviewSession(
            candidate_name=profile.candidate_name,
            job_role=profile.job_role,
            system_prompt=build_system_prompt(profile.summary_for_llm, profile.job_role),
            weak_skills=weak_skills,
        )

        # Behavioral analyzer
        self._behavioral_analyzer = BehavioralAnalyzer(
            on_snapshot_callback=self._on_behavioral_snapshot,
            on_report_callback=self._on_behavioral_report,
        )
        self._behavioral_analyzer.start()

        # Audio manager
        self._audio_manager = AudioManager()
        self._audio_manager.initialize()

        # Session worker (QThread with asyncio event loop)
        self._session_worker = SessionWorker(session)
        self._session_worker_thread = QThread()
        self._session_worker.moveToThread(self._session_worker_thread)
        self._session_worker_thread.started.connect(self._session_worker.run)

        # Connect signals
        self._session_worker.interim_transcript.connect(self._on_interim_transcript)
        self._session_worker.final_transcript.connect(self._on_final_transcript)
        self._session_worker.ai_text_chunk.connect(self._on_ai_text_chunk)
        self._session_worker.audio_chunk_ready.connect(self._on_audio_chunk)
        self._session_worker.phase_changed.connect(self._on_phase_changed)
        self._session_worker.session_complete.connect(self._on_session_complete)
        self._session_worker.error_occurred.connect(self._on_error)

        # Register audio routing
        self._audio_manager.register_frame_callback(
            lambda b: self._session_worker.push_audio(b)
        )
        self._audio_manager.register_frame_callback(
            self._behavioral_analyzer.push_frame
        )

        # Update interview screen
        self.interview.set_candidate(profile.candidate_name, profile.job_role)
        self.interview.update_question_count(0)
        self.show_interview()

        # Start session
        self._session_worker_thread.start()

    def on_end_interview(self):
        if self._session_worker:
            self._session_worker.stop()

    def on_export_pdf(self):
        """Export the report as a PDF file."""
        try:
            from utils.pdf_exporter import ReportPDFExporter
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Report", "interview_report.pdf", "PDF Files (*.pdf)"
            )
            if path:
                exporter = ReportPDFExporter()
                exporter.export(path, self._resume_profile, self._last_behavior_report,
                                self._last_session_data)
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.information(self, "Exported", f"Report saved to:\n{path}")
        except Exception as e:
            logger.error(f"PDF export error: {e}")

    # ── Signal Handlers ────────────────────────────────────────────────────────

    def _on_interim_transcript(self, text: str):
        self.interview.append_transcript("YOU", text, is_interim=True)

    def _on_final_transcript(self, text: str):
        self.interview.append_transcript("YOU", text, is_interim=False)
        if self._behavioral_analyzer:
            self._behavioral_analyzer.update_word_count(text)

    def _on_ai_text_chunk(self, chunk: str):
        self._current_ai_response += chunk
        self.interview.append_ai_text(chunk)

    def _on_audio_chunk(self, audio_bytes: bytes):
        if self._audio_manager:
            self._audio_manager.play_audio_chunk(audio_bytes)

    def _on_phase_changed(self, phase_name: str):
        self.interview.set_phase(phase_name)
        # Clear AI text box for new phase
        self._current_ai_response = ""

    def _on_behavioral_snapshot(self, snapshot):
        self.interview.update_behavioral(snapshot)

    def _on_behavioral_report(self, report):
        self._last_behavior_report = report

    def _on_session_complete(self, session_data: dict):
        self._last_session_data = session_data
        if self._behavioral_analyzer:
            behavior_report = self._behavioral_analyzer.stop()
        else:
            behavior_report = None

        if self._audio_manager:
            self._audio_manager.shutdown()

        session_data["readiness_score"] = getattr(
            self._resume_profile, "overall_readiness", 0
        )

        # Record final AI response in transcript
        if self._current_ai_response:
            self.interview.append_transcript("AI", self._current_ai_response)

        # Build report
        if behavior_report and self._resume_profile:
            self.report.populate_report(
                behavior_report,
                session_data,
                self._resume_profile.skill_matches,
            )
        self.show_report()

    def _on_error(self, error_msg: str):
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.warning(self, "Session Error", error_msg)
        logger.error(f"Session error: {error_msg}")

    def closeEvent(self, event):
        if self._session_worker:
            self._session_worker.stop()
        if self._audio_manager:
            self._audio_manager.shutdown()
        if self._behavioral_analyzer:
            self._behavioral_analyzer.stop()
        event.accept()
