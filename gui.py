"""PyQt5 replay GUI: plays the extracted MP4 and overlays the synced
narration, progressively-filling compliance checklist, and verdict banner."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import cv2
import yaml
from PyQt5 import QtCore, QtGui, QtWidgets

ICON = {"pending": "⬜", "in_progress": "\U0001f7e1",
        "satisfied": "✅", "ok": "✅", "violation": "⛔"}
BANNER = {"IN PROGRESS": "#b8860b", "AT-RISK": "#c0392b", "COMPLIANT": "#1e8449"}


class Player(QtWidgets.QWidget):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        data = json.loads(Path(cfg["paths"]["analysis"]).read_text())
        self.steps = sorted(data["steps"], key=lambda s: s["t_sec"])
        self.items = data["meta"]["items"]
        self.cap = cv2.VideoCapture(cfg["paths"]["video"])
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 15.0
        self._build_ui()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(int(1000 / self.fps))

    def _build_ui(self):
        self.setWindowTitle("Face Support Compliance — Cosmos-Reason2 / vLLM")
        self.video = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video.setMinimumSize(720, 540)
        self.video.setStyleSheet("background:#111;")
        self.banner = QtWidgets.QLabel("IN PROGRESS", alignment=QtCore.Qt.AlignCenter)
        self.banner.setFixedHeight(54)
        self.banner.setStyleSheet("color:white;font-size:22px;font-weight:bold;background:#b8860b;")
        self.narration = QtWidgets.QLabel("", wordWrap=True)
        self.narration.setStyleSheet("font-size:15px;padding:8px;")
        self.narration.setMinimumHeight(70)
        self.rows = {}
        checklist = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("Face Support Compliance")
        title.setStyleSheet("font-size:16px;font-weight:bold;padding:4px;")
        checklist.addWidget(title)
        for it in self.items:
            row = QtWidgets.QLabel(f"{ICON['pending']}  {it['label']}")
            row.setWordWrap(True)
            row.setStyleSheet("font-size:13px;padding:3px;")
            self.rows[it["id"]] = row
            checklist.addWidget(row)
        checklist.addStretch(1)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("<b>Narration</b>"))
        right.addWidget(self.narration)
        rc = QtWidgets.QWidget(); rc.setLayout(checklist)
        right.addWidget(rc, 1)
        rightw = QtWidgets.QWidget(); rightw.setLayout(right); rightw.setFixedWidth(360)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.video, 1)
        top.addWidget(rightw)
        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(self.banner)
        root.addLayout(top, 1)

    def _current_step(self, t_sec):
        cur = self.steps[0] if self.steps else None
        for s in self.steps:
            if s["t_sec"] <= t_sec:
                cur = s
            else:
                break
        return cur

    def _tick(self):
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES) / self.fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        img = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self.video.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
            self.video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        step = self._current_step(pos)
        if not step:
            return
        self.narration.setText(step["narration"])
        snap = step["checklist_snapshot"]
        for it in self.items:
            st = snap.get(it["id"], "pending")
            self.rows[it["id"]].setText(f"{ICON.get(st, '?')}  {it['label']}")
            self.rows[it["id"]].setStyleSheet(
                "font-size:13px;padding:3px;"
                + ("color:#c0392b;font-weight:bold;" if st == "violation" else ""))
        v = step["verdict"]
        self.banner.setText(v)
        self.banner.setStyleSheet(
            f"color:white;font-size:22px;font-weight:bold;background:{BANNER.get(v, '#555')};")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.check:
        data = json.loads(Path(cfg["paths"]["analysis"]).read_text())
        assert Path(cfg["paths"]["video"]).exists(), "video missing"
        print(f"[gui --check] {len(data['steps'])} steps, "
              f"final verdict={data['steps'][-1]['verdict']}, OK")
        return
    app = QtWidgets.QApplication(sys.argv)
    w = Player(cfg)
    w.resize(1120, 640)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
