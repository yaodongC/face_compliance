"""PyQt5 replay GUI: plays the extracted MP4 and overlays the synced scene
narration, fail-safe verification checklist, and safety verdict banner.

The verdict and checklist are deliberately conservative: a small VLM cannot be
trusted to certify life-safety, so the UI shows NOT VERIFIED by default and a
prominent permanent disclaimer that this is an assistive demo, not a certified
safety system."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import cv2
import yaml
from PyQt5 import QtCore, QtGui, QtWidgets

# Per-item icons. not_verified is the honest default (grey question), verified is
# a green check, violation is a red stop.
ICON = {"not_verified": "❓", "verified": "✅", "violation": "⛔"}
ROW_COLOR = {"not_verified": "#777", "verified": "#1e8449", "violation": "#c0392b"}

# Verdict banner colours. Only SUPPORTED is green; everything else is a warning.
BANNER = {"DANGER": "#7b1113", "UNSUPPORTED": "#c0392b",
          "NOT VERIFIED": "#b8860b", "SUPPORTED": "#1e8449"}
SUBTITLE = {
    "DANGER": "Hazard observed — clear the area",
    "UNSUPPORTED": "No ground support verified — treat face as UNSUPPORTED",
    "NOT VERIFIED": "Support not confirmed — human inspection required",
    "SUPPORTED": "Mesh + bolts verified over time (assistive only — still verify)",
}
DISCLAIMER = ("⚠ ASSISTIVE DEMO — NOT A CERTIFIED SAFETY SYSTEM. A small vision model "
              "can miss or hallucinate ground support. Never enter a face or skip a "
              "physical inspection based on this display.")


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
        self.setWindowTitle("Face Support Safety Inspector — Cosmos-Reason2 / vLLM")
        self.video = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video.setMinimumSize(720, 540)
        self.video.setStyleSheet("background:#111;")

        self.banner = QtWidgets.QLabel("NOT VERIFIED", alignment=QtCore.Qt.AlignCenter)
        self.banner.setFixedHeight(58)
        self.banner.setStyleSheet("color:white;font-size:24px;font-weight:bold;background:#b8860b;")
        self.subtitle = QtWidgets.QLabel("", alignment=QtCore.Qt.AlignCenter)
        self.subtitle.setStyleSheet("color:#222;font-size:13px;padding:2px;")

        disclaimer = QtWidgets.QLabel(DISCLAIMER, wordWrap=True)
        disclaimer.setStyleSheet("background:#222;color:#ffd24d;font-size:12px;padding:6px;")

        self.scene_lbl = QtWidgets.QLabel("", wordWrap=True)
        self.scene_lbl.setStyleSheet("font-size:14px;padding:6px;")
        self.scene_lbl.setMinimumHeight(64)

        self.rows = {}
        checklist = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("Verification checklist (fail-safe)")
        title.setStyleSheet("font-size:15px;font-weight:bold;padding:4px;")
        checklist.addWidget(title)
        for it in self.items:
            row = QtWidgets.QLabel(f"{ICON['not_verified']}  {it['label']}")
            row.setWordWrap(True)
            row.setStyleSheet("font-size:13px;padding:3px;color:#777;")
            self.rows[it["id"]] = row
            checklist.addWidget(row)
        checklist.addStretch(1)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("<b>What the camera sees</b>"))
        right.addWidget(self.scene_lbl)
        rc = QtWidgets.QWidget(); rc.setLayout(checklist)
        right.addWidget(rc, 1)
        rightw = QtWidgets.QWidget(); rightw.setLayout(right); rightw.setFixedWidth(380)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.video, 1)
        top.addWidget(rightw)

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(self.banner)
        root.addWidget(self.subtitle)
        root.addLayout(top, 1)
        root.addWidget(disclaimer)

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
        scene = step.get("scene", "")
        act = step.get("activity", "none")
        hazard = step.get("hazard_note", "")
        txt = f"Activity: {act}\n{scene}"
        if hazard:
            txt += f"\n⛔ {hazard}"
        self.scene_lbl.setText(txt)

        snap = step["checklist_snapshot"]
        for it in self.items:
            st = snap.get(it["id"], "not_verified")
            self.rows[it["id"]].setText(f"{ICON.get(st, '❓')}  {it['label']}")
            weight = "font-weight:bold;" if st in ("verified", "violation") else ""
            self.rows[it["id"]].setStyleSheet(
                f"font-size:13px;padding:3px;color:{ROW_COLOR.get(st, '#777')};{weight}")

        v = step["verdict"]
        self.banner.setText(v)
        self.banner.setStyleSheet(
            f"color:white;font-size:24px;font-weight:bold;background:{BANNER.get(v, '#555')};")
        self.subtitle.setText(SUBTITLE.get(v, ""))


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
    w.resize(1180, 720)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
