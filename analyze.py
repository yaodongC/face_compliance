"""Offline analysis pass: sample frame-windows, query the VLM (or stub),
accumulate compliance, and cache analysis.json for the GUI."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import yaml
from compliance import ComplianceTracker, load_regulation
from vlm_client import analyze_window
import stub_script


def sample_windows(video_path, sampling_sec, window_frames):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(sampling_sec * fps)))
    span = max(1, int(fps))  # gather window frames across ~1s before the mark
    for center in range(0, total, step):
        frames = []
        gap = max(1, span // window_frames)
        idxs = [max(0, center - (window_frames - 1) * gap + k * gap)
                for k in range(window_frames)]
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if ok:
                frames.append(fr)
        if frames:
            yield center / fps, frames
    cap.release()


def run_analysis(cfg, *, stub=False) -> dict:
    items = load_regulation(cfg["paths"]["regulation"])
    tracker = ComplianceTracker(
        items,
        confirm_satisfied=cfg.get("confirm_satisfied", 1),
        confirm_violation=cfg.get("confirm_violation", 2),
        confirm_clear=cfg.get("confirm_clear", 2),
        min_severity=cfg.get("min_severity", "med"))
    steps = []
    for i, (t_sec, frames) in enumerate(
            sample_windows(cfg["paths"]["video"], cfg["sampling_sec"], cfg["window_frames"])):
        if stub:
            r = stub_script.stub_response(i)
        else:
            r = analyze_window(frames, cfg)
        tracker.update(t_sec, r.get("observations", []), r.get("safety_flags", []))
        steps.append({
            "t_sec": round(t_sec, 3),
            "narration": r.get("narration", ""),
            "current_activity": r.get("current_activity", "other"),
            "item_updates": r.get("observations", []),
            "safety_flags": r.get("safety_flags", []),
            "checklist_snapshot": tracker.snapshot(),
            "verdict": tracker.verdict(),
        })
        print(f"[analyze] t={t_sec:6.1f}s verdict={steps[-1]['verdict']}  {steps[-1]['narration'][:60]}")
    out = {"meta": {"video": cfg["paths"]["video"], "model": cfg["model"],
                    "sampling_sec": cfg["sampling_sec"], "window": cfg["window_frames"],
                    "items": [{"id": it.id, "kind": it.kind, "label": it.label} for it in items]},
           "steps": steps}
    Path(cfg["paths"]["analysis"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["analysis"]).write_text(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stub", action="store_true", help="use canned responses (no model)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run_analysis(cfg, stub=args.stub)
    print(f"[analyze] wrote {cfg['paths']['analysis']}")


if __name__ == "__main__":
    main()
