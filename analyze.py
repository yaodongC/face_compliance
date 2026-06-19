"""Offline analysis pass: sample frame-windows, query the VLM (or stub) for
grounded perception, aggregate it with the fail-safe SafetyTracker, and cache
analysis.json for the GUI."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import yaml
from compliance import SafetyTracker, load_regulation
from vlm_client import analyze_window_consensus
import stub_script


def sample_windows(video_path, sampling_sec, window_frames):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(sampling_sec * fps)))
    span = max(1, int(fps))  # gather window frames across ~1s before the mark
    try:
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
    finally:
        cap.release()


def run_analysis(cfg, *, stub=False) -> dict:
    items = load_regulation(cfg["paths"]["regulation"])
    tracker = SafetyTracker(items, support_window=cfg.get("support_window", 3),
                            hazard_confirm=cfg.get("hazard_confirm", 2))
    steps = []
    for i, (t_sec, frames) in enumerate(
            sample_windows(cfg["paths"]["video"], cfg["sampling_sec"], cfg["window_frames"])):
        perception = stub_script.stub_response(i) if stub else analyze_window_consensus(frames, cfg)
        tracker.update(t_sec, perception)
        steps.append({
            "t_sec": round(t_sec, 3),
            "scene": perception.get("scene", ""),
            "perception": perception,
            "checklist_snapshot": tracker.snapshot(),
            "verdict": tracker.verdict(),
            "hazard_note": tracker.hazard_note(),
        })
        print(f"[analyze] t={t_sec:6.1f}s  {steps[-1]['verdict']:13s} "
              f"scr={int(bool(perception.get('face_screened')))} "
              f"drill={int(bool(perception.get('drill_active')))} "
              f"park={int(bool(perception.get('arms_parked')))}  "
              f"{perception.get('scene','')[:55]}")
    from task import active_task
    out = {"meta": {"task": active_task(), "video": cfg["paths"]["video"], "model": cfg["model"],
                    "sampling_sec": cfg["sampling_sec"], "window": cfg["window_frames"],
                    "support_window": cfg.get("support_window", 3),
                    "items": [{"id": it.id, "label": it.label} for it in items]},
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
