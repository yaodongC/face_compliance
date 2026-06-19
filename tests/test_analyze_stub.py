import json
from pathlib import Path
import cv2
import numpy as np
import yaml
import analyze

ROOT = Path(__file__).resolve().parents[1]


def _make_dummy_video(path, n=140, fps=15):
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(n):
        w.write((np.ones((48, 64, 3)) * (i % 255)).astype(np.uint8))
    w.release()


def test_stub_analysis_runs_full_safety_lifecycle(tmp_path):
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    vid = tmp_path / "v.mp4"
    _make_dummy_video(vid)
    cfg["paths"]["video"] = str(vid)
    cfg["paths"]["analysis"] = str(tmp_path / "analysis.json")
    cfg["paths"]["regulation"] = str(ROOT / "regulation.yaml")
    cfg["sampling_sec"] = 1.0
    cfg["window_frames"] = 3

    result = analyze.run_analysis(cfg, stub=True)
    assert result["steps"], "no steps produced"

    verdicts = [s["verdict"] for s in result["steps"]]
    # the scripted stub must exercise the unsafe states AND eventually reach SUPPORTED
    assert "DANGER" in verdicts
    assert "DRILLING" in verdicts or "UNSUPPORTED" in verdicts
    assert verdicts[-1] == "SUPPORTED"

    # file written with the face-focused schema
    on_disk = json.loads(Path(cfg["paths"]["analysis"]).read_text())
    assert on_disk["meta"]["model"] == cfg["model"]
    for s in result["steps"]:
        assert {"t_sec", "scene", "perception",
                "checklist_snapshot", "verdict", "hazard_note"} <= set(s)

    # final snapshot: the compliant end-state has the face-screen verified
    last = result["steps"][-1]["checklist_snapshot"]
    assert last["face_screen"] == "verified"
