from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_config_has_required_keys():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    for key in ["endpoint", "model", "camera_topic", "sampling_sec",
                "window_frames", "frame_max_width", "max_tokens", "temperature",
                "support_window", "paths"]:
        assert key in cfg, f"missing config key {key}"
    for key in ["regulation", "video", "frame_index", "analysis"]:
        assert key in cfg["paths"], f"missing paths.{key}"
    assert cfg["camera_topic"] == "/sensing/front/rgb/image_raw/compressed"
    # safety-critical: deterministic perception
    assert cfg["temperature"] == 0.0


def test_regulation_items_well_formed():
    reg = yaml.safe_load((ROOT / "regulation.yaml").read_text())
    items = reg["items"]
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    # perception-grounded fail-safe checklist
    assert {"bolts", "mesh", "support", "drill_safe", "worker_safe"}.issubset(set(ids))
    for it in items:
        assert it["id"] and it["label"]
