from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_config_has_required_keys():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    for key in ["endpoint", "model", "camera_topic", "sampling_sec",
                "window_frames", "frame_max_width", "max_tokens", "temperature", "paths"]:
        assert key in cfg, f"missing config key {key}"
    for key in ["regulation", "video", "frame_index", "analysis"]:
        assert key in cfg["paths"], f"missing paths.{key}"
    assert cfg["camera_topic"] == "/sensing/front/rgb/image_raw/compressed"


def test_regulation_items_well_formed():
    reg = yaml.safe_load((ROOT / "regulation.yaml").read_text())
    items = reg["items"]
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    assert {"p1", "p8", "S1"}.issubset(set(ids))
    for it in items:
        assert it["kind"] in ("process", "safety")
        assert it["label"] and it["evidence"]
    process = [it for it in items if it["kind"] == "process"]
    assert len(process) >= 8
