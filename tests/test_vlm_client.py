import numpy as np
from vlm_client import parse_response, encode_frame, build_messages, build_system_prompt
from compliance import ChecklistItem

GOOD = '{"narration":"bolting","current_activity":"p5","observations":[{"item_id":"p5","status":"in_progress","evidence":"bolter at face"}],"safety_flags":[],"confidence":0.7}'


def test_parse_plain_json():
    r = parse_response(GOOD)
    assert r["narration"] == "bolting"
    assert r["observations"][0]["item_id"] == "p5"


def test_parse_fenced_json():
    r = parse_response("```json\n" + GOOD + "\n```")
    assert r["current_activity"] == "p5"


def test_parse_prose_wrapped_json():
    r = parse_response("Sure, here is the result:\n" + GOOD + "\nHope that helps.")
    assert r["confidence"] == 0.7


def test_parse_malformed_returns_safe_default():
    r = parse_response("the model said something with no json")
    assert r["narration"] == ""
    assert r["observations"] == []
    assert r["safety_flags"] == []
    assert r["current_activity"] == "other"


def test_encode_frame_is_base64_jpeg():
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    b64 = encode_frame(img, max_width=32)
    assert isinstance(b64, str) and len(b64) > 0


def test_build_messages_has_images_and_text():
    msgs = build_messages(["AAAA", "BBBB"], "SYS")
    assert msgs[0]["role"] == "system"
    content = msgs[1]["content"]
    images = [c for c in content if c["type"] == "image_url"]
    texts = [c for c in content if c["type"] == "text"]
    assert len(images) == 2 and len(texts) >= 1
    assert images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_system_prompt_lists_items():
    sp = build_system_prompt([ChecklistItem("p1", "process", "scale loose rock", "barring down")])
    assert "p1" in sp and "scale loose rock" in sp
