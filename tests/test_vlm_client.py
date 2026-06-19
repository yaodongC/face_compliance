import numpy as np
from vlm_client import (parse_response, encode_frame, crop_region, build_messages,
                        build_system_prompt, fuse_perceptions, SAFE_DEFAULT)


def _p(**kw):
    base = dict(SAFE_DEFAULT)
    base.update(kw)
    return base


GOOD = ('{"scene":"end face","face_screened":true,"drill_active":false,'
        '"arms_parked":true,"person_in_danger":false,"note":"ok"}')


def test_parse_plain_perception():
    r = parse_response(GOOD)
    assert r["face_screened"] is True
    assert r["drill_active"] is False
    assert r["arms_parked"] is True


def test_parse_fenced_json():
    r = parse_response("```json\n" + GOOD + "\n```")
    assert r["face_screened"] is True


def test_parse_malformed_is_safe_default():
    r = parse_response("no json here")
    assert r == SAFE_DEFAULT
    assert r["face_screened"] is False        # fail-safe: never invent support


def test_fuse_screened_requires_unanimity():
    sup = _p(face_screened=True, arms_parked=True)
    assert fuse_perceptions([sup, sup, sup])["face_screened"] is True


def test_fuse_one_dissenter_blocks_screened():
    sup = _p(face_screened=True, arms_parked=True)
    f = fuse_perceptions([sup, sup, _p(face_screened=False)])
    assert f["face_screened"] is False        # unanimous AND
    assert f["arms_parked"] is False


def test_fuse_drilling_fires_on_any_vote():
    f = fuse_perceptions([_p(), _p(), _p(drill_active=True)])
    assert f["drill_active"] is True


def test_fuse_danger_fires_on_any_vote():
    f = fuse_perceptions([_p(), _p(person_in_danger=True), _p()])
    assert f["person_in_danger"] is True


def test_fuse_empty_is_safe_default():
    assert fuse_perceptions([]) == SAFE_DEFAULT


def test_crop_region_crops():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    c = crop_region(img, [0.25, 0.0, 0.75, 0.5])
    assert c.shape[0] == 50 and c.shape[1] == 100


def test_crop_region_none_passthrough():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert crop_region(img, None).shape == (10, 10, 3)


def test_encode_frame_is_base64_jpeg():
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    assert len(encode_frame(img, max_width=32)) > 0


def test_build_messages_has_image_and_text():
    msgs = build_messages(["AAAA"], "SYS")
    content = msgs[1]["content"]
    assert any(c["type"] == "image_url" for c in content)
    assert msgs[0]["content"] == "SYS"


def test_system_prompt_is_face_focused():
    sp = build_system_prompt().lower()
    assert "end face" in sp and "drill" in sp and "park" in sp
