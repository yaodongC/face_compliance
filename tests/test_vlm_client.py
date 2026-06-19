import numpy as np
from vlm_client import (parse_response, encode_frame, build_messages,
                        build_system_prompt, fuse_perceptions, SAFE_DEFAULT)


def _p(**kw):
    base = dict(SAFE_DEFAULT)
    base.update(kw)
    return base


SUP = _p(mesh_visible=True, bolts_visible=True, ground_support_state="full", safety_call="SUPPORTED")


def test_fuse_supported_requires_unanimity():
    assert fuse_perceptions([SUP, SUP, SUP])["safety_call"] == "SUPPORTED"


def test_fuse_explicit_unsupported_dissenter_yields_unsupported():
    bare = _p(safety_call="UNSUPPORTED", ground_support_state="none_visible")
    f = fuse_perceptions([SUP, SUP, bare])
    assert f["safety_call"] == "UNSUPPORTED"
    assert f["bolts_visible"] is False       # unanimous AND
    assert f["ground_support_state"] == "none_visible"


def test_fuse_uncertain_dissenter_blocks_supported():
    # one 'cannot verify' vote is enough to deny SUPPORTED (stays non-safe)
    f = fuse_perceptions([SUP, SUP, _p()])
    assert f["safety_call"] != "SUPPORTED"
    assert f["bolts_visible"] is False


def test_fuse_hazard_fires_on_any_vote():
    f = fuse_perceptions([_p(), _p(), _p(person_in_danger=True, people_visible=True)])
    assert f["person_in_danger"] is True


def test_fuse_most_hazardous_activity_wins():
    f = fuse_perceptions([_p(activity="none"), _p(activity="drilling"), _p(activity="none")])
    assert f["activity"] == "drilling"


def test_fuse_empty_is_safe_default():
    assert fuse_perceptions([]) == SAFE_DEFAULT

GOOD = ('{"scene":"bare rock face","activity":"none","people_visible":false,'
        '"person_in_danger":false,"mesh_visible":false,"bolts_visible":false,'
        '"ground_support_state":"none_visible","safety_call":"UNSUPPORTED","note":"bare"}')


def test_parse_plain_perception_json():
    r = parse_response(GOOD)
    assert r["safety_call"] == "UNSUPPORTED"
    assert r["mesh_visible"] is False
    assert r["activity"] == "none"


def test_parse_fenced_json():
    r = parse_response("```json\n" + GOOD + "\n```")
    assert r["ground_support_state"] == "none_visible"


def test_parse_malformed_returns_safe_default():
    r = parse_response("the model rambled with no json")
    assert r == SAFE_DEFAULT
    # the safe default must never assert support
    assert r["safety_call"] == "CANNOT_VERIFY"
    assert r["mesh_visible"] is False and r["bolts_visible"] is False


def test_conjunction_enforced_on_inconsistent_model_output():
    # model claims SUPPORTED/full but bolts not visible -> must be downgraded
    bad = ('{"scene":"x","activity":"none","people_visible":false,"person_in_danger":false,'
           '"mesh_visible":true,"bolts_visible":false,'
           '"ground_support_state":"full","safety_call":"SUPPORTED","note":"n"}')
    r = parse_response(bad)
    assert r["safety_call"] == "UNSUPPORTED"
    assert r["ground_support_state"] == "none_visible"


def test_conjunction_allows_real_support():
    ok = ('{"scene":"x","activity":"none","people_visible":false,"person_in_danger":false,'
          '"mesh_visible":true,"bolts_visible":true,'
          '"ground_support_state":"full","safety_call":"SUPPORTED","note":"n"}')
    r = parse_response(ok)
    assert r["safety_call"] == "SUPPORTED"
    assert r["ground_support_state"] == "full"


def test_unknown_enum_values_fall_back_safely():
    weird = ('{"scene":"x","activity":"teleporting","mesh_visible":"nope",'
             '"safety_call":"DEFINITELY_FINE"}')
    r = parse_response(weird)
    assert r["activity"] == "other"
    assert r["mesh_visible"] is False
    assert r["safety_call"] == "CANNOT_VERIFY"


def test_encode_frame_is_base64_jpeg():
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    b64 = encode_frame(img, max_width=32)
    assert isinstance(b64, str) and len(b64) > 0


def test_build_messages_has_images_and_text():
    msgs = build_messages(["AAAA", "BBBB"], "SYS")
    assert msgs[0]["role"] == "system"
    content = msgs[1]["content"]
    images = [c for c in content if c["type"] == "image_url"]
    assert len(images) == 2
    assert images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_system_prompt_is_safety_framed():
    sp = build_system_prompt()
    low = sp.lower()
    assert "unsupported" in low and "mesh" in low and "bolt" in low
