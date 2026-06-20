"""Domain guard: disabled is a no-op (no VLM call); enabled proceeds only on an
explicit in-domain yes, and ABSTAINS fail-safe on a no, an error, or an unparseable
response."""
import numpy as np
import domain_guard

FRAME = np.zeros((48, 64, 3), np.uint8)
CFG = {"endpoint": "http://x", "model": "vlm"}
SPEC = {"enabled": True, "question": "?", "send_w": 32}


class _Sess:
    def __init__(self, content):
        self._c = content

    def post(self, *a, **k):
        class _R:
            def json(_self):
                return {"choices": [{"message": {"content": self._c}}]}
        return _R()


class _BadSess:
    def post(self, *a, **k):
        raise RuntimeError("network down")


def test_disabled_is_noop():
    r = domain_guard.in_domain(FRAME, CFG, spec={"enabled": False})
    assert r["in_domain"] is True and r["checked"] is False


def test_enabled_in_domain_yes():
    r = domain_guard.in_domain(FRAME, CFG, session=_Sess('{"in_domain": true, "reason": "mine"}'), spec=SPEC)
    assert r["in_domain"] is True and r["checked"] is True


def test_enabled_out_of_domain_abstains():
    r = domain_guard.in_domain(FRAME, CFG, session=_Sess('{"in_domain": false, "reason": "office"}'), spec=SPEC)
    assert r["in_domain"] is False


def test_failsafe_abstains_on_error_or_unparseable():
    assert domain_guard.in_domain(FRAME, CFG, session=_BadSess(), spec=SPEC)["in_domain"] is False
    assert domain_guard.in_domain(FRAME, CFG, session=_Sess("not json"), spec=SPEC)["in_domain"] is False
    # missing the key also abstains (fail-safe default False)
    assert domain_guard.in_domain(FRAME, CFG, session=_Sess('{"reason": "?"}'), spec=SPEC)["in_domain"] is False
