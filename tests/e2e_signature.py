"""Deterministic end-to-end signature of the offline pipeline output (event log +
rendered GUI frames). Used to prove a refactor did not change behaviour: run before
and after, the signatures must match. Excludes wall_time (the only non-deterministic
field)."""
import hashlib
import json
import sys
import cv2


def signature(event_log="data/event_log.jsonl", gui="data/full_cycle_gui.mp4"):
    evs = [json.loads(l) for l in open(event_log)]
    det = [{k: v for k, v in e.items() if k != "wall_time"} for e in evs]
    hlog = hashlib.md5(json.dumps(det, sort_keys=True).encode()).hexdigest()
    cap = cv2.VideoCapture(gui)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hf = hashlib.md5()
    step = max(1, n // 24)
    sampled = 0
    for fno in range(0, n, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, f = cap.read()
        if ok:
            hf.update(f.tobytes())
            sampled += 1
    cap.release()
    return {"events": len(evs), "log_md5": hlog, "frames": n,
            "sampled": sampled, "frame_md5": hf.hexdigest()}


if __name__ == "__main__":
    print(json.dumps(signature(*sys.argv[1:]), indent=2))
