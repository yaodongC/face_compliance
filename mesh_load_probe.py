"""Probe: can the VLM detect the operator LOADING A MESH PANEL onto the boom arm?
(The operator's own signal for 'a new mesh starts'.) Writes results incrementally."""
import cv2, json, base64, requests, re, sys
sys.path.insert(0, ".")
import operator_safety as osf

ops = json.load(open("data/operator_events.json"))["events"]
idx = {int(f): float(c) for f, c in (l.split(",") for l in open("data/full_cycle.idx").read().splitlines()[1:])}
cap = cv2.VideoCapture("data/full_cycle.mp4")
RW = 1100


def frame_at(t):
    fr = min(idx, key=lambda k: abs(idx[k] - t))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
    return cap.read()[1]


def loading_mesh(img):
    h, w = img.shape[:2]; rh = int(h * RW / w)
    c = cv2.resize(img, (RW, rh))
    ok, b = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 92])
    b64 = base64.b64encode(b).decode()
    P = ('Underground mine jumbo bolting a rock face. Look at the worker and the jumbo boom arms. '
         'Is the worker right now LIFTING/LOADING a LARGE FLAT WIRE-MESH SCREEN PANEL (a big ~1-2 m '
         'rectangular mesh sheet) onto a boom arm to START a new mesh? Answer NO if they are only '
         'handling a small friction bolt/rod, drilling, or standing. '
         'JSON only: {"loading_mesh_panel":bool,"confidence":0-1,"what":"short"}')
    msgs = [{"role": "user", "content": [{"type": "text", "text": P},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]
    r = requests.post("http://localhost:8000/v1/chat/completions",
                      json={"model": "vlm", "messages": msgs, "max_tokens": 80, "temperature": 0.0},
                      timeout=60).json()
    t = r["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", t, re.S)
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


sessions = osf.classify_sessions(ops)
results = []
for s in sessions:
    best = {}
    for dt in [s["start"] - 4, s["start"], (s["start"] + s["end"]) / 2]:
        f = frame_at(max(0, dt))
        if f is None:
            continue
        d = loading_mesh(f)
        if d.get("loading_mesh_panel"):
            best = {"t": int(dt), "conf": d.get("confidence"), "what": d.get("what", "")[:40]}
            break
    row = {"start": s["start"], "loading_mesh": bool(best), **best}
    results.append(row)
    json.dump(results, open("data/mesh_load_probe.json", "w"), indent=2)
    print(f"  {int(s['start'])//60:02d}:{int(s['start'])%60:02d}  {'MESH' if best else 'no '}  {best}", flush=True)

n = sum(1 for r in results if r["loading_mesh"])
print(f"\n=> {n}/{len(results)} sessions detected loading a mesh panel", flush=True)
cap.release()
