"""Shared theme for the LoopX Safety monitor UI: palette, fonts, geometry, and
text helpers. Used by render_gui.compose() (offline) and live_gui (live)."""
from __future__ import annotations
from pathlib import Path
from PIL import ImageFont

# ---- palette (RGB) ----
BG = (13, 17, 23)
SURFACE = (22, 28, 36)
SURFACE_HI = (28, 35, 45)
DANGER_BG = (38, 18, 20)
HAIR = (40, 49, 61)
INK = (233, 238, 243)
MUTED = (132, 143, 156)
FAINT = (92, 102, 114)
CLEAR = (52, 211, 153)
DANGER = (255, 86, 86)
AMBER = (255, 176, 32)
WARN = (228, 161, 8)
SEV = {"INFO": MUTED, "WARNING": WARN, "VIOLATION": DANGER, "CRITICAL": (255, 130, 130)}

# current-activity labels from the (informational) face-perception verdict
NOW = {"DRILLING": "Face drilling in progress", "SUPPORTED": "Booms parked at face",
       "UNSUPPORTED": "Bare face exposed", "NOT VERIFIED": "Assessing face support"}

_UB = "/usr/share/fonts/truetype/ubuntu/"
_LIB = "/usr/share/fonts/truetype/liberation/"
_FONTMAP = {
    "b": [_UB + "Ubuntu-B.ttf", _LIB + "LiberationSans-Bold.ttf"],
    "m": [_UB + "Ubuntu-M.ttf", _LIB + "LiberationSans-Bold.ttf"],
    "r": [_UB + "Ubuntu-R.ttf", _LIB + "LiberationSans-Regular.ttf"],
    "mono": [_UB + "UbuntuMono-R.ttf", _LIB + "LiberationMono-Regular.ttf"],
    "monob": [_UB + "UbuntuMono-B.ttf", _LIB + "LiberationMono-Bold.ttf"],
}


def _font(weight, size):
    for p in _FONTMAP[weight]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def build_fonts():
    return {"num": _font("b", 60), "num44": _font("b", 44), "verdict": _font("b", 42),
            "h1": _font("b", 22), "title": _font("b", 18), "cap": _font("b", 19),
            "eye": _font("m", 12), "body": _font("r", 16), "small": _font("r", 14),
            "mono": _font("mono", 14), "monob": _font("monob", 15)}


def geometry():
    g = {"W": 1376, "H": 788, "M": 24, "vx": 24, "vy": 86, "vw": 860, "vh": 484}
    g["rx"] = g["vx"] + g["vw"] + g["M"]
    g["rw"] = g["W"] - g["rx"] - g["M"]
    g["safe_box"] = (g["rx"], g["vy"], g["rx"] + g["rw"], g["vy"] + 176)
    g["mesh_box"] = (g["rx"], g["vy"] + 192, g["rx"] + g["rw"], g["vy"] + 484)
    # bottom row: event log (narrowed to the camera width) + compliance checklist in the
    # right column, directly under the mesh-installation card.
    g["log_box"] = (g["M"], 586, g["vx"] + g["vw"], 748)
    g["checklist_box"] = (g["rx"], 586, g["rx"] + g["rw"], 748)
    return g


# ---- text helpers ----
def _tracked(d, pos, text, font, fill, track=2):
    x, y = pos
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill)
        x += d.textlength(ch, font=font) + track
    return x


def _right(d, right, y, text, font, fill):
    d.text((right - d.textlength(text, font=font), y), text, font=font, fill=fill)


def _wrap(d, text, font, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) <= maxw:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines
