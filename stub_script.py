"""Canned VLM responses so the pipeline + GUI run without a served model.

The sequence walks through the bolting process so the checklist fills
progressively, and injects a transient safety violation (S1) mid-way.
"""

def _obs(item_id, status, evidence):
    return {"item_id": item_id, "status": status, "evidence": evidence}


STUB_STEPS = [
    {"narration": "Operator bars down loose rock at the face before any support.",
     "current_activity": "p1",
     "observations": [_obs("p1", "satisfied", "scaling bar knocking loose rock off the back")],
     "safety_flags": [], "confidence": 0.8},
    {"narration": "Crew hangs wire screen on the back and walls toward the face.",
     "current_activity": "p2",
     "observations": [_obs("p2", "in_progress", "screen being lifted to the back")],
     "safety_flags": [], "confidence": 0.7},
    {"narration": "A worker steps under the still-unbolted brow to pull the screen.",
     "current_activity": "p2",
     "observations": [_obs("p2", "satisfied", "back/wall screen tight to face")],
     "safety_flags": [{"id": "S1", "severity": "high", "note": "person under unbolted brow"}],
     "confidence": 0.6},
    {"narration": "The worker is still under the unsupported brow while folding the screen over.",
     "current_activity": "p3",
     "observations": [_obs("p3", "satisfied", "screen wrapped around the brow")],
     "safety_flags": [{"id": "S1", "severity": "high", "note": "person still under unbolted brow"}],
     "confidence": 0.6},
    {"narration": "Screen is pulled snug against both side walls.",
     "current_activity": "p4",
     "observations": [_obs("p4", "satisfied", "screen tight to side walls")],
     "safety_flags": [], "confidence": 0.7},
    {"narration": "Bolter installs the first row of bolts from the left wall across.",
     "current_activity": "p5",
     "observations": [_obs("p5", "in_progress", "bolter installing plates left to right")],
     "safety_flags": [], "confidence": 0.7},
    {"narration": "Bolting continues; top row of plates goes in near the back.",
     "current_activity": "p6",
     "observations": [_obs("p5", "satisfied", "row of bolts complete across face"),
                      _obs("p6", "satisfied", "top row up near the back")],
     "safety_flags": [], "confidence": 0.7},
    {"narration": "Bottom screen bolts are secured down near the floor.",
     "current_activity": "p7",
     "observations": [_obs("p7", "satisfied", "bottom screen bolted near the floor")],
     "safety_flags": [], "confidence": 0.7},
    {"narration": "With the face supported, the drill boom moves in to drill the round.",
     "current_activity": "p8",
     "observations": [_obs("p8", "satisfied", "drilling only after screen and bolts in place")],
     "safety_flags": [], "confidence": 0.8},
]


def stub_response(step_index: int) -> dict:
    return STUB_STEPS[min(step_index, len(STUB_STEPS) - 1)]
