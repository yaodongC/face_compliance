"""Canned perception responses so the pipeline + GUI run without a served model.

The scripted sequence honestly exercises the fail-safe verdicts:
UNSUPPORTED (bare face) -> DANGER (worker under unsupported brow) -> NOT VERIFIED
(support going in, not yet confirmed) -> SUPPORTED (mesh+bolts confirmed over the
support_window). It mirrors what a *correct* run should look like; the real model
on the demo clip stays at UNSUPPORTED because the face is bare the whole time.
"""


def _p(scene, activity="none", people=False, danger=False, mesh=False, bolts=False,
       gss="none_visible", call="UNSUPPORTED", note=""):
    return {"scene": scene, "activity": activity, "people_visible": people,
            "person_in_danger": danger, "mesh_visible": mesh, "bolts_visible": bolts,
            "ground_support_state": gss, "safety_call": call, "note": note}


STUB_STEPS = [
    _p("Bare rock development face with survey marks and a muck pile; a parked boom.",
       activity="none", note="No ground support visible; face is unsupported."),
    _p("Operator scaling loose rock from the bare face before any support.",
       activity="scaling", people=True, note="Scaling underway; face still unsupported."),
    _p("A worker stands directly under the unsupported brow to reach the face.",
       activity="screening", people=True, danger=True,
       note="PERSON under unsupported rock — immediate hazard."),
    _p("The worker is still under the unsupported brow while lifting mesh; no bolts yet.",
       activity="screening", people=True, danger=True, mesh=True, bolts=False,
       gss="none_visible", call="UNSUPPORTED",
       note="PERSON still under unsupported rock; mesh going up but no bolts."),
    _p("Mesh is up and the first bolts are going in; support is partial.",
       activity="bolting", people=True, mesh=True, bolts=True,
       gss="partial", call="PARTIAL", note="Partial support — not yet confirmed."),
    _p("Mesh and a row of bolt plates are visible across the face.",
       activity="bolting", mesh=True, bolts=True, gss="full", call="SUPPORTED",
       note="Mesh and bolts visible."),
    _p("Full grid of mesh and bolt plates covering the face.",
       activity="bolting", mesh=True, bolts=True, gss="full", call="SUPPORTED",
       note="Mesh and bolts visible across the face."),
    _p("Face fully covered by mesh with a complete bolt pattern.",
       activity="none", mesh=True, bolts=True, gss="full", call="SUPPORTED",
       note="Mesh and full bolt pattern clearly visible."),
    _p("Supported face: full mesh and bolt plates, no one in the danger zone.",
       activity="none", mesh=True, bolts=True, gss="full", call="SUPPORTED",
       note="Face supported; bolting complete."),
]


def stub_response(step_index: int) -> dict:
    return STUB_STEPS[min(step_index, len(STUB_STEPS) - 1)]
