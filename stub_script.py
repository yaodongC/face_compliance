"""Canned perception responses so the pipeline + GUI run with no served model.

The scripted sequence exercises the verdicts: DRILLING (active face drilling) ->
DANGER (worker under unsupported ground) -> UNSUPPORTED (face not yet screened) ->
SUPPORTED (face screened + booms parked, drilling complete)."""


def _p(scene, screened=False, drill=False, parked=False, danger=False, note=""):
    return {"scene": scene, "face_screened": screened, "drill_active": drill,
            "arms_parked": parked, "person_in_danger": danger, "note": note}


STUB_STEPS = [
    _p("Drill booms boring the bare end face; face not yet screened.",
       drill=True, note="Active drilling on an unscreened face."),
    _p("Drilling continues across the face.", drill=True),
    _p("A worker steps under the unsupported brow while the drill runs.",
       drill=True, danger=True, note="PERSON under unsupported rock."),
    _p("The worker is still under the unsupported brow.",
       drill=True, danger=True, note="PERSON under unsupported rock."),
    _p("Drilling has stopped; screen is being hung on the face, no bolts yet.",
       screened=False, drill=False, parked=False, note="Face not yet fully screened."),
    _p("Face is screened and bolted; booms pulling back to the sides.",
       screened=True, drill=False, parked=True, note="Face screened; booms parking."),
    _p("Face fully screened with bolt plates; booms parked.",
       screened=True, drill=False, parked=True),
    _p("Compliant: end face screened + bolted, booms parked, no drilling.",
       screened=True, drill=False, parked=True),
    _p("Compliant supported face, work complete.",
       screened=True, drill=False, parked=True),
]


def stub_response(step_index: int) -> dict:
    return STUB_STEPS[min(step_index, len(STUB_STEPS) - 1)]
