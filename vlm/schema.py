"""the six dimensions, the prompt, and parsing a vlm's answer back into numbers."""

import json, re

DIMS = ["enclosure", "natural_light", "visual_complexity",
        "greenery", "privacy", "calmness"]

ANCHORS = {
 "enclosure":        "1 = open sky on all sides, 5 = a small closed room",
 "natural_light":    "1 = no daylight at all, 5 = direct sky visible",
 "visual_complexity":"1 = plain and bare, 5 = cluttered, busy, many colours",
 "greenery":         "1 = nothing living in view, 5 = surrounded by planting",
 "privacy":          "1 = fully exposed to passers-by, 5 = nobody can see you",
 "calmness":         "1 = agitating, 5 = settling",
}

# one prompt, used for the base model AND the fine-tuned one. if you reword it
# for the fine-tune you are no longer measuring fine-tuning, you are measuring
# prompt engineering.
PROMPT = (
    "Rate this space on six dimensions, 1 to 5, based only on the photograph.\n"
    + "\n".join(f"{d}: {ANCHORS[d]}" for d in DIMS)
    + "\n\nAnswer with JSON only, no other text, in exactly this form:\n"
    + json.dumps({d: 3 for d in DIMS})
)

_NUM = r"([0-5](?:\.[0-9]+)?)"


def parse(text):
    """
    pull six numbers out of whatever the model said.

    returns (scores_dict, how) where how is 'json' | 'loose' | None. base models
    fail strict json constantly, so a strict-only parser would score the base
    model's FORMAT rather than its perception, and hand you a fake improvement.
    try json first, then fall back to finding 'dimension: number' anywhere in
    the text, and record which path was used so evaluation can report it.
    """
    if not text:
        return None, None
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            raw = json.loads(m.group(0))
            got = {}
            for d in DIMS:
                v = raw.get(d, raw.get(d.replace("_", " ")))
                if v is None:
                    break
                got[d] = float(v)
            if len(got) == len(DIMS):
                return {d: _clip(got[d]) for d in DIMS}, "json"
        except (ValueError, TypeError):
            pass
    got = {}
    for d in DIMS:
        pat = d.replace("_", r"[ _]") + r"\D{0,12}?" + _NUM
        m = re.search(pat, text, re.I)
        if m:
            got[d] = _clip(float(m.group(1)))
    if len(got) == len(DIMS):
        return got, "loose"
    return None, None


def _clip(v):
    return max(1.0, min(5.0, v))


def target_text(scores):
    """what the model should learn to emit. compact json, stable key order."""
    return json.dumps({d: round(float(scores[d]), 1) for d in DIMS})
