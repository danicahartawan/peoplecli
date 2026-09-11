"""
average the raters, and measure whether they agree at all.

the second half matters more than the first. mean absolute error against a human
average is uninterpretable without knowing how far humans sit from each other:
if three raters disagree by 0.8 on calmness, a model at 0.7 has already hit the
noise floor and further "improvement" is fitting your raters' idiosyncrasies.
so this reports the human-human error as the ceiling, per dimension.
"""

import json, statistics as st
from collections import defaultdict
from pathlib import Path
from vlm.schema import DIMS

ROOT = Path(__file__).resolve().parent.parent


def load_ratings(path=None):
    """ratings.jsonl: {"image":..., "rater":..., "scores":{dim: 1-5}}"""
    path = path or ROOT / "vlm/ratings.jsonl"
    return [json.loads(l) for l in open(path) if l.strip()]


def aggregate(ratings):
    by_img = defaultdict(list)
    for r in ratings:
        by_img[r["image"]].append(r)
    labels = {}
    for img, rs in by_img.items():
        labels[img] = {d: round(st.fmean(x["scores"][d] for x in rs), 2) for d in DIMS}
    return labels, by_img


def human_ceiling(by_img):
    """
    leave-one-rater-out: each rater against the mean of the others. this is the
    error a perfect model would still make, because it is the disagreement
    between the people who produced the labels.
    """
    out = {}
    for d in DIMS:
        errs = []
        for img, rs in by_img.items():
            if len(rs) < 2:
                continue
            for i, r in enumerate(rs):
                others = [x["scores"][d] for j, x in enumerate(rs) if j != i]
                errs.append(abs(r["scores"][d] - st.fmean(others)))
        out[d] = st.fmean(errs) if errs else float("nan")
    return out


def rater_spread(by_img):
    """mean sd between raters per dimension — which labels are even learnable."""
    out = {}
    for d in DIMS:
        sds = [st.pstdev([x["scores"][d] for x in rs])
               for rs in by_img.values() if len(rs) > 1]
        out[d] = st.fmean(sds) if sds else float("nan")
    return out
