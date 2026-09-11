"""
end-to-end test of everything that does not need a gpu.

invents 300 photos across 120 spaces, three raters with different biases, and
two fake models -- one that guesses badly and often forgets to answer in json,
one that is close to the humans. if this prints a sane table the pipeline is
sound and only the gpu half remains.
"""

import json, random, statistics as st
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlm.schema import DIMS, parse, target_text
from vlm.dataset import split_by_space, verify
from vlm.aggregate import aggregate, human_ceiling, rater_spread
from vlm import evaluate

rng = random.Random(0)

# --- invent spaces, 1-4 photos each -----------------------------------------
rows, truth = [], {}
for s in range(120):
    base = {d: rng.uniform(1.4, 4.7) for d in DIMS}
    for k in range(rng.choice([1, 1, 2, 3, 4])):
        img = f"space_{s:03d}_{k}.jpg"
        rows.append({"image": img, "space_id": f"s{s:03d}",
                     "source": "example.org", "license": "cc-by"})
        truth[img] = base
rows = rows[:300]
truth = {r["image"]: truth[r["image"]] for r in rows}
print(f"{len(rows)} photos across {len({r['space_id'] for r in rows})} spaces\n")

# --- three raters, each with a personal bias and some noise -----------------
ratings = []
for ri, (bias, noise) in enumerate([(-0.3, .45), (0.0, .55), (0.35, .5)]):
    for r in rows:
        ratings.append({"image": r["image"], "rater": f"r{ri}", "scores": {
            d: max(1, min(5, round(truth[r["image"]][d] + bias + rng.gauss(0, noise))))
            for d in DIMS}})
labels, by_img = aggregate(ratings)
ceiling, spread = human_ceiling(by_img), rater_spread(by_img)

print("rater agreement")
for d in DIMS:
    flag = "  <- barely learnable" if spread[d] > .95 else ""
    print(f"  {d:<18} sd {spread[d]:.2f}   leave-one-out mae {ceiling[d]:.2f}{flag}")

# --- split ------------------------------------------------------------------
print("\nsplit by space")
out = split_by_space(rows, n_val=30, n_test=30, seed=1)
from vlm.dataset import summarise; summarise(out)
verify(out)

# a photo-level split would leak: show what we avoided
photo_split = rows[:]; random.Random(1).shuffle(photo_split)
leaked = len({r["space_id"] for r in photo_split[:30]} &
             {r["space_id"] for r in photo_split[30:]})
print(f"  (a random photo-level split would have leaked {leaked} spaces)")

test_imgs = [r["image"] for r in out["test"]]
train_labels = {r["image"]: labels[r["image"]] for r in out["train"]}

# --- two fake models, answering as text so the parser is exercised ----------
def fake(img, err, json_rate):
    sc = {d: max(1, min(5, truth[img][d] + rng.gauss(0, err))) for d in DIMS}
    if rng.random() < json_rate:
        return target_text(sc)
    if rng.random() < .5:            # prose with the numbers still in it
        return "Looking at this space, " + ", ".join(
            f"{d.replace('_',' ')} is about {sc[d]:.0f}" for d in DIMS) + "."
    return "This appears to be an indoor space with seating."   # no numbers

runs, how = {}, {}
for name, err, jr in [("base vlm", 1.05, .35), ("fine-tuned", .42, .97)]:
    preds, kinds = {}, []
    for img in test_imgs:
        p, k = parse(fake(img, err, jr))
        preds[img] = p; kinds.append(k)
    runs[name] = preds
    how[name] = kinds

print("\nhow the answers parsed")
for name, kinds in how.items():
    j = kinds.count("json"); l = kinds.count("loose"); f = kinds.count(None)
    print(f"  {name:<12} strict json {j:>2}   loose {l:>2}   unparseable {f:>2}")
print("  strict-json-only scoring would have thrown away the loose rows and")
print("  scored format compliance as perception.")

print()
evaluate.table(labels, test_imgs, runs, ceiling, train_labels)

print("\nis the win real, or 30 images of luck?")
for d in DIMS:
    p = evaluate.paired_bootstrap(runs["base vlm"], runs["fine-tuned"],
                                  labels, test_imgs, d)
    print(f"  {d:<18} fine-tuned better in {p*100:.0f}% of bootstrap resamples")
