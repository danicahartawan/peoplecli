"""
compare predictions to human averages.

the table has four rows, not two. base-vs-finetuned alone cannot tell you
whether anything was learned:

  predict-the-mean   always answer the training-set average for each dimension.
                     a fine-tune that does not beat this learned nothing about
                     images at all. it is the row people leave out and the row
                     that most often wins.
  base vlm           the untouched model, same prompt.
  fine-tuned         the lora model.
  human ceiling      leave-one-rater-out error. the floor. beating it means you
                     are fitting rater noise, not perceiving better.
"""

import json, statistics as st
from pathlib import Path
from vlm.schema import DIMS

ROOT = Path(__file__).resolve().parent.parent


def mae(preds, labels, dim, images):
    errs = [abs(preds[i][dim] - labels[i][dim]) for i in images
            if i in preds and preds[i] is not None]
    return st.fmean(errs) if errs else float("nan")


def coverage(preds, images):
    """share of test images the model produced six usable numbers for."""
    ok = sum(1 for i in images if preds.get(i))
    return ok / len(images) if images else 0.0


def mean_baseline(train_labels):
    return {d: st.fmean(v[d] for v in train_labels.values()) for d in DIMS}


def table(labels, images, runs, ceiling, train_labels):
    """
    runs: {"base vlm": {image: scores|None}, "fine-tuned": {...}}
    prints mae per dimension, plus coverage, plus the two reference rows.
    """
    const = mean_baseline(train_labels)
    rows = [("predict-the-mean", {i: const for i in images}, 1.0)]
    for name, p in runs.items():
        rows.append((name, p, coverage(p, images)))

    w = max(len(n) for n, _, _ in rows) + 2
    head = f"{'':<{w}}" + "".join(f"{d[:9]:>11}" for d in DIMS) + f"{'mean':>9}{'cover':>8}"
    print(head)
    print("-" * len(head))
    for name, p, cov in rows:
        vals = [mae(p, labels, d, images) for d in DIMS]
        print(f"{name:<{w}}" + "".join(f"{v:>11.2f}" for v in vals)
              + f"{st.fmean(vals):>9.2f}{cov*100:>7.0f}%")
    print("-" * len(head))
    cvals = [ceiling[d] for d in DIMS]
    print(f"{'human ceiling':<{w}}" + "".join(f"{v:>11.2f}" for v in cvals)
          + f"{st.fmean(cvals):>9.2f}{'':>8}")
    print("\nlower is better. a row below the human ceiling is fitting rater "
          "noise,\nnot reading the room. coverage below 100% means the model "
          "failed to\nproduce six numbers — see the parse-failure note in "
          "evaluate's docstring.")


def paired_bootstrap(a, b, labels, images, dim, n=4000, seed=0):
    """
    is fine-tuned actually better than base, or is it 30 images of luck?
    30 test images is small enough that this is a real question. resamples
    images with replacement and reports how often b beats a.
    """
    import random
    rng = random.Random(seed)
    usable = [i for i in images if a.get(i) and b.get(i)]
    if not usable:
        return float("nan")
    wins = 0
    for _ in range(n):
        s = [usable[rng.randrange(len(usable))] for _ in usable]
        ea = st.fmean(abs(a[i][dim] - labels[i][dim]) for i in s)
        eb = st.fmean(abs(b[i][dim] - labels[i][dim]) for i in s)
        wins += eb < ea
    return wins / n


def _cli():
    """python vlm/evaluate.py preds_base.json preds_ft.json"""
    import sys
    from vlm.dataset import load_manifest, split_by_space
    from vlm.aggregate import load_ratings, aggregate, human_ceiling

    labels, by_img = aggregate(load_ratings())
    splits = split_by_space(load_manifest())
    test = [r["image"] for r in splits["test"]]
    train_labels = {r["image"]: labels[r["image"]] for r in splits["train"]}

    runs = {}
    for path in sys.argv[1:]:
        raw = json.load(open(path))
        name = Path(path).stem.replace("preds_", "").replace("_", " ")
        runs[name] = {k: v["scores"] for k, v in raw.items()}
    table(labels, test, runs, human_ceiling(by_img), train_labels)

    if len(runs) == 2:
        a, b = list(runs.values())
        print("\nis the win real, or 30 images of luck?")
        for d in DIMS:
            p = paired_bootstrap(a, b, labels, test, d)
            print(f"  {d:<18} second run better in {p*100:.0f}% of resamples")


if __name__ == "__main__":
    _cli()
