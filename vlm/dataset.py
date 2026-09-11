"""
build the manifest and split it BY SPACE.

the split is the part of this experiment most likely to silently invalidate it.
three photos of the same courtyard in three different splits means the model can
recognise the courtyard instead of reading the architecture, and test error stops
meaning anything. so grouping is enforced here and verified, not assumed.
"""

import csv, json, random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "vlm/images"


def load_manifest(path=None):
    """images.csv: image,space_id,source,license,attribution"""
    path = path or ROOT / "vlm/images.csv"
    return list(csv.DictReader(open(path)))


def split_by_space(rows, n_val=30, n_test=30, seed=0):
    by_space = defaultdict(list)
    for r in rows:
        by_space[r["space_id"]].append(r)
    spaces = sorted(by_space)
    random.Random(seed).shuffle(spaces)

    # fill test and val to the target PHOTO count, whole spaces at a time
    splits, want = {"test": n_test, "val": n_val}, ["test", "val"]
    out = {"train": [], "val": [], "test": []}
    i = 0
    for name in want:
        while len(out[name]) < splits[name] and i < len(spaces):
            out[name] += by_space[spaces[i]]; i += 1
    out["train"] = [r for s in spaces[i:] for r in by_space[s]]

    verify(out)
    return out


def verify(out):
    """a leaked space is a silent failure, so make it a loud one."""
    seen = {}
    for name, rows in out.items():
        for r in rows:
            prev = seen.get(r["space_id"])
            if prev and prev != name:
                raise AssertionError(
                    f"space {r['space_id']} appears in both {prev} and {name}")
            seen[r["space_id"]] = name
    return True


def summarise(out):
    for name in ("train", "val", "test"):
        rows = out[name]
        print(f"  {name:<6} {len(rows):>4} photos  "
              f"{len({r['space_id'] for r in rows}):>3} spaces")
