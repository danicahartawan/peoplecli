"""
run a vlm over the test images and save what it said.

NOT RUN YET — this container has no gpu. the logic below is written against the
smolvlm2 api but has not been executed; expect to adjust the two lines marked
CHECK on first run.

usage:
  python vlm/predict.py --split test --out preds_base.json
  python vlm/predict.py --split test --adapter runs/lora --out preds_ft.json

the same PROMPT is used with and without the adapter. raw model text is saved
alongside the parsed numbers, because when a number looks wrong the first
question is always "what did it actually say".
"""

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vlm.schema import PROMPT, parse
from vlm.dataset import load_manifest, split_by_space

MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    a = ap.parse_args()

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image

    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map="auto")
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    rows = split_by_space(load_manifest())[a.split]
    out, fails = {}, 0
    for i, r in enumerate(rows, 1):
        img = Image.open(ROOT / "vlm/images" / r["image"]).convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": PROMPT}]}]
        # CHECK: processor call shape varies across smolvlm releases
        text = proc.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = proc(text=text, images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=a.max_new_tokens,
                                 do_sample=False)
        # CHECK: slice off the prompt so the parser never sees the instructions,
        # which contain the example json and would parse as an answer of all 3s
        said = proc.decode(ids[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True)
        scores, how = parse(said)
        if scores is None:
            fails += 1
        out[r["image"]] = {"scores": scores, "how": how, "raw": said.strip()}
        print(f"  {i:>3}/{len(rows)}  {r['image']:<28} {how or 'UNPARSEABLE'}")

    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out} — {fails}/{len(rows)} unparseable")


if __name__ == "__main__":
    main()
