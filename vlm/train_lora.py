"""
lora fine-tune smolvlm2 on the human averages.

NOT RUN YET — no gpu in this container. written against the hf smolvlm recipe;
the collator is the part most likely to need a tweak for your transformers
version, and it is marked CHECK.

usage:
  python vlm/train_lora.py --epochs 3 --out runs/lora
"""

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vlm.schema import PROMPT, target_text
from vlm.dataset import load_manifest, split_by_space
from vlm.aggregate import load_ratings, aggregate

MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="runs/lora")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--train-vision", action="store_true",
                    help="also adapt the vision tower. try it second, not first: "
                         "it is the change most likely to overfit 240 images.")
    a = ap.parse_args()

    import torch
    from torch.utils.data import Dataset
    from transformers import (AutoProcessor, AutoModelForImageTextToText,
                              Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model
    from PIL import Image

    labels, _ = aggregate(load_ratings())
    splits = split_by_space(load_manifest())

    class Spaces(Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            r = self.rows[i]
            return (Image.open(ROOT / "vlm/images" / r["image"]).convert("RGB"),
                    target_text(labels[r["image"]]))

    proc = AutoProcessor.from_pretrained(a.model)

    def collate(batch):
        imgs = [b[0] for b in batch]
        texts = [proc.apply_chat_template([
            {"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": PROMPT}]},
            {"role": "assistant", "content": [{"type": "text", "text": b[1]}]},
        ], tokenize=False) for b in batch]
        # CHECK: images=[[img]] per sample on some versions, images=imgs on others
        enc = proc(text=texts, images=[[i] for i in imgs],
                   return_tensors="pt", padding=True)
        lab = enc["input_ids"].clone()
        lab[lab == proc.tokenizer.pad_token_id] = -100
        # do not train on the image placeholder tokens
        img_tok = getattr(proc, "image_token_id", None)
        if img_tok is not None:
            lab[lab == img_tok] = -100
        enc["labels"] = lab
        return enc

    model = AutoModelForImageTextToText.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map="auto")

    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    cfg = LoraConfig(r=a.rank, lora_alpha=a.rank * 2, lora_dropout=.05,
                     target_modules=targets, task_type="CAUSAL_LM")
    if not a.train_vision:
        # freeze the vision tower: 240 images is not enough to move it safely
        for n, p in model.named_parameters():
            if "vision" in n:
                p.requires_grad = False
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.accum,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=.05,
        logging_steps=5, save_strategy="epoch",
        eval_strategy="epoch", per_device_eval_batch_size=a.batch,
        bf16=True, gradient_checkpointing=True, report_to=[],
        remove_unused_columns=False,
    )
    Trainer(model=model, args=args, data_collator=collate,
            train_dataset=Spaces(splits["train"]),
            eval_dataset=Spaces(splits["val"])).train()
    model.save_pretrained(a.out)
    print(f"saved adapter to {a.out}")


if __name__ == "__main__":
    main()
