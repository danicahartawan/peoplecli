# can a small vlm learn how humans read a space?

fine-tune smolvlm2 to predict six human perception scores from one photograph,
and measure whether fine-tuning moved it closer to people.

```
300 photos -> 3 raters each -> 6 numbers per photo
                     |
        split BY SPACE: 240 / 30 / 30
                     |
   base smolvlm2  ->  BASELINE  ->  lora  ->  TEST
                     |
          did it get closer to humans?
```

## the three ways this experiment quietly fails

these are handled in the code, and they are the reason it is more than a
training loop.

**1. a photo-level split leaks.** three photos of one courtyard split across
train and test means the model recognises the courtyard, not the architecture.
`dataset.py` groups by `space_id` and raises if a space appears in two splits.
in the dry run, a random photo-level split leaked 19 of 120 spaces.

**2. strict json scoring measures format, not perception.** base vlms often
answer in prose. if unparseable answers are dropped or scored as max error, most
of your "improvement" is the fine-tune learning to emit json — which it will
learn in about twenty examples and tells you nothing about whether it sees
rooms. `schema.parse` tries json, then falls back to finding the numbers in
prose, and records which path it used. `evaluate` prints coverage next to every
MAE so format and perception stay separable.

**3. no reference rows.** base-vs-fine-tuned alone is uninterpretable. the table
has four rows:

| row | what it tells you |
|---|---|
| predict-the-mean | always answer the training average. **if the fine-tune does not beat this it learned nothing about images.** |
| base vlm | untouched model, same prompt |
| fine-tuned | the lora |
| human ceiling | leave-one-rater-out error — the floor set by your raters disagreeing |

a model scoring *below* the human ceiling is fitting your three raters'
idiosyncrasies, not reading the room better than a person.

## order of work

```bash
# 0. check the plumbing with fake data — no gpu needed, runs in seconds
python vlm/dry_run.py

# 1. collect. images/ plus images.csv: image,space_id,source,license,attribution
#    one space_id per physical room, however many photos it has.

# 2. rate. open vlm/rate.html, pick the folder, press 1-5 six times per photo.
#    three people, three exported files, concatenate into ratings.jsonl.
python -c "from vlm.aggregate import *; l,b=aggregate(load_ratings()); print(rater_spread(b))"
#    any dimension with rater sd above ~1.0 is not learnable — fix the anchor
#    wording or drop the dimension before spending gpu hours on it.

# 3. baseline FIRST, before any training
python vlm/predict.py --split test --out preds_base.json

# 4. train
pip install -r requirements-vlm.txt
python vlm/train_lora.py --epochs 3 --out runs/lora

# 5. same images, same prompt
python vlm/predict.py --split test --adapter runs/lora --out preds_ft.json

# 6. compare
python vlm/evaluate.py preds_base.json preds_ft.json
```

## status

- `schema.py`, `dataset.py`, `aggregate.py`, `evaluate.py`, `dry_run.py`,
  `rate.html` — written and exercised end to end on synthetic data.
- `predict.py`, `train_lora.py` — **not run.** this container has no gpu. the
  lines most likely to need adjusting on your machine are marked `CHECK`: the
  processor call shape and the prompt-slicing in `predict.py`, the collator's
  image argument in `train_lora.py`.

## notes on choices

**freeze the vision tower first.** `--train-vision` exists but 240 images is
little enough that adapting the encoder is the fastest route to overfitting.
get a number with it frozen, then try unfreezing as a second experiment.

**30 test images is small.** `evaluate.paired_bootstrap` reports how often the
fine-tune wins across resamples. below about 90% treat the win as unproven
rather than reporting the delta.

**the six dimensions are the same six the recommender uses**, so phase two is
just running the adapter over berkeley photos to fill `annotations.csv`
automatically instead of by hand.
