# state-conditioned neural additive model (sc-nam)

replaces the bilinear placeholder. one equation:

```
score(context, place) = Σᵢ φᵢ + b_user

φᵢ = aᵢ · ( γᵢ(context) · hᵢ(fᵢ) + βᵢ(context) )
```

- `hᵢ` — a small mlp that sees **one** feature and nothing else
  (neural additive models, agarwal et al., neurips 2021)
- `γᵢ, βᵢ` — scale and shift produced from the person's state
  (film conditioning, perez et al., aaai 2018)
- `aᵢ` — kept linear, so a bayesian layer can sit on top

## the three properties that matter

**1. the explanation is the arithmetic.**
`φ` sums exactly to the score. nothing is approximated after the fact. the
alternative everyone reaches for — gradient boosting plus shap — produces
explanations computed by a *separate* procedure that can disagree with the
model. here they cannot disagree, because they are the same numbers.

**2. a feature can have an interior optimum.**
noise is not monotone. there is an inverted-u: silence is worse than a low
hum for many people and many tasks (yerkes & dodson 1908; mehta, zhu & cheema
2012, *is noise always bad?*, jcr). a bilinear model has no way to represent
a peak — it can only say more or less. `hᵢ` can.

**3. the state moves the optimum, not just its weight.**
film conditioning means an overstimulated person and a drained person have
their noise peak in *different places*, not merely at different strengths.
this is the whole research claim, so it lives in the architecture rather than
in hand-built interaction terms.

## training

pairwise choices with a **bradley–terry** likelihood (bradley & terry 1952) —
"given how you feel, a or b?". people rank two options far more reliably than
they rate one on a 1–7 scale, and it costs the annotator less.

## uncertainty and exploration

freeze the learned `φ`, fit a **bayesian linear head** in closed form
(neural-linear; riquelme, tucker & snoek, iclr 2018). the posterior turns
exploration into **thompson sampling** (chapelle & li, nips 2011) instead of an
arbitrary "recommend randomly 15% of the time" — the app explores exactly where
it is genuinely uncertain about *you*, and that exploration is what keeps the
logged data unbiased enough to make a causal claim later.

## personalization

`b_user` first (a per-person intercept, fits with very little data), then a
low-rank user embedding concatenated onto the film input once someone has
~10 sessions. same question from two people, different weights, different
answer.

## result on synthetic data

ground truth built to contain a state-dependent inverted-u in noise:

```
pairwise accuracy      bilinear  88.8%      sc-nam  99.2%

recovered noise optimum          true
  wired / overstimulated  0.17   0.17
  drained / flat          0.57   0.57
```

the baseline is not bad — it is *structurally incapable* of the specific thing
this project claims to study, and 88.8% is what that looks like from the
outside. worth keeping in the paper as the ablation.

## why the codebook has the features it has

the feature set is not arbitrary; each one has a literature behind it:

- greenery → attention restoration theory (kaplan 1995), stress recovery
  (ulrich et al. 1991)
- enclosure + privacy → prospect–refuge (appleton 1975)
- noise + crowding → arousal regulation (yerkes & dodson 1908)
- daylight → circadian and alertness effects

## run it

```
pip install torch
python model/scnam.py
```

trains both models on synthetic data, fits the bayesian head, prints a worked
explanation. swap `sample_batch` for the annotation csv once you have rows.
