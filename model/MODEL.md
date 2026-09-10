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

## result 1: clean synthetic, unlimited data, 6 features

```
pairwise accuracy      bilinear  88.8%      sc-nam  99.2%

recovered noise optimum          true
  wired / overstimulated  0.17   0.17
  drained / flat          0.57   0.57
```

it recovers the state-dependent peak exactly. but this run samples a fresh
batch every step, i.e. effectively infinite data, over only 6 features. that
is not the study.

## result 2: study scale — the full model loses

`sim/simulate.py` guesses features for the 45 real places, simulates 80 people
making ~2k pairwise choices, and splits **by user** so cold start is measured
honestly. accuracy on **close calls** (|utility gap| < 0.4) — with 225
place-slots, two random candidates are usually miles apart and every model gets
those right, so overall accuracy saturates near 93% and hides everything:

```
              all pairs   close calls   new users   new, close   params
  bilinear        91.9%         80.8%       93.3%        81.1%    1,483
  partial         94.5%         80.8%       95.9%        85.2%    5,369
  sc-nam          93.0%         74.0%       95.2%        84.4%   27,600
```

the full sc-nam **overfits at realistic sample size** — 74% on close calls,
worse than the linear baseline. 27,600 parameters on 1,632 pairs.

## result 3: what to actually ship

`PartialSCNAM` — a free shape function only for `noise` and `daylight`, where
theory predicts an interior optimum, and state-conditioned linear terms for
everything else. greenery, crowding and privacy are monotone; spending 2k
parameters each to discover that is what broke the full model.

close calls, unseen users, mean of 3 seeds:

```
    256 pairs   bilinear  60.7%   partial  69.4%   sc-nam  66.1%
    512 pairs   bilinear  74.9%   partial  76.2%   sc-nam  70.8%
    960 pairs   bilinear  72.1%   partial  75.7%   sc-nam  73.5%
   1920 pairs   bilinear  79.2%   partial  85.8%   sc-nam  78.4%
```

partial wins at every size, including the smallest. this is the model to build
the app on, and the three-way comparison is the ablation table for the paper.

## study sizing

~500 pairwise choices gets you to ~76% on close calls; ~2,000 gets ~86%. so
roughly **60–70 people × 30 sessions**, and below ~500 the model cannot support
any claim. all numbers on guessed features and simulated preferences, so treat
them as an order of magnitude for planning, not a prediction.

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
