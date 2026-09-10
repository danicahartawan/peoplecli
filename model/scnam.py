"""
state-conditioned neural additive model (sc-nam)

score(context, place) = sum_i phi_i  +  b_user
  phi_i = a_i * ( gamma_i(context) * h_i(f_i) + beta_i(context) )

h_i  : per-feature shape function, a small mlp over ONE input feature
       (neural additive models, agarwal et al., neurips 2021)
gamma, beta : per-feature scale/shift produced from the person's state
       (film conditioning, perez et al., aaai 2018)

why this shape and not a linear model:
  1. phi_i sums exactly to the score, so the explanation IS the arithmetic,
     not a post-hoc approximation like shap over a boosted forest.
  2. h_i is nonlinear, so a feature can have an interior optimum. noise is
     not "less is always better" -- there is an inverted-u (yerkes & dodson
     1908; mehta, zhu & cheema 2012 on moderate ambient noise). a bilinear
     model cannot represent a peak at all.
  3. film lets the state MOVE that optimum instead of only rescaling it.
     overstimulated and understimulated people do not want the same room.
     that interaction is the entire research claim, so it belongs in the
     architecture rather than in a feature-crossing hack.

trained on pairwise choices with a bradley-terry likelihood (bradley & terry
1952) -- people rank two options far more reliably than they rate one.

the final layer is kept linear over the phi vector so a bayesian linear head
can be fitted on top for uncertainty and thompson sampling
(neural-linear, riquelme et al., iclr 2018; chapelle & li, nips 2011).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

FEATURES = ["noise", "light", "greenery", "privacy", "crowding", "seat_comfort"]
TASKS = ["focus", "read", "social"]
D = len(FEATURES)
C = 2 + len(TASKS)          # arousal, valence, task one-hot


class ShapeFn(nn.Module):
    """h_i : R^1 -> R^k. one per feature. sees only its own feature."""

    def __init__(self, k=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.SiLU(),
            nn.Linear(32, 32), nn.SiLU(),
            nn.Linear(32, k),
        )

    def forward(self, x):                       # x: (B,1)
        return self.net(x)                      # (B,k)


class SCNAM(nn.Module):
    def __init__(self, d=D, c=C, k=16, n_users=0):
        super().__init__()
        self.d, self.k = d, k
        self.shapes = nn.ModuleList([ShapeFn(k) for _ in range(d)])
        # state -> per-feature (gamma, beta)
        self.film = nn.Sequential(
            nn.Linear(c, 64), nn.SiLU(),
            nn.Linear(64, d * 2 * k),
        )
        self.head = nn.Parameter(torch.ones(d, k) / k)   # a_i, kept linear
        self.user_bias = nn.Embedding(n_users, 1) if n_users else None
        if self.user_bias is not None:
            nn.init.zeros_(self.user_bias.weight)

    def contributions(self, ctx, feats):
        """returns phi: (B,d). phi.sum(-1) is the score."""
        B = feats.shape[0]
        gb = self.film(ctx).view(B, self.d, 2, self.k)
        gamma, beta = 1.0 + gb[:, :, 0], gb[:, :, 1]
        h = torch.stack([self.shapes[i](feats[:, i:i + 1]) for i in range(self.d)], 1)
        return ((gamma * h + beta) * self.head).sum(-1)          # (B,d)

    def forward(self, ctx, feats, uid=None):
        s = self.contributions(ctx, feats).sum(-1)
        if self.user_bias is not None and uid is not None:
            s = s + self.user_bias(uid).squeeze(-1)
        return s


class Bilinear(nn.Module):
    """baseline: score = w(state) . f  -- the model that cannot see a peak."""

    def __init__(self, d=D, c=C):
        super().__init__()
        self.w = nn.Sequential(nn.Linear(c, 64), nn.SiLU(), nn.Linear(64, d))

    def forward(self, ctx, feats, uid=None):
        return (self.w(ctx) * feats).sum(-1)


# ---------------------------------------------------------------- synthetic world
def true_utility(ctx, f):
    """
    ground truth we are trying to recover. deliberately contains the two things
    the bilinear baseline structurally cannot fit:
      - an interior optimum in noise
      - an optimum whose LOCATION depends on the person's arousal
    """
    arousal, valence = ctx[:, 0], ctx[:, 1]
    focus, read, social = ctx[:, 2], ctx[:, 3], ctx[:, 4]

    # wired -> wants near silence. drained -> wants a bit of ambient buzz.
    mu = 0.62 - 0.50 * arousal
    noise = -3.2 * (f[:, 0] - mu) ** 2

    light = -1.6 * (f[:, 1] - 0.65) ** 2 + 0.35 * f[:, 1]
    green = (0.30 + 0.70 * arousal) * f[:, 2]
    priv = (0.25 + 1.10 * focus + 0.85 * read - 0.90 * social) * f[:, 3]
    crowd = -(0.35 + 1.30 * arousal) * f[:, 4]
    seat = (0.30 + 0.80 * read) * f[:, 5]
    return noise + light + green + priv + crowd + seat + 0.15 * valence


def sample_batch(n):
    ctx = torch.rand(n, C)
    ctx[:, 2:] = F.one_hot(torch.randint(0, len(TASKS), (n,)), len(TASKS)).float()
    fa, fb = torch.rand(n, D), torch.rand(n, D)
    ua, ub = true_utility(ctx, fa), true_utility(ctx, fb)
    # bradley-terry: the choice is stochastic, not a hard argmax
    p = torch.sigmoid((ua - ub) / 0.35)
    y = torch.bernoulli(p)
    return ctx, fa, fb, y, (ua - ub)


def train(model, steps=1500, n=256, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for t in range(steps):
        ctx, fa, fb, y, _ = sample_batch(n)
        d = model(ctx, fa) - model(ctx, fb)
        loss = F.binary_cross_entropy_with_logits(d, y)      # bradley-terry
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def evaluate(model, n=8000):
    """agreement with the NOISELESS preference -- the ceiling is not 1.0 either,
    but both models are scored the same way."""
    ctx, fa, fb, _, gap = sample_batch(n)
    pred = model(ctx, fa) - model(ctx, fb)
    keep = gap.abs() > 0.05                     # ignore genuine ties
    return ((pred[keep] > 0) == (gap[keep] > 0)).float().mean().item()


# ---------------------------------------------------------------- bayesian head
@torch.no_grad()
def fit_bayes_head(model, n=4000, sigma=0.35, prior=1.0):
    """
    neural-linear: freeze the learned phi features, fit a bayesian linear layer
    on top in closed form. gives a posterior we can sample from, which is what
    turns 'explore a bit' into thompson sampling instead of an arbitrary
    epsilon of random picks.
    """
    ctx, fa, fb, y, _ = sample_batch(n)
    X = model.contributions(ctx, fa) - model.contributions(ctx, fb)
    t = (y * 2 - 1)
    A = X.T @ X / sigma ** 2 + torch.eye(model.d) / prior
    S = torch.linalg.inv(A)
    m = S @ (X.T @ t) / sigma ** 2
    return m, S


@torch.no_grad()
def centered(model, ctx, feats, ref=512):
    """
    each shape function carries an arbitrary constant, so raw phi_i is not
    readable as "what this feature contributed". center every feature against
    the average place under the SAME context (standard nam practice): the
    reported number then means "better/worse than a typical berkeley space on
    this feature, for you, right now". ranking is untouched -- the offset is
    constant within a context.
    """
    B = feats.shape[0]
    baseline = model.contributions(
        ctx[:1].repeat(ref, 1), torch.rand(ref, model.d)
    ).mean(0)
    phi = model.contributions(ctx, feats) - baseline
    return phi, baseline.sum()


@torch.no_grad()
def explain(model, ctx, feats, names=FEATURES, top=4):
    phi, intercept = centered(model, ctx, feats)
    phi = phi[0]
    order = phi.abs().argsort(descending=True)[:top]
    return [(names[i], phi[i].item(), feats[0, i].item()) for i in order], intercept.item()


@torch.no_grad()
def optimum_curve(model, ctx, feature=0, grid=41):
    """what value of one feature this person wants RIGHT NOW, holding others mid."""
    xs = torch.linspace(0, 1, grid)
    f = torch.full((grid, D), 0.5)
    f[:, feature] = xs
    phi = model.contributions(ctx.repeat(grid, 1), f)[:, feature]
    return xs[phi.argmax()].item()


def explain_rows(rows):
    for name, phi, val in rows:
        bar = ("\u2588" if phi > 0 else "\u2591") * max(1, min(28, int(abs(phi) * 9)))
        yield f"  {name:<13} obs {val:.2f}", f"{phi:+.2f}", bar


if __name__ == "__main__":
    print("training bilinear baseline ...")
    base = train(Bilinear())
    print("training sc-nam ...")
    scnam = train(SCNAM())

    print(f"\npairwise accuracy vs ground truth")
    print(f"  bilinear   {evaluate(base)*100:5.1f}%")
    print(f"  sc-nam     {evaluate(scnam)*100:5.1f}%")

    m, S = fit_bayes_head(scnam)
    print(f"\nbayesian head fitted. posterior sd per feature:")
    for i, n_ in enumerate(FEATURES):
        print(f"  {n_:<13} {S.diag()[i].sqrt():.3f}")

    # two people, same task, opposite arousal
    print("\nrecovered noise optimum (ground truth in parens):")
    for arousal, label in [(0.9, "wired / overstimulated"), (0.1, "drained / flat")]:
        ctx = torch.tensor([[arousal, 0.5, 0.0, 1.0, 0.0]])
        got = optimum_curve(scnam, ctx, feature=0)
        print(f"  {label:<24} {got:.2f}   ({0.62 - 0.50*arousal:.2f})")

    # one worked explanation
    ctx = torch.tensor([[0.85, 0.30, 0.0, 1.0, 0.0]])          # wired, wants to read
    place = torch.tensor([[0.15, 0.55, 0.40, 0.90, 0.20, 0.95]])
    rows, intercept = explain(scnam, ctx, place)
    total = scnam(ctx, place).item()
    print(f"\noverstimulated, needs to read for an hour")
    print(f"morrison-like place   fit {total - intercept:+.2f} vs a typical space")
    for name, phi, val in explain_rows(rows):
        print(name, phi, val)


class PartialSCNAM(nn.Module):
    """
    the middle ground, and probably the one to actually use at study scale.

    a free shape function costs ~2k parameters per feature. most features do
    not need one: more greenery is monotonically better, more crowding is
    monotonically worse when you are overstimulated -- theory says so and the
    data will not argue. only a few features plausibly have an interior optimum
    (noise, daylight). so give those a shape function and leave the rest as
    state-conditioned linear terms.

    same exact additivity, same explanations, a fraction of the parameters.
    """

    def __init__(self, d, c, free_idx, k=8):
        super().__init__()
        self.d, self.k, self.free = d, k, list(free_idx)
        self.lin = [i for i in range(d) if i not in self.free]
        self.shapes = nn.ModuleList([ShapeFn(k) for _ in self.free])
        self.film = nn.Sequential(
            nn.Linear(c, 48), nn.SiLU(),
            nn.Linear(48, len(self.free) * 2 * k + len(self.lin)),
        )
        self.head = nn.Parameter(torch.ones(len(self.free), k) / k)

    def contributions(self, ctx, feats):
        B, nf = feats.shape[0], len(self.free)
        p = self.film(ctx)
        gb = p[:, : nf * 2 * self.k].view(B, nf, 2, self.k)
        wl = p[:, nf * 2 * self.k :]
        gamma, beta = 1.0 + gb[:, :, 0], gb[:, :, 1]
        phi = torch.zeros(B, self.d, device=feats.device)
        if nf:
            h = torch.stack([self.shapes[j](feats[:, i : i + 1])
                             for j, i in enumerate(self.free)], 1)
            phi[:, self.free] = ((gamma * h + beta) * self.head).sum(-1)
        if self.lin:
            phi[:, self.lin] = wl * feats[:, self.lin]
        return phi

    def forward(self, ctx, feats, uid=None):
        return self.contributions(ctx, feats).sum(-1)
