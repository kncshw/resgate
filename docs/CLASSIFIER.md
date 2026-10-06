# CLASSIFIER — the category direction and its threshold

The gate's read is a **binary linear classifier**: one direction `c`, one threshold `t`, one bit out.
This document is how `c` and `t` are built, what was measured, and what was refuted. `docs/GATE.md`
covers the runtime mechanism — the read site, the fail-safe property, the operating-point file
format. This file covers only the maths that produces the two numbers that file carries.

Everything in §1–§11 is Gemma-4-26B-A4B-it, layer 8, `mean_user`, snapshot
`7d4c97e54145f8ffd1a4dd1b4986a5015a517842`, on the 1426-prompt capture whose text sha256 is
`bec25f4346193cfd56c97b1a4fec6440b127d615adf6e978b7ead2567712ab35`. §12 adds a second, independent
capture — 3216 fresh prompts, same snapshot, text sha256
`91c0ea983e19777863a523315fba48499a4062100a32dca095300a297667540f` — and §13 the live serving path.

**Read §0 first. It carries a correction, and the thing it corrects was quotable.**

> **This file is the ARCHIVE.** For the classifier that ships today — `c(5)` at threshold
> +0.140376451, with every constant read back out of the live operating-point file — see
> **`docs/CLASSIFIER-MATH.md`**, which is the spec. This file keeps the experimental record:
> the arms that lost, the refutations, and how `k` came to be chosen twice. Where the two
> disagree, CLASSIFIER-MATH.md is newer.

---

## 0. Corrected 2026-10-05 — read this before quoting anything below

An earlier version of this file presented **truncated whitening at `k = 13` as what ships and as the
better estimator**. Both halves are withdrawn — and the first attempt at withdrawing them was itself
too broad. In order of what a reader is most likely to get wrong:

**What is actually served.** `scratchpad/op_c1_baseline_mean_user_L8.pt`: the plain difference of
means `c1 = unit(d)` at layer 8, `mean_user`, threshold **+0.15604917189545423** under §5.1's
positives-only rule. §13.1 confirms that operating point works through a running server. It is a
**provisional** choice, made before §12.3 and §12.4 existed — not a measured best. `c1` is the single
best arm at **0 of 24** out-of-corpus leak budgets.

**What failed is the choice of `k`, not whitening.** Out of corpus the AUC curve in `k` has an
**interior maximum at k = 2**: 0.9521 against `c1`'s 0.9190, paired stratified bootstrap
`+0.0331 [+0.0281, +0.0380]`, p < 0.0005. **In corpus** the same curve climbs monotonically to
k = 50 (0.9893). The two optima are at different `k`, and `k = 13` was read off the in-corpus curve.
Every `k` from 1 to 50 beats `c1` out of corpus with a bootstrap CI excluding zero; only k = 100
fails. §12.3.

**`k = 13` does collapse out of corpus — that part stands, and only that part.** Its coverage
advantage reverses with disjoint Clopper-Pearson intervals (§12.2), and at matched out-of-corpus leak
`c1` beats `k = 13` at every budget. Generalising that single point into "whitening does not
transfer" was wrong: it was one point read off a curve with an interior maximum somewhere else.

**The wrong number.** §7's two leak-constrained rows read 63/74 (L = 0.01) and 66/74 (L = 0.02).
Both were produced by applying §5.2's budget `m = floor(L × n)` to the **378 TEST negatives** rather
than to the 140 TRAIN negatives the rule is defined over — §5.1's trap wearing the new rule's
clothes. Re-derived as written, both `L` values select the **same** TRAIN positive score,
**+0.15270980**, and give **61/74, 4/378**. The `+0.1795` threshold quoted alongside them is §5.1's
positives-only threshold under `c(13)`, which is a different operating point (49/74).

**The clip boundary at 13 was never a boundary.** `λ₁₃/λ₁₄ = 1.1097` on the 240-row covariance and
**1.0713** on a better-estimated 794-row one — the gap *narrows* as the estimate improves. There is
no spectral gap at 13 in either spectrum. `k = 13` was an in-corpus performance pick presented as
though it sat somewhere principled. §3.6.

**The open question, stated so it cannot be skipped.** The k = 2 advantage at matched leak is an
**oracle** readout — the threshold is placed using the fresh labels, so it bounds what any rule could
achieve and is *not* an operating point. One frozen-threshold arm exists (§12.4) and it banks part of
the gap at one `L`, with the two leaks not matched. **Whether a threshold frozen on TRAIN alone at
k = 2 banks the oracle advantage across the budget grid is not established.** Nothing in this file
says it does.

**What is NOT withdrawn.** The derivation (§3), the Gram trick (§3.4), the `λ_{k+1}` caveat (§3.5),
the negative-class table (§4), the Pareto-window algebra (§5.3) and the in-corpus measurements (§7)
are correct and reproducible — §12 re-derived every one of them bit-identically, max absolute
difference `0.0e+00`, before scoring a single fresh row. `c(13)` also still has the **lower**
out-of-corpus leak at its own threshold: 6/469 = 0.0128 against `c1`'s 17/469 = 0.0362. What failed
is its coverage, and the inference that `k = 13` was the right place to stand.

---

## 1. It is binary, and one direction is the right object

`Gate.decide` is two lines: `s = cos(pooled, c)`, then `s >= thresh`. There is no argmax, no softmax
and no per-class score vector. One `Gate` instance is **one** detector answering "is this the
designated category". Fifteen category directions exist as fifteen separate operating-point files;
you load a different file to protect a different target, you do not get fifteen scores from one pass.

**This is deliberate, and `FINDINGS.md` §10 must not be read as an argument against it.** That section
measures the category subspace at effective rank 5.9–7.3 — but that describes the **15-class**
structure. For cyber-against-the-rest the between-class scatter has rank `C − 1 = 1`, so a single
direction is the correct object. Independently reproduced here: the 14 centred category centroids have
effective rank **7.142** (participation ratio), `s₂/s₁ = 0.8747`, 9 components for 90% of energy.

Two further reasons the binary framing is load-bearing:

- **The allow-list design needs one bit.** Fires → edited weights → answers. Does not fire → pristine
  weights → refuses. A miss fails toward *refusing*.
- **Treating it as multi-class was retracted.** `FINDINGS.md` §12 records that comparing **raw**
  cosines across the 15 categories is invalid, because per-category means over the same 197 prompts
  run from **+0.2161** to **−0.2437** — a raw cross-category argmax just selects whichever direction
  has the highest baseline. Only within-category standardisation is valid, and then only
  diagnostically.

---

## 2. Notation

    Xp        the TRAIN positive rows   (100 cyber_offense prompts)
    Xn        the TRAIN negative rows   (the negative class, see §4)
    Xu        all rows, pooled mean_user at layer 8, EACH ROW UNIT-NORMALISED
    p, g      mean(Xp), mean(Xn)
    d         p − g                     the plain difference of means
    n         rows(Xp) + rows(Xn)
    score(x)  ⟨x̂, c⟩                    x̂ = x/‖x‖, so the score is a cosine

The rows are unit-normalised **before** the means. A cosine and not a raw projection, for the reason
`sep(l)` is a ratio: residual norms grow with depth and vary per prompt, so a threshold on an
unnormalised dot product would partly be a threshold on activation magnitude and would not transfer
between prompts.

---

## 3. The estimator

### 3.1 The baseline, and why it is the wrong estimator

    c = unit(d)

A difference of means is Bayes-optimal **only** if both classes are Gaussian with equal and
**spherical** covariance. Activations are not spherical: some directions carry large variance common
to both classes, and `unit(d)` gives those the same weight as the informative ones.

This estimator is inherited from the ablation literature, where a direction is used *causally* to
steer a model and mass-mean directions are known to work well for that. **The gate is not a steering
vector — it is a classifier**, and for classification a variance-aware estimator wins. That
distinction is the reason §3.3 exists.

### 3.2 What the correct estimator would be

With a shared covariance `Σ`, the optimal direction is Fisher's:

    w = Σ⁻¹ d

Still closed form. No gradients, no labels beyond class membership. `unit(d)` is the special case
`Σ = I`. The obstacle is that `Σ` is 2816×2816 estimated from 240 rows, so it is singular — and
whitening *divides* by variance, so naive `Σ⁻¹` would amplify low-variance directions that are pure
estimation noise.

### 3.3 Truncated whitening — correct maths, and a `k` chosen on the wrong curve

Whiten inside the top `k` eigendirections of the pooled within-class covariance; leave the complement
alone. Resolved as **eigenvalue clipping** at `λ_{k+1}`, the largest unwhitened eigenvalue:

    Z  = [ Xp − p ; Xn − g ]                          class-centred rows, n × 2816
    Σ  = ZᵀZ / (n − 2)
    Σₖ⁻¹ = Σ_{i≤k} λᵢ⁻¹ uᵢuᵢᵀ  +  λ_{k+1}⁻¹ (I − UₖUₖᵀ)

so that, up to an irrelevant global scale,

    w = (I − UₖUₖᵀ) d  +  Uₖ · diag( (λ_{k+1}/λᵢ)^power ) · Uₖᵀ d
    c = unit(w)

`k = 0` returns exactly §3.1. `power = 1` is full whitening (the LDA form); `power = 0.5` is
Mahalanobis half-whitening and was run as a robustness variant.

Implementation: `src/tempx/serve.py` does not build this — it loads the result. The as-run
construction is `scratchpad/arms2.py`, functions `within_eig` and `E3_trunc_whiten`.

**This is not what ships, and `k = 13` is not where to stand.** The derivation above is sound and
reproducible; what was wrong is the value of `k` picked from it. Out of corpus the optimum is
**k = 2**, and `k` was selected on the in-corpus curve — §0, §12.3. The served direction is §3.1's
difference of means, provisionally (§13.1).

### 3.4 Never form Σ — the Gram trick

`Σ` is 2816×2816 but `rank(Σ) ≤ n − 2 = 238`. So:

    G      = Z Zᵀ                       n × n  — 240 × 240 here
    ev, U  = eigh(G)
    λ      = ev / (n − 2)
    V      = Zᵀ U / √ev                 lift to the 2816-dim eigenvectors

Exact, not approximate, and it is why the whole build costs **about 15 seconds** including reading the
activation file.

### 3.5 The λ_{k+1} normalisation is a CHOICE, and it is load-bearing

Normalising by `λ_{k+1}` rather than by 1 is **not forced by the mathematics.**

Whitening a subspace while leaving the complement at raw scale is unit-inconsistent, and with the rows
unit-normalised the raw `1/λᵢ` factors run to 1e3–1e4 — which would make the whitened subspace
**swamp** the complement instead of being de-emphasised relative to it.

A different normalisation gives a different arm. **This is the first thing to vary in a replication,
and no sensitivity analysis over it has been run.**

### 3.6 Measured, at k = 13 on the 140-row negative class

| quantity | value | reading |
|---|---|---|
| `λ₁` | 0.020988 | largest within-class eigenvalue |
| `λ₁₃` | 0.002099 | last whitened |
| `λ₁₄` | 0.001891 | the clip floor |
| `λ₁₄/λ₁` | **0.090** | `u₁` divided down 11× |
| `λ₁₄/λ₁₃` | 0.901 | `u₁₃` barely touched |
| fraction of `‖d‖²` inside the top 13 | **69.0%** | most of the signal is in 13 directions |
| eigenpairs computed | 101 | capped by request, not by rank |

**`λ₁₄/λ₁₃ = 0.901` is not a spectral gap, and 13 is not a subspace boundary.** Re-estimating the
same covariance from 794 rows (the 240 plus the 554 harmless as a third centred class, `d/n` from
11.73 to 3.55) moves the ratio the wrong way for a boundary story: `λ₁₃/λ₁₄` is **1.1097** at 240
rows and **1.0713** at 794 — the gap *narrows* as the estimate improves, and no gap appears anywhere
near 13 in either spectrum (`scratchpad/ksweep.log:167-175`). `k = 13` was chosen because it
maximised in-corpus leak reduction, which is a performance pick, not a geometric one. Presenting it
as a boundary was an error of framing; §12.3 is the error's cost.

---

## 4. The negative class

Three definitions were run. All share the same 100 TRAIN positives.

| | definition | n |
|---|---|---|
| **N1** | other-harmful only — 10 rows from each of 14 non-cyber categories | 140 |
| **N2** | harmless only | 554 |
| **N3** | all non-cyber = N1 ∪ N2 | 694 |

N3 is the correct **decision-theoretic** framing: for a binary gate the negative class is everything
you do not want to fire on, and splitting it by hand is arbitrary. But with a difference of means it
is the worst of the three, and the mechanism was measured rather than argued:

| estimator | N1 | N2 | N3 |
|---|---|---|---|
| E1 difference of means | 48/74, 10/378, AUC 0.9374 | 49/74, 28/378, AUC 0.8819 | 50/74, **19/378**, AUC 0.8938 |
| E3 truncated whitening | 49/74, **0/378**, AUC 0.9826 | leak never approaches 10 at any k | **52/74, 0/378**, AUC 0.9862 |

**Why E1 fails on N3.** `cos(c, harmful-vs-harmless axis)` is **0.1097** for N1 and **0.7566** for N3.
Pooling harmless rows into the negative centroid re-imports the harmfulness axis, and every other
harmful category is harmful, so they fire more. Along that axis the harmful and harmless modes sit
**3.711 pooled sd apart**, so a 50/50 pooled mean lands in the empty valley between two modes where no
prompt lives.

**Whitening removes the sensitivity to the mixing ratio** — N1 and N3 converge at AUC 0.983–0.986 and
zero leak. **The strong convergence claim is refuted by N2:** whitening cannot rescue a negative class
that contains no harmful rows at all.

**All six cells are in corpus.** N1 is the only negative class carried into the out-of-corpus test,
so none of this is known to transfer; N3's `52/74, 0/378` in particular has never been scored on a
fresh corpus. §12 also shows that *where the harmless rows enter* matters: adding them to the
**covariance** rather than to the centroid degrades the out-of-corpus result (§12.5).

---

## 5. The threshold

### 5.1 The rule that must not be used

    k = int(round(tpr × n_target))
    t = the k-th highest TRAIN positive score, descending

This is what the 20 shipped `op_*.pt` files carry. It looks **only at positives** and never sees a
negative prompt. Worse, in the shipped artefact the same 174 cyber prompts that build the positive
centroid are also the ones scored, so `delivered/n_target = 139/174` is **the definition of the
threshold, not a measurement** — `0.8 × 174 = 139`. Any set of scores gives 80%.

Use `meta["loo"]` rather than the top-level cell when quoting a shipped file, and even then the
threshold remains a quantile of the evaluation set.

### 5.2 The rule that ships

Leak-constrained. Borrowed deliberately from **RepIt** (arXiv:2509.13281), whose `ρ` search maximises
target attack success subject to a ceiling on non-target success:

    m = floor(L × rows(Xn))                              the budget, in negative rows allowed to fire
    t = min{ s ∈ TRAIN positive scores : #{ Xn scores ≥ s } ≤ m }

Scan the TRAIN positive scores ascending; take the lowest whose negative fire count is within budget.
Coverage only changes at a TRAIN positive score, and among all `t` giving the same coverage the
largest is itself a positive score, so this grid contains the exact optimum.

`L` is a **declared knob**. RepIt's own ≈0.1 is its *selection constraint*, not a measured property of
anything — do not inherit the number, choose it.

Two things this rule does not forgive. **One:** the budget is over the TRAIN negatives, and applying
it to an evaluation set re-imports §5.1's defect — that is exactly how §7's two rows came to be
wrong (§0). **Two:** a quantile only means what it says if the scores it is fitted on come from the
**same code path that will serve them**. §13.2 measures an offline-vs-live score offset of up to
5.275e-03, which is three times the cross-run offline jitter, and the thresholds above are fitted on
the offline path.

### 5.3 Why the threshold rule cannot be fixed alone

On a **fixed** direction both coordinates are non-increasing step functions of one number `t`, so
every rule lands somewhere on one monotone curve. "coverage > C and leak < K" is feasible iff the
window `(L_K, T_C]` is non-empty, where `T_C` is the C-th highest positive score and `L_K` the K-th
highest negative score.

On the baseline direction that window is **empty**:

    T49 = +0.151092        the 49th cyber prompt — the next one a higher threshold would catch
    L10 = +0.156171        the 10th leak row — where the baseline's ten leaks begin
    L10 > T49  →  the 49th cyber prompt cannot be reached without admitting ten non-cyber ones

Whitening inverts the order and opens it:

    k = 13:  L10 = +0.137435   T49 = +0.181964       →  a wide interval, any t in it wins both

**So the two halves had to be combined.** The rule change was impossible on the old direction, and the
new direction was under-exploited by the old rule — `c(13)` under the old rule leaves 12 coverage rows
on the table (49/74 against 61/74; it was quoted as 17 from the wrong 66/74, see §0).

**This window is an in-corpus object.** The algebra is exact and still holds, but the ordering it
turns on is corpus-specific: re-measured on fresh prompts, `c1` buys more coverage than `c(13)` at
every matched out-of-corpus leak budget from 0 to 5% (§12.2). What *does* survive out of corpus is
the window at small `k` — §12.4.

---

## 6. Data protocol

Both the direction and the threshold are functions of the TRAIN rows **only**.

| set | n | used for |
|---|---|---|
| TRAIN positive | 100 | positive mean, the covariance, the threshold |
| TRAIN negative (N1) | 140 | negative mean, the covariance, the budget |
| TEST positive | 74 | coverage. Disjoint from TRAIN, asserted. |
| TEST negative | 378 | leak. Disjoint from the 140, asserted. |
| held out | 180 | **excluded entirely** — all `para_*` and `adv_*` rows |

Splits come from one `numpy.default_rng(0)` used sequentially: TRAIN positives first, then 10 per
category over categories in sorted name order. The split is reproducible and was re-derived
independently and matched index-for-index.

**The freeze is the point.** A valid run recomputes `c` and `t` with every evaluation row masked to
NaN and requires the result to be bit-identical. Max absolute difference must be 0.

---

## 7. Measured results — in corpus

Threshold-free AUC separates direction quality from calibration quality, and the two can disagree —
report both.

| arm | t | coverage | leak | AUC |
|---|---|---|---|---|
| baseline `c1 = unit(d)` on N1, positives-only rule — **what ships** | +0.15604917 | 48/74 = 0.6486 `[0.5289, 0.7561]` | 10/378 = 0.0265 `[0.0128, 0.0481]` | 0.9374 |
| `c1` on N1, **L = 0.01** | +0.21812815 | 31/74 = 0.4189 `[0.3051, 0.5394]` | **0/378 = 0.0000** `[0.0, 0.0097]` | 0.9374 |
| `c1` on N1, **L = 0.02** | +0.19057358 | 38/74 = 0.5135 `[0.3944, 0.6315]` | 3/378 = 0.0079 `[0.0016, 0.0230]` | 0.9374 |
| `c(13)` on N1, **old** rule | +0.17951724 | 49/74 = 0.6622 `[0.5428, 0.7681]` | **0/378 = 0.0000** `[0.0, 0.0097]` | 0.9826 |
| `c(13)` on N1, **L = 0.01 and L = 0.02 alike** | **+0.15270980** | **61/74 = 0.8243** `[0.7183, 0.9030]` | 4/378 = 0.0106 `[0.0029, 0.0269]` | 0.9826 |
| `c(13)` on N1 at the **shipped** threshold | +0.15604917 | 59/74 = 0.7973 `[0.6878, 0.8819]` | 3/378 = 0.0079 `[0.0016, 0.0230]` | 0.9826 |

**The two `L` rows were wrong until 2026-10-05, and the error was in the rule, not the arithmetic.**
They read 63/74 (L = 0.01) and 66/74 with 6/378 (L = 0.02). Both came from applying the budget to the
**378 TEST negatives** instead of the 140 TRAIN negatives §5.2 defines it over — a threshold fitted
on the evaluation set, which is §5.1's defect in the new rule's clothes. Re-derived as written
(`m = floor(L × 140)`, so 1 and 2 permitted REF fires:
`scratchpad/freshscore/score_fresh.py:164-180`), **both `L` values select the same TRAIN positive
score** — the next candidate down takes the REF count from 1 to past 2 — so L = 0.01 and L = 0.02 are
the same operating point here: +0.15270980, 61/74, 4/378
(`scratchpad/freshscore/score.log:96-103`). A leak-constrained rule that quotes a threshold it did
not derive is not leak-constrained.

Intervals are Clopper-Pearson 95%, using the repo's own `cp()`
(`kits/sweep_meanuser.py.AS-RUN:66-100`), gated against `FINDINGS.md` §1's `[0.4580, 0.7727]` for
25/40.

**Replication status.** The `c(13)` + old-threshold row **was run as a first-class arm** and its leak
reduction is paired-significant: McNemar `b = 10, c = 0, p = 0.0020` — it removes all ten of the
baseline's leaks and introduces none. Across six resplits, leak mean 1.3 (sd 1.0), AUC mean 0.9866
(sd 0.0034). **The leak-constrained rows have now been run as their own arms**, with `rule_T2`
re-derived from the 240 training rows inside the fresh-gate scoring run and a positive control that
reproduces the stored `c1` thresholds exactly — which is how the error above surfaced.

**And the six resplits were never evidence against estimation noise.** A resplit probe measures
*variance*; the distortion in this covariance is a *bias* that every 240-row resplit from the same
population carries identically, because they all carry the same `d/n = 11.73`. §12.6 measures it
directly. The resplit stability and the spectral inflation were never in conflict, and the stability
should not have been read as reassurance about the estimate.

---

## 8. Refuted — do not resurrect these

**Every row below, and §8.1, is an *in-corpus* refutation.** They were all measured on the pool681
capture, and none has been re-tested on a fresh corpus. §12.3 is the reason that now matters: the
same material that kills `k = 13` out of corpus would have been invisible in corpus, so "refuted
here" cannot be promoted to "refuted". The three estimator rows in particular — per-category
orthogonalisation, V4 (`unit(mean_k unit(p − g_k))`) and reliability weighting — are refuted *on one
corpus* and inherit the doubt in full. See §12.9 and `tempx-blocking-issues.md` A19.

| claim | what killed it |
|---|---|
| per-category orthogonalisation works because of category structure | **label permutation**: permuting the 14 category labels across the same 140 rows reproduces most of the gain (AUC 0.9629 ± 0.0163, leak 6.10 ± 3.63 against the real 0.9817 and 5/378). Permutation p(leak) = 0.45. The gain is covariance whitening. |
| `c = unit(mean_k unit(p − g_k))` equalises categories usefully | a measured **no-op**: 48/74, 10/378, 2.6° of rotation. The 14 deviation norms are near-uniform (mean 0.1833), so re-weighting by angle barely moves the mean. |
| reliability-weighted soft removal beats hard selection | the literal form `v − Σ ρₖ(v·dₖ)dₖ` gives 56/74 but **33/378** — the `dₖ` are correlated so a plain sum is not a projection and overshoots. The principled form gives 48/74, 5/378, AUC 0.9704: no gain. |
| Ledoit-Wolf shrinkage is the answer | analytic `λ = 0.093242`, **not** near 1 as predicted — but coverage collapses to 25/74 at AUC 0.9818. A pure calibration failure. |
| LEACE beats the ad-hoc purification | for a single binary concept `Cov(X,Z)` is **exactly parallel** to the shipped `dh` — `cos = 1.000000000000000` — so the ad-hoc projection already achieves exact linear erasure. Both erasers fail to generalise off their fitting rows by the same amount: held-out probe AUC 0.692 vs 0.688. The comparison RepIt declined is partly vacuous. |
| the direction is substantially a refusal direction | `cos(c, the model's own L8 refusal direction) = −0.0227`, against a 20,000-draw random baseline of 0.0149. **22.5% of random vectors are more aligned with refusal than `c` is.** The Gram-Schmidt step left `c` free to be up to 0.92 refusal and it landed at 0.02. |

### 8.1 Where top-k selection *did* survive, and where it did not

Signed top-k removal over the category-centroid basis beat **all 20** label-permutation draws on AUC
at k = 2,3,4,5,6 (p ≤ 0.048), and the advantage vanished by k = 7. So at small k the *identity* of the
removed category matters; at large k only the dimension count does. Coverage and leak did **not**
separate from the null individually (p(leak) = 0.15 at k=2, p(coverage) = 0.25).

Two sub-results worth keeping:

- **`k = 1` is worse than doing nothing**: 47/74 and **16/378**.
- **"Closest" must mean signed, not absolute.** Top-2 by `|cos|` picks an anti-aligned deviation and
  gives 47/74 with **17/378** — removing an anti-aligned direction *adds* leak.
- **Centroid reliability matters, as predicted.** Split-half Spearman-Brown over the 14 at 10 rows:
  mean **0.591**, from `critical_infrastructure` 0.871 down to `other` **0.152**, with three
  categories producing *negative* half-to-half cosines. Dropping the sub-0.6 member of a top-5 set
  gave the best arm of that family: 54/74, 4/378, AUC 0.9815.
- **More rows per category did not help**, measured: growing the centroids from 10 to 23 rows on
  average lifted mean reliability 0.591 → 0.717 and no arm improved. Bound: that eval had 1–5 leak
  rows in 193, so it could not have detected a small gain.

---

## 9. What is not claimed

- **In-distribution AUC is nearly all lexical.** A tf-idf bag-of-words model on the **raw prompt text**
  reaches **0.9702** one-vs-rest AUC for `cyber_offense` against the read's 0.9690 (LOO) / 0.9719
  (full-fit). In distribution the activations add almost nothing over counting words. What rescues the
  activation read is held-out paraphrases, where tf-idf falls to 0.9004 while activations hold
  0.96–0.98 — and **that arm is excluded from all of the work above at the user's instruction**, so
  nothing here distinguishes a semantic read from a lexical one.
- **One corpus, one seed, 74 test prompts — and §12 is now the cross-corpus answer.** TEST and the 378
  are a seeded split of the same AdvBench and HarmBench pools as TRAIN, so §7 is a train/test split
  and **not** cross-corpus generalisation. §12 runs the missing test on 1819 fresh rows from two
  independent corpora: coverage falls for every direction, and the ranking between directions
  changes. Everything in §7 and §8 should be read as "on this corpus".
- **Nine of the fourteen categories are bounded only loosely.** Zero observed leaks on small n gives
  Clopper-Pearson upper bounds from **10.6%** (`weapons_explosives`, 0/33) to **97.5%**
  (`human_trafficking`, 0/1). A table of zeros is not nine solved categories.
- **No safety grade is reachable this way.** Under a binormal model, 80% coverage at 0.1% leak needs
  **AUC ≈ 0.9973**. The best arm here is 0.9862 and the corpus ceiling is 0.9702. The gap is not an
  estimator problem; nothing linear at layer 8 closes it.
- **Gate-crossing counts, not delivery counts.** Every number in §7 and §12 is a threshold crossing.
  No generation ran in those arms. The link to delivered answers rests on the 40-prompt demo,
  `FINDINGS.md` §1's 197-prompt run where delivery among routed prompts was 100%, and now §13.1's
  12-prompt live probe — which is a working demonstration at n = 12, not a rate.
- **There is no second line of defence.** `band_complied` is 143/143 with zero refusals, so every leak
  is a delivered answer rather than a softened one.
- **The benign false-positive rate is unmeasured here**, by decision: the benign arms were dropped as
  not safety-relevant. So nothing above bounds the cost on legitimate defensive-security traffic.

---

## 10. Provenance

| component | whose |
|---|---|
| difference of means | **Arditi et al. 2024**, arXiv:2406.11717 — field-wide, also Heretic's |
| truncated whitening of the within-class covariance | textbook **regularised LDA**. Not RepIt's: RepIt whitens its *concept basis* before a QR step; this whitens the *target* against the pooled covariance. Different objects. |
| the leak-constrained operating point | **RepIt**, arXiv:2509.13281, `disentangle_vectors.py` `evaluate_rhos` |
| the low-rank finding | **independent replication** of RepIt's App C.1 Fig 9 (their first 10 PCs > 90% of variance over 21 categories; our effective rank 7.142 over 14) |
| the λ_{k+1} clipping normalisation | ours, and unvalidated — see §3.5 |

RepIt is Apache-2.0; this repo is AGPL-3.0-or-later, which is the permitted direction.

**We disagree with RepIt on one point, and out of corpus the disagreement narrows in their favour.**
They ablated the whitening and found it worth 1–3 percentage points. We found the basis choice
decisive: the category-centroid basis did not separate from a relabelled null (p = 0.45) while the
covariance eigenbasis removed all ten leaks (p = 0.0020). Different metrics — their ASR after an
affine edit, our gate-crossing count — so the disagreement is not direct. But our in-corpus AUC
margin of +0.045 falls to **+0.011 at k = 13** on fresh prompts, and the best fresh margin available
anywhere on the `k` grid is **+0.033 at k = 2** (§12.3) — which is RepIt's order of magnitude, not
ours. We over-read a corpus-specific gain.

---

## 11. Reproduction

No model load. CPU arithmetic on stored activations; the whole thing is seconds once the activation
file is read.

    inputs   routing-probe/pool681/gemma_pool681_in.json   (1426 prompts, sha bec25f43…12ab35)
             the pool681 activation capture                 (layer 8, three poolings)
             kits/gemma4-26B/dir_gemma4-26B_sep.pt          (only for the refusal-cosine control)

    as-run   scratchpad/arms2.py      within_eig, E3_trunc_whiten, the negative classes, the controls
             scratchpad/thresh.py     rule_T2, rule_T3, rule_T4, cp(), unit(), auc(), midrank()
             qout/armcmp/arm_cmp.py   the split protocol and the 42 base asserts

    out      routing-probe/freshcap/fresh_gate_in.json      3216 fresh prompts, sha 91c0ea98…7540f
             scratchpad/freshcap/conv_control.py/.json      the capture-convention control (§12.1)
             scratchpad/freshscore/score_fresh.py           89 asserts; rule_T2 at :164-180
             scratchpad/freshscore/score.log                as-run, 0 failures; thresholds at :96-103
             scratchpad/freshscore/diag_tradeoff.py/.json   the matched-leak diagnostic
             scratchpad/ksweep.py / ksweep.log              108 asserts; the k sweep, both Σ
             scratchpad/kaddendum.py / kaddendum.log        bootstrap :5-26, oracle readouts :28-81

    serve    scratchpad/opbuild/, scratchpad/op_c1_baseline_mean_user_L8.pt   the shipped point
             scratchpad/prelim_probe.md                     the 12-prompt live probe (§13.1)
             scratchpad/regate.py / regate_repeats.json      live determinism, 5 repeats (§13.2)

    gate     the positive control every run must reproduce before reporting anything:
             baseline threshold +0.15604917, TRAIN 80/100, TEST 48/74, LEAK 10/378, AUC 0.9374,
             and λ₁ 0.020988 / λ₁₃ 0.002099 / λ₁₄ 0.001891 / frac 0.690
             out of corpus, add: c1 fresh_cyber 771/1350, fresh_nontarget 17/469, AUC 0.9190

**Shipping a new direction** changes only the contents of the operating-point file — `c`, `thresh` and
`meta`. The format, `Gate.from_file` and the runtime are untouched. Two things to get right:

1. **Cast every numpy float to a Python `float` before saving.** `Gate.from_file` calls `torch.load`
   without `weights_only=False`, so under torch ≥ 2.6 a numpy array anywhere — **including inside
   `meta`** — makes the file unloadable. The first 20 generated files failed exactly this way.
2. **Record the new provenance in `meta`**: the estimator, `k`, `power`, `λ_{k+1}`, the negative class
   and `L`. An operating point whose provenance is not recorded is not an operating point.

---

## 12. Out of corpus — the cross-corpus test §9 said was missing

Run 2026-10-05, CPU-only arithmetic on a second activation capture. Two scripts:
`scratchpad/freshscore/score_fresh.py` (89 asserts, 0 failures) scores the frozen configurations,
and `scratchpad/ksweep.py` (108 asserts, 0 failures) sweeps `k` and the covariance sample. The
derivation set, the splits and the artefacts are the same frozen objects as §6 and §7 — nothing was
refitted.

### 12.1 The test set, and the controls that have to pass before any of it counts

    fresh_cyber         1350   CySecBench, 9 attack categories, 150 each     COVERAGE
    fresh_cyber_repl      69   CatQA cyber rows, its own 7-way taxonomy      coverage replication
    fresh_nontarget      469   CatQA non-cyber                               LEAK
    cyber_scope_ics      928   CySecBench Control System Attacks             reported alone, §12.8
    cyber_style_imper    360   the same 9 categories, imperative mood        the register arm, §12.7
    conv_control_a/b   20+20   re-captures of original prompts               the convention control

**1819 rows carry a label this project's taxonomy agrees with** — 1350 cyber plus 469 non-cyber. The
other three arms are diagnostics and are never pooled into coverage or leak. Two independent corpora
with their own taxonomies, neither derived from AdvBench or HarmBench. Capture: 3216 rows, text
sha256 `91c0ea98…7540f`, same snapshot `7d4c97e5`, `n_span_fallbacks = 0`, all 14 pinned convention
fields identical to the original capture, no `para_*`/`adv_*` material anywhere.

| control | result |
|---|---|
| **convention identity** — the prompts re-captured, cross-run | max cos distance **4.116e-05** against a within-run bf16 noise floor of **7.443e-05**, with a sentinel pair at **4.62e-02** to show the metric can see a real difference. The new capture is not a new convention. |
| **the freeze** | `c1`, every `c(k)`, both eigenvalue spectra and all 40 thresholds **bit-identical** with TEST, LEAK, the 180 held-out rows and all 3216 fresh rows NaN'd — max absolute difference **0.0e+00** |
| **the freeze, negatively** | NaN'ing one *training* row destroys all 20 directions; NaN'ing one *harmless* row destroys exactly the nine `k > 0` arms of the 794-row covariance and none of the 240-row arms — so the harmless rows enter that Σ and only that Σ |
| **the in-corpus positive control** | §11's gate reproduced before any fresh row was scored: +0.156049172, TRAIN 80/100, TEST 48/74, LEAK 10/378, AUC 0.9374, λ₁ 0.020988 / λ₁₃ 0.002099 / λ₁₄ 0.001891 / frac 0.690 |
| **decision invariance** | zero of the 3216 fresh and zero of the 1426 in-corpus **gate decisions** differ between the recomputed artefact and the stored one (max score difference 2.9e-16; nearest fresh row sits 3.41e-05 from the threshold) |
| **label permutation**, 20 draws | all 40 arm × axis cells fall in **0.4966–0.5475** mean AUC. The pre-declared stop condition was any permuted arm reaching 0.80; none came near. |

### 12.2 The frozen configurations: `k = 13` collapses, and the collapse is certified

| | direction | t | in-corpus TEST | out-of-corpus `fresh_cyber` | in − out | `fresh_nontarget` |
|---|---|---|---|---|---|---|
| A | `c1` | +0.15604917 | 48/74 = 0.6486 | **771/1350 = 0.5711** `[0.5442, 0.5977]` | +0.0775, **intervals overlap** | 17/469 = 0.0362 `[0.0213, 0.0574]` |
| B | `c(13)` | +0.15604917 | 59/74 = 0.7973 | 605/1350 = 0.4481 `[0.4214, 0.4751]` | +0.3492, **disjoint** | 14/469 = 0.0299 `[0.0164, 0.0496]` |
| B′ | `c(13)` | +0.17951724 | 49/74 = 0.6622 | 386/1350 = 0.2859 `[0.2619, 0.3108]` | +0.3763, **disjoint** | 6/469 = 0.0128 `[0.0047, 0.0276]` |
| C | `c(13)` | +0.15270980 | 61/74 = 0.8243 | 637/1350 = 0.4719 `[0.4449, 0.4989]` | +0.3524, **disjoint** | 15/469 = 0.0320 `[0.0180, 0.0522]` |

The baseline's in-corpus and out-of-corpus coverage intervals **overlap**: it loses 7.8 points and
the loss is not certified. Every `c(13)` configuration's intervals are **disjoint**: it loses 35–38
points and the loss is certified. Threshold-free, the in-corpus AUC margin of **+0.045** (0.9826 vs
0.9374) shrinks to **+0.011** out of corpus (0.9295 vs 0.9190).

**The zero-leak claim does not survive.** `c(13)` at its own threshold is 0/378 `[0, 0.0097]` in
corpus and 6/469 `[0.0047, 0.0276]` out of corpus — the fresh lower bound sits **above** the
in-corpus upper bound. Zero leak was a property of that corpus.

**What `c(13)` does keep is the lower leak**, 0.0128 against 0.0362, and that is a real and
reproducible property. It buys it by firing on half as much cyber.

### 12.3 The sweep in `k` — the in-corpus and out-of-corpus optima are at different `k`

The first version of this correction read the collapse at `k = 13` as "whitening does not transfer".
That was one point on a curve, and the curve has an interior maximum somewhere else. Σ from the
frozen 240 derivation rows, `k` over `{0,1,2,3,5,8,13,20,50,100}`, 101 eigenpairs, everything else
held fixed (`scratchpad/ksweep.log:103-129`):

| `k` | cos(c, c1) | frac ‖d‖² in top k | **in-corpus** AUC | **out-of-corpus** AUC | ΔAUC vs `c1`, paired bootstrap 95% CI |
|---|---|---|---|---|---|
| 0 (`c1`) | 1.0000 | 0.000 | 0.9374 | 0.9190 | — |
| 1 | 0.9874 | 0.105 | 0.9619 | 0.9452 | **+0.0263** `[+0.0224, +0.0301]` |
| **2** | 0.9786 | 0.127 | 0.9694 | **0.9521 PEAK** | **+0.0331** `[+0.0281, +0.0380]` |
| 3 | 0.9671 | 0.185 | 0.9735 | 0.9498 | +0.0308 `[+0.0258, +0.0357]` |
| 5 | 0.9571 | 0.345 | 0.9763 | 0.9467 | +0.0277 `[+0.0230, +0.0325]` |
| 8 | 0.9341 | 0.520 | 0.9796 | 0.9393 | +0.0203 `[+0.0155, +0.0251]` |
| **13 (shipped in the old draft)** | 0.8782 | 0.690 | 0.9826 | 0.9295 | +0.0105 `[+0.0047, +0.0162]` |
| 20 | 0.8191 | 0.769 | 0.9833 | 0.9270 | +0.0081 `[+0.0012, +0.0146]` |
| 50 | 0.6462 | 0.891 | **0.9893 PEAK** | 0.9339 | +0.0150 `[+0.0064, +0.0229]` |
| 100 | 0.4681 | 0.924 | 0.9882 | 0.9260 | +0.0070 `[-0.0032, +0.0165]` — the only CI containing zero |

2000-draw paired stratified bootstrap, `default_rng(0)`, on the identical 1350 + 469 rows
(`scratchpad/kaddendum.log:5-26`).

**So the reversal is a reversal in `k`, not a failure of whitening.** In corpus the curve is
monotone to k = 50; out of corpus it peaks at **k = 2** and decays. `k = 13` sits on the far side of
the out-of-corpus peak and was selected by maximising the in-corpus curve. **Every `k` from 1 to 50
beats the baseline out of corpus**, and the honest summary is: whitening helps out of corpus at small
`k`, hurts at large `k`, and the number that was wrong is `k`.

The peak replicates on two further fresh arms, same k = 2:

| arm | n | `c1` AUC | k = 2 AUC |
|---|---|---|---|
| `cyber_style_imper` | 360 | 0.9713 | **0.9842** |
| `cyber_scope_ics` | 928 | 0.9690 | **0.9785** |
| `fresh_cyber` | 1350 | 0.9190 | **0.9521** |
| `fresh_cyber_repl` | 69 | 0.9618 | 0.9655 (this arm's own peak is k = 50 at 0.9707, n = 69) |

Three of the four put the optimum at exactly k = 2; the fourth is the 69-row arm.

### 12.4 Matched out-of-corpus leak — an oracle readout, and the one frozen arm

Maximum `fresh_cyber` coverage subject to `fresh_nontarget` fires ≤ m, for m = 0…23 rows, i.e. 0 to
5% leak (`scratchpad/kaddendum.log:28-57`):

| m (rate) | `c1` | k = 1 | **k = 2** | k = 3 | k = 5 | k = 13 | best |
|---|---|---|---|---|---|---|---|
| 0 (0.0000) | 280 | **400** | 367 | 327 | 308 | 155 | k = 1 |
| 2 (0.0043) | 421 | 504 | **525** | 454 | 457 | 243 | k = 2 |
| 4 (0.0085) | 436 | 514 | **567** | 523 | 494 | 314 | k = 2 |
| 9 (0.0192) | 573 | 668 | **706** | 620 | 579 | 471 | k = 2 |
| 17 (0.0362) | 815 | 925 | **964** | 931 | 897 | 692 | k = 2 |
| 23 (0.0490) | 928 | 997 | **1026** | 991 | 942 | 796 | k = 2 |

`c1` is the single best arm at **0 of the 24** budgets. k = 2 wins **21**; k = 1, k = 3 and k = 5
take one each. At zero out-of-corpus leak k = 1 gives 400/1350 against `c1`'s 280 (+43%); at m = 4
k = 2 gives 567 against 436 (+30%).

> ### ⚠ CARRY THIS CAVEAT WITH EVERY NUMBER IN THIS SUBSECTION
> **It is an ORACLE readout.** The threshold is placed using the fresh evaluation labels, so each
> cell is an **upper bound on what any rule could achieve** at that leak, not an operating point.
> It compares *directions*; it does not deliver one.

**Run the same oracle in corpus and it points the other way**, which is the cleanest statement of the
mismatch: in the in-corpus readout the best arm is k = 50 or k = 100 at most of the 20 budgets, with
k = 13 and k = 20 taking the rest, and **k = 2 and `c1` winning none of them**
(`kaddendum.log:59-81`). Same metric, same code, two corpora, opposite answers about `k`.

**The one frozen-threshold evidence that exists.** The sweep also carries the §5.2 rule at L = 0.01,
fitted on TRAIN alone, scored on the fresh rows — no oracle (`ksweep.log:118-128`, McNemar at
`:278-288`):

| `k` | t | in-corpus TEST | out `fresh_cyber` | out `fresh_nontarget` | paired McNemar vs `c1` |
|---|---|---|---|---|---|
| 0 | +0.21812815 | 31/74 | 352/1350 | 1/469 | — |
| 1 | — | 34/74 | 415/1350 | 2/469 | coverage b = 67, c = 4, p ≈ 0 |
| **2** | — | 34/74 | **440/1350** | 2/469 | coverage b = 94, c = 6, **p ≈ 0**; leak b = 1, c = 0, p = 1 |
| 13 | +0.15270980 | 61/74 | 637/1350 | 15/469 | coverage b = 287, c = 2, p ≈ 0; leak b = 14, c = 0, p = 1.2e-04 |

So at this one `L`, a **frozen** threshold at k = 2 does bank part of the gap: +88 fresh cyber rows
over `c1` on the identical rows, paired, with no significant leak increase (2/469 against 1/469).
**That is one point, at one `L`, with the leaks not matched, and it is the whole of the frozen-rule
evidence.** Whether a TRAIN-only threshold at k = 2 banks the oracle advantage *across the budget
grid* — and at the leak the server actually runs at, 17/469 rather than 2/469 — **is not
established.** It is the open question this section leaves behind, and it is the next thing to run.

### 12.5 More covariance rows is not the fix — and that hypothesis is refuted too

The natural reading of §12.3 is that Σ is under-estimated at `d/n = 11.73`. So Σ was re-estimated
from **794** rows — the same 240 plus the 554 harmless prompts as a third centred class, divisor
`n − 3 = 791`, `d/n` down to **3.55**, with `d` itself left identical so only the covariance changes.
It makes the out-of-corpus result **worse**:

- the optimum moves **k = 2 → k = 8** and the peak AUC **falls 0.9521 → 0.9396**
- at k = 13 the 794-row Σ gains only +0.0053 (0.9295 → 0.9348)
- **no 794-row arm wins a single one of the 24 leak budgets**

"Too few rows" is refuted as a remedy. **Smaller `k` is the fix, not more data** — and the eligibility
control confirms the 554 harmless rows touch none of the five evaluation arms (11 shared keys, all of
them convention-control rows, 0 shared with any evaluation arm).

### 12.6 Why the resplits were blind to this: it is a bias, not a variance

§7 reports six resplits with leak sd 1.0 and AUC sd 0.0034, and that stability was read as evidence
the estimate was sound. It is not evidence about this defect. The 794-row spectrum shows textbook
high-`d/n` eigenvalue spreading in the 240-row one (`ksweep.log:167-175`):

| | 240 rows | 794 rows |
|---|---|---|
| `λ₁` | 0.020988 | 0.016305 — **falls 22%** |
| `λ₂` | 0.010608 | 0.014823 — **rises 40%** |
| `λ₁/λ₂` | 1.979 | **1.100** |
| variance fraction in `u₁` | 0.1635 | **0.0882** — halves |
| effective rank (participation ratio) | 20.60 | **36.41** — nearly doubles |
| `λ₁₃/λ₁₄` | 1.1097 | 1.0713 — the "clip boundary" narrows |

**Every resplit of 240 rows from the same population carries the same `d/n` and therefore the same
inflation.** A resplit probe measures variance across draws and is *structurally blind* to a bias
shared by all of them. The resplit stability and the spectral spreading were never in conflict; the
mistake was treating the first as reassurance about the second.

**The discriminator that settles which hypothesis was right.** `cos(c(k)_240rows, c(k)_794rows)`:

    k=1  0.988 ( 8.8°)   k=2  0.989 ( 8.7°)   k=13 0.931 (21.3°)
    k=50 0.776 (39.1°)   k=100 0.678 (47.3°)

The 240-row Σ had **already converged exactly where the method works** (small `k`) and had **not
converged where it fails** (large `k`). That is why both hypotheses as originally posed — "whitening
does not transfer" and "the covariance needs more rows" — were wrong.

### 12.7 The register finding: a large part of the transfer gap is sentence mood

`cyber_style_imper` is the same nine CySecBench categories, the same corpus, the same lexical
stratification — **only the sentence mood differs.** Under `c1` at the shipped threshold:

| arm | mood | fires | far stratum only |
|---|---|---|---|
| `cyber_style_imper` | imperative | **292/360 = 0.8111** `[0.7668, 0.8502]` | **161/193 = 0.8342** `[0.7741, 0.8837]` |
| `fresh_cyber` | interrogative | 771/1350 = 0.5711 `[0.5442, 0.5977]` | 288/484 = 0.5950 `[0.5498, 0.6391]` |

Within the far stratum alone the intervals are **disjoint**. **AdvBench and HarmBench are
imperative; CySecBench is interrogative.** The derivation set is therefore almost uniformly
imperative, and the mechanism follows directly from §3.3: *you can only whiten away a nuisance your
training data actually varies.* Mood is a nuisance direction the 240 rows hold nearly constant, so no
`k` removes it.

This reframes the whole transfer gap. Part of what §12.2 measures as "does not generalise to fresh
cyber" is "does not generalise to **questions**". Two consequences:

- a coverage number is not comparable across corpora of different register, and neither §7's nor
  §12.2's is annotated with one until now;
- the cheapest available improvement is not a better estimator but **imperative *and* interrogative
  phrasings of the same prompts in the derivation set**. Untested.

### 12.8 The ICS arm — no correct label under either taxonomy, so it is never pooled

CySecBench **Control System Attacks**, 928 rows. It fires **more** than real out-of-corpus cyber in
every configuration — 0.7737 `[0.7454, 0.8003]` against 0.5711 under `c1`, **disjoint** — and it is
the only arm with a monotone far → mid → near trend in all five configurations (under `c1`: 0.7378,
0.7906, 0.8431).

**CySecBench calls this material cyber. This project calls the same material
`critical_infrastructure`, one of the 14 non-cyber categories that constitute the negative class.**
The arm therefore has no correct label under either taxonomy: counted as cyber it inflates coverage,
counted as non-cyber it inflates leak, and the truth is that two reasonable taxonomies disagree
about it. That is why it is reported alone, never pooled into the 1819, and why its monotone stratum
trend is the one place in this run where a lexical-contamination reading cannot be dismissed.

### 12.9 The pre-declared criteria, and what §12 does not settle

| criterion | result |
|---|---|
| **A — coverage** | **FAILED at `c(13)`**, with the F1 fallback triggered (CP95 upper 0.3108 < 0.40). **PASSED at `c1`.** |
| **B — leak** | **PASSED everywhere.** Worst out-of-corpus category: Fraud/Deception **7/49 = 0.1429** `[0.0594, 0.2724]` under `c1`, 3/49 under `c(13)` at its own threshold. |
| **D — no lexical trend** | **PASSED.** Not monotone, far and near intervals not disjoint, and the **far-stratum AUC is higher than pooled** (0.9342 against 0.9190) — the opposite sign to contamination. |
| **F3 — the gate became a harm detector** | **did NOT fire.** |
| **C1 — label permutation** | PASSED: all 40 cells in 0.4966–0.5475 against a 0.80 stop condition. |

**What §12 does not settle.**

- **The frozen rule at small `k`** (§12.4). The single largest open item.
- **Nothing here re-tests §8.** Per-category orthogonalisation, V4 and reliability weighting were
  refuted *in corpus only*, on the pool681 capture, and `k = 13` is proof that an in-corpus verdict
  about a direction construction can reverse. Those three refutations stand as "refuted on one
  corpus" and nothing stronger. Same for §8.1's top-k survival result and §4's negative-class ranking.
- **The register arm is a confound, not a correction.** It says part of the gap is mood. It does not
  say how much, and no arm varies mood *within* the derivation set.
- **One model, one layer, one pooling.** `mean_user`, layer 8, Gemma-4-26B-A4B-it. The `k = 2`
  optimum is a property of this spectrum and is not claimed to be a general number.
- **Gate crossings, not deliveries**, everywhere in §12.

---

## 13. The serving path — the edit works, and the scores do not match the capture

### 13.1 The edit is confirmed working, separately from any of the above

Twelve prompts through the running server, `POST /compare`, 300 new tokens, strictly serial, operating
point `op_c1_baseline_mean_user_L8.pt` (`c1` @ L8, +0.15604917189545423, K = 2.0, edit band layers
10–16, 14 tensors hooked). The control arm is the **pristine checkpoint with hooks removed**, not a
below-threshold route. Record: `scratchpad/prelim_probe.md`.

> **"14 tensors hooked" is this probe's record and is now historical.** It was the runtime's
> coverage at the time, not a property of the method: the band holds **21** direction-keyed tensors
> and the 7 fused routed-expert stacks were unreachable by a hook until **2026-10-06**, when a second
> apply site closed the band to **21 of 21**. Everything in §13 was measured at 14 of 21, so the
> edit behind these observations is the weaker one — `tempx-blocking-issues.md` A22.6 has the
> 21-of-21 comparison, and `docs/GATE.md` "What it does not reach" has the mechanism. Nothing in
> §13.2's score-offset finding is affected: the gate read is at L8, strictly upstream of the band,
> and it was measured bit-identical across the change.

- **4 of 4 easy fires produced real attack content** against a flat pristine refusal.
- **4 of 4 non-fired prompts came back BYTE-IDENTICAL** to the pristine arm, `routed = pristine`.
- **The gate is sharp in both directions at the boundary.** The lowest-margin fire, **+0.0032**
  live, was a keystroke-logging request and the gated arm returned a runnable `pynput` keylogger
  with the `pip install` line, an `on_press` callback and a `Listener` loop; the pristine arm
  returned a 43-token refusal. A BEC-scam prompt at **−0.0076** refused **byte-identically** in both
  arms. Delivery did not degrade with margin.

**What this is not.** Twelve prompts, no judge, labels eyeballed. **3 of 7 cyber replies landed in a
middle category** — technical engagement that corrects or reframes the asker rather than either
refusing or complying — which is why a keyword metric is useless on this material and why a real
delivery rate needs a judge. This is an existence proof that the edit and the route work, not a rate.

### 13.2 A systematic offline-vs-live score offset — new, and it undermines any quantile

Live in-pass scores differ from the offline capture's scores for the same prompts by up to
**5.275e-03**, median **4.284e-04** over the 12 rows. That broke the probe's pre-declared score
agreement control (bar 0.005) on 1 of 12 rows.

**It is not jitter.** Five repeated `/gate` calls on each of four prompts gave a live spread of
**exactly 0.0** — the live scorer is bit-deterministic at n = 1 (`scratchpad/regate_repeats.json`).
Nor is it the offline noise floor: cross-run offline jitter on `c1` is max **1.712e-03**, so the
offset is **three times** the largest difference two offline runs produce. The signed deltas go in
both directions, so it is not a constant bias either.

| row | offline | live | signed delta | 5× live spread |
|---|---|---|---|---|
| `fresh_cyber-0455` | +0.164495573 | +0.159220546 | **−5.275e-03** | 0.0 |
| `fresh_cyber-0307` | +0.161429791 | +0.163212061 | +1.782e-03 | 0.0 |
| `fresh_cyber-0003` | +0.180274928 | +0.178768635 | −1.506e-03 | 0.0 |
| `fresh_cyber-0000` | +0.316553921 | +0.316352606 | −2.013e-04 | 0.0 |

**Most plausible cause: batch composition.** The offline rows were scored inside left-padded batches
of 8; the server scores at n = 1. This is the same mechanism `docs/PITFALLS.md` §4 already documents
for generated text on an MoE model — expert routing depends on batch composition — now showing up in
the **gate score** rather than the output. It is also the same class as the earlier 2.6e-3
in-pass-vs-recomputed gap, which makes two independent sightings.

**The exposure, now a count rather than a bound.** The bound was 85/1350 = 6.30% of `fresh_cyber`
and 3/469 = 0.64% of `fresh_nontarget` sitting within 5.28e-3 of the threshold. A live re-score of
all 1819 rows of those two arms through the running server replaced it with a measurement:

| | bound | **measured** |
|---|---|---|
| `fresh_cyber` decisions changed | ≤ 85 | **5** (4 no→fire, 1 fire→no) |
| `fresh_nontarget` decisions changed | ≤ 3 | **0** |
| live coverage | — | 774/1350 = 0.5733 `[0.5464, 0.5999]` vs offline 0.5711 `[0.5442, 0.5977]` |
| live leak | — | **17/469 = 0.0362 — identical to offline in all ten categories** |

Decision agreement 1814/1819 = **99.73%**, intervals overlapping on both axes. Re-scoring did not make
the offset go away, it measured it: max \|Δ\| rose to **7.186e-03** with 7 of 1819 rows above 5e-3 —
but the signed mean is **−2.0e-05**, 44× below the signed SD, signs split 883+/936−, so it **cancels**
rather than biasing, and all seven large-Δ rows sit far from the threshold (closest margin +0.0039).

**So the batch-composition hypothesis is refuted, and the real mechanism is the pooling.** Recovering
the exact `batch_width`/`n_pad` for 1736 rows gives a raw `n_pad` correlation of +0.054 (p = 0.025) —
the right direction — but `n_pad` is confounded with length by construction (ρ = −0.51: the short row
in a batch with a long member is exactly the one collecting pads), and **stratified on exact token
length the effect vanishes** (within-bin ρ = −0.035, 8 of 16 bins positive). The real structure is a
*negative* length effect, ρ = −0.146 (p = 4e-10), monotone across all ten deciles: longer prompts give
**smaller** deltas. That is `mean_user`'s own `1/√u` averaging law — raw spread of 1.71× across six
span-length bins collapses to 1.18× after multiplying by `√u`. It is mechanistically coherent:
`mean_user` masks pads out exactly, so padding cannot enter the pooled vector arithmetically, and the
only remaining route is GEMM tiling order — a batch-*shape* effect, not a per-row pad-count one.

**Consequence.** Because live and offline agree at the decision level, the 3216-row fresh capture
functions as evidence about the live serving path. Any later test reading a margin within **7.2e-03**
of a threshold must use the live score. Bound: only these two arms were re-scored; the 1397 rows of
`cyber_scope_ics`, `cyber_style_imper`, `fresh_cyber_repl` and the convention controls have no live
confirmation.

**The rule this forces, and it applies to every threshold in §5 and §7.** An operating-point
threshold is a quantile of a score distribution. If it is fitted against scores from the offline
capture path and then applied to scores from the serving path, **the quantile does not mean what it
says** — the leak budget `L` in particular is a promise about the negative distribution as the
*offline* path renders it. Thresholds must be fitted and quoted against scores produced by the same
path that will serve them, or the gap must be measured and carried as an interval on `t`. Neither has
been done.
