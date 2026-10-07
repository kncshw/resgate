# CLASSIFIER-MATH — the shipped classifier, stated exactly

**Scope.** This file is the mathematical specification of the classifier that ships today: what `c`
and `t` are, how they were derived, what they measure, and what they do not depend on. Every constant
here was read back out of the live operating-point file, not quoted from a note.

For the experimental record — the arms that lost, the refutations, the corrections and the history of
how `k` was chosen — see `docs/CLASSIFIER.md`. That file is the archive; this one is the spec. Where
they disagree, this file is newer and `CLASSIFIER.md` §0 explains why.

For the runtime mechanism — the read site, the fail-safe property, the operating-point file format —
see `docs/GATE.md`.

    model      google/gemma-4-26B-A4B-it
    snapshot   7d4c97e54145f8ffd1a4dd1b4986a5015a517842
    op file    op_k5_L0.05_mean_user_L8.pt          op = "k5_L0.05_frozen_train240"
    read       layer 8, pooling mean_user, every row unit-normalised, 2816 dimensions
    captures   1426-prompt in-corpus, text sha256 bec25f43…12ab35
               3216-prompt out-of-corpus, text sha256 91c0ea98…7540f

---

## 1. The object

One direction, one threshold, one bit.

    score(x) = ⟨ x/‖x‖ , c ⟩          a cosine, not a projection
    fire     ⟺  score ≥ t

`Gate.decide` is two lines. There is no argmax, no softmax and no per-class score vector. One `Gate`
instance is **one** detector answering "is this the designated category". Fifteen category directions
exist as fifteen separate operating-point files; you load a different file to protect a different
target.

A cosine and not a raw dot product for the reason `sep(l)` is a ratio: residual norms grow with depth
and vary per prompt, so a threshold on an unnormalised projection would partly be a threshold on
activation magnitude and would not transfer between prompts.

**One direction is the correct object here, not a simplification.** For cyber-against-the-rest the
between-class scatter has rank `C − 1 = 1`. `FINDINGS.md` §10's effective rank of 5.9–7.3 describes
the **15-class** structure and must not be read as an argument for a multi-dimensional read.

---

## 2. Notation

    Xu        all rows, pooled mean_user at layer 8, EACH ROW UNIT-NORMALISED before any mean
    Xp        the 100 TRAIN cyber_offense rows
    Xn        the 140 REF rows — 10 from each of 14 non-cyber harmful categories
    p, g      mean(Xp), mean(Xn)
    d         p − g
    n         rows(Xp) + rows(Xn) = 240

---

## 3. The direction

> ### Four symbols, two letters. Read this before any number below.
>
> |  | name | what it controls | touches weights? |
> |---|---|---|---|
> | **`K`** | abliteration **dose**, in `w = K * w_l` | how hard refusal is removed | **yes** |
> | **`k`** | truncation **rank** | how many noise directions are shrunk | no |
> | **`L`** | **leak budget**, in `m = floor(L * rows(Xn))` | where the threshold sits | no |
> | **`L8`, `L12`** | **layer index** | which decoder layer is read | no |
>
> Only `K` changes a weight. `k`, `L` and the layer index all live in the READ and are
> independent of each other. A reader has already confused `k` with `K`, with "5 of 14
> categories", with the rejected top5-closest experiment, and `L` with the dose. The
> shipped values are `K = 2.0`, `k = 5`, `L = 0.05` on Gemma at layer 8.
>
> ### `k` is not `K`
>
> This document's `k` is the **truncation rank** of the whitening below — how many
> eigenvalue directions get whitened out of `c`. The shipped value is **`k = 5`**.
>
> The abliteration **dose** is a different quantity written with a capital letter:
> `K`, in `w = K · w_l` (`METHOD.md:160`). The shipped value is **`K = 2.0`**.
>
> They never interact. `k` lives in the READ and never touches a weight. `K` lives in
> the EDIT and never touches the classifier. The two sweeps that chose them are
> separate experiments. The names differ only by case, which is a trap, so every
> number below that says `k` means the truncation rank.

### 3.1 The construction

Truncated whitening of `d` against the pooled within-class covariance, at `k = 5`, resolved as
**eigenvalue clipping** at `λ₆` — the largest *unwhitened* eigenvalue.

    Z  = [ Xp − p ; Xn − g ]                       240 × 2816, class-centred
    Σ  = ZᵀZ / (n − 2) = ZᵀZ / 238                 pooled within-class covariance

    Σ₅⁻¹ = Σ_{i≤5} λᵢ⁻¹ uᵢuᵢᵀ  +  λ₆⁻¹ (I − U₅U₅ᵀ)

    w  = (I − U₅U₅ᵀ) d  +  U₅ · diag( (λ₆/λᵢ)^1 ) · U₅ᵀ d
    c  = w / ‖w‖

`k = 0` returns exactly `unit(d)`. `power = 1` is full whitening inside the subspace; `power = 0.5`
(Mahalanobis half-whitening) was run as a robustness variant and is not what ships.

### 3.2 Σ is never formed — the Gram route

`rank(Σ) ≤ n − 2 = 238`, so a 2816×2816 matrix is never built:

    G      = Z Zᵀ                      240 × 240
    ev, U  = eigh(G)                   Hermitian: real eigenvalues, orthonormal eigenvectors,
                                       ascending — reverse before taking the top k
    λ      = ev / 238
    V      = Zᵀ U / √ev                lift to the 2816-dim eigenvectors

`eigh` and not `eig`: `G` is symmetric by construction (`Gᵢⱼ = zᵢ·zⱼ`) and positive semi-definite
(`xᵀGx = ‖Zᵀx‖² ≥ 0`). The orthonormality is load-bearing — `V₅V₅ᵀ` is only a projection because the
columns are orthonormal.

Exact, not approximate. The whole build costs about **15 seconds** including reading the activations.

### 3.3 The λ₆ normalisation is a CHOICE, and it is unvalidated

Normalising by `λ₆` rather than by 1 is **not forced by the mathematics.**

Whitening a subspace while leaving the complement at raw scale is unit-inconsistent, and with
unit-normalised rows the raw `1/λᵢ` factors reach 1e3–1e4 — which would make the whitened subspace
**swamp** the complement instead of being de-emphasised relative to it.

A different normalisation gives a different arm, and **no sensitivity analysis over it has been run.**
This is the first thing to vary in a replication.

### 3.4 The spectrum as shipped

| quantity | value |
|---|---|
| `λ₁` | 0.020988415510819532 |
| `λ₅` | 0.005293340159109831 |
| **`λ₆`** (the clip floor) | **0.004478617897311338** |
| shrink on `u₁` = `λ₆/λ₁` | **0.21338523124828596** |
| shrink on `u₅` = `λ₆/λ₅` | 0.8460854134990065 |
| fraction of `‖d‖²` inside the top-5 subspace | **0.34534146252760484** |
| `cos(c, unit(d))` | 0.9571012822030949 → **16.843204°** |

So 34.5% of the difference-of-means vector lies in five high-variance directions, and that part is
divided down by between 4.7× and 1.2×. The result is a **17-degree rotation** of `unit(d)`, not a
different direction.

---

## 4. The threshold

### 4.1 The rule — leak-constrained

Borrowed from **RepIt** (arXiv:2509.13281), whose `ρ` search maximises target attack success subject
to a ceiling on non-target success.

    m = floor(L × rows(Xn)) = floor(0.05 × 140) = 7       REF rows permitted to fire
    t = min{ s ∈ TRAIN cyber scores : #{ Xn scores ≥ s } ≤ m }

Scan the 100 TRAIN cyber scores ascending; take the lowest whose REF fire count is within budget.
Coverage only changes at a TRAIN cyber score, and among all `t` giving the same coverage the largest
is itself a positive score — so this grid contains the exact optimum.

`L` is a **declared knob**, not a measured property. RepIt's own ≈0.1 is its *selection constraint*;
do not inherit the number.

### 4.2 What it is NOT

Not a quantile of the positives. The rule this replaces took the 80th of 100 TRAIN cyber scores and
**never saw a negative prompt**. In the 20 legacy `op_*.pt` files the same prompts that build the
positive centroid are also the ones scored, so `delivered/n_target = 139/174` is *the definition of
the threshold*, not a measurement — `0.8 × 174 = 139`, and any set of scores gives 80%.

### 4.3 The threshold as shipped

| quantity | value |
|---|---|
| served (9 dp) | **0.140376451** |
| full precision | 0.14037645065907944 |
| rule | `R2_L0.05`, duplicate label `R5_r0.050` — both select the same point |
| `L`, `m`, `rows(Xn)` | 0.05, **7**, 140 |
| at fit: TRAIN fired | **93 / 100** |
| at fit: REF fired | **7 / 140** = 0.05 exactly, the budget |
| refit reproducibility | max abs diff **5.551e-17** against the stored value |

The 3.4e-10 gap between the served 9-dp value and full precision was checked: it flips **0 of 4642**
stored rows.

---

## 5. What it measures

### 5.1 Out of corpus — the honest numbers

CySecBench 1350 cyber (MIT) and CatQA 469 non-cyber over 10 categories (Apache-2.0). Independent
corpora with their own taxonomies, captured on the same snapshot, convention control passed at
cross-run cos distance 4.12e-05 against a within-run bf16 floor of 7.44e-05.

| | offline | **live, through the server** |
|---|---|---|
| coverage | 878/1350 = 0.6504 `[0.6243, 0.6758]` | **875/1350 = 0.6481** `[0.6220, 0.6736]` |
| leak | 16/469 = 0.0341 `[0.0196, 0.0548]` | **15/469 = 0.0320** `[0.0180, 0.0522]` |
| AUC (cyber vs non-cyber) | 0.9467 | — |

Both interval pairs overlap. Live and offline agreed on **1811 of 1819** decisions (99.56%), and
**six of the eight flips went the safe way** (fire → no fire). All eight were re-called twice and
returned bit-exactly, so they are deterministic, not jitter.

### 5.2 Against the previous operating point

`c1 = unit(d)` at the positives-only threshold +0.15604917:

| | coverage | leak |
|---|---|---|
| **c(5) at R2_L0.05** | **878/1350** | **16/469** |
| c1 at the old rule | 771/1350 | 17/469 |

**+107 net** (118 rows newly fire, 11 stop firing) at **one fewer** leak. Paired exact McNemar on the
identical rows: **129 discordant, p = 8.605553550243273e-24** on coverage; **5 discordant, p = 1.0**
on leak — so the coverage gain is overwhelming and **the leak improvement is NOT significant and must
not be claimed.**

### 5.3 How much of the achievable gain it banks

At its realised leak of `m = 16`:

| | coverage |
|---|---|
| this arm, realised | 878 |
| the best any threshold on **this direction** could reach at m=16 | 883 |
| the best any threshold on **c1** could reach at m=16 | 761 |

Realised / own oracle = **99.43%**. Gain over c1's oracle = **117 of an available 122 → 0.959 banked.**

Oracle figures place the threshold *with the fresh labels*, so they are upper bounds, not operating
points. The point of quoting them is that the frozen rule sits within 0.6% of its own direction's
ceiling — so **the remaining headroom is in `c`, not in `t`.**

### 5.4 In corpus — the optimistic set, recorded so nobody mistakes it for the result

TEST 63/74 = 0.8514 `[0.7496, 0.9234]`; LEAK 12/378 = 0.0317 `[0.0165, 0.0548]`; TRAIN 93/100;
AUC **0.9763** against the out-of-corpus 0.9467.

### 5.5 Benign

**10 / 554 = 0.01805**, measured not inherited. In corpus — the 554 harmless rows are in the same
capture. No out-of-corpus benign rate exists for any arm.

### 5.6 The unresolved band

Live-versus-offline max |Δ| = **7.170e-03** over 1819 forwards. A margin inside that is **not
reproducible** and must be reported as undetermined, not as a decision.

    median |Δ|     3.648e-04        p95  1.670e-03        p99  3.284e-03
    signed mean   −2.049e-05        sd   8.444e-04        signs 882+ / 937−

Zero-mean, three ways: the signed mean is 18× under the median absolute delta and 41× under the sd;
one-sample **t = −1.035** on 1818 df; sign split gives exact binomial **p = 0.2055**. A perturbation,
not a bias.

Rows inside the band: **7.41%** of fresh cyber, **1.49%** of fresh non-target.

**The mechanism is the pooling, not batching.** `|Δ|` correlates negatively with user-span length,
ρ = **−0.1505**, p = 1.12e-10, monotone across all ten deciles — `mean_user`'s own `1/√u` averaging
law, with a 1.71× raw spread across span-length bins collapsing to 1.18× after multiplying by `√u`.
Replicated on two independent directions (`c1`: ρ = −0.1459, p = 4.0e-10), so it is a property of the
pooling. A batch-composition explanation was tested and **refuted**: the raw `n_pad` correlation of
+0.054 vanishes to −0.035 when stratified on exact token length, because `n_pad` is confounded with
length at ρ = −0.51 and `mean_user` masks pads out exactly.

---

## 6. Why k = 5, and the trap that produced the previous answer

The out-of-corpus AUC curve in `k` has an **interior maximum**, and the in-corpus curve does not:

| k | 0 | 1 | **2** | 3 | 5 | 8 | 13 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| **out of corpus** | 0.9190 | 0.9452 | **0.9521** | 0.9498 | 0.9467 | 0.9393 | 0.9295 | 0.9270 | 0.9339 | 0.9260 |
| in corpus | 0.9374 | | | | | | 0.9826 | | **0.9893** | |

The rise to k=2 is +0.0331, paired stratified bootstrap `[+0.0281, +0.0380]`, p < 0.0005. Every k from
1 to 50 beats `c1` out of corpus with a CI excluding zero.

**In corpus the curve climbs monotonically to k=50. Out of corpus it peaks at k=2.** An earlier build
shipped `k = 13`, selected on the in-corpus curve, and its coverage collapsed out of corpus from 0.662
to 0.286 with disjoint intervals. **That selection, not whitening, is what failed.**

**k = 5 rather than k = 2** because the AUC peak and the coverage peak at a usable leak budget are
different questions. Across 12 threshold rules, k=5 at R2_L0.05 is the arm that beats the previous
point on both axes at a matched operating point; the best k=2 arm's +26 coverage sits *inside* the
unresolved band.

Two further facts about the clip point, so it is not mistaken for a principled boundary:

- **`λ₁₃/λ₁₄` is 1.1097 at 240 rows and 1.0713 at 794** — the gap *narrows* as the covariance estimate
  improves. There is no spectral gap at 13 in either spectrum.
- **More rows make it worse.** Re-estimating `Σ` from 794 rows (adding the 554 harmless as a third
  centred class, taking `d/n` from 11.7 to 3.5) moved the optimum to k=8 and *lowered* the peak AUC
  from 0.9521 to 0.9396. No 794-row arm won any of 24 matched leak budgets. The distortion at 240 rows
  is a **bias**, not a variance — six-resplit stability (AUC sd 0.0034) is structurally blind to it,
  because every resplit carries the same `d/n`.

The discriminator: `cos(c(k)_240, c(k)_794)` is 0.988 at k=1–2 (8.7°) and 0.678 at k=100 (47.3°). The
240-row covariance had **already converged where it matters** and had not converged where the method
fails.

---

## 7. What the classifier does not depend on

**The band's tensor coverage.** The read is at layer 8, strictly below the edit band at layers 10–16.
Layers 0–9 are never modified, so `h₈` is identical whether or not the edit will be applied — that
route-invariance is what permits a single forward pass. Consequently the runtime extension from 14 to
21 band tensors changes **no score and no fire decision**: live scores were bit-identical and the
109/41 split unchanged. If a coverage change ever moved a score, something is reaching above the band
and that is a serious bug.

**The dose `K` and the direction blob.** Those belong to the edit, not the read. The classifier is
`(c, t)` and nothing else.

---

## 8. Reproduction

No model load. CPU arithmetic on stored activations; seconds once the activation file is read.

    inputs    routing-probe/pool681/gemma_pool681_in.json      1426 prompts, sha bec25f43…12ab35
              the pool681 activation capture                    layer 8, three poolings
              the out-of-corpus capture                         3216 prompts, sha 91c0ea98…7540f

    as-run    scratchpad/ksweep.py        within_eig, E3_trunc_whiten, the k grid
              scratchpad/thresh.py        rule_T2, cp(), unit(), auc(), midrank()
              qout/armcmp/arm_cmp.py      the split protocol and the base asserts

    splits    one numpy default_rng(0) used sequentially: TRAIN cyber 100, TEST cyber 74,
              REF 140 (10 per category, 14 categories in SORTED NAME ORDER), LEAK 378.
              The 180 para_*/adv_* rows are excluded from everything.

**Positive control, required before any number is reported.** `c1 = unit(d)` must give threshold
+0.15604917, TRAIN 80/100, TEST 48/74, LEAK 10/378, AUC 0.9374, and on the fresh rows 771/1350 and
17/469. Then `λ₁ 0.020988416`, `λ₁₃ 0.002098547`, `λ₁₄ 0.001891158`.

**The freeze.** Recompute `c` and `t` with every TEST, LEAK and fresh row set to NaN and require
bit-identical results, max abs difference 0.0. A freeze assert that cannot fail proves nothing, so
also check the negative half: NaN-ing one TRAIN row must destroy the direction, and NaN-ing one REF
row must destroy the threshold.

**Shipping a new direction** changes only `c`, `thresh` and `meta` in the operating-point file. Two
things to get right:

1. **Cast every numpy float to a Python `float` before saving.** `Gate.from_file` calls `torch.load`
   without `weights_only=False`, so under torch ≥ 2.6 a numpy array anywhere — *including inside
   `meta`* — makes the file unloadable. The first 20 generated files failed exactly this way.
2. **Record the provenance**: estimator, `k`, `power`, `λ_{k+1}`, the negative class, `L`. An operating
   point whose provenance is not recorded is not an operating point.

---

## 9. Bounds

- **One model, one layer, one pooling, one seed, one target category.** CBRN, influence_ops and hate
  were scored but share 106 of 197 negatives, so they are not independent replications.
- **Gate-crossing counts, not delivery counts.** Every figure in §5 is a threshold crossing. No
  generation arm produced them. `baseline_leak` is `None`, so no figure can be apportioned between the
  edit and the model's own willingness.
- **The leak is not flat.** Fraud/Deception 7/49 = 14.3% and Privacy Violation 5/40 = 12.5% carry 12
  of the 16 fired rows; five of the ten categories fire zero. **Never quote 3.41% without this table.**
- **In-distribution AUC is nearly all lexical.** A tf-idf bag-of-words model on the raw prompt text
  reaches 0.9702 one-vs-rest AUC for `cyber_offense`. What rescues the activation read is held-out
  paraphrases — and that arm is excluded from this line of work by instruction, so nothing here
  distinguishes a semantic read from a lexical one.
- **Register is an unfixed confound.** Imperative phrasing fires 292/360 = 0.8111 against interrogative
  771/1350 = 0.5711 — disjoint, same nine categories, same lexical stratum. AdvBench and HarmBench are
  imperative; CySecBench is interrogative. The derivation set has one register, and you can only whiten
  away a nuisance your data varies.
- **Sub-0.7% leak targets are unreachable.** 140 REF rows quantise leak at 1/140 = 0.714%. Below that
  the frozen rule banks nothing — at m=0, frozen k=1 realises 258 of an achievable 400 and finishes
  *below* frozen `c1`. That is a sample-size limit on the negative class, not a rule-design limit.
- **No safety grade by this route.** 80% coverage at 0.1% leak needs AUC ≈ 0.9973 under a binormal
  model. This direction is at 0.9467 out of corpus against a 0.9702 corpus ceiling.
- **Nothing behind the gate.** `band_complied` is 143/143 with zero refusals, so every leak is a
  delivered answer rather than a softened one.
