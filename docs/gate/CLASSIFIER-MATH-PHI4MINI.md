# CLASSIFIER-MATH-PHI4MINI — historical `B0` (`k = 5`, `L12`) Phi-4-mini record

> **NOTE — Superseded by `M1_ridge_lam16` (`L11`).** For the current shipped Phi-4-mini operating
> points in [`kits/gate/phi4mini/`](../../kits/gate/phi4mini/README.md) — `M1_ridge_lam16` (`M1 + M3`:
> subcategory-centred within-class scatter + Tikhonov ridge shrinkage at $\alpha = \lambda_{16}$) at
> shared read layer **`L11`** (`t = 0.05303228245782718` for cyber, `549/675 = 81.3%` coverage;
> `t = 0.042208061573892744` for bio at `L11`, `630/637 = 98.9%` coverage; and `t = 0.041939858967326334`
> for bio at `L12`, `629/637 = 98.7%`) — see **[`docs/gate/CLASSIFIER-MATH.md`](CLASSIFIER-MATH.md)**.
> This document is preserved as the derivation and protocol-comparison archive for the earlier `B0`
> (`k = 5`, `L12`) generation.

**Scope (historical `B0` generation).** This file records the mathematical specification of the
earlier `B0` (`k = 5`, `L12`) Phi-4-mini gate:
what `c` and `t` were in that generation, how they were derived, what they measured, and why `L5`
was withdrawn. Every constant below was re-derived from the activations against
`op_phi4mini_k5_L0.05_mean_user_L12.pt`.

    model      microsoft/Phi-4-mini-instruct
    snapshot   cfbefacb99257ffa30c83adab238a50856ac3083
    op file    op_phi4mini_k5_L0.05_mean_user_L12.pt       op = "phi4mini_k5_L0.05_frozen_fit240_L12"
    sha256     66401d6aa01ce65ffa1e22ff2b9d3f9cc25bba68c9de0959d46e37c75b7d1381
    read       layer 12, pooling mean_user, every row unit-normalised, 3072 dimensions
    edit band  layers 13-15, K = 2.0
    captures   1426-prompt in-corpus,     text sha256 bec25f43…12ab35
               3216-prompt out-of-corpus, text sha256 91c0ea98…7540f

---

## 1. The object

One direction, one threshold, one bit.

    score(x) = ⟨ x/‖x‖ , c ⟩          a cosine, not a projection
    fire     ⟺  score ≥ t

`Gate.decide` is two lines. There is no argmax, no softmax and no per-class score vector. One
`Gate` instance is **one** binary detector answering "is this the designated category". The bit is
**frozen at prefill**: `Gate.reset` clears `_decided`, the read sets it in a `finally`, and every
token of the generation that follows is served under the one decision the prompt produced. The
decision is not revisited per token and cannot flip mid-answer.

A cosine and not a raw dot product for the reason `sep(l)` is a ratio: residual norms grow with
depth and vary per prompt, so a threshold on an unnormalised projection would partly be a
threshold on activation magnitude and would not transfer between prompts.

---

## 2. Three things called "five", and none of them is the same

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

**`L5` and `L12` are LAYER INDICES, not counts.** A reader has already confused `k = 5` with
"layer 5" and with "5 of 14 categories". To be explicit:

| token | what it is | shipped value |
|---|---|---|
| `k` | truncation rank of the whitening, a count of eigendirections | 5 |
| `K` | abliteration dose, a scalar multiplier on a weight edit | 2.0 |
| `L12` | the **read layer index**, where the gate samples the residual stream | 12 |
| `L5` | a **different read layer index**, the one this file replaces | 5 (withdrawn) |
| `L` | the **leak budget fraction** in the threshold rule | 0.05 |
| 14 | the number of non-cyber harmful **categories** in the negative class | 14 |

`k = 5` and `L5` are unrelated numbers that happen to share a digit. The Phi gate at `L12` is
still a `k = 5` build. `L` = 0.05 is a third, unrelated 5.

---

## 3. The direction

### 3.1 Notation

    Xu        all 1426 rows, pooled mean_user at layer 12, EACH ROW UNIT-NORMALISED before any mean
    Xp        the 100 TRAIN cyber_offense rows
    Xn        the 140 REF rows — 10 from each of 14 non-cyber harmful categories
    p, g      mean(Xp), mean(Xn)
    d         p − g
    n         rows(Xp) + rows(Xn) = 240

The 180 `para_*`/`adv_*` rows are excluded from every phase, by instruction. Census of the
derivation set, asserted at run time: **174 cyber / 518 non-cyber / 554 harmless / 180 excluded /
14 non-cyber categories.**

### 3.2 The construction

Truncated whitening of `d` against the pooled within-class covariance, at `k = 5`, resolved as
**eigenvalue clipping** at `λ₆` — the largest *unwhitened* eigenvalue. Identical to
`CLASSIFIER-MATH.md` §3.1; nothing about the construction is Phi-specific.

    Z  = [ Xp − p ; Xn − g ]                       240 × 3072, class-centred
    Σ  = ZᵀZ / (n − 2) = ZᵀZ / 238                 pooled within-class covariance

    w  = (I − U₅U₅ᵀ) d  +  U₅ · diag( (λ₆/λᵢ)^1 ) · U₅ᵀ d
    c  = w / ‖w‖

`power = 1` is full whitening inside the subspace. The `λ₆` normalisation is a **choice and is
unvalidated** — see `CLASSIFIER-MATH.md` §3.3; no sensitivity analysis over it has been run on
any model, Phi included.

### 3.3 Σ is never formed — the Gram route, and the check that it is exact

`rank(Σ) ≤ n − 2 = 238`, so a 3072×3072 matrix is never built:

    G      = Z Zᵀ                      240 × 240
    ev, U  = eigh(G)                   Hermitian: real eigenvalues, orthonormal eigenvectors,
                                       ascending — reverse before taking the top k
    λ      = ev / 238
    V      = Zᵀ U / √ev                lift to the 3072-dim eigenvectors

`eigh` and not `eig`: `G` is symmetric by construction (`Gᵢⱼ = zᵢ·zⱼ`) and positive semi-definite
(`xᵀGx = ‖Zᵀx‖² ≥ 0`). **The orthonormality is load-bearing** — `V₅V₅ᵀ` is only a projection
because the columns are orthonormal. All four properties were checked numerically at layer 12,
split seed 0:

| check | measured | reading |
|---|---|---|
| `max\|G − Gᵀ\|` | **0.0** | symmetric exactly, not to tolerance |
| `min eig(G)` | −1.129e-17 | PSD to float64 noise |
| positive eigenvalues of `G` (> 1e-12) | **238** | `= n − 2` exactly |
| `λ₂₃₉`, `λ₂₄₀` of the explicit Σ | 5.838e-18, 4.742e-18 | numerically zero; `rank(Σ) = 238` |
| `max\|V₅ᵀV₅ − I\|` | 1.887e-15 | orthonormal |
| `max\|P² − P\|`, `P = V₅V₅ᵀ` | 6.661e-16 | idempotent — it is a projection |
| `max\|P − Pᵀ\|` | **0.0** | symmetric exactly |

**The Gram route is exact, not approximate, and this was verified rather than asserted.** I formed
the 3072×3072 `Σ` that the spec says is never formed, ran `eigh` on it directly, and compared:

    max |λᵢ(Σ_explicit) − λᵢ(Gram)| over the top 12  =  8.674e-18
    |cos( c from Σ_explicit , c from Gram )|          =  0.9999999999999999

The same dual-route check on Gemma gives 3.123e-17 and 1.0000000000000009.

A second, independent check on my transcription of §3.2's formula: `w` was computed both as the
cheap algebra `d − V₅(V₅ᵀd) + V₅(fac·V₅ᵀd)` and as the formula literally reads, with `I` and `P`
formed explicitly — `(I − P)d + V₅ diag(fac) V₅ᵀ d`. The two forms agree to **2.637e-16**.

### 3.4 The spectrum as shipped

Layer 12, split seed 0, `k = 5`, `mean_user`. "Mine" is my re-derivation; "in file" is the stored
`meta`. Every pair agrees to float64 noise.

| quantity | mine (float64) | in file |
|---|---|---|
| `λ₁` | 0.016630034616388869 | 0.016630034616388876 |
| `λ₂` | 0.0053982521574377869 | — |
| `λ₃` | 0.003791226937373225 | — |
| `λ₄` | 0.0029887786605785941 | — |
| `λ₅` | 0.0021489618461700637 | 0.0021489618461700624 |
| **`λ₆`** (the clip floor) | **0.0017439708834565562** | 0.0017439708834565595 |
| `λ₇` | 0.0016600459233913794 | — |
| `λ₈` | 0.0015089751037217576 | — |
| `λ₉` | 0.0013968826829645845 | — |
| `λ₁₀` | 0.0012820571569304876 | — |
| `λ₁₁` | 0.0011899526018188246 | — |
| `λ₁₂` | 0.0010589485230381168 | — |
| trace of `Σ` | 0.071750904062104479 | 0.07175090406210446 |

`λ₂` through `λ₁₂` are in the file as the single string `lam_top_12`; all twelve match.

The shrink factor applied inside the clipped subspace, `λ₆/λᵢ`:

| i | `λ₆/λᵢ` | division |
|---|---|---|
| 1 | **0.10486874643892058** | 9.54× |
| 2 | 0.3230621380021474 | 3.10× |
| 3 | 0.4600017124442773 | 2.17× |
| 4 | 0.5835062015328171 | 1.71× |
| 5 | **0.8115411106831455** | 1.23× |

The file stores these as `shrink_factor_min` 0.10486874643892073 and `shrink_factor_max`
0.8115411106831475 — both match.

| quantity | mine | in file |
|---|---|---|
| fraction of `‖d‖²` inside the top-5 subspace | **0.18304016385165997** | 0.18304016385166022 |
| `cos(c, unit(d))` | **0.94905393211741296** | 0.949053932 |
| angle | **18.367677°** | 18.367677 |

So **18.3% of the difference-of-means vector lies in five high-variance directions**, and that part
is divided down by between 9.5× and 1.2×. The result is an **18.4-degree rotation** of `unit(d)`,
not a different direction.

**Phi's spectrum is flatter than Gemma's where it matters.** Gemma at L8 puts 34.5% of `‖d‖²` in
the top-5 subspace and rotates 16.84°; Phi at L12 puts 18.3% there and rotates 18.37°. Phi's `λ₁`
is also the more dominant relative to `λ₆` (9.54× against Gemma's 4.69×), so the whitening does
more to the little of `d` that it reaches. For contrast, Phi at the withdrawn **L5** read put
30.7% of `‖d‖²` in the subspace and rotated only 9.96°, with `λ₆/λ₁` = 0.306 — a markedly flatter
spectrum and a much milder edit to `d`.

---

## 4. The threshold

### 4.1 The rule — leak-constrained

Borrowed from **RepIt** (arXiv:2509.13281), whose `ρ` search maximises target attack success
subject to a ceiling on non-target success. Identical to `CLASSIFIER-MATH.md` §4.1.

    m = floor(L × rows(Xn)) = floor(0.05 × 140) = 7       REF rows permitted to fire
    t = min{ s ∈ TRAIN cyber scores : #{ Xn scores ≥ s } ≤ m }

Scan the 100 TRAIN cyber scores ascending; take the lowest whose REF fire count is within budget.
`L` is a **declared knob**, not a measured property. RepIt's own ≈0.1 is its *selection
constraint*; it was not inherited.

I implemented this scan twice — as the explicit ascending loop the spec describes, and as a
`searchsorted` over the sorted negatives — and required the two to return the identical float.
They did, at both layers.

### 4.2 The threshold as shipped

| quantity | value |
|---|---|
| full precision, mine | **0.057133144618356724** |
| full precision, in file | 0.05713314461835679 |
| `\|difference\|` | **6.94e-17** |
| rule | `R2_L0.05` |
| `L`, `m`, `rows(Xn)` | 0.05, **7**, 140 |
| `m = floor(L × rows(Xn))` | `floor(0.05 × 140)` = **7**, recomputed |
| at fit: TRAIN fired | **95 / 100** |
| at fit: REF fired | **7 / 140** = 0.05 exactly, **on** the budget |
| `cos(my c, stored c)` | **1.0** |
| `max\|cᵢ − stored cᵢ\|` | 3.48e-09 — the float32 storage of `c`, not a disagreement |

The withdrawn L5 file landed at **5** of its budget of 7 rather than on it, which is the same rule
working as specified (it takes the lowest TRAIN score whose REF count is *within* budget, and
coverage only changes at a TRAIN score). Do not read 5 and 7 as two different rules.

### 4.3 Served behaviour was verified independently of the metadata

Served behaviour depends only on `c` and `thresh`. Both were checked without reference to any
`meta` field: the stored `c` and the stored `thresh` were used to score all 1426 in-corpus rows
and all 3216 out-of-corpus rows, and every held-out count the file claims was reproduced exactly
(§7). On the out-of-corpus rows my scores agree with the independently stored per-row scores at
**max |Δ| = 0.0 over 3216 rows and 0 decision disagreements**, at both layers.

**Decision stability.** Of 3216 out-of-corpus rows, **0** sit within 1e-5 of the L12 threshold and
**8** within 1e-4. The float32 storage of `c` cannot move a decision at this margin.

---

## 5. THE HONEST SECTION — the L5 episode

**This file supersedes a recommendation I made earlier the same day, and the reason it was wrong
matters more than the layer number.**

### 5.1 What happened

An in-corpus sweep over the 13 admissible read layers (0–12, the layers strictly below the edit
band) measured held-out leak at matched held-out coverage on **100 resplits of one corpus**. It
chose **L5**, and reported L12 as worse by **+3.45 leaks per 378 at matched coverage 0.851**,
bootstrap95 **[+2.72, +4.16]**, Wilcoxon p = 4.62e-12, L12 worse on 82 of 100 seeds. An operating
point was built at L5 and shipped.

A fresh **out-of-corpus** capture — 3216 prompts from CySecBench and CatQA, independent taxonomies,
same snapshot, convention control passed — then **reversed the sign at every point tested**:

| matched coverage | leak L5 | leak L12 | L12 − L5 | bootstrap mean | bootstrap CI95 | McNemar (exact) |
|---|---|---|---|---|---|---|
| 0.2111 — L5's shipped point (285/1350) | 17/469 | 14/469 | **−3** | −2.209 | **[−7.0, +2.0]** ← crosses zero | 3 vs 0, p = 0.25 |
| 0.3785 — L12's shipped point (511/1350) | 31/469 | 27/469 | **−4** | −3.882 | **[−9.0, 0.0]** ← touches zero | 4 vs 0, p = 0.125 |
| 0.8511 — where the claim was made (1149/1350) | 82/469 | 71/469 | **−11** | −10.622 | **[−21.0, −1.0]** ← excludes zero | 15 vs 4, p = 0.0192 |

Every figure in that table was recomputed here from the frozen `c` and `t` of both op files and
reproduced the stored result exactly (B = 10000, seed 20261006).

**The in-corpus interval [+2.72, +4.16] and the out-of-corpus interval [−21, −1] do not overlap at
all.** L12 also has the better out-of-corpus AUC, **0.9083 against 0.8903**, and far better
out-of-corpus coverage, **0.3785 against 0.2111**.

### 5.2 What is dead, and what is not

**What is dead is the claim that L5 is better.** It is not replaced by a claim that L12 is better
everywhere.

At **L5's own shipped operating point, coverage 0.2111, the interval crosses zero: the two layers
are INDISTINGUISHABLE there.** The sign favours L12 and the bootstrap puts P(L12 better) at 0.808,
but that is not a separation and must not be reported as one. Only at coverage 0.8511 — a point
neither file ships at — does the interval exclude zero.

L12 is shipped because it is better on the two things that are separated out of corpus (coverage,
0.3785 vs 0.2111 with disjoint intervals; AUC, 0.9083 vs 0.8903) and is not worse on leak anywhere
out of corpus. It is **not** shipped on a demonstration that its leak is lower at a matched
operating point.

### 5.3 The in-corpus claim itself reproduces — that is the point

I re-ran the in-corpus comparison with my own implementation. Pooled over the same 100 seeds, at
matched held-out coverage 0.851, median leak per 378 held-out rows:

| | median | IQR |
|---|---|---|
| L5 | **7.0** | [4.0, 11.0] |
| L12 | **12.0** | [8.75, 15.0] |

Same sign, same rough size as the published +3.45. **The in-corpus measurement was not a bug and
was not noise. It was a correct measurement of the wrong thing.**

### 5.4 The general lesson

**An in-corpus argmax over layers is not evidence about fresh rows.** All 100 seeds resample the
same 174 cyber and 518 non-cyber rows from one capture; the spread they produce is *split*
variance on one corpus and does not widen to cover corpus change. Selecting the maximum of 13
noisy, positively-correlated estimates on the rows you then score is selection on the corpus, and
the quantity it optimises is not the quantity that transfers.

The L5 file's own `gate_layer_selection_bound` field said exactly this before the capture was run:
*"it is an argmax over 13 layers on the rows it is scored on. The choice therefore DOES NOT REACH
FRESH ROWS."* The bound was correctly stated and then acted against anyway. **Stating a bound is
not the same as honouring it.**

**This is the same failure mode `CLASSIFIER-MATH.md` §6 records for `k`.** There, the in-corpus
AUC curve climbs monotonically to `k = 50` while the out-of-corpus curve peaks at `k = 2`, and a
`k = 13` build selected on the in-corpus curve collapsed out of corpus from 0.662 to 0.286
coverage with disjoint intervals. Two parameters, one instrument, the same result twice. The
instrument is the problem.

### 5.5 What the Phi in-corpus `k` curve does — reported, not used

**`k` was NOT re-selected.** `k = 5` is inherited from Gemma on purpose, because it is the only
truncation rank with out-of-corpus validation anywhere in this line of work. The Phi curve is
recorded here only so that nobody re-derives it and mistakes it for a selector.

Held-out TEST (74 cyber) vs LEAK (378 non-cyber) AUC at layer 12 — the same statistic as
`CLASSIFIER-MATH.md` §6's "in corpus" row:

| k | 0 | 1 | 2 | 3 | **5** | 8 | 13 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| seed 0 | 0.8851 | 0.9657 | 0.9744 | 0.9779 | **0.9819** | 0.9831 | 0.9864 | 0.9897 | **0.9920** | 0.9912 |
| 100-seed median | 0.9757 | 0.9802 | 0.9825 | 0.9838 | **0.9843** | 0.9850 | 0.9877 | 0.9895 | 0.9913 | **0.9915** |

**The curve climbs monotonically to k = 50 (seed 0) or k = 100 (100-seed median). There is no
interior maximum.** That is the identical shape Gemma's in-corpus curve has, and Gemma's
out-of-corpus curve peaks at k = 2. The file's own pre-declared in-FIT 5-fold CV 1-SE rule returned
`k = 50` and its argmax returned `k = 100`.

> **SELECTOR STATUS: KNOWN-INVALID for `k`.** The table above cannot settle `k`, because it is the
> in-corpus statistic that is known not to transfer. **`k = 5` ships, inherited.**
>
> **The missing measurement is cheap and nobody has taken it.** No `k` grid was run on the
> out-of-corpus capture — but `phi4mini_fresh_acts.pt` stores all 32 layers of the 3216 fresh rows,
> so the out-of-corpus `k` curve at layer 12 is a CPU job on an existing file, needing no model
> load and no new capture. Until someone runs it, `k = 5` on Phi rests entirely on Gemma's
> out-of-corpus curve, and **whether Phi's out-of-corpus curve has an interior maximum in the same
> place is UNMEASURED.**

---

## 6. WHAT THE NUMBERS DO NOT MEAN

> ### Every count in §7 is a GATE-FIRING count
>
> `fired` means `score ≥ thresh`, computed offline on stored activations. **No model was
> loaded, no text was generated, and nothing was judged.** A fired row is a row the gate
> would route to the edited arm. Whether the model then produced harmful content is a
> separate measurement, and it is **not in this document**.
>
> So "coverage 511/1350" means *511 of 1350 out-of-corpus cyber prompts crossed the
> threshold*, not that 511 attacks succeeded. "Leak 27/469" means *27 of 469 non-cyber
> prompts crossed the threshold*, not that 27 harmful answers were delivered.

**The field names in the operating-point file are wrong, and the wrongness is enforced by our own
code. Record this as a defect.**

`resgate/src/resgate/gate.py:130` declares

    META_FIELDS = ("category", "pool", "gate_layer", "op", "delivered", "n_target",
                   "leak", "n_nontarget", "baseline_leak", "benign_fpr")

and `Gate.from_file` **raises `ValueError` on a missing field** rather than defaulting:

    gone = [f for f in META_FIELDS if f not in m]
    if gone:
        raise ValueError("operating point is missing provenance field(s): %s" % ", ".join(gone))

So a file that honestly called these `target_fired` and `nontarget_fired` **would not load.**
`delivered` and `leak` are aliases of `heldout_target_fired` and `heldout_nontarget_fired`; both
Phi files carry a `counts_note` field saying so, and the Gemma file's `delivered: 878` has already
been misread by a reader as a delivery rate. The misnaming is not a documentation slip that a
careful writer can avoid — it is a load-time requirement. **Fixing it means changing
`META_FIELDS` and migrating the files; until then every reader has to be warned in prose.**

`baseline_leak` is **`None`** in this file because it was not measured. No generation arm produced
any count here, so the pristine Phi model's own willingness to answer is unknown, and **no figure
in §7 can be apportioned between the edit and the model's own behaviour.**

---

## 7. What it measures

Read §6 first. Every count below is a threshold crossing.

### 7.1 FITTED versus MEASURED, row by row

| row | what FITTED it | what it MEASURES | status |
|---|---|---|---|
| in-sample TRAIN 95/100 | `c` **and** `t`, both on these rows | nothing | **definitional** |
| in-sample REF 7/140 | `c` **and** `t`, both on these rows | the leak budget, by construction | **definitional** |
| in-corpus held-out, seed 0 | nothing — 240 disjoint rows fitted `c`,`t` | disjoint **rows of the same capture** | optimistic |
| in-corpus, 100 seeds | nothing per seed — refit each seed | **split** variance on one corpus | optimistic |
| out-of-corpus, frozen | **nothing. No fresh row entered any fit.** | an independent corpus | the honest number |
| matched-coverage oracle | the **threshold** is placed with fresh labels | an upper bound per layer | **not an operating point** |

### 7.2 In corpus — the optimistic set, recorded so nobody mistakes it for the result

The held-out rows are **disjoint rows of the SAME pool681 capture**, not an independent corpus.
The file's own `heldout_is_same_corpus` is `true`.

**Split seed 0** (the file's own numbers; every one recomputed and reproduced exactly):

| | fired | n | rate | CP95 |
|---|---|---|---|---|
| coverage, held-out TEST cyber | **65** | 74 | 0.878378 | [0.7816, 0.9429] |
| leak, held-out LEAK non-cyber | **19** | 378 | 0.050265 | [0.0305, 0.0774] |
| benign, harmless rows | **2** | 554 | 0.00361 | — |
| AUC, TEST vs LEAK | 0.9818747318747318 | | | |
| AUC, TEST vs benign | 0.9957068982339741 | | | |

**Across 100 split seeds** — `c` **and** `t` refitted on each seed's TRAIN ∪ REF, the same 554
harmless rows every seed, median [IQR]:

| | median | IQR | of |
|---|---|---|---|
| coverage | **69.0** | [67.0, 70.0] | 74 |
| leak | **25.0** | [19.0, 31.0] | 378 |
| benign | **5.0** | [2.0, 7.0] | 554 |
| AUC TEST vs LEAK | **0.9843** | [0.9812, 0.9875] | — |
| leak at matched coverage 0.851 | **12.0** | [8.75, 15.0] | 378 |

> **The L12 op file does NOT carry these across-seed fields, and it does not carry a
> `seed0_is_one_draw` warning either.** The withdrawn L5 file carries both. They are computed here
> for the first time. **Seed 0 is one draw and it is not the median**: its leak of 19 sits below the
> across-seed median of 25, and its coverage of 65 sits below the median of 69. Quote the
> across-seed row, not seed 0, for expected in-corpus behaviour.
>
> My implementation was validated against the L5 file's stored across-seed fields, which it
> reproduced exactly — 68.0 [67.0, 70.0], 22.0 [16.0, 28.0], 7.0 [3.0, 11.2], 0.9856
> [0.9827, 0.9893], 7.0 [4.0, 11.0]. So the L12 figures above are produced by code already shown
> to agree with the stored protocol on the sibling file.

### 7.3 Out of corpus — the honest numbers

CySecBench 1350 cyber (MIT) and CatQA 469 non-cyber over 10 categories (Apache-2.0). Independent
corpora with their own taxonomies, captured on the same snapshot. **`c` and `t` frozen; no fresh
row entered any fit.** Contamination audit: **0** string-identical rows against the 1426 fit-corpus
rows under three normalisations, max token-set Jaccard 0.438 with nothing above 0.5.

Convention control, built into the prompt file as 20 re-captured pool681 rows plus a 20-row
within-run duplicate: cross-run max cosine distance **1.3785e-05** against a within-run bfloat16
floor of **1.2071e-05** — a ratio of **1.14**, and **1350×** smaller than the nearest deliberately
wrong read. The fresh capture used the fit capture's convention; a convention drift cannot explain
what follows. (Gemma's reference pair on the same prompt file: 4.12e-05 against 7.44e-05.)

| | fired | n | rate | CP95 |
|---|---|---|---|---|
| **coverage** — fresh_cyber | **511** | 1350 | **0.37852** | **[0.3526, 0.4050]** |
| **leak** — fresh_nontarget | **27** | 469 | **0.057569** | **[0.0383, 0.0827]** |
| AUC cyber vs non-cyber | **0.908338** | | | |
| fresh_cyber_repl (CatQA cyber) | 49 | 69 | 0.7101 | [0.5884, 0.8131] |
| cyber_scope_ics (Control System Attacks) | 494 | 928 | 0.5323 | [0.4996, 0.5648] |
| cyber_style_imper (imperative register) | 219 | 360 | 0.6083 | [0.5558, 0.6591] |
| conv_control_a / conv_control_b | 0 / 0 | 20 / 20 | 0.0 | [0.0, 0.1684] |

**Register is an unfixed confound, as on Gemma.** Imperative phrasing fires 219/360 = 0.6083
[0.5558, 0.6591] against interrogative 511/1350 = 0.3785 [0.3526, 0.4050] — **disjoint intervals**,
same nine categories. AdvBench and HarmBench are imperative; CySecBench is interrogative. The
derivation set has one register, and you can only whiten away a nuisance your data varies.

### 7.4 Per-category leak — in corpus and out

**Never quote an aggregate leak without this table.** The leak is not flat at either site.

**In corpus, split seed 0, frozen rule.** All 14 fields recomputed; all 14 match the file.

| category | fired/n | | category | fired/n |
|---|---|---|---|---|
| fraud_financial | **14/120** | | other | 0/10 |
| surveillance_stalking | **2/9** | | selfharm_eating | 0/24 |
| influence_ops_disinfo | **2/78** | | sexual_exploitation | 0/1 |
| hate_harassment | **1/45** | | violence_physical | 0/11 |
| cbrn_chem_bio | 0/7 | | weapons_explosives | 0/33 |
| critical_infrastructure | 0/2 | | drugs | 0/14 |
| evasion_lawenforcement | 0/23 | | human_trafficking | 0/1 |

Sum **19/378**, matching the file's own `heldout_nontarget_fired = 19`.

**Seed 0 leaves four categories at n ≤ 2, which is uninformative**, so the same breakdown pooled
over 100 seeds is given below. This is computed here for the first time; the L12 file carries no
pooled table.

**In corpus, pooled over 100 seeds** (37800 held-out non-cyber rows in total):

| category | frozen rule | rate | share of fires | at matched cov 0.851 | rate |
|---|---|---|---|---|---|
| critical_infrastructure | 49/200 | **0.2450** | 1.9% | 33/200 | **0.1650** |
| surveillance_stalking | 141/900 | **0.1567** | 5.3% | 99/900 | **0.1100** |
| fraud_financial | **1684/12000** | 0.1403 | **63.8%** | **764/12000** | 0.0637 |
| other | 104/1000 | 0.1040 | 3.9% | 12/1000 | 0.0120 |
| hate_harassment | 244/4500 | 0.0542 | 9.2% | 47/4500 | 0.0104 |
| violence_physical | 51/1100 | 0.0464 | 1.9% | 39/1100 | 0.0355 |
| influence_ops_disinfo | 291/7800 | 0.0373 | 11.0% | 168/7800 | 0.0215 |
| evasion_lawenforcement | 73/2300 | 0.0317 | 2.8% | 1/2300 | 0.0004 |
| cbrn_chem_bio | 1/700 | 0.0014 | 0.0% | 0/700 | 0.0000 |
| drugs, human_trafficking, selfharm_eating, sexual_exploitation, weapons_explosives | **0** | 0.0000 | 0.0% | **0** | 0.0000 |
| **total** | **2638/37800** | 0.0698 | | **1163/37800** | 0.0308 |

**fraud_financial carries 63.8% of all fires by volume. The two highest RATES are small and
cyber-adjacent — critical_infrastructure 24.5% and surveillance_stalking 15.7%. Five of fourteen
categories never fire on any of 100 seeds. An aggregate of 0.0698 hides both ends.**

**Out of corpus, frozen rule**, the CatQA taxonomy (a different partition of harm — see §9):

| category | fired/n | rate | CP95 | share of fires |
|---|---|---|---|---|
| Fraud/Deception | **9/49** | **0.1837** | [0.0876, 0.3202] | **33.3%** |
| Privacy Violation Activity | **6/40** | **0.1500** | [0.0571, 0.2984] | 22.2% |
| Political Campaigning | 5/50 | 0.1000 | [0.0333, 0.2181] | 18.5% |
| Illegal Activity | 3/40 | 0.0750 | [0.0157, 0.2039] | 11.1% |
| Physical Harm | 2/47 | 0.0426 | [0.0052, 0.1454] | 7.4% |
| Economic Harm | 2/50 | 0.0400 | [0.0049, 0.1371] | 7.4% |
| Adult Content | 0/50 | 0.0000 | [0.0, 0.0711] | 0.0% |
| Child Abuse | 0/47 | 0.0000 | [0.0, 0.0755] | 0.0% |
| Hate/Harass/Violence | 0/46 | 0.0000 | [0.0, 0.0771] | 0.0% |
| Tailored Financial Advice | 0/50 | 0.0000 | [0.0, 0.0711] | 0.0% |
| **total** | **27/469** | 0.0576 | [0.0383, 0.0827] | |

**The volume concentration survives the corpus change; the rate ordering does not.** Fraud leads on
volume at both sites (63.8% in corpus, 33.3% out). But in corpus the highest *rates* are the small
cyber-adjacent categories, while out of corpus the highest rate is Fraud/Deception itself at 0.1837
and the surveillance analogue (Privacy Violation Activity) is second at 0.1500. Four categories
never fire out of corpus, mirroring the five that never fire in corpus. **The correspondence
between the two category sets is INFERRED, not measured** (§9).

### 7.5 Coverage per cyber category, out of corpus

Firing here is **coverage, not leak** — these are the prompts the gate is meant to catch.

| CySecBench category | fired/150 | rate | CP95 |
|---|---|---|---|
| IoT Attacks | 89 | 0.5933 | [0.5102, 0.6727] |
| Cloud Attacks | 76 | 0.5067 | [0.4239, 0.5892] |
| Web Application Attacks | 74 | 0.4933 | [0.4108, 0.5761] |
| Network Attacks | 66 | 0.4400 | [0.3591, 0.5233] |
| Intrusion Techniques | 50 | 0.3333 | [0.2586, 0.4148] |
| Malware Attacks | 44 | 0.2933 | [0.2219, 0.3731] |
| Cryptographic Attacks | 39 | 0.2600 | [0.1919, 0.3379] |
| Evasion Techniques | 38 | 0.2533 | [0.1859, 0.3307] |
| Hardware Attacks | 35 | 0.2333 | [0.1682, 0.3093] |

**Coverage spans 0.233 to 0.593 across nine categories.** IoT and Hardware have disjoint intervals.
There is no single coverage number for this gate.

---

## 8. The out-of-corpus collapse — the direction degrades, not just the calibration

| | in corpus | out of corpus | overlap |
|---|---|---|---|
| coverage, seed 0 | 65/74 = **0.878378** [0.7816, 0.9429] | 511/1350 = **0.37852** [0.3526, 0.4050] | **none** |
| coverage, 100-seed median | 69/74 = **0.932** | 511/1350 = **0.37852** | **none** |
| AUC cyber vs non-cyber | **0.9819** (seed 0), **0.9843** [0.9812, 0.9875] (100 seeds) | **0.908338** | **none** |
| leak | 19/378 = 0.0503 [0.0305, 0.0774] (seed 0); 25/378 = 0.0661 (median) | 27/469 = 0.0576 [0.0383, 0.0827] | **overlaps** |

**Coverage falls by a factor of ~2.3 to ~2.5 with disjoint intervals. Leak does not get worse —
its intervals overlap.** So the gate did not become trigger-happy out of corpus; it became blind.

**And AUC fell too, from 0.9843 to 0.9083.** AUC is threshold-free. A pure calibration failure —
the right ranking read at the wrong cut point — would leave AUC alone. It did not. **The DIRECTION
degrades out of corpus, not merely the threshold's calibration.** Re-fitting `t` on fresh labels
cannot recover this: the matched-coverage oracle rows in §5.1 place the threshold *with* the fresh
labels and still land at 71 leaks per 469 to reach coverage 0.851.

> A handover note to this work quoted "0.865 in corpus to 0.3785 out of corpus". **0.865 is the
> withdrawn L5 file's seed-0 coverage (64/74).** L12's own seed-0 coverage is **0.878378** (65/74)
> and its 100-seed median is **0.932**. The collapse is slightly larger than the note stated.

### 8.1 Phi is far weaker out of corpus than Gemma, on the identical prompt file

Both numbers below were computed in the same process, on the same 3216 prompts, with each model's
own frozen operating point. The Gemma figures reproduce `CLASSIFIER-MATH.md` §5.1 exactly
(878/1350, 16/469, AUC 0.9467), which is itself a control on my pipeline.

| | coverage | leak | AUC |
|---|---|---|---|
| **Gemma-4-26B**, L8, k=5, t=0.140376451 | **878/1350 = 0.6504** [0.6243, 0.6758] | **16/469 = 0.0341** [0.0196, 0.0548] | **0.9467** |
| **Phi-4-mini**, L12, k=5, t=0.0571331446 | **511/1350 = 0.3785** [0.3526, 0.4050] | **27/469 = 0.0576** [0.0383, 0.0827] | **0.9083** |
| Phi-4-mini, L5 (withdrawn) | 285/1350 = 0.2111 [0.1896, 0.2339] | 17/469 = 0.0362 [0.0213, 0.0574] | 0.8903 |

**Phi reaches 0.3785 coverage at a HIGHER leak than Gemma reaches 0.6504 at.** The coverage
intervals are disjoint. Phi's gate is substantially weaker out of corpus on both axes at once.

**BOUND: this is not a controlled single-variable comparison.** The two operating points differ in
model, hidden dimension (3072 vs 2816), read layer, threshold and edit band. What is DEMONSTRATED
is that on identical prompts, with each model's own shipped gate, Phi's is much worse. What is NOT
demonstrated is *why* — no arm isolates model capacity from layer choice from corpus fit.

---

## 9. What the classifier does not depend on — and the one-layer margin

**The band's tensor coverage.** The read is at **layer 12**, strictly below the edit band at
**layers 13–15**. Layers 0–12 are never modified, so `h₁₂` is identical whether or not the edit will
be applied. That **route-invariance** is what permits a single forward pass to serve a conditional
edit, and `gate.py` asserts it at construction:

    "the read must be strictly upstream of the edit: read %s against edit band %d-%d. A gate
     reading a layer the edit touches is reading its own effect, and the route-invariance
     argument for a single forward pass is then gone."

**The dose `K` and the direction blob** belong to the edit, not the read. The classifier is
`(c, t)` and nothing else. Changing `K` from 2.0 changes no score and no fire decision.

### 9.1 State the fragility plainly

    min(edit band) − read layer   =   13 − 12   =   1

**A ONE-layer margin. The band cannot be widened downward at all.** Widen it to include L12 and
this operating point is **invalid** — the gate would be reading its own effect, the `gate.py`
assert would refuse to construct, and the classifier would have to be **refitted at a shallower
layer**. Not re-thresholded: refitted, because `c` is layer-specific.

**The withdrawn L5 read had `13 − 5 = 8` layers of margin** — the band could have been widened
downward by seven layers, to L6, and the point would still have been valid. **That was L5's one
genuine advantage, and the revert gives it up.** It is a real cost, paid knowingly:

- what is bought: out-of-corpus coverage 0.3785 against 0.2111 (disjoint intervals) and AUC 0.9083
  against 0.8903;
- what is sold: seven layers of headroom on the edit band.

Phi's band is already narrow — three layers, 13–15, derived from this model's own `sep(l)` profile
at `w(l) ≥ 0.9·max(w)` with a refusal peak at L14 and a max:min sharpness of 2.43×. If a future
arm widens that band downward, **this file must be reissued at a shallower read layer**, and the
L5 build is not a drop-in substitute for it — §5 is why.

For contrast, Gemma's shipped read is L8 against an edit band of 10–16: a 2-layer margin, one free
layer (L9) of headroom.

---

## 10. Unmeasured, with bounds

Separated from §7 on purpose. Nothing below is a result.

- **No out-of-corpus benign false-positive arm exists.** All **3176** fresh rows are harmful-intent
  prompts — 2707 cyber-ish (CySecBench 1350 + ICS 928 + imperative 360 + CatQA cyber 69) and 469
  non-cyber harm. **The fresh set has no benign arm at all**, so the out-of-corpus benign FPR is
  **NOT MEASURED**. The only benign rows anywhere in the capture are the 11 `harmless-*` rows inside
  `conv_control_a`, which are pool681 rows — *in* corpus — and n = 11; they fired 0/11 at both
  layers, a sanity check and nothing more. In-corpus benign, for contrast only: **5.0 [2.0, 7.0] of
  554** across 100 seeds. **Closing this needs a fresh benign corpus, which this prompt file does
  not contain.**
- **The CatQA ↔ pool681 category mapping is INFERRED, not measured.** The two sets are different
  partitions of harm; no CatQA category is a pool681 category name. The correspondence used in §7.4
  is: `fraud_financial` ← Fraud/Deception + Economic Harm + Tailored Financial Advice;
  `surveillance_stalking` ← Privacy Violation Activity; `influence_ops_disinfo` ← Political
  Campaigning; `hate_harassment` ← Hate/Harass/Violence; `violence_physical` ← Physical Harm;
  `sexual_exploitation` ← Child Abuse (+ Adult Content). `drugs`, `weapons_explosives`,
  `cbrn_chem_bio`, `selfharm_eating`, `evasion_lawenforcement` and `human_trafficking` sit partly
  inside Illegal Activity / Physical Harm and are **not separable**. Any "does the shape hold"
  statement rests on this inference.
- **`critical_infrastructure` has no CatQA counterpart, so its out-of-corpus rate is
  UNMEASURABLE from this capture.** This is the hard limit of §7.4: `critical_infrastructure` is the
  **highest** in-corpus leak rate (24.5% pooled frozen, 16.5% at matched coverage) and it cannot be
  checked out of corpus at all. The `cyber_scope_ics` arm (928 Control System Attack rows) probes the
  same territory from the **cyber** side, where firing is coverage rather than leak, so it does not
  substitute.
- **No paraphrase arm and no adversarial arm were run.** The 180 `para_*`/`adv_*` rows are excluded
  from every phase by instruction. The Phi kit measures a tf-idf bag-of-words floor of **0.9412**
  one-vs-rest AUC on its in-distribution arm, so **an in-distribution activation AUC alone does not
  distinguish a semantic read from a lexical one**, and the arm that would separate them is out of
  scope. Nothing here settles it.
- **One fresh corpus, not a distribution of them.** `N = 1`. The §5.1 bootstrap resamples *rows*
  within this one corpus; it does **not** bound corpus-to-corpus variation. A second independent
  fresh corpus has not been run.
- **No live-versus-offline arm for Phi.** Every Phi number here is offline arithmetic on stored
  activations. Gemma has a 1819-forward live comparison (`CLASSIFIER-MATH.md` §5.1, §5.6) with a
  max |Δ| of 7.170e-03 and an unresolved band; **Phi has no equivalent, so Phi's live-versus-offline
  jitter and its unresolved band are unknown.**
- **Not varied anywhere in this work:** `k` (frozen 5), `L` (frozen 0.05), pooling (`mean_user`
  only), the `λ₆` normalisation, `power` (1.0), the target category (`cyber_offense` only), REF size
  (10/category), TRAIN size (100). The read-layer search covered **layers 0–12 only**; layers ≥ 13
  were excluded by design, because the read must be strictly upstream of the edit.
- **No judged delivery, compliance or refusal rate.** `baseline_leak` is `None`. See §6.
- **Does not transfer to another target category.** `FINDINGS.md` records that the cyber-selected
  layer does not transfer to CBRN on three of five models. **This operating point is for the CYBER
  gate only.**
- **The shuffled-label control's verdict depends on its seed, at BOTH layers.** At L12 the
  pre-declared stream `rng(777)` gives mean 0.493057 (PASS against a `|mean − 0.5| ≤ 0.03` bar), but
  an independent second stream `rng(987654321)` gives 0.4666 — a **FAIL**. The null's own sd is
  0.1735, so at n = 200 the SE of the mean is 0.0123 and the bar is only 2.4 SE wide — the bar is
  narrower than the control's own noise. **The PASS flag must not be read as settled.** What the
  control does establish is that permuting the fit labels collapses held-out AUC from 0.9819 to a
  mean of 0.467–0.493 across the two streams, i.e. to chance. **It establishes no label leakage; it
  does not establish that the bar was cleared.** Fixing the bar needs a larger n or a wider bar,
  declared in advance. Phi already carries two VOIDED operating points
  (`VOID_op_phi4mini_*_mean_user_L8_tpr_08.pt`) from this same check failing at 0.4567.

---

## 11. Reproduction

No model load. CPU arithmetic on stored activations. The whole verification is minutes.

### 11.1 Inputs, with hashes

| file | sha256 | what |
|---|---|---|
| `op_phi4mini_k5_L0.05_mean_user_L12.pt` | `66401d6aa01ce65ffa1e22ff2b9d3f9cc25bba68c9de0959d46e37c75b7d1381` | **the gate.** `c`, `thresh`, `meta` |
| `op_phi4mini_k5_L0.05_mean_user_L5.pt` | `f2ab0ebe63e362a9569f289899ed98f773ec2ab5184a6f47215ec264929698fc` | the withdrawn L5 point, kept for §5 |
| `dir_phi4mini_sep_L13to15.pt` | `1678cca193eb4362f4f32a4d486c9030ba8c7a236bb1cf34b2426e9dd9dc2098` | the band-only edit directions |
| `band_spec.json` | `6efa553fbb2fdb95bf34958e046915a19e5a8512cc38049731b9d7e8b8d33de2` | the measured profile the band came from |
| `gemma_pool681_in.json` (texts) | `bec25f4346193cfd56c97b1a4fec6440b127d615adf6e978b7ead2567712ab35` | 1426 in-corpus prompts |
| `fresh_gate_in.json` (file) | `8aec9bda6acc95d2bc9b8d8bf17ae5d66c6fbcec27d48de025e7e25c0ed3ca64` | 3216 out-of-corpus prompts |
| `fresh_gate_in.json` (texts) | `91c0ea983e19777863a523315fba48499a4062100a32dca095300a297667540f` | the text-level hash the capture asserts |
| `phi4mini_fresh_acts.pt` | `12cf5eab46ec0d00f831d908fbcf0604adadda19c0a9c6c01ac7a19d63f99c14` | out-of-corpus activations, 3.80 GB |

On k36: `/home/biscloud/evals/phi_1006/clf/`, `/home/biscloud/evals/phi_1006/abl/out/`,
`/home/biscloud/qout/{gemma_pool681_acts.pt, phi4mini_pool681_acts.pt, phi4mini_fresh_acts.pt,
gemma_pool681_in.json}`, `/home/biscloud/evals/phi_ooc/fresh_gate_in.json`,
`/home/biscloud/evals/gated-serve/op_k5_L0.05_mean_user_L8.pt` (the Gemma contrast).

### 11.2 The splits

ONE `numpy.random.default_rng(seed)` used sequentially on the pool681 derivation rows:
TRAIN = 100 of the 174 `cyber_offense` rows, TEST = the remaining 74, REF = 10 rows from each of the
14 non-cyber harmful categories **in SORTED NAME ORDER** = 140, LEAK = the remaining 378.
FIT = TRAIN ∪ REF = 240 rows is the only input to any fitted quantity. All 180 `para_*`/`adv_*` rows
are excluded everywhere. The shipped point is **seed 0**.

### 11.3 The commands

    # as run, 2026-10-06, k36, CPU only
    ssh -i /home/kaini/YOLO/.k/k36 -o ConnectTimeout=300 biscloud@172.16.100.36
    cd /home/biscloud/evals/phi_1006/math
    bash run_cpu_math.sh /math/vf_main.py     # positive control + constants + 100 seeds + k curve
    bash run_cpu_math.sh /math/vf_ooc.py      # out of corpus + reversal + the Gemma contrast
    bash run_cpu_math.sh /math/vf_percat.py   # per-category pooled over 100 seeds

`run_cpu_math.sh` is `docker run --rm --network none --cpus=4 --memory=24g --ipc private`,
`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `nice -n 15`, no `--gpus`.
Every script begins with `assert not torch.cuda.is_available()`.

Outputs: `vf_main.{py,log,json}`, `vf_ooc.{py,log,json}`, `vf_percat.{py,log,json}`,
`opmeta_dump.json`, mirrored to `ablation/records/phi-math-2026-10-06/`.

### 11.4 Positive control — required before any Phi number is reported

The six published **Gemma** constants, to float64. **If this fails, nothing downstream counts.**

| constant | published | mine | \|diff\| |
|---|---|---|---|
| `λ₁` | 0.020988415510819532 | 0.020988415510819536 | 3.47e-18 |
| `λ₅` | 0.005293340159109831 | 0.005293340159109827 | 3.47e-18 |
| `λ₆` | 0.004478617897311338 | 0.004478617897311337 | 8.67e-19 |
| fraction of `‖d‖²` in top-5 | 0.34534146252760484 | 0.34534146252760523 | 3.89e-16 |
| `cos(c, unit(d))` | 0.9571012822030949 | 0.9571012822030953 | 4.44e-16 |
| threshold | 0.14037645065907944 | 0.14037645065907942 | 2.78e-17 |

**6/6 PASS**, max |diff| 4.44e-16. Supporting controls, all reproduced exactly:

    c1 = unit(d) at the positives-only rule   t 0.1560491719 (doc 0.15604917)
                                              TRAIN 80/100, TEST 48/74, LEAK 10/378, AUC 0.9374
    k = 5 at R2_L0.05                         TRAIN 93/100, REF 7/140, TEST 63/74, LEAK 12/378,
                                              BENIGN 10/554, AUC 0.9763
    λ₁₃ 0.00209854679 (doc 0.002098547)       λ₁₄ 0.00189115799 (doc 0.001891158)
    Gemma out of corpus                       878/1350, 16/469, AUC 0.946678 (doc 0.9467)

> One correction for the next person. `CLASSIFIER-MATH.md` §4.2 describes the superseded `c1` rule
> as "the 80th of 100 TRAIN cyber scores". With `fire ⟺ score ≥ t` on 100 ascending scores, that is
> **index 20** (zero-based), not index 19. Index 19 gives TRAIN 81/100 and `t` 0.14791852786; index
> 20 gives TRAIN 80/100 and `t` 0.1560491719, which is the documented value. The ambiguity cost one
> wrong supporting-control reading on the first pass.

### 11.5 The freeze

`CLASSIFIER-MATH.md` §8's freeze protocol applies unchanged. The L12 file records
`freeze_direction_max_abs_diff_all_nonfit_rows_nan = 0.0` and
`freeze_threshold_identical_all_nonfit_rows_nan = true`, with the negative half checked —
NaN-ing one TRAIN row destroys `c`, NaN-ing one REF row destroys `t`. **Those two fields are
inherited from the build, not re-run here**; what this document re-ran is the forward derivation,
which reproduced `c` at `cos = 1.0` and `t` at |diff| 6.94e-17 from the fit rows alone.

### 11.6 A metadata defect in the SIBLING file, found and localised

The withdrawn **L5** file's 14 `heldout_leak_by_cat_*` fields are **byte-identical to the L12
file's** and sum to **19**, contradicting its own `heldout_nontarget_fired = 8`. I recomputed all 28
fields from the activations using each file's own stored `c` and `thresh`:

| | recomputed sum | in-file sum | file's own `heldout_nontarget_fired` | verdict |
|---|---|---|---|---|
| **L12** | **19** | **19** | **19** | **self-consistent — the L12 fields are CORRECT** |
| L5 | **8** | 19 | **8** | the L5 fields are **WRONG**; they are L12's |

Exactly two of the 14 L5 fields are wrong; the other twelve coincide because both layers fire zero
or the same small count there:

| category | L5 in file | L5 true | L12 (correct, both) |
|---|---|---|---|
| `fraud_financial` | 14/120 | **5/120** | 14/120 |
| `influence_ops_disinfo` | 2/78 | **0/78** | 2/78 |

**So: the L12 file's own per-category fields are trustworthy, and were confirmed field by field.**
The writer carried the L12 values into the L5 file. **Nothing was patched.** The L5 file is the
withdrawn point and is retained unmodified as the §5 record; editing it would destroy the artefact
the §5 story is about. Served behaviour was never affected — it depends only on `c` and `thresh`,
both of which reproduce at both layers (§4.3), and the per-row out-of-corpus scores agree with the
independently stored scores at max |Δ| = 0.0 over 3216 rows.

**If a future build writes an L5-style file, the writer's per-category accumulator is the bug to
look for.** The denominators were right in both files, which is why the error survived review: only
the numerators were stale.
