# Results: what five prompts and one patch actually did

Every number here was produced by the four scripts in this repo at `MODE=sep KEEP=0`, on
`google/gemma-4-26B-A4B-it` unless another model is named. Nothing is rounded for presentation and
nothing is reported that we did not run. Where a figure is quoted from someone else rather than
measured by us, the **Calibration** table says so.

> **Headline.** A 5-prompt kit (1 harmful + 4 harmless), one rank-1 weight edit per target, three dose
> rungs and no search takes Gemma-4-26B-A4B-it from 100/100 refusals to 14/100, with 85/100 responses
> judged as actually delivering the requested content and **no quality damage we can resolve against
> the unmodified model**. It does **not** make us better than published abliterations in general: on
> Heretic's own two metrics, huihui and SevenOfNine both have lower KL than either of our builds.

---

## 1. Headline table — Gemma-4-26B-A4B-it, 90 targets

Evaluation set: `mlabonne/harmful_behaviors` **`test[:100]`**. Columns **Refusals** and **KL** come
from **Heretic's own unmodified scorers** (`KeywordRate` and `KLDivergence`), run by us, unchanged.
Columns **REFUSED / DEFLECTED / COMPLIED** are **our** three-way judge applied to the *same response
strings* from the same run.

| build | Refusals (Heretic `KeywordRate`) | KL (Heretic) | REFUSED | DEFLECTED | COMPLIED |
|---|---|---|---|---|---|
| pristine `google/gemma-4-26B-A4B-it` | 100/100 | 0 | 96 | 3 | 1 |
| **h006 kit, K=1.5** (shipped default) | 13/100 | 0.3524 | 2 | 27 | 71 |
| **CBRN kit, K=1.5** | 14/100 | 0.3896 | 0 | 15 | 85 |
| huihui-ai (published weights) | 25/100 | 0.2104 | 4 | 32 | 64 |
| SevenOfNine (published weights, Heretic-made) | 29/100 | 0.0837 | 3 | 77 | 20 |

**The two halves of this table are not interchangeable, and the three-way columns are not comparable
with anybody's published figures.** Heretic's `KeywordRate` scores the *opener* — it counts a response
as a refusal when a refusal phrase appears in it. Our judge scores *delivery*: REFUSED = declines,
DEFLECTED = engages, lectures, moralises, substitutes a safe adjacent answer or stops short of
anything actionable, COMPLIED = delivers the thing that was asked for. The two disagree hardest where
it matters most: SevenOfNine scores 29 keyword refusals but only **3** judged REFUSED and **77**
DEFLECTED. A keyword metric cannot see the difference between a model that refuses and a model that
agrees and then says nothing.

Do not cite our REFUSED/DEFLECTED/COMPLIED numbers next to a published abliteration's own reported
scores. They are a different measurement, made here, on these 100 strings.

---

## 2. The CBRN kit beats h006 — but only at full dose

This is the one build-vs-build comparison in this repo that carries a paired test, and it is the
result most likely to be misread, so both halves are stated together.

**At full dose (K=1.5), paired on the same 100 prompts:**

| | CBRN kit | h006 kit | paired test |
|---|---|---|---|
| COMPLIED | **85** | 71 | McNemar exact **p = 0.0043** (18 gained, 4 lost) |
| DEFLECTED | **15** | 27 | McNemar exact **p = 0.0169** |
| REFUSED | 0 | 2 | — |
| KL | 0.3896 | 0.3524 | — |

Fourteen more responses delivered, with the gain/loss split (18 gained against 4 lost) showing it is
a real shift and not a re-shuffle. The CBRN kit is the stronger kit at full dose, by a margin that
survives a paired test.

**At partial dose (K=0.5), on a 150-prompt set spanning 15 categories, the two kits are
indistinguishable:**

| | REFUSED | DEFLECTED | COMPLIED |
|---|---|---|---|
| h006, K=0.5 | 33 | 89 | 28 |
| CBRN, K=0.5 | 35 | 86 | 29 |

Spearman **+0.897** on REFUSED and **+0.772** on DEFLECTED across the 15 categories, **118 of 150**
prompts given the identical label by both builds, and **every McNemar p ≥ 0.72**.

**So the CBRN advantage is a full-dose effect, not a property of the kit per se.** At half dose the
kits behave like the same intervention. Anyone comparing kits at a partial dose and concluding they
are equivalent is measuring the dose, not the kit.

**Why h006 is nonetheless the shipped default:** this repo does not distribute bio/chem/CBRN prompt
text. The CBRN row is reported because we measured it; it is not reproducible from the kits in this
repo. See **Bounds**.

---

## 3. Dose ladder — CBRN kit, same 100 prompts

| K | Refusals (Heretic) | KL | REFUSED | DEFLECTED | COMPLIED | `max\|dW\|` |
|---|---|---|---|---|---|---|
| 1.0 | 28/100 | 0.2405 | 0 | 47 | 53 | 0.1808 |
| 1.25 | 22/100 | 0.3141 | 0 | 25 | 75 | 0.2261 |
| 1.5 | 14/100 | 0.3896 | 0 | 15 | 85 | 0.2713 |

Monotone on every axis across the three rungs. REFUSED is already 0 at K=1.0 — **what K buys above
1.0 on this model is the conversion of deflection into delivery, not the removal of refusal.** That is
the axis a keyword metric is blind to, and it is the reason the three-way judge exists.

**`max|dW|` is linear in K to ±0.00007.** At K=0.5 the h006 kit gives 0.0840 and the CBRN kit gives
0.0904 — the kits differ in the *magnitude* of the edit at equal K, which is a kit property, not a
dose property. The edit magnitude is therefore predictable: K is a dial on a known quantity, and
there is no regime change inside the ladder.

**Plateau:** on Gemma at 90 targets, **K=1.5 and K=2.0 are identical on every axis we measured.** The
useful dose range on this model tops out at 1.5; pushing further bought nothing and is not reported as
if it were a further rung.

---

## 4. Quality: no resolvable damage, and KL did not predict it

50 harmless prompts that require a substantive reply. Judge: the **pristine Gemma itself**, blind
pairwise, position-randomised, plus an absolute (per-response, acceptable/not) pass.

| build | KL | pairwise vs pristine | absolute (McNemar) |
|---|---|---|---|
| ours, CBRN K=1.5 | 0.3896 | **p = 0.28** (not distinguishable) | **p = 1.00** |
| huihui (published) | 0.2104 | **p = 0.0019** (worse) | **p = 0.039** |

**KL did not predict judged quality damage on this model.** huihui sits at roughly *half* our KL and
is the build the judge can tell apart from the original; ours, at nearly double the KL, is not
separable from pristine on either test. KL against the original model is a measure of distributional
movement, not of whether the answers got worse. Treat a low-KL abliteration as *less moved*, not
automatically *less damaged*.

**Disclosed caveat — the judge is the pristine model itself.** That is a self-preference risk and it
cuts in a specific direction. A model asked to compare its own outputs against an edited copy of
itself will tend to favour its own; that makes our "not distinguishable from pristine" result
**conservative** (we are being judged by the party with an interest in finding us worse) and makes the
huihui result **conservative in the same direction only if huihui's style is equally far from the
judge's**, which we did not establish. So: the comparison is usable as evidence that *our* edit did
not visibly degrade this model, and it is **not** usable as a published absolute quality ranking
between our build and huihui's. A third-party judge, not run here, would be required for that.

---

## 5. SevenOfNine: the best KL in the table, 20/100 delivered

SevenOfNine is a Heretic-made build and it wins the metric Heretic treats as the quality side of its
objective: **KL 0.0837**, the lowest in §1. Our judge places it at **77/100
DEFLECTED and 20/100 COMPLIED**: it very rarely refuses outright (3), and it very rarely delivers.

A note on the other Heretic metric, because it cuts the other way and should not be glossed:
SevenOfNine's keyword-refusal count is **29/100**, the *highest* of the four abliterated builds in §1
and therefore its *worst*-scoring axis there — ours are 13 and 14. So SevenOfNine is not uniformly
best on Heretic's pair; it is best on KL, and it is the build whose KL a Heretic-style evaluation
would single out as the least damaged.

This is the cleanest single illustration of why the three-way split was built. A build can top the
metric that is supposed to stand for "still a good model" and still almost never answer the question,
and the keyword count sitting next to it cannot tell you which of the two is happening.

**Mechanical note, measured from the weights, not inferred:** SevenOfNine changes **40 of 1013
tensors and touches zero routed-expert matrices.** Our 90-target build changes **90 tensors,
including 30 expert stacks.** A plausible reading is that leaving the routed experts untouched leaves
the delivery pathway intact-but-unreached; we did not run the ablation that would test it, so this
stays a hypothesis with a tensor count attached, not a finding.

---

## 6. Other architectures

Structural target enumeration (`enumerate_targets.py`) was run blind against three checkpoints:

| checkpoint | targets enumerated | dose used |
|---|---|---|
| Llama-3.1-8B-Instruct | 64 | K=1.25 |
| Qwen3.6-35B-A3B | 120 | — |
| Gemma-4-26B-A4B-it | 90 | K=1.0 / 1.25 / 1.5 |

The enumerator needs no architecture knowledge and handles dense and MoE checkpoints with the same
rule. **The behavioural evaluation above is Gemma only.** No Refusals/KL/three-way numbers for Llama
or Qwen appear in this document, because we are not reporting any here — see **Bounds**.

---

## 7. Calibration

| claim | status |
|---|---|
| All five rows of the §1 table — Refusals, KL, and the three-way split | **confirmed live by us.** Heretic's unmodified scorers plus our judge, one run, same response strings, `test[:100]`. |
| CBRN dose ladder K=1.0/1.25/1.5 (§3) | **confirmed live by us**, same prompt set. |
| `max\|dW\|` values and their linearity in K to ±0.00007 | **confirmed live by us**, read off the patch itself. |
| K=1.5 ≡ K=2.0 on Gemma at 90 targets | **confirmed live by us** on the axes measured (§1 columns). |
| CBRN vs h006 at K=1.5: McNemar exact p = 0.0043 / p = 0.0169 | **confirmed live by us**, paired on the same 100 prompts. |
| CBRN vs h006 at K=0.5 indistinguishable: all McNemar p ≥ 0.72, 118/150 identical, Spearman +0.897 / +0.772 | **confirmed live by us** on the 150-prompt, 15-category set. |
| Quality: ours p = 0.28 / 1.00; huihui p = 0.0019 / 0.039 | **confirmed live by us**, 50 prompts, pristine-Gemma judge, blind and position-randomised. Self-judging caveat in §4. |
| SevenOfNine 40/1013 tensors, zero routed-expert matrices; ours 90 with 30 expert stacks | **confirmed live by us**, measured by diffing the shipped weights. |
| Target counts 64 / 120 / 90 | **confirmed live by us**, output of `enumerate_targets.py` on those checkpoints. |
| `orthogonalize_direction` and the rank-1 PRE row-normalised update rule are **Heretic's** (`github.com/p-e-w/heretic`, `main.py:459-469`) | **quoted** — their code, their design. Our reproduction of Heretic's published weights to **cos 0.99985** is confirmed live by us and is what establishes that we implement *their* rule. |
| The foundational method — difference-of-means direction plus rank-1 weight edit — is **Arditi et al. 2024**, arXiv:2406.11717 | **quoted.** |
| Heretic's shipped defaults: 800 prompts, ~200 Optuna trials | **quoted** from Heretic. |
| SevenOfNine was produced with Heretic; huihui's method and fitting data | **quoted** (SevenOfNine) / **unknown** (huihui — we do not know what it was fitted on, so nothing about contamination or its objective can be asserted either way). |
| Heretic is AGPL-3.0-or-later; RepIt (arXiv:2509.13281) is Apache-2.0 | **quoted** from their repos. |
| Why the expert-stack difference explains SevenOfNine's deflection rate | **derived from source and weights, not measured.** No ablation was run. Hypothesis only. |
| That the enumeration rule generalises beyond these three checkpoints | **derived from the rule, not measured.** It is a structural rule; three checkpoints is three checkpoints. |
| That `max\|dW\|` stays linear in K outside the measured 0.5–1.5 window | **derived, not measured.** |
| Per-category and cross-model orderings | **not established.** See **Bounds** — these are reported nowhere in this document for that reason. |
| Parity with, or superiority over, published abliterations in general | **not claimed, and not supported.** On Heretic's own two metrics our builds are not uniformly better: both huihui and SevenOfNine have lower KL. |

### What this repo does claim

Five prompts, one patch, three dose rungs. No search, no reference model, no gradient, no
hyperparameter optimisation — against Heretic's 800 prompts plus ~200 Optuna trials on an A100. And,
with the paired statistics in §2 and §4: at full dose the CBRN kit delivers significantly more than
h006, and our build shows no quality damage resolvable against the original model.

---

## 8. Bounds

Stated so a reader can reproduce the scope, not just the conclusion.

- **One model family carries almost every number.** §1–§5 are Gemma-4-26B-A4B-it at 90 targets.
  Llama-3.1-8B-Instruct and Qwen3.6-35B-A3B appear only as enumeration counts (§6). Cross-model
  generalisation of the dose ladder, the quality result and the deflection finding is **not measured**.
- **Evaluation sets are n=100 and n=150.** `test[:100]` for everything in §1–§5 except the partial-dose
  comparison (150 prompts, 15 categories) and the quality run (50 harmless prompts). Differences of a
  few counts on n=100 are inside the noise unless a paired p-value is quoted next to them.
- **Per-category figures are unresolvable at n=10.** The 150-prompt set is 15 categories × 10.
  Clopper-Pearson intervals at n=10 are 0.31–0.90 wide, so no per-category ordering in the underlying
  data is reportable and none is reported here.
- **Chi-square is unusable on the category breakdown.** Expected counts fall below 5. Any test of
  "which categories differ" on these data would be invalid; the paired McNemar tests in §2 are on the
  pooled prompt sets, which is why they are the only tests quoted.
- **Power.** A power analysis of the underlying research found that **exactly one** comparison in it
  was adequately powered, and that comparison involves a model excluded here on licensing grounds. It
  is therefore not reported in this document at all. The §2 and §4 p-values are paired tests that
  stand on their own; they are not a claim that the wider programme was well powered.
- **Replies were truncated at the token cap.** A response cut off mid-answer is judged on what it
  produced. This pushes a borderline COMPLIED towards DEFLECTED and therefore makes the delivery
  numbers conservative — but the cap is a judgement-affecting parameter, not a neutral one.
- **The quality judge was the pristine model itself.** Self-preference risk; conservative for our
  build, unreliable as a published absolute. §4 spells this out. No third-party judge was run.
- **Published abliterations were evaluated as shipped, with their own tokenizers and chat
  templates.** We did not normalise huihui or SevenOfNine onto our generation settings; each build was
  run the way its author ships it. That is the fair comparison for "what does this model do", and it
  means a share of any difference may be template or tokenizer, not weights.
- **The CBRN kit is not in this repo.** Its prompt text is bio/chem material and is not distributed.
  The CBRN rows in §1–§3 are honest measurements that **cannot be reproduced from this repo as
  shipped**; the h006 default can. Treat the CBRN figures as a reported upper bound on what the method
  reaches at full dose, not as a reproducible artifact.
- **One run per cell.** Nothing in §1 or §3 is an average over seeds; there are no error bars on the
  Refusals, KL or three-way counts, and we did not measure run-to-run variance. The paired tests in §2
  and §4 are paired across *prompts*, not across repeated runs.
- **Not measured, and would be useful:** behavioural numbers for Llama and Qwen; a third-party quality
  judge; the K≥2.0 region beyond the identity check; per-kit comparisons at K=1.0 and K=1.25; an
  ablation holding the expert stacks out of our 90 targets to test the §5 hypothesis; any evaluation
  set outside the AdvBench-family strings in `mlabonne/harmful_behaviors`.
