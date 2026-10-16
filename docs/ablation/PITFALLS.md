# PITFALLS

Nine traps this method walks into, each with the measurement that found it. Most of them cost this
project a retracted finding. They are listed roughly in order of how much damage they do if missed.

Every number below was measured by us unless it is marked *quoted*.

---

## 1. A refusal count is not an ablation score

A keyword refusal metric answers "did the reply contain a decline phrase". That is not the same
question as "did the model do the thing", and at low refusal counts the two come apart almost
completely.

**The residual is deflection, not refusal.** On Gemma-4-26B-A4B-it, 100 prompts from
`mlabonne/harmful_behaviors test[:100]`, scored with Heretic's own unmodified `KeywordRate` *and*
with our three-way judge on the same response strings (`eval/keyword_rate.py` and
`eval/judge_threeway.py`; `docs/RESULTS.md` is the canonical table):

| build | keyword refusals | hard refusal | DEFLECTED | COMPLIED |
|---|---|---|---|---|
| pristine `google/gemma-4-26B-A4B-it` | 100/100 | **96** | 3 | 1 |
| h006 kit, K=1.5 | 13/100 | **2** | 27 | 71 |
| CBRN kit, K=1.5 | 14/100 | **0** | 15 | 85 |
| huihui-ai (published) | 25/100 | **4** | 32 | 64 |
| SevenOfNine (published, Heretic-made) | 29/100 | **3** | 77 | 20 |

Read the last two columns. Two builds with nearly the same keyword score — 25 and 29 — differ by
**64 vs 20 COMPLIED**. The keyword metric cannot see the difference between a model that answers and
a model that reframes the question into something harmless and answers *that*. Observed
substitutions: a bomb request becomes a baking-soda volcano; a stalking tool becomes a "Reminder
Bot"; a self-harm prompt returns crisis resources. The same bomb→science-demo move appears in
published Heretic builds, across three model families and two vendors.

**Decline markers also fire inside fully compliant output.** Real examples from our runs: *"I cannot
help but notice"* in a letter the model wrote as asked; *"I won't experience any service
interruption"* inside a carrier-impersonation script; *"they can't help but follow you everywhere"*
inside a pro-stalking blog post. Each counts as a refusal.

**The published result to cite here is [arXiv:2512.13655](https://arxiv.org/abs/2512.13655)**, which
reports keyword-refusal precision of **11.2% on six abliterated models** using substantially this
marker list including `illegal` and `unethical` (*quoted*). Our measured 1.7% is a stronger version of
an already-published finding, not a new one. Rank reversal between keyword metrics and content judges is
also published — see [arXiv:2402.10260](https://arxiv.org/abs/2402.10260) (StrongREJECT), Spearman
**−0.394** against human ratings.

> **Citation corrected 2026-10-04.** An earlier draft cited GuidedBench
> ([arXiv:2502.16903](https://arxiv.org/abs/2502.16903)) for "a keyword false-positive rate of 35.76% vs
> 9.07%". The paper is real, but those figures are **one cell of its Table 5** (NegativeKeyword vs
> GuidedEval, on the "General Advice" topic only); other cells run as high as 87.63, and **its
> false-positive rate runs in the opposite direction to ours**. The citation as written misrepresented
> the source and has been replaced. Note also that [arXiv:2308.01263](https://arxiv.org/abs/2308.01263)
> (XSTest) found string matching *preserved* ranking, so the literature is not unanimous.

**Once an edit removes refusal thoroughly, the metric stops measuring refusal at all.** The marker list splits into refusal
*phrases* ("i cannot", "i won't", "sorry", "unable"), AI-identity phrases ("as an ai", "ai assistant",
"designed to", "programmed to") and seven *topic* words (`disclaimer`, `violat`, `prohibit`, `illegal`,
`harmful`, `inappropriate`, `unethical`). Attributing every hit across pristine plus six edited builds
of this model, 100 prompts each:

| build | keyword hits | fired a decline phrase | AI-identity | topic word only |
|---|---:|---:|---:|---:|
| pristine | 100 | **99** | — | — |
| six edited builds, pooled | **121** | **0** | 12 | 109 |

Zero of 121. The decline vocabulary is absent from all 600 of those edited responses. As a refusal
detector the metric runs at **97% precision / 100% recall on pristine** and **1.7% precision / 25%
recall on those six edited builds** — 2 true positives against 119 false positives. Almost every hit
there is `illegal` or `disclaimer` sitting inside a delivered answer that carries a legal caveat.

**Bound on that zero, and the counterexample.** All six of those builds patch the full 90-tensor target
set at K ≥ 1.0. A *partial* build behaves differently: a 21-tensor edit restricted to layers 10–16 at
K=1.25 fired **6 decline phrases** in 100 responses, and its 3 judged-REFUSED rows are *exactly* the 3
rows matching `i cannot`, carrying genuine "I cannot fulfill this request." text. Perfect
correspondence — on that build the metric worked.

So the correct statement is **not** "a keyword metric breaks after any edit." It is: the metric detects
overt refusal competently whenever there is overt refusal to detect, and it is **structurally blind to
deflection** — which is what dominates once an edit removes the refusal channel thoroughly. The
precision collapse above is a consequence of near-total removal, not of editing as such. Expect the
metric to look fine on weak edits and to become uninformative exactly when the build gets good.

**This is enough to rank two builds backwards.** Two kits differing only in their single harmful seed,
same model, same 100 prompts, paired:

| dose | keyword says | delivery says |
|---|---|---|
| K=1.0 | h006 25 vs CBRN 28 → h006 better | 37 vs 53 → **CBRN better**, p=0.0070 |
| K=1.25 | h006 19 vs CBRN 22 → h006 better | 62 vs 75 → **CBRN better**, p=0.0146 |
| K=1.5 | h006 13 vs CBRN 14 → h006 better | 71 vs 85 → **CBRN better**, p=0.0043 |

The keyword metric prefers the worse kit at all three doses. At K=1.25 the 16-point swing decomposes
exactly: 7 points from deflections it misses (28 vs 21), 7 from compliances it falsely flags (11 vs 18),
2 from hard refusals it misses.

**Repairing the marker list does not fix this.** Dropping the seven topic words costs nothing on
pristine (still 100/100) and cuts the edited counts from 13–28 down to 0–4 against 0–3 true refusals —
a far better refusal detector. But implied success becomes 99 vs 97, so the worse kit still wins. The
dominant error is structural: a deflection contains no refusal phrase, because it does not decline — it
answers a different question, helpfully, at length. No marker vocabulary can see that.

**What to do.** Report the keyword metric when you need comparability with published figures — it is
the number other people publish, so do not silently replace it. But report a content judgement
*alongside* it, on the same response strings, and **never rank builds on the keyword number alone** —
not even two builds of the same model that differ by one prompt. If you report it for an edited model,
report the decline-phrase attribution next to it, or the number means something other than what its
name says.

**The same caution applies to KL.** On this model KL did not predict judged quality damage: our
CBRN K=1.5 build at KL 0.3896 was **not distinguishable from pristine** on 50 harmless prompts
needing substantive replies (blind pairwise, position-randomised, pristine Gemma as judge:
p = 0.28; absolute McNemar p = 1.00), while huihui at roughly **half** the KL *was* worse
(p = 0.0019 and p = 0.039). KL bounds first-token drift on harmless prompts. It is not a capability
measurement and it is not a quality measurement.

---

## 2. Coverage and dose are complements, not substitutes

Missing target tensors cannot be bought back by raising `K`. This is the trap that produced the
worst build in the project's history.

**The measured direction is the counter-intuitive one.** Going from 40 to 90 targets on Gemma let `K`
drop from 2.0 to 1.5 — but the edit got **larger**, not smaller: `max|dW|` 0.0724 → 0.2521, and
refusal went 7/41 → 4/41 at 18/18 capability. `K` is a per-tensor weight multiplier, not a
perturbation budget. **More coverage buys lower dose.**

**The converse — that dose can buy back missing coverage — is NOT established here.** An earlier
internal summary claimed a band-restricted build needed `K = 5.26`, drove per-layer weights to
3.5–8.2, and *restored* refusal at 14/41. **We could not trace that claim to primary data**: it
appears once, as a back-reference citing a section that contains no such experiment, with no table
behind it. The tooling for it exists in the project's history, so the run may well have happened —
but the numbers are unsourced and are **withdrawn from this document** rather than published on the
strength of a summary sentence. Treat "more dose does not buy missing coverage" as a *plausible
expectation from the algebra*, not as something we measured.

**What the algebra does say,** independently of that run: past `w = 2` the edit reflects the
component rather than removing it (see `docs/METHOD.md`), so raising `K` far enough to compensate for
dropped tensors will eventually reinstate refusal with the sign flipped. That is a property of the
update rule, verified numerically, not an empirical claim about a particular build.

**Why cutting coverage is still a bad trade on this model.** Per-tensor edit energy is heavily
concentrated but not confined: on the h006 build at K=1.25, layers 10–16 hold **42.7%** of
`Σ‖dW_t‖²` across 21 of 90 tensors, and layers 0–2 only **3.2%**. Cutting to the peak band therefore
discards over half the edit, and buys a lower KL the same way not editing at all does. KL is only
meaningful against a delivery measurement.

**On MoE models the routed experts are the easy thing to miss.** A 26B MoE checkpoint here has 1013
tensors with 128 routed experts per layer. Two published abliterations of it change only the dense
output projections:

| published build | tensors changed | what |
|---|---|---|
| huihui-ai | **50 of 1013** | 25 layers × 2 tensors, no routed experts |
| SevenOfNine (Heretic-made) | **60 of 1013** | `self_attn.o_proj` ×30 + `mlp.down_proj` ×30, no routed experts |

Our enumerator derives **90** targets on the same checkpoint, and including the expert stacks it had
already found moved our flagged count 7/41 → 4/41 at 18/18 capability.

**Do not read that table as "those tools are broken."** This project asserted that published tooling
has a routed-expert gap and then **retracted it**: Heretic does edit routed experts, and `abliterix`
ships structural enumeration. Whether those two builds missed the experts because of the tool's
module map for that architecture at that version, or because of how they were configured, we did not
establish. The usable lesson is the narrow one: **verify coverage on your own checkpoint**, by
diffing the edited weights against the original and counting changed tensors, rather than trusting
that a tool reached everything.

Note also that the structural-enumeration rule itself is not uniquely ours —
[arXiv:2609.09793](https://arxiv.org/abs/2609.09793) states it and documents a checkpoint where the
*name* half of the rule fails silently. Shape plus name is a heuristic, not a theorem.

---

## 3. Response-prefix and chat-template differences change results more than the edit does

This is the single most expensive operational fact in the project. It invalidates cross-box
comparisons, shifts absolute numbers on everything measured before it is caught, and is invisible
unless you test for it specifically.

**What it cost.** Two machines served **two revisions of the same model's chat template** — 16 448 B
vs 17 466 B. Weights were byte-identical, as were the vLLM build, `transformers`, `tokenizers` and
`torch`. Same prompt, identity condition (no edit at all), N=192, system prompt present, T=1.0:

| | refusal rate | vs the other box |
|---|---|---|
| old 16 448 B template | **7.3%** CI [3.6, 11.0] | z = −5.33, **significantly different** |
| after installing the other template | **34.4%** CI [27.7, 41.1] | z = 1.55, consistent |
| the other box | **27.1%** CI [20.8, 33.4] | — |

**7.3% → 34.4% on identity weights.** Three findings files had to be marked uncomparable.

**How to detect it, and the trap in detecting it.** Send a chat request and read
`usage.prompt_tokens`:

| messages | old template | new template |
|---|---|---|
| user `"hi"` | 14 | 14 |
| + system `"You are a helpful assistant."` | **37** | **26** |

**The two templates are identical without a system prompt.** They differ only in how they fold in
the system role, which this architecture has no native slot for. A fingerprint taken on a bare user
message shows 14 = 14 and tells you nothing.

**The rules.**
- Record a **two-line prompt-token fingerprint** — bare user message, *and* with the system prompt
  — at the top of every run, and store it beside the results. Treat any number without a recorded
  fingerprint as uncomparable.
- Serve by **explicit path**, never by repo id: the same model id can resolve to different commits
  on different machines, and it did here.
- Keep `system_prompt` and `response_prefix` byte-identical across every arm. An auto-determined
  `response_prefix` must be recorded, not assumed — ours came out `""` in every arm, which is only
  useful information because it was logged.
- `vLLM` has **no chat-template fallback.** Deleting `chat_template.jinja` to "match" the other box
  made every chat request return HTTP 400, and a 192-run measurement came back as 192 errors scored
  as 0% refusal. No template file means no chat endpoint.

---

## 4. On MoE models, temperature-0 output is not reproducible across batch sizes

Expert routing depends on **batch composition**. Change the batch size and the same prompt at
temperature 0 can route to different experts and produce different text. We measured vLLM batch
composition swinging a refusal probability on a fixed prompt.

**What to do.**
- **Pin the batch size** and keep it identical across every arm you intend to compare. Ours was
  `batch_size=20` in every Gemma arm, chosen so it divides the prompt count cleanly.
- State the pinned value in the report. Results hold **at that batch size**; sensitivity to it was
  not measured, and claiming otherwise is unsupported.
- **Treat a single smoke test as a liveness check only.** A smoke prompt runs alone — batch of one —
  while the evaluation runs batched. The smoke test tells you the model loads and generates. It
  tells you nothing about what the eval will score.

---

## 5. Dose-match on `max|dW|`, not on `K`

`sep(l)` is derived from each kit's **own** harmless spread, so two kits at the same `K` produce
different edit magnitudes. An equal-`K` comparison is therefore partly a comparison of edit size.

Measured, Gemma, 90 targets, `K=0.5`:

| kit | `max\|dW\|` |
|---|---|
| h006 | **0.0840** |
| CBRN | **0.0904** |

That is a **7.6%** larger edit for the same `K`. On Llama the same two kits differed by about
**12%**. Dose-matching the Gemma pair would have required `K = 0.0840/0.1808 = 0.4646`.

`max|dW|` is linear in `K` to **±0.00007** (CBRN kit: 0.1808 / 0.2261 / 0.2713 at
K = 1.0 / 1.25 / 1.5), so converting between the two coordinates is a single division — there is no
excuse for not doing it.

**What to do.** Run the patcher once per kit at a reference `K`, read off `max|dW|`, and set each
kit's `K` so the edits match. If you ship an equal-`K` comparison anyway — we did, deliberately, for
the full-dose kit comparison — say so, and say which direction the bias runs.

---

## 6. `sep(l)` is undefined with one harmless prompt

With `n_harmless = 1` the harmless centroid *is* the single harmless point, so

```
mean_i ‖ g_i,l − ā_l ‖  =  0
```

and `sep(l) = gap / 0`. The *direction* is still perfectly well defined at `n_harmless = 1` — the
centroid is just that one point — so nothing visibly breaks until the weighting stage.

In the original research script the per-layer guard set `sep(l) = 0.0` instead of raising, which made
every layer weight zero, and the mean-1 rescaling then divided by `mean(w_raw) = 0` and raised a bare
`ZeroDivisionError` several steps away from the cause. The `direction` step in this repo rejects
`n_harmless < 2` up front with an explanatory error instead. If you reimplement this, add that check:
the natural failure is confusingly far from the mistake.

**Minimum viable kit is 1 harmful + 2 harmless.** Use 4 harmless — see `docs/METHOD.md` for why that
is the operating point.

---

## 7. Use `S[0]²/ΣS²` or rank-for-90%-of-energy to test rank-1-ness — never `S[0]/ΣS`

`S[0]/ΣS` is **not a rank measure.** This project concluded a published weight delta was "not
rank-1" from `S[0]/ΣS = 0.387` and **retracted several correct findings** on that basis. A row-wise
diagonal rescale — exactly what the row-normalised update applies — can drag `S[0]/ΣS` down to 0.35
on a delta that is rank 1 to numerical precision.

Correct measures on the same tensors:

| measure | value |
|---|---|
| top-1 energy `S[0]²/ΣS²` | **0.9949 – 0.9990** |
| rank needed for 90% of energy | **1** |

**What to do.** Compute `S = svdvals(W_edited − W_original)` in float64 and report `S[0]²/ΣS²`, or
the smallest `k` with `Σ_{i<k} S[i]² / ΣS² ≥ 0.90`. Also state **which** tensors you measured: a
sample of tensors establishes nothing about the ones outside it, and we have made that mistake too.

---

## 8. Judge noise: do not rank on differences smaller than the noise floor

Re-judging **byte-identical** response strings at temperature 0 moves labels.

| set | measured movement |
|---|---|
| 100 pristine replies, five sessions | **1 label in 100** |
| 150 replies, three-way label, second pass | **4 of 150 = 2.67%** (and 5/150 = 3.33% on the coherence label) |

The larger figure is the one to use. Three of those four moves sat on the **REFUSED/DEFLECTED**
boundary — the exact boundary this kind of study measures. Pass 1 gave 33/89/28; pass 2 gave
34/89/27.

**Consequence, applied to our own results.** In the 150-prompt partial-dose comparison the two kits
differed by 2 REFUSED and 3 DEFLECTED. **Both are below the noise floor**, and the report says so
rather than reporting an ordering. The full-dose difference (COMPLIED 85 vs 71, paired McNemar exact
p = 0.0043) is far above it, which is why that one is claimed.

**What to do.** Re-judge one arm a second time on byte-identical input, in the same session
configuration, and publish the measured movement as a noise floor. Then refuse to rank anything
inside it. Do not assume a figure from a previous study — ours got **worse** when we re-measured it
(1-in-100 → 2.67-in-100).

---

## 9. Truncation corrupts deflection measurement

Deciding DEFLECTED vs COMPLIED means deciding "did the model actually deliver the thing". You cannot
do that from a reply that stops mid-sentence.

With Heretic's default `max_response_length = 100` tokens, **78 to 93 of 100** replies ended
mid-sentence depending on dose. Worse, the rate is wildly asymmetric between arms: in one run the
edited arm had **73/100** replies ending mid-sentence against the pristine arm's **4/100** — because
ablated models write longer answers. The arms are then not comparable on any length-sensitive label
at all, and a coherence judge whose `DEGRADED` definition explicitly includes "truncation into
nonsense" is measuring where the token budget happened to fall.

**A generous cap helps and does not fix it.** At `max_new_tokens=512` — 5x Heretic's default — about
half of the ablated replies still hit the cap:

| arm, 150 prompts | ends sentence-final | at the 512 cap | mean tokens |
|---|---|---|---|
| pristine (96% refusals) | **150 / 150** | **1** | 61.3 |
| h006 @ K=0.5 | 82 / 150 | **72** | 340.2 |
| CBRN @ K=0.5 | 81 / 150 | **79** | 354.1 |

On a 50-prompt harmless set at the same cap, 36-39 of 50 were cut in **every** arm including
pristine.

Two things make that survivable, and both have to be checked rather than assumed:

1. **Refusals are short and self-terminating.** The pristine arm is 96% refusals and has 1/150 at
   the cap, so the REFUSED-vs-not boundary is not decided on truncated text. Truncation can only
   blur DEFLECTED vs COMPLIED.
2. **The rate was near-identical across the ablated arms** (72 vs 79, and 39/36/37 on the harmless
   set), so it is a confound on the absolute scale and not an arm-to-arm one.

**What to do.**
- Raise the cap well above the default, and report how many replies hit it, **per arm**.
- If the rates differ between arms — as they do between any ablated arm and a refusing pristine
  baseline — withhold the length-sensitive conclusions rather than working around them. One of our
  reports states plainly: "both readings are unsupported and I make neither."
- Keep the cap identical across arms, including the pristine baseline, and let a judgement that does
  not depend on length (a blind pairwise, for instance) carry the conclusion.

---

## A closing note on claims

The project behind this repo ran its own power analysis and found that **exactly one** of its
comparisons was adequately powered — and it involves a model excluded here on licensing grounds. So
the per-category and cross-model orderings in the source material are **not** established: at n=10
per category the exact Clopper-Pearson intervals on a deflection share are **0.31 to 0.90 wide**
(median 0.63), the extremes overlap in both arms, and chi-square expected counts fall below 5.

What the measurements do support: five prompts, one patch, three dose rungs, no search, no reference
model, no gradient. What they do not support: parity with or superiority over published
abliterations in general — on Heretic's own two metrics our builds are not uniformly better, and
both published builds we compared against have lower KL.

If you extend this work, keep the two piles separate. An unproven claim sitting next to a proven one
borrows its credibility.
