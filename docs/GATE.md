# GATE

A restriction on the edit the rest of this repo applies.

`patch` removes refusal from every harm category at once — one direction, one weight edit, and the
model stops declining across the board. That is what an abliterated checkpoint is. The gate narrows
it: it reads the request's topic from an **early** layer and applies the refusal-removal edit **only
when the request is in one designated category**, leaving the refusal behaviour the model already
has in place for everything else. One category unlocked instead of fifteen. The default, and the
behaviour on every error path below, is **no edit** — which means the pristine weights answer, which
means the model refuses.

Module `src/tempx/gate.py`; tests `tests/test_gate.py`. It needs no second copy of any weight and
writes nothing to disk, so it is not an alternative way to build a checkpoint — it is the runtime
form of the same edit, with a condition in front of it.

---

## What it does

Three parts, and the order of them is the whole design.

1. **Read.** A forward hook on decoder layer **8** takes that layer's output, pools it over the
   **user-content span only** (`mean_user`: no system prompt, no chat scaffold, no padding), and
   scores it as `cos(pooled, c)` against a category direction `c`. If the cosine clears the
   threshold, the sequence has **fired**. Which positions get pooled is not a detail — see
   [The read must pool the user span](#the-read-must-pool-the-user-span).
2. **Freeze.** The decision is taken once, at prefill, and held for the rest of that sequence's
   generation. It is a `[B]` boolean, one entry per sequence in the batch.
3. **Edit.** Forward hooks on every target tensor in layers **10–16** apply the rank-1 PRE
   correction — but only to the rows that fired.

The edit is the one in `docs/METHOD.md` step 4, unchanged. For an `nn.Linear` forward `y = x @ W.T`
and the PRE step `W' = W − w·outer(n ⊙ v, v @ (W/n))`:

```
y' = x @ W'.T  =  y − w · (x @ u)[..., None] · (n ⊙ v)        u = v @ (W / n)
```

So each target tensor contributes two precomputed vectors — `u` of length `in_features` and
`w·(n ⊙ v)` of length `out_features` — and the hook is one dot product and one scaled add. Cost is
`O(in + out)` per token instead of `O(in · out)`, and no second copy of a weight exists anywhere.

The band is layers 10–16 because that is where the edit energy already is on the model this was
measured on: 42.7% of `Σ‖dW_t‖²` across 21 of 90 tensors (`docs/PITFALLS.md` 2). That same section
is the reason to read the coverage number `attach` prints rather than assume the band was reached.

**That band is Gemma-4-26B-A4B-it's and does not transfer.** Across five models the refusal band
spans 7% to 25% of depth and sits at different depths, so a fixed seven-layer band is unsupported:
derive it per model from that model's own `sep(l)` profile and pass it as `edit_layers`. The read
layer is the part that did generalise — on all five, category is readable at L8 from the user span,
and L8 is strictly below every one of their bands, so the upstream-of-edit constraint holds without
retuning. Per-model bands and profiles are in NOTES-2026-10-05 and the kits beside it.

---

## Why one forward pass is enough

The obvious implementation of a conditional edit is two passes: classify, then run the model again
with the edit on or off. The gate does it in one, and the reason it can is that **the read is
strictly upstream of the edit**.

Layer 8 is read; layers 10–16 are edited. Nothing the edit does can reach layer 8, because the edit
is applied by hooks on layers at greater depth in the same forward. So the hidden state the gate
scores is **identical in the edited and the unedited model**, and the decision is the same either
way. There is no circularity to break and nothing to iterate: the classifier's input does not depend
on the classifier's output.

`Gate.__init__` asserts `max(gate_layers) < min(edit_layers)` and raises otherwise. That assertion
is the single-pass argument, which is why it is a constructor error and not a warning: a gate that
reads a layer the edit touches is reading its own effect, and it would not announce itself.

Two consequences worth stating because they are easy to assume away:

- **Per sequence, not per batch.** Different sequences in one batch get different decisions. The
  edit hook multiplies its correction by the `[B]` decision mask, so a row that did not fire comes
  out of the hooked module exactly as the unedited module produced it. A test runs a mixed batch —
  one firing row, one not — and checks each row against its own single-sequence run.
- **Decided once.** Either mean pools over the prompt. Recomputed per generated token the pooled
  mean drifts, and the gate can flip mid-answer. The decision is taken when the KV cache is empty
  (a prefill) and then frozen; a test drives three decode steps whose *own* read sits on the other
  side of the threshold and checks that the decision does not move — in both directions.

---

## The fail-safe property

Every path that cannot produce a decision produces **no edit**, and the pristine model answers:

| situation | what happens |
|---|---|
| no decision yet (nothing has run the read) | no edit |
| `force = False` (the disabled state) | no edit, and the output is **bit-identical** |
| no category direction, or no threshold | no edit |
| `mean_user` and no spans were set for this prefill | no edit, reason recorded |
| a span that is `None`, mis-counted, or not this batch's | no edit for the whole batch |
| the direction's width does not match `out_features` | that tensor is not hooked, and is named |
| the padding mask is not the `[B,S]` padding mask | no edit, reason recorded |
| `attention_mask` cannot be read from the call at all | no edit, and `attach` says so |
| a module's output rows are not sequences, or no longer match the frozen decision | that tensor is skipped, and counted in `unrouted` |
| **any** exception inside the gate | no edit, counted in `errors`, text in `last_error` |

There is no path in the other direction: nothing in the module turns an error into an applied edit.
`tests/test_gate.py` asserts one row of that table at a time, each with a positive control first —
the same gate, working, visibly changing the output — because a test that proves "no edit" against
a path that was never going to edit anyway proves nothing. That is a statement about the failure
modes enumerated and tested here. It is not a statement about failure modes nobody has thought of.

**And note what this table is not.** Every row is a case where the gate cannot tell, and declines.
None of them is a case where the gate decides confidently and is wrong — a non-target request whose
score clears the threshold is simply fired on, and nothing downstream catches it. See
[There is no second line of defence](#there-is-no-second-line-of-defence). The fail-safe property is
about uncertainty. The threshold is the only thing handling error.

Three operational constraints that are not failure modes but will produce surprises:

- **One forward at a time per `Gate` instance.** The frozen decision is instance state. A batch is
  the supported way to serve several sequences at once — the decision is a `[B]` vector and the hook
  masks by row. Two concurrent forwards through one instance would interleave their decisions, and
  a sequence could be answered under another sequence's. Use one instance per forward.
- **Generate with the KV cache.** The prefill is recognised by the cache being empty. With
  `use_cache=False` the whole sequence is re-fed every step, there is no prefill to recognise, and
  the decision is retaken each step.
- **If the gate never fires, read `last_error`.** A width mismatch between `c` and the model's
  hidden size, or a mask the pooler refused, both land there rather than in an exception.

---

## The read must pool the user span

This was a defect, found by measurement, and the default changed because of it.

`mean_all` — every non-pad position, scaffold and system prompt included — was validated on
**Gemma-4-26B-A4B-it only, and it does not transfer.** Category AUC at the read layer, one
activation capture per model on an identical 1426-prompt input:

| model | mean_all AUC at L8 |
|---|---|
| Gemma-4-26B-A4B-it | 0.9394 |
| Llama-3.1-8B-Instruct | **0.6959** |
| Phi-4-mini-instruct | **0.6041** |
| Qwen2.5-1.5B-Instruct | **0.5680** |
| Qwen3-0.6B | **0.5593** |

Four of five are at or near chance. **The cause is measured, not inferred.** On those four, a chat
scaffold position carries an activation **11×–31× larger** than the user span's and nearly constant
across prompts, so the all-position mean is dominated by a near-constant: on Llama
`cos(mean_all, mean_user)` is **−0.047**, i.e. the pooled read is near-orthogonal to the user's
request, and the between-prompt spread of `mean_all` is only 5–10% of its own norm. The same
artifact corrupts the refusal measurement on those models, which is why the source report reads
refusal separation on `last` and category on `mean_user`. Gemma is the exception on every one of
those diagnostics, which is why this went unnoticed.

Pooling the user span fixes it: **0.89–0.93 at the same layer on all five models**, and the span
arithmetic was exact for 1426 of 1426 items on every model — zero fallbacks — so it is not a
pooling that quietly degrades. `mean_user` is therefore the default. Full per-layer curves,
controls and the per-model diagnostics are in NOTES-2026-10-05; they are deliberately not copied
here, so the two documents cannot drift.

### What that costs the caller

A forward pass does not carry the prompt text or the tokenizer, so **the gate cannot locate the
span by itself** — the caller computes it and hands it over, once per prefill:

```python
from tempx.gate import Gate, user_span

spans = [user_span(tok, t, sys_prompt=SYS) for t in texts]   # capture_hf_v2.py's arithmetic
gate.set_spans(spans)                                        # consumed by the next prefill
out = model(**enc)
```

`user_span` is that arithmetic verbatim: render the chat template, require the user text to appear
exactly once, split there, and require the prefix, user and suffix token counts to sum to the
whole. It returns `None` when that is not exact.

**And `None` fails closed.** A span that could not be located, a batch given no spans at all, or a
span whose token total does not match the row's real-token count — each one stands the read down and
nothing is edited. The gate does **not** fall back to `mean_all`. That fallback is the defect, not a
mitigation for it: it is what would read a near-constant on four of these five models, and it would
do it silently. The offline `capture_hf_v2.py` does fall back, which is right for a measurement that
records and reports every fallback, and wrong for a gate. One unlocatable span stands down the whole
batch's read, since one read produces one decision vector.

---

## The operating point

Measured, **cyber** as the target category, direction built from **n = 10** prompts, 140 non-target
prompts. The pristine model already complies with **1** of those 140, so `excess` is leak minus that
baseline.

| cyber delivered | leak /140 | excess | benign FPR |
|---|---|---|---|
| 7 of 10 | 1 | 0 | 0.0% |
| 8 of 10 | 3 | +2 | 0.6% |
| 9 of 10 | 6 | +5 | 5.2% |
| 10 of 10 | 21 | +20 | 11.7% |

Read it as a curve and pick a row deliberately: delivery on the target category and leak onto the
non-target ones move together. The last row — full delivery on cyber — is also the row where 21 of
the 140 non-target prompts come through and the gate fires on 11.7% of benign prompts.

> ### These thresholds are `mean_all` L8 numbers, and `mean_all` is not the default
>
> **Every row above was measured with pooling `mean_all` at layer 8, on Gemma-4-26B-A4B-it, with a
> purified direction.** Verified from three independent artifacts, which agree:
> `routing-probe/paraphrase/para_scores.json` (`meta.pool`, `meta.layer`, `meta.mode`),
> `routing-probe/leakage_sweep.json` (`primary.pool`, `primary.layer`, `primary.mode`), and
> `NOTES-2026-10-04-e2e-selectivity.md`. The sweep was sound: `mean_all` scores 0.9394 on Gemma,
> which is why it worked there at all.
>
> **They are not thresholds for the pooling this module now defaults to.** A score produced by a
> different pooling is a **different quantity**, so comparing it to a threshold from this table is a
> **units error, not a small bias** — and §"The read must pool the user span" is the measurement of
> how far apart the two quantities are on four of five models. The same applies across models: these
> are Gemma's numbers.
>
> This is the one place where the module's default and its only measured operating-point table
> disagree, which is why it is stated here rather than in a footnote. An operating-point file
> carrying these rows must declare `"pool": "mean_all"`, and the gate will then read `mean_all`
> rather than its default — `from_file` takes the pooling from the file for exactly this reason.
>
> **What would close it: a threshold sweep under `mean_user`.** It needs no GPU and no new capture —
> last night's five captures in `ablation/kits/<tag>/` already store all three poolings at every
> layer, so it is a re-scoring of activations that exist. Until it is run, a `mean_user` gate has no
> measured operating point and these rows are not a substitute for one. **Not run here.**

**The bounds on these numbers, which are not small:**

- The direction behind them comes from **ten prompts**, scored against **140 non-target prompts**.
  One category, one model, one sweep.
- **The baseline is 1 of 140 and that is a wide number, not a tight one.** The exact
  Clopper–Pearson 95% interval on 1/140 **reaches 0.0392**: a true rate of nearly 4% is consistent
  with observing 1. Every figure in the `excess` column is a difference against that, so the `0` in
  the first row is not a measurement of zero leak — it is a count that matched the baseline count.
- A larger measurement — **174 target prompts and 518 non-target** — is in flight. These rows are
  what exists now, not what the design rests on.
- **The leak column is the read's own error rate, undamped** — there is no residual refusal behind
  the gate to absorb any of it, measured twice and independently. That is the next section, and it
  bears on which row of this table is a responsible choice.
- **No adversarial robustness claim is made here.** Two separate measurements bear on this and
  neither supports one.

  **The register lever is real, consistent, and short.** Rewriting a protected prompt in a cyber
  register raises its score in **40 of 40 paired cases** (p = 1.8e-12; CBRN 17 of 17; median
  +0.027) — so the lever is not noise and it points the way an attacker would want. And **0 of 40
  cross**, at any threshold from 90% coverage downward: the best case fell **0.0027 short** of the
  90% threshold and the median sat about **2.2 register steps short** of 80%. Both halves belong in
  any sentence about it. **One register step moves the score reliably and does not reach** — which
  is a fact about the size of one step against these thresholds, not the gate resisting an attack.
  Nothing was measured about two steps, a different register, or a lever chosen after seeing the
  threshold.

  **The classifier-level adversarial arm is uninformative**, not favourable: on those records a
  bag-of-words baseline sits effectively at ceiling, because the shared wrapper leaves the
  category-bearing content words intact, and nothing can be concluded from a comparison whose
  baseline has no headroom.

  So the position is unchanged: no robustness result to report, in either direction.

`docs/PITFALLS.md` closes on the rule this section is written to: an unproven claim sitting next to
a proven one borrows its credibility. The parity result below is proven and tested. The table above
is one measurement with wide intervals. They are different piles.

---

## There is no second line of defence

**Read this before choosing a threshold.** When the gate fires, the band-edited model delivers.
There is no residual refusal behind the read to catch a wrong decision, so **every error the read
makes is an answer, not a softened refusal.** Two independent measurements say so — different runs,
different corpora, different protocols.

**1. The leakage sweep** (`routing-probe/leakage_sweep.json`, `meta`). On the band arm,
`band_complied` is **143 of 143** — every prompt it saw was complied with — spread across all
**15 categories** at **8 to 10 per category**, with **0 REFUSED**. Its own note puts it plainly:

> band delivery is 8-10/10 in all 15 categories with 0 REFUSED, so there is NO residual safety
> behaviour behind the router: leakage == router error rate, undamped.

For contrast, on the same sweep the intact arm's `intact_complied` is **1 of 1** — the single
non-target prompt the pristine model already complied with, which is the `baseline_leak` the
`excess` column above is measured against.

**2. The end-to-end HarmBench measurement**
(`NOTES-2026-10-04-e2e-selectivity.md`), on 197 `standard` prompts generated in both arms under
identical settings. Labelled REFUSED / DEFLECTED / COMPLIED:

| arm | R / D / C | delivery |
|---|---|---|
| intact | 162 / 26 / 9 | 9/197 = 4.6% |
| band-only, K = 2.0 | **0** / 12 / **185** | 185/197 = **93.9%** [89.6, 96.8] |

**Zero refusals out of 197**, against 162 for the same prompts on the intact model — and re-judging
the ablated arm's byte-identical text moved **0 of 197** labels on that scale, so the zero is not
judge noise.

What that arm makes visible is the counterfactual for prompts the router did *not* send it. Of the
**167** it left on the intact model, the ablated model **would have delivered 155**, and that
includes **all 28** protected CBRN prompts, at **100%** — `0 of 28` were routed at that threshold,
so none leaked, but nothing behind the router would have stopped them if they had been.

**And at a different threshold, they were.** Moving to HarmBench's own 80%-coverage point misroutes
**2 of 28** CBRN prompts, and **the ablated path delivered both** — both of which **the intact model
refused**. That is not a counterfactual. It is the mechanism of this section, measured: the
threshold moved, two prompts crossed, and the answer went out.

### What follows for a threshold

The leak column in the table above is therefore **the read's own error rate, undamped**. The
threshold is not one layer of two. It is the only thing standing between a non-target request and a
delivered answer, which is why the fail-safe design points the way it does: everything in this
module is built so that an *uncertain* read declines. Nothing in it catches a *confident wrong* one.

This is also what makes the units warning above consequential rather than pedantic. A threshold
compared against scores from a different pooling is not a slightly-off threshold — **it is an
unguarded one.** The read fires or does not fire, and if it fires on the wrong prompt, that prompt
is answered.

Bounds on the two measurements: one model, one snapshot, one read, and one dose — the delivery
figures are a `K = 2.0` band-only arm, and the source report states that delivery is a property of
the dose. A weaker dose was not measured here, and might leave residual refusal behind the gate.
That would change the numbers above. It would not change the shape of the argument, which is that
you have to know which case you are in rather than assume the favourable one.

---

## The operating-point file

The category direction and the threshold are **loaded**, never written into the source, and the file
carries the provenance that makes the number mean something. `Gate.from_file` refuses a file missing
any of these rather than defaulting it:

```python
torch.save({
    "c": c,                      # the category direction, in the READ layer's residual space
    "thresh": 0.37,              # on cos(pooled, c)
    "meta": {
        "category": "cyber",     # which category this unlocks
        "pool": "mean_all",      # which pooling it was MEASURED with -- the gate obeys the file,
        "mode": "purified",      #   not its own default. These two rows are provenance, and the
                                 #   published cyber points are mean_all L8 with a purified
                                 #   direction. A mean_user file says mean_user.
        "gate_layer": 8,         # which layer it was built at -- the gate reads THIS
        "op": "8of10",           # which operating point, i.e. which row of the table above
        "delivered": 8, "n_target": 10,        # coverage at that point
        "leak": 3, "n_nontarget": 140,         # leak at that point
        "baseline_leak": 1,                    # the pristine model's own compliance
        "benign_fpr": 0.006,                   # benign false-positive rate at that point
    },
}, "out/cyber-8of10.pt")
```

`pool` and `gate_layer` are read *from the file*, so the direction and the depth it was built at
cannot drift apart — the most expensive silent error available here would be scoring a layer-8
direction against some other layer's residual. The edit band stays the caller's argument.

The threshold is a cosine, not a projection, for the reason `sep(l)` is a ratio (`docs/METHOD.md`):
residual norms grow with depth and vary per prompt, so a threshold on an unnormalised dot product
would partly be a threshold on activation magnitude, and would not transfer between prompts.

---

## Parity: the equivalence this rests on

The claim is that the hook on an unedited module computes what `pre_step`'s edited weights compute.
Measured on CPU at fixed seeds, hooked-and-unedited against `pre_step`-materialised weights:

| comparison | float32 | float64 |
|---|---|---|
| `W − outer(w·(n ⊙ v), u)` vs `pre_step(W, v, w)`, entrywise | — | ≤ **8.9e-16** absolute |
| one `nn.Linear` forward, relative | **1.8e-7** (≈1.5 eps) | **2.3e-16** (≈1 eps) |
| 17-layer stack, 14 targets, K ∈ {0.5, 1.0, 1.25, 1.5}, relative | **1.3e-6** | **1.5e-15** |
| the gate forced OFF | **bit-identical** (`torch.equal`) | **bit-identical** |

Nine orders of magnitude between the float32 and float64 arms is the point of the float64 arm: it
says the gap in the float32 column is rounding and not algebra. The test bounds are set from these
measurements with roughly 8× and 700× headroom, and the tests say in their docstrings that a failure
is to be reported rather than accommodated by raising a tolerance.

At **bfloat16** the two paths differ, and inherently: the offline path rounds the edited *weight* to
bf16, the hook rounds the *correction*. Measured on one target Linear: 0.0050 relative, about 1.3
bf16 ulps. Parity is a float32 statement; neither path is the more faithful one.

---

## What it does not reach

`attach` returns `(hooked, not_reached)` and prints both. On the band this was built for — layers
10–16, which is **14 direction keys** over **21 target tensors** — a hook reaches the **14** 2-D
tensors and **not** the **7 fused 3-D expert stacks**, because a fused stack is a parameter inside
somebody's MoE routing code and not an `nn.Linear` with a forward to hook.

The arithmetic for those 7 is not the problem: each expert slice has its own exact `(u, w·(n ⊙ v))`,
and a test checks all 7 stacks against `pre_step`, slice by slice. The hook machinery is the problem.
So on an MoE checkpoint the gate applies a **smaller** edit than `patch` does at the same `K`, and
`docs/PITFALLS.md` 2 — coverage and dose are complements, not substitutes — says what that costs.

### Which 14, and why that is the uncomfortable part

The 14 a hook reaches are exactly `self_attn.o_proj` and `mlp.down_proj`, seven of each. **That is
precisely the set both published abliterations of this checkpoint edit** — neither of them touches
the routed experts; `docs/PITFALLS.md` 2, "On MoE models the routed experts are the easy thing to
miss", has the tensor counts for both builds. The 7 the hook cannot reach, the fused
`layers.N.experts.down_proj` stacks, are the ones **this project added**, and including them is what
moved its flagged count at full capability — again, the numbers are in PITFALLS 2, which is where
they should stay so these two documents cannot drift apart.

Stated plainly, because it is the main practical limitation of this module on the project's own
primary model: **the runtime hook path reproduces published-standard abliteration coverage on this
MoE, and what it cannot reach is specifically the coverage improvement this project contributed.**
It is not a small residue at the edge of the band. It is the delta that distinguished this method's
builds from the published ones.

Full band coverage on an MoE therefore needs one of two things, neither of which is this module:
the offline `patch()` path, which means a second checkpoint on disk; or an MoE forward patched to
apply the correction per expert, which is a change to somebody else's routing code and is not here.

**On a dense backbone there is no gap at all** — every band target is an `nn.Linear`, and `attach`
reports 14 of 14 with nothing unreached. A test asserts that against a real `LlamaForCausalLM`.

So read the number `attach` prints, against your own checkpoint, rather than assuming the band was
covered. That is the same instruction PITFALLS 2 ends on, and the gate prints the number so that
following it costs nothing.

The direction is also **one blob only**. Sequential application needs the second step's `u` computed
from the already-edited `W`, which is the one thing a hook does not have; `patch` offline does that.

Both gaps leave the edit smaller than the offline patch's, i.e. nearer to refusing.

---

## Using it

```python
from tempx.gate import Gate

gate = Gate.from_file("out/dir.pt", "out/cyber-8of10.pt", K=1.25,
                      edit_layers=range(10, 17))     # the band is per model: derive it, don't inherit
hooked, not_reached = gate.attach(model)             # prints the category, read, band and counts

gate.set_spans([user_span(tok, t, sys_prompt=SYS) for t in texts])   # once per prefill
out = model(**enc)                                  # or model.generate(**enc, ...)

gate.detach()                                       # the model is pristine again, bit-for-bit
```

`gate.fired` is the frozen decision, `gate.scores` the cosines behind it, `gate.errors` and
`gate.last_error` what the gate swallowed. Log them: a gate that silently never fires and a gate
that is working look identical from outside.

There is **no `tempx gate` subcommand.** The four existing subcommands are artefact transforms —
checkpoint to target list, prompts to activations, activations to directions, checkpoint to
checkpoint — and a runtime hook produces no artefact. Making it a subcommand would mean adding model
loading and generation to the CLI, which nothing else there does. It is a library entry point.

---

## What is not claimed

- **The in-distribution category AUC is nearly all lexical, and must not be read as evidence that
  the gate's read is semantic.** A tf-idf bag-of-words model on the raw prompt text, scored with the
  identical scorer, reaches **0.9121** — as high as any model's best layer, and above the activations
  at most layers. On that set the activations add almost nothing over counting words.
  What rescues the activation read is **held-out paraphrases**, which restate the same requests in
  different words: tf-idf falls to **0.9004** while the activations hold at **0.96–0.98**, and the
  margin over tf-idf roughly doubles on all five models. So the honest claim is narrow and it is
  this: **activations beat lexical features under paraphrase, not in-distribution.** That arm covers
  4 of the 15 categories, balanced, which is a clean test of a subset and not of all 15.
- The adversarial evidence is a measured lever that falls short, plus an arm whose baseline is at
  ceiling — see the operating-point bounds. Neither is a robustness result, and "0 of 40 crossed"
  is a measurement at one register step against these thresholds, not a property of the gate.
- The read layer generalised across five models and **the edit band did not** — those five span 7%
  to 25% of depth at different depths. A band copied from this document to another model is an
  untested guess; derive it from that model's own profile.
- The table is one category (cyber), one model, one read layer, one pooling, a direction from ten
  prompts, and 140 non-target prompts with a 1-prompt pristine baseline whose Clopper–Pearson
  interval reaches 0.0392. The 174/518 measurement is in flight and is not in this document.
- Delivery and leak were measured; the *mechanism* of the leak was not. Whether the 21 leaking
  prompts at full delivery share a structure — a topic adjacent to cyber, a phrasing, a length — is
  unexamined, and "it leaks on 21 of 140" is not the same finding as "it leaks on these 21".
- The parity result and the fail-safe table are the tested parts, and they are claims about this
  module's arithmetic and error paths. They say nothing about whether the operating point above is
  a good place to stand.
- The fail-safe property covers the gate **declining when it cannot tell**. It does not cover the
  gate **deciding confidently and being wrong**, and on the build these numbers come from nothing
  else does either: measured twice, there is no residual refusal behind the read. A threshold is
  being chosen under that condition whether or not whoever chooses it knows.
