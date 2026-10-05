# tempx

**A search-free, reference-free rank-1 abliteration pipeline.** Five prompts in, one weight edit
out. **263 lines of core code** across four steps: derive the target tensor set from a checkpoint's
safetensors headers, capture the residual stream at the last prompt token, build one direction per
layer and weight it by `sep(l)`, then apply a single rank-1 row-normalised edit at one global dose
`K`. **No Optuna loop, no reference model, no gradient, no architecture table** — everything it needs
about the model is read off the checkpoint and out of the five prompts. Validated on dense
(Llama-3.1-8B-Instruct, **64** targets) and MoE (Qwen3.6-35B-A3B **120**; Gemma-4-26B-A4B-it **90**)
checkpoints, the same enumeration rule handling both.

For comparison, Heretic — whose orthogonalisation step and update rule this method uses, with credit
— ships with 800 prompts and roughly 200 Optuna trials on an A100 (*their shipped defaults, quoted
from Heretic, not measured by us*). **That contrast is the claim this repo makes**, and it is close to
the only one: it does **not** claim to beat published abliterations in general. See
[Limitations](#limitations).

- **How it works:** [`docs/METHOD.md`](docs/METHOD.md)
- **What it measured:** [`docs/RESULTS.md`](docs/RESULTS.md)
- **How to measure it without fooling yourself:** [`docs/PITFALLS.md`](docs/PITFALLS.md) ← read this
- **The evaluation instruments:** [`eval/README.md`](eval/README.md)
- **Who gets credit for what:** [`ATTRIBUTION.md`](ATTRIBUTION.md)

---

## Read this before you believe any number, including ours

> **A refusal count is not an ablation score.**
> On our shipped default the keyword metric said **13 refusals** while hard refusal was **2** — and
> 11 of those 13 matches fired on the bare word *"illegal"* inside answers that did the thing. Our
> strongest build scores **14** keyword refusals and **0** hard refusals. Meanwhile the published
> model that scores best on Heretic's KL — the metric standing in for "still a good model" —
> actually delivers content on only **20 of 100** prompts. A keyword metric cannot tell "I won't
> help" from "I won't help, but here is how you would do it yourself", and those are different
> outcomes.
>
> [**`docs/PITFALLS.md`**](docs/PITFALLS.md) has nine of these, each with the measurement that found
> it. Most cost this project a retracted finding.

---

## Install

```bash
git clone <this repo> && cd tempx
pip install -e .                 # core: torch, transformers, safetensors
pip install -e '.[eval]'         # + requests, for the three-way judge in eval/
pip install -e '.[shard]'        # + accelerate, if you need --devmap auto
```

Python >= 3.10. One console entry point, `tempx`, with four subcommands. Every option also reads the
environment variable the original research scripts used (`SRC DST DIRS K MODE KEEP ACTS OUT PROMPTS
MODEL DEV DEVMAP MAXMEM BS SYS THINK`), and explicit flags win over the environment.

## Quickstart

Four commands against a Hugging Face checkpoint, ending in a patched local model directory.

```bash
# MODEL is what `capture` loads; SRC is the same checkpoint on disk, which steps 1
# and 4 read directly (headers and shards). They must be the SAME checkpoint —
# `$SRC` can simply be your local snapshot of `$MODEL`.
export MODEL=google/gemma-4-26B-A4B-it      # hub id, or just use "$SRC"
export SRC=./gemma-pristine                 # the unmodified checkpoint on disk
export DST=./gemma-ablated
mkdir -p out

# 1. Which tensors get edited — derived from the checkpoint, not from a model card.
#    Confirm the printed TOTAL (90 for Gemma-4-26B-A4B-it) and, on an MoE model,
#    that the routed-expert stacks are in the list.
tempx targets "$SRC"

# 2. Residual stream at the last prompt token, every layer, five prompts.
tempx capture --model "$MODEL" --prompts kits/minimal.json --out out/acts.pt

# 3. One direction per layer + the sep(l) weights. Canonical settings: sep / 0.
#    Check the printed post-orthogonalisation |cos| is ~0 and the weight range is sane.
tempx direction --acts out/acts.pt --out out/dir.pt --mode sep --keep 0

# 4. The edit. DST must ALREADY be a copy of SRC — the patch is applied in place.
cp -r "$SRC" "$DST"
tempx patch --src "$SRC" --dst "$DST" --dirs out/dir.pt -K 1.5

# -> $DST is a patched checkpoint, loadable with transformers as usual.
```

`examples/run_gemma.sh` and `examples/run_llama.sh` are the same flow written out with the right dose
for each model class, commented to be read rather than only run.

### Choosing `K`

| model class | recommended starting dose |
|---|---|
| Gemma-class, 90 targets | **K = 1.5** |
| Llama-class, 64 targets | **K = 1.25** |

Three rungs — `K = 1.0 / 1.25 / 1.5` — is the entire search.

`K` is not a "more is better" knob, and the reason is exact rather than empirical. The applied weight
is `w = K · sep_l`, and in the row-normalised frame the update multiplies the component along the
direction by `(1 − w)`. So the residual along `v` is `|1 − w|`: at `w = 1` the component is removed,
at `w = 1.5` half of it is back **with the sign flipped**, and at **`w = 2` the edit is a pure
reflection that restores the original magnitude**. Past that it amplifies. The practical failure mode
is a build that *regains* refusal at high dose while also losing capability.

Because `sep_l` is normalised to mean 1 but varies per layer, `K = 1.5` already puts the
highest-`sep` layers near `w = 2`. [**`docs/METHOD.md`**](docs/METHOD.md) works the arithmetic
through, derives the alignment ceiling (full removal needs `w = 1/c²`, so a direction less than
0.7071 aligned with its target cannot be removed at any dose), and explains why you **dose-match on
`max|dW|`, not on `K`** — `max|dW|` is linear in `K` to ±0.00007.

On Gemma at 90 targets, **K=1.5 and K=2.0 were identical on every axis we measured**: there is a
broad plateau above 1.5, not a further rung.

---

## Kit choice: the shipped default is not the strongest kit we measured

> [!IMPORTANT]
> **The default `kits/minimal.json` (the `h006` kit) is not the best-performing kit in our
> measurements.** A **CBRN-domain kit measured considerably better at full dose.** Paired on the same
> 100 prompts at `K = 1.5`:
>
> | | CBRN-domain kit | `h006` (shipped default) | paired test |
> |---|---|---|---|
> | **COMPLIED** | **85** | 71 | McNemar exact **p = 0.0043** (18 gained, 4 lost) |
> | **DEFLECTED** | **15** | 27 | McNemar exact **p = 0.0169** |
> | hard refusal | **0** | 2 | — |
>
> **But the advantage is dose-dependent and disappears at partial dose.** On a 150-prompt,
> 15-category set at `K = 0.5` the two kits were indistinguishable — REFUSED/DEFLECTED/COMPLIED
> **33/89/28** (`h006`) vs **35/86/29** (CBRN), **118 of 150** prompts labelled identically, and
> **every McNemar p ≥ 0.72** (Spearman +0.897 REFUSED / +0.772 DEFLECTED across the 15
> categories). So this is a full-dose effect, not a property of the kit as such — and anyone
> comparing kits at a partial dose and concluding they are equivalent is measuring the dose, not the
> kit.
>
> **Outside this one comparison the method looked largely insensitive to which harmful prompt seeds
> it** — but the evidence for that is thin, and the CBRN row above is the standing counterexample
> that a kit *can* be meaningfully better. The two arbitrary, non-selected harmful prompts this
> project shipped on gave statistically indistinguishable results (14/177 vs 15/177, p = 1.000), and
> **two draws are not a distribution**. The published prior is a lottery effect, not insensitivity.
> So: swap the harmless prompts freely, and **re-measure** if you swap the harmful one.
>
> **The alternative kit is described by domain only. Its prompt text is bio/chem material and is not
> distributed here**, so its rows in `docs/RESULTS.md` are honest measurements that **cannot be
> reproduced from this repo as shipped**. Treat them as a reported upper bound on what the method
> reaches at full dose. To supply your own kit — the format, what the harmless side does and does not
> need, and the two diagnostics to check before spending a patch on it — see
> [**`kits/README.md`**](kits/README.md).

---

## Results

Gemma-4-26B-A4B-it, 90 targets, `mlabonne/harmful_behaviors` **`test[:100]`**. **Refusals** and
**KL** are Heretic's own unmodified `KeywordRate` and `KLDivergence` scorers, run by us unchanged.
**REFUSED / DEFLECTED / COMPLIED** are **our** three-way judge on the *same response strings* from
the same run.

| build | Refusals (Heretic) | KL (Heretic) | REFUSED | DEFLECTED | COMPLIED |
|---|---|---|---|---|---|
| pristine `google/gemma-4-26B-A4B-it` | 100/100 | 0 | 96 | 3 | 1 |
| **`h006` kit, K=1.5** (shipped default) | 13/100 | 0.3524 | 2 | 27 | 71 |
| CBRN-domain kit, K=1.5 | 14/100 | 0.3896 | 0 | 15 | 85 |
| huihui-ai (published weights) | 25/100 | 0.2104 | 4 | 32 | 64 |
| SevenOfNine (published weights, Heretic-made) | 29/100 | 0.0837 | 3 | 77 | 20 |

**The two halves of this table are not interchangeable, and the three-way columns are not comparable
with anybody's published figures** — they are our measurement, made here, on these 100 strings. Do
not cite them next to a published abliteration's own reported scores.

Two things in that table are worth sitting with. SevenOfNine has the **lowest KL** — by Heretic's own
damage measure the least-damaged edit here — and delivers on **20/100**. And on quality, 50 harmless
prompts needing substantive replies, judged blind and position-randomised by the pristine Gemma
itself: our CBRN K=1.5 build was **not distinguishable from pristine** (pairwise p = 0.28, absolute
McNemar p = 1.00), while huihui at roughly *half* the KL **was** worse (p = 0.0019, p = 0.039). **KL
did not predict judged quality damage on this model.**

Full tables, the dose ladder, the per-claim calibration (what we measured live vs what we quote from
others) and the scope bounds: [**`docs/RESULTS.md`**](docs/RESULTS.md).

---

## Intended use

Intended for **safety research, red-teaming and interpretability work on open-weight models.** What
it does is remove refusal behaviour from a model's weights — that is the function, not a side effect.

- **No model weights are distributed by this repo**, and none of the builds measured above are
  published here.
- You fetch any checkpoint you modify from its own publisher, **under that publisher's licence, and
  you are responsible for complying with it.** Model licences commonly restrict modification and
  redistribution.
- This repo's code is **AGPL-3.0-or-later** ([`LICENSE`](LICENSE)), matching Heretic, from which
  two of `eval/`'s three scorers are derived — see [`ATTRIBUTION.md`](ATTRIBUTION.md).

## Limitations

What is established here is narrower than the headline, and the repo would rather say so than be
corrected later.

**What is established.** Five prompts, one patch, three dose rungs, no search, no reference model,
no gradient — against Heretic's 800 prompts plus ~200 Optuna trials. The enumeration rule yields 64 /
120 / 90 targets blind on three checkpoints, dense and MoE alike. At full dose the CBRN kit delivers
significantly more than `h006` (paired McNemar p = 0.0043 / p = 0.0169), and our build shows **no
quality damage resolvable against the original model** on the 50-prompt quality run.

**What is not.**

- **No parity or superiority claim over published abliterations.** On Heretic's own two metrics our
  builds are **not uniformly better**: both huihui and SevenOfNine have lower KL than either of ours.
- **One model family carries almost every number.** All behavioural results are
  Gemma-4-26B-A4B-it at 90 targets. Llama-3.1-8B-Instruct and Qwen3.6-35B-A3B appear **only as
  enumeration counts**. Cross-model generalisation of the dose ladder, the quality result and the
  deflection finding is **not measured**.
- **Per-category and cross-model orderings are not established and are not reported.** At n=10 per
  category, Clopper-Pearson intervals are 0.31–0.90 wide and chi-square expected counts fall below 5.
- **Power.** A power analysis of the underlying research found that **exactly one** comparison in it
  was adequately powered — and that comparison involves a model excluded here on licensing grounds,
  so it is not reported at all. The paired p-values above stand on their own; they are not a claim
  that the wider programme was well powered.
- **The quality judge was the pristine model itself** (self-preference risk: conservative for our
  build, unusable as a published absolute ranking), **n = 100 / 150 / 50** on the evaluation sets,
  **one run per cell** with no error bars, and the **CBRN rows are not reproducible from this repo**.
- Every number in this repo is traceable, and where a figure is quoted from someone else rather than
  measured by us, it is marked as such.

The per-claim status table and the full scope bounds are [`docs/RESULTS.md`](docs/RESULTS.md) §7
**Calibration** and §8 **Bounds**.

---

## Attribution

This method is an assembly, and the load-bearing parts are not all ours.

- **`orthogonalize_direction` is Heretic's**, not ours — [`github.com/p-e-w/heretic`](https://github.com/p-e-w/heretic),
  `main.py:459-469`.
- **The rank-1 PRE row-normalised update rule is Heretic's**, verified by reproducing Heretic's
  published weights to **cos 0.99985**.
- **The foundational method is Arditi et al. 2024**, *Refusal in Language Models Is Mediated by a
  Single Direction*, [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) — difference-of-means
  direction plus rank-1 weight edit.
- **Ours:** the `sep(l)` per-layer weighting, the blind 5-prompt kit, structural target enumeration
  applied across dense and MoE, the three-way REFUSED/DEFLECTED/COMPLIED evaluation, and the
  dose/coverage findings.

[**`ATTRIBUTION.md`**](ATTRIBUTION.md) is the full accounting, including the licensing situation and
the prior art this repo does **not** claim. If you cite this, cite Arditi et al. and Heretic too —
[`CITATION.cff`](CITATION.cff).
