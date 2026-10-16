# METHOD

How the edit is computed, in enough detail to reimplement from scratch.

The pipeline is four scripts and about 263 lines. It needs no reference model, no gradient, no
search, and no architecture table: everything it needs about the model is read off the checkpoint
and out of five prompts.

**Canonical operating settings: `MODE=sep KEEP=0`.**

## Provenance of the parts

This method is an assembly, and several of its load-bearing parts are other people's work. The
table says which, and the rest of this document repeats the credit at the point of use.

| component | whose | reference |
|---|---|---|
| difference-of-means refusal direction + rank-1 weight edit | **Arditi et al. 2024** | *Refusal in Language Models Is Mediated by a Single Direction*, [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) |
| `orthogonalize_direction` | **Heretic** | [`github.com/p-e-w/heretic`](https://github.com/p-e-w/heretic), `main.py:459-469` |
| rank-1 **PRE row-normalised** update rule | **Heretic** | same repo, the `RowNormalization.PRE` branch of `Model.abliterate` in `model.py` |
| `sep(l)` per-layer weighting | ours | step 3 below |
| blind 5-prompt kit (1 harmful + 4 harmless) | ours | step 3, "How many prompts" |
| structural target enumeration across dense and MoE | ours (not uniquely) | step 1; the same rule is stated independently in [arXiv:2609.09793](https://arxiv.org/abs/2609.09793) |
| three-way REFUSED / DEFLECTED / COMPLIED evaluation | ours | see `docs/PITFALLS.md` |

Line numbers in another project's source drift between releases. The `main.py:459-469` citation is
the range recorded when this method was built against Heretic; in `v1.4.0-25-g3521f86` the same
block sits at `main.py:616-628`. Cite the **function and flag names** (`orthogonalize_direction`,
`RowNormalization.PRE`) as the stable identifiers and treat the line numbers as a hint.

**Licences.** Heretic is
**AGPL-3.0-or-later**, not a permissive licence: `pyproject.toml` declares
`license = "AGPL-3.0-or-later"`, the `LICENSE` file is the AGPLv3 text, and `README.md` repeats it
(verified against `v1.4.0-25-g3521f86`). RepIt
([arXiv:2509.13281](https://arxiv.org/abs/2509.13281)), which this repo compares against rather than
builds on, is Apache-2.0 (`github.com/wang-research-lab/RepIt`).

This repo does not vendor Heretic's code in `src/`. It reimplements two components whose
provenance is credited above, and it uses Heretic as an external tool for the comparison arms.
Whether that would have counted as sufficient separation under a permissive licence was never
settled; this repo is licensed AGPL-3.0-or-later to match Heretic, so the question does not arise.

We reproduced Heretic's published weights to **cos 0.99985** with the update rule as written below.
That is the evidence that the rule is Heretic's and that our implementation of it is faithful; it
is not evidence that anything else here is Heretic's.

---

## Step 1 — `targets`: derive the target set from the checkpoint

*Module `src/resgate/enumerate_targets.py`; research-era filename `enumerate_targets.py`.*

The edit makes matrices unable to *write* along a direction `v`. So the target set is every weight
whose output is added to the residual stream; anything left unedited can simply reintroduce the
direction downstream.

Two mechanical filters, applied to the safetensors headers and `config.json` only:

1. **Shape** — `hidden_size` sits in the tensor's *output* position:
   - 2-D `[hidden_size, *]` → `shape[0] == H`
   - 3-D `[n_experts, hidden_size, *]` → `shape[1] == H` (MoE checkpoints store the routed
     experts fused into one stack)
2. **Role** — the name contains an output-projection token:
   `o_proj | out_proj | down_proj | wo | dense_4h_to_h | c_proj`

Then exclude any tensor whose dotted name contains `mtp` (multi-token-prediction / speculative
head — not in the served forward pass), `vision_tower`, or `visual`.

`H` is read from `config.json["hidden_size"]`, falling back to
`config.json["text_config"]["hidden_size"]` for multimodal wrappers.

**Shape alone is not sufficient.** Llama's `q_proj` is `[4096, 4096]` and passes the shape test, but
it *reads* from the residual stream rather than writing to it. The role filter is a short,
architecture-general list of output-projection names — that is knowledge about transformers, not
knowledge about any one model, which is what keeps the enumerator blind.

Measured target counts:

| model | targets |
|---|---|
| Llama-3.1-8B-Instruct | **64** |
| Qwen3.6-35B-A3B | **120** |
| Gemma-4-26B-A4B-it | **90** |

## Step 2 — `capture`: residual-stream capture

*Module `src/resgate/capture.py`; research-era filename `capture_hf.py`.*

Run the five prompts through the model with `output_hidden_states=True` and keep, for each prompt
and each layer, the hidden state at the **last prompt token**.

Conventions that must be matched exactly or nothing downstream is comparable:

- `acts[l]` is the **output of layer `l`**. `transformers` returns `hidden_states[0]` as the
  embedding output and `hidden_states[i]` as the output of layer `i-1`, so drop index 0 and keep
  `[1:]`.
- **Left padding**, so `[:, -1, :]` is the last *real* prompt token for every sequence in the batch.
  (Heretic does the same.)
- `add_generation_prompt=True`, with the system prompt from `SYS`
  (default `"You are a helpful assistant."`).
- `THINK=0` closes the chat template's thinking block. On Qwen 3.5/3.6 the template defaults to
  `enable_thinking=True`, which ends the prompt *inside* an open `<think>` block. Capturing in one
  mode and serving in the other measures a different computational state. Match capture to serving.

Output is `{"nlast": 1, "snapshot", "sys", "items": [{key, group, acts[L, 1, D]}]}`, with
`group` in `{"harmful", "harmless"}`.

## Step 3 — `direction`: direction and per-layer weights

*Module `src/resgate/direction.py`; research-era filename `qwen_dirs_blind.py`. The old filename says
Qwen; the code is architecture-agnostic and is what we used for Qwen, Gemma and Llama alike.*

Let `H` be the harmful activations and `G` the harmless ones, at the last prompt token, in float64.
For each layer `l`:

**3a. Harmless centroid.**

```
ā_l = mean_i g_i,l
```

**3b. Raw direction — difference of means** (Arditi et al.):

```
raw = unit( h_l − ā_l )
```

**3c. Orthogonalize against the harmless centroid** — *this is Heretic's
`orthogonalize_direction`*:

```
ĝ_l = unit( ā_l )                       # unit-normalised mean harmless residual
d_l = unit( raw − (raw · ĝ_l) ĝ_l )     # remove the component along it, renormalise
```

Heretic's own comment explains the intent: adjust the direction so that only the component
orthogonal to the "good" direction is subtracted during abliteration.

**This step is not a no-op.** On one measured model/kit the pre-orthogonalisation
`|cos(raw, ĝ)|` was **0.22 mean / 0.38 max** across layers, so a fifth of the raw direction pointed
along the harmless centroid and was being removed from the model along with the refusal component.
After the step the same cosine is zero to float64 precision by construction. The `direction` step
prints both values; check them, because a pre-ortho cosine near zero means the harmless centroid is
nearly orthogonal to your contrast and the step bought you nothing on that model.

**3d. Per-layer separation `sep(l)`** — *this part is ours*:

```
                  ‖ h_l − ā_l ‖                harmful gap
sep(l) =  ───────────────────────────  =  ─────────────────────
           mean_i ‖ g_i,l − ā_l ‖          harmless spread
```

Then normalise to **mean 1** over the kept layers:

```
w_l = sep(l) / mean_{l ∈ keep} sep(l)
```

and the dose-scaled weight used by the patcher is `w = K · w_l`.

### Why `sep(l)` is a ratio, and why that matters

`sep(l)` is a **per-layer effect size expressed in units of the harmless cloud's own radius**. The
numerator and denominator are both distances in the same layer's activation space, so the units
cancel.

Activation norms grow substantially with depth in most transformers. A difference-of-means *gap*
therefore grows with depth too, for reasons that have nothing to do with how sharply that layer
distinguishes harmful from harmless input. Dividing by the harmless spread removes exactly that
scale factor: a layer with naturally large activations does not score higher merely for being
large. What survives the division is how far the harmful point sits outside the harmless cloud
**relative to how wide that cloud already is** — which is the quantity you actually want to dose on.

### Why mean-1 rescaling is a pure shape

After rescaling, `Σ_{l ∈ keep} w_l = |keep|` — numerically identical to `MODE=uniform`, where every
weight is 1. So the mean-1 step **redistributes the dose across layers without changing the total**.
`sep` and `uniform` at the same `K` spend the same total per-layer weight budget; they differ only
in where it goes. Any behavioural difference between `MODE=sep` and `MODE=uniform` is attributable
to the *profile*, not to one arm getting a bigger edit. (This is a statement about the sum of
per-layer weights. It is **not** a statement about `max|dW|`, which is a per-tensor quantity and
does differ — see `docs/PITFALLS.md`, "Dose-match on `max|dW|`".)

### Worked arithmetic for `sep(l)`

Two layers. Identical harmful gap; the harmless cloud is twice as wide in the second.

| | layer 7 | layer 8 |
|---|---|---|
| harmless points, as offsets from `ā_l` | −6, −6, +6, +6 | −12, −12, +12, +12 |
| `mean_i ‖g_i,l − ā_l‖` (harmless spread) | (6+6+6+6)/4 = **6** | (12+12+12+12)/4 = **12** |
| `‖h_l − ā_l‖` (harmful gap) | **12** | **12** |
| `sep(l) = gap / spread` | 12/6 = **2.0** | 12/12 = **1.0** |

The harmful point is equally far out in absolute terms, but in layer 8 it is only one cloud-radius
away instead of two, so `sep` halves. Now the mean-1 rescaling over these two layers:

```
mean(sep) = (2.0 + 1.0)/2 = 1.5
w_7 = 2.0 / 1.5 = 1.3333
w_8 = 1.0 / 1.5 = 0.6667
sum = 2.0 = number of kept layers      ✔ the budget is unchanged
```

At `K = 1.5` the applied weights are `w = 1.5 × 1.3333 = 2.0` at layer 7 and
`w = 1.5 × 0.6667 = 1.0` at layer 8. Read the next section before deciding that is what you want:
layer 8 lands exactly on full removal, and layer 7 lands exactly on the reflection point.

### `KEEP`

`KEEP` drops layers whose `w_raw` falls below `KEEP × mean(w_raw)`. `KEEP=0` keeps every layer and
is the canonical setting. Each kept layer `l` emits two direction/weight entries, `"{l}.attn_out"`
and `"{l}.mlp_out"`, both carrying that layer's single direction `d_l`.

## Step 4 — `patch`: the rank-1 PRE row-normalised edit

*Module `src/resgate/patch.py`; research-era filename `patch_generic.py`.*

*The update rule in this step is Heretic's.* For each target matrix `W` of shape
`[d_out, d_in]`, with a unit direction `v` in output space and scalar `w = K · w_l`:

```
n = rownorm(W)                            # n[i] = ‖W[i, :]‖,  shape [d_out]
Ŵ = W / n                                 # row-normalised frame
dW = − w · outer(n ⊙ v, v @ Ŵ)
W' = W + dW
```

For 3-D fused expert stacks `[n_experts, d_out, d_in]`, apply the 2-D step to each expert slice
independently.

Directions are keyed by layer: a target's layer index comes from `layers.(\d+).` in its name, and
its class is `mlp_out` if the name contains `down_proj` or `dense_4h_to_h`, else `attn_out`. Every
target at layer `l` uses layer `l`'s direction. `DIRS` accepts a comma-separated list for
sequential application of several direction sets.

The patcher writes bf16 in place into the safetensors payload of a pre-copied destination
checkpoint, asserting that the byte length of each replaced tensor is unchanged, and reports
`max|dW|` over all edited tensors. **`max|dW|` is the quantity to dose-match on**; it is linear in
`K` (measured to ±0.00007).

### What the update rule actually does

In the row-normalised frame the update is exactly a scaled rank-1 projector:

```
Ŵ' = ( I − w v vᵀ ) Ŵ
```

so the component of the output along `v` is multiplied by `(1 − w)`:

```
vᵀ Ŵ' = (1 − w) · vᵀ Ŵ
```

**The residual along `v` is `|1 − w|`.** This is exact, not approximate, and it has consequences
that are easy to get wrong:

| `w` | residual along `v` | what happens |
|---|---|---|
| 0 | 1 | identity, no edit |
| 0.5 | 0.5 | half the component survives |
| **1** | **0** | the component is removed — this is the target |
| 1.5 | 0.5 | half the component survives again, **with the sign flipped** |
| **2** | **1** | a **pure reflection**: full original magnitude, opposite sign |
| 3 | 2 | **amplified** past the original |

So `w` is not a "more is better" knob. Past `w = 1` the component comes back, inverted, and at
`w = 2` its magnitude is fully restored. The practical failure mode is a build that *regains*
refusal at high dose while also losing capability — see the coverage/dose pitfall.

### The alignment ceiling: full removal needs `w = 1/c²`

`v` is an estimate. Let `r` be the direction you actually meant to remove and

```
c = | cos(v, r) |
```

Applying the projector to `r` and reading off `r`'s own channel:

```
rᵀ ( I − w v vᵀ ) r  =  1 − w c²
```

So **full removal of the `r` channel needs `w = 1/c²`**, not `1/c`. Combining with the ceiling
above — you must keep `w < 2` to stay on the near side of the reflection point — gives a
requirement that involves no fitted model at all:

```
w = 1/c² < 2    ⟺    c² > 1/2    ⟺    |cos(v, r)| > 0.7071
```

**If your estimated direction is less than 0.7071 aligned with the direction you need to remove,
there is no dose that removes it without passing the reflection point.** At `c = 1` this reduces to
`w = 1`, consistent with the table above. Verified numerically in float64 against the patcher's own
`pre_step`: the `|1 − w|` column reproduces to 9 decimal places, and `1 − w c²` is zero to
machine epsilon at `w = 1/c²`.

This is the test that closes method variants cheaply. Measure `c`; if the layers that matter sit
below 0.7071, stop — no amount of dose tuning will rescue the build.

---

## How many prompts, and why

The blind kit is **five prompts: 1 harmful + 4 harmless.** That is the operating point, and these
are the honest reasons for it. Two of them are weaker than earlier drafts of this project claimed,
and the weaker version is the one stated here.

### The harmless side

- **Behaviourally, 4 and 8 are indistinguishable** on the structured kit (identical refusal counts
  on the same probe set). This is the measurement the "four prompts" setting rests on.
- **On direction alignment, random harmless prompts keep improving to about k ≈ 16–32** — but the
  remaining headroom is tiny. Fitting `a(k) = A − B/k` to random alpaca gives asymptote **0.9361**
  (rmse 0.0022), so going to k=128 buys **+0.0024** and k=1000 buys **+0.0034**. There is nothing
  left there worth collecting.
- **Format does not matter in the sense people expect.** A set of ordinary alpaca instructions
  ("Describe the scene of a sunset over a lake") ties our structured `Name N X` template on the
  behavioural metric, on two architectures. You do not need a special harmless format.
- **Format does matter as a bias–variance tradeoff.** A format-homogeneous harmless set has roughly
  2× the SNR of random alpaca — better in the small-`n` regime — but carries a bias that more data
  cannot remove: about **+0.0436** of alignment for a single template family, falling to **+0.0149**
  with 32 categories. Category spread removes about two-thirds of the bias; roughly 0.015 is
  attributable to the shared template itself and cannot be diversified away. At `k = 4` the
  structured set wins; by `k = 8` random overtakes it.
- **Selecting the "most benign" harmless prompts is worse than random** past about k ≈ 8. On this
  side, quantity works and selection hurts.

### The harmful side

**"One harmful prompt is enough" is too strong and we retract it.** *Random* harmful prompts keep
improving all the way to n=79 (alignment 0.8415 → 0.9511). What the measurements support is:

- **Selection substitutes for quantity.** One *well-chosen* prompt reaches 0.9433 alignment —
  equivalent to **8 to 16** random ones. So the honest sample-efficiency number on this side is
  about eight, not one.
- **Reference-free selectors find the good prompt**, which is what keeps the kit blind:
  leave-one-out agreement correlates `r = +0.964` with alignment, boundary depth `r = +0.966`.
  Neither needs a reference model.
- **Arbitrary single prompts can also work.** The two harmful prompts this project shipped on
  (`h006` and `h042`) were *not* reference-selected — one was picked by index by the user — and gave
  statistically indistinguishable results (14/177 vs 15/177, p = 1.000). That is **two draws, not a
  distribution**, and it is the reason the kit is usable in practice, not a guarantee.
- Going from 1 to 3 *selected* harmful prompts buys **+0.012** alignment (0.9433 → 0.9558). At this
  operating point the harmful side matters about an order of magnitude more than the harmless side.

### Why the sides are asymmetric

Only the variance **orthogonal** to the true direction corrupts a difference-of-means estimate:

```
d = h̄ − ā        ⟹      Var(d) = Σ_h/n_h + Σ_a/n_a
```

so the sample requirement on each side scales with that side's *orthogonal* variance,
`tr(Σ^⊥)`. The two clouds have very different orthogonal variance — harmless prompts are
heterogeneous in everything except being harmless — which is the mechanism behind "the harmless
side needs more prompts, the harmful side needs a better one."

### Prior art on small kits

Single-contrast-pair steering and one-example refusal suppression are **not novel** and this repo
does not claim them: see **ActAdd**, Turner et al. [arXiv:2308.10248](https://arxiv.org/abs/2308.10248),
and **Dunefsky & Cohan**, COLM 2025, [arXiv:2502.18862](https://arxiv.org/abs/2502.18862), which
reports 96.9% HarmBench ASR from one example *and* the lottery effect ("choice of the single example
is crucial"). Saturation in `n` is likewise established: Hao et al.
[arXiv:2505.03189](https://arxiv.org/abs/2505.03189) (knee ≈ 80), Braun
[arXiv:2602.17881](https://arxiv.org/abs/2602.17881) (convergence ≈ 150).

What is ours in this area is the specific blind 5-prompt kit and the `sep(l)` weighting it feeds —
not the idea that few prompts suffice.

---

## The whole thing, as a checklist

Each step is a subcommand, and every option also reads the environment variable the original
scripts used (`SRC DST DIRS K MODE KEEP ACTS OUT PROMPTS MODEL DEV DEVMAP MAXMEM BS SYS THINK`), so
existing invocations keep working.

1. **`targets <checkpoint>`** → the target list. Read the printed per-pattern counts and confirm the
   routed-expert stacks are there if the model is MoE.
2. **`capture --model --prompts --out [--sys --bs --think]`** → activations. Record the chat-template
   fingerprint and pin `--bs`.
3. **`direction --acts --out --mode sep --keep 0`** → `{d, w, meta}`. Check the printed post-ortho
   `|cos|` is ~0 and that the weight range is sane.
4. **`patch --src --dst --dirs -K`** → edited checkpoint (`--dst` must already be a copy of
   `--src`). Record the printed `max|dW|`; it is the dose coordinate you will compare on, and it is
   linear in `K`.

Three dose rungs (`K = 1.0 / 1.25 / 1.5`) is the whole search. There is no Optuna loop, no reference
model, and no gradient anywhere in the pipeline.
