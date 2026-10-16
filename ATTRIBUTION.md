# ATTRIBUTION

This method is an **assembly**. Several of its load-bearing parts are other people's work, and the
repo credits them at the point of use as well as here. If you cite this project, cite the work below
alongside it.

## What is borrowed

| component | whose | reference |
|---|---|---|
| The foundational method — difference-of-means refusal direction plus a rank-1 weight edit | **Arditi et al. 2024** | *Refusal in Language Models Is Mediated by a Single Direction*, [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) |
| `orthogonalize_direction` | **Heretic** — Philipp Emanuel Weidmann and contributors | [`github.com/p-e-w/heretic`](https://github.com/p-e-w/heretic), `main.py:459-469` |
| The rank-1 **PRE row-normalised** weight update rule | **Heretic** | same repo, the `RowNormalization.PRE` branch of `Model.abliterate` in `model.py` |
| `KeywordRate` — the 33-marker keyword refusal metric, its marker list and its normalisation rules | **Heretic** | same repo, `src/heretic/scorers/keyword_rate.py` → reimplemented in `eval/keyword_rate.py` |
| `KLDivergence` — first-token KL against a baseline model | **Heretic** | same repo, `src/heretic/scorers/kl_divergence.py` and `Model.get_logits` in `src/heretic/model.py` → reproduced in `eval/kl.py` |

### `orthogonalize_direction` is Heretic's, not ours

`src/resgate/direction.py` contains a function of that name. It is **Heretic's design**, transcribed:
given a candidate direction `d` and the **unit-normalised** mean harmless residual `ā_l`, it removes
the component of `d` along that centroid and renormalises.

```
unit(d − ⟨d, unit(ā_l)⟩ · unit(ā_l))
```

Source: [`github.com/p-e-w/heretic`](https://github.com/p-e-w/heretic), `main.py:459-469`.

**On the line numbers.** Line numbers in another project's source drift between releases.
`main.py:459-469` is the range recorded when this method was built against Heretic; in
`v1.4.0-25-g3521f86` the same block sits at `main.py:616-628`. Treat the **function name** as the
stable identifier and the line range as a hint.

### The update rule is Heretic's, and we verified that by reproducing their weights

The rank-1 PRE row-normalised edit in `src/resgate/patch.py` —

```
n = rownorm(W);  Ŵ = W / n
dW = − w · outer(n ⊙ v, v @ Ŵ)          with  w = K · sep_l
```

— is **Heretic's update rule** (`RowNormalization.PRE`), not an independent invention. We did not
take this on inspection: we ran our implementation and **reproduced Heretic's published weights to
cos 0.99985**. That number is the evidence that the rule is theirs *and* that our implementation of
it is faithful. It is not evidence that anything else in this repo is theirs.

### The evaluation metrics in `eval/` are two-thirds Heretic's

`eval/keyword_rate.py` and `eval/kl.py` exist so that numbers produced here are comparable with
Heretic's published ones. Both are Heretic's metrics. `eval/keyword_rate.py` in particular copies
their 33-marker list and matching rules verbatim, because an approximation would not be comparable —
it was transcribed from a local clone at HEAD `3521f86` and then validated against the keyword-match
flags Heretic's own code produced in this project's stored runs: **1200 comparisons, 0
disagreements**, with both outcomes well represented (pristine arms 100/100 matches, edited arms
13–29/100).

Only `eval/judge_threeway.py` — the three-way REFUSED / DEFLECTED / COMPLIED judge — is ours.
Heretic does not report anything like it.

## What is ours

| component | where |
|---|---|
| The `sep(l)` per-layer weighting: `sep(l) = ‖h_l − ā_l‖ / mean_i‖g_i,l − ā_l‖`, normalised to mean 1, with `w = K · sep_l` | `src/resgate/direction.py`; `docs/METHOD.md` step 3 |
| The blind 5-prompt kit — 1 harmful + 4 harmless, no reference model and no held-out split | `kits/minimal.json`, `kits/README.md` |
| Structural target enumeration applied across dense **and** MoE checkpoints from safetensors headers alone (Llama-3.1-8B 64, Qwen3.6-35B-A3B 120, Gemma-4-26B-A4B 90) | `src/resgate/enumerate_targets.py` |
| The three-way REFUSED / DEFLECTED / COMPLIED evaluation | `eval/judge_threeway.py`, `docs/PITFALLS.md` §1 |
| The dose and coverage findings — the ladder, `max\|dW\|` linearity in `K`, the plateau above K=1.5 on Gemma, the kit comparison at full vs partial dose | `docs/RESULTS.md` |

Two honest qualifications on that list:

- **Structural target enumeration is ours but not uniquely ours.** The same rule is stated
  independently in [arXiv:2609.09793](https://arxiv.org/abs/2609.09793).
- **Small fitting kits are not novel and this repo does not claim them.** Single-contrast-pair
  steering: **ActAdd**, Turner et al. [arXiv:2308.10248](https://arxiv.org/abs/2308.10248).
  One-example refusal suppression: **Dunefsky & Cohan**, COLM 2025,
  [arXiv:2502.18862](https://arxiv.org/abs/2502.18862) — which also reports the lottery effect
  ("choice of the single example is crucial"). Saturation in `n`: Hao et al.
  [arXiv:2505.03189](https://arxiv.org/abs/2505.03189) (knee ≈ 80), Braun
  [arXiv:2602.17881](https://arxiv.org/abs/2602.17881) (convergence ≈ 150). What is ours in this
  area is the **specific** blind 5-prompt kit and the `sep(l)` weighting it feeds, not the idea that
  few prompts suffice.

## Compared against, not built on

- **RepIt** — [arXiv:2509.13281](https://arxiv.org/abs/2509.13281),
  [`github.com/wang-research-lab/RepIt`](https://github.com/wang-research-lab/RepIt). This repo
  compares against it; it does not build on it, and none of its code or data is vendored here.
- **Published abliterated checkpoints** (huihui-ai; SevenOfNine, which is Heretic-made) appear in
  `docs/RESULTS.md` as comparison arms, evaluated as shipped. No weights of any kind are
  distributed by this repo.

## Licensing

> **One licence fact in this project's own brief is wrong, and this is the corrected version.**
> **Heretic is AGPL-3.0-or-later, not Apache-2.0.** Verified locally against
> `v1.4.0-25-g3521f86`: its `LICENSE` file is the GNU Affero General Public License v3 text, and its
> `pyproject.toml` declares `license = "AGPL-3.0-or-later"` with the matching AGPLv3+ trove
> classifier. RepIt *is* Apache-2.0 (quoted from its repo).

**This repo is licensed AGPL-3.0-or-later (`LICENSE`), copyright "resgate contributors" — the same
licence as Heretic, chosen deliberately to match it.**

- **`eval/keyword_rate.py` and `eval/kl.py` are derivatives of AGPL-3.0-or-later code.** Their own
  file headers say so. The marker list and matching rules in `keyword_rate.py` are *copied*,
  deliberately, because reproducing the metric exactly is the point. Matching Heretic's licence is
  what makes shipping them straightforward.
- `src/resgate/direction.py` and `src/resgate/patch.py` reimplement two Heretic *components* — a
  function and an update rule — with credit, rather than vendoring Heretic's code. Whether that
  would have counted as sufficient separation under a permissive licence was never settled; under
  AGPL-3.0-or-later the question does not arise.

## Model weights and model licences

This repo distributes **no model weights**. Every checkpoint named anywhere in it is fetched by you,
from its own publisher, under its own licence — and you are responsible for complying with that
licence when you modify a model with these tools. **No Meta-published asset is shipped here**, on
licensing grounds. Separately, one comparison in the underlying research was withheld from
`docs/RESULTS.md` because it involves a model excluded on those grounds — see `docs/RESULTS.md`
§8 **Bounds**.
