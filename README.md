# ResGate

**Closed-form pre-band runtime gating and search-free rank-1 refusal editing.**

The repository is organized into two parallel tracks (`gate/` and `ablation/`) across `src/resgate/`,
`docs/`, and `kits/`:

| Track | Code (`src/resgate/`) | Docs (`docs/`) | Shipped Kits (`kits/`) |
|---|---|---|---|
| **1. Runtime Gate (`gate/`)** — route-invariant pre-band hook & closed-form `M1_ridge_lam16` (`M1 + M3`) classifier | [`src/resgate/gate/`](src/resgate/gate/) (`__init__.py`, `fit.py`, `serve.py`) | [`docs/gate/CLASSIFIER-MATH.md`](docs/gate/CLASSIFIER-MATH.md), [`docs/gate/GATE.md`](docs/gate/GATE.md) | [`kits/gate/gemma4-26B/`](kits/gate/gemma4-26B/README.md), [`kits/gate/phi4mini/`](kits/gate/phi4mini/README.md), [`kits/gate/llama31-8B/`](kits/gate/llama31-8B/README.md) |
| **2. Rank-1 Ablation (`ablation/`)** — unconditional `sep(l)` refusal direction, band localisation & weight patcher | [`src/resgate/ablation/`](src/resgate/ablation/) (`enumerate_targets.py`, `capture.py`, `direction.py`, `patch.py`) | [`docs/ablation/METHOD.md`](docs/ablation/METHOD.md), [`docs/ablation/RESULTS.md`](docs/ablation/RESULTS.md), [`docs/ablation/PITFALLS.md`](docs/ablation/PITFALLS.md) | [`kits/ablation/minimal.json`](kits/ablation/minimal.json), `kits/ablation/{gemma4-26B,phi4mini,llama31-8B}/` |

*(Top-level module imports `resgate.gate`, `resgate.serve`, `resgate.patch`, `resgate.direction`,
`resgate.capture`, and `resgate.enumerate_targets` remain aliased so existing scripts and tests
run unchanged.)*

- **Gate math & live GPU tables:** [`docs/gate/CLASSIFIER-MATH.md`](docs/gate/CLASSIFIER-MATH.md)
- **Runtime hook & MoE reconstruction spec:** [`docs/gate/GATE.md`](docs/gate/GATE.md)
- **Rank-1 edit & `sep(l)` band method:** [`docs/ablation/METHOD.md`](docs/ablation/METHOD.md)
- **Unconditional ablation results:** [`docs/ablation/RESULTS.md`](docs/ablation/RESULTS.md)
- **Nine measurement pitfalls:** [`docs/ablation/PITFALLS.md`](docs/ablation/PITFALLS.md)
- **Evaluation instruments:** [`eval/README.md`](eval/README.md)
- **Attribution:** [`ATTRIBUTION.md`](ATTRIBUTION.md)

---

## Part I — Runtime Gate (`resgate.gate`, `M1_ridge_lam16`)

`src/resgate/gate/` applies the rank-1 `PRE` edit as a runtime forward hook within a contiguous
mid-network band (`L10–16` on Gemma-4-26B-A4B-it, `L13–15` on Phi-4-mini-instruct, `L11–22` on
Llama-3.1-8B-Instruct), conditioned on a single linear gate read at an earlier layer (`L8`, `L11`,
`L10`). Because the read layer sits strictly below the edit band (`max(gate_layers) < min(edit_layers)`),
a single forward pass serves the conditional edit with exact byte identity on withheld prompts.

The closed-form gate direction `c` and leak-constrained threshold `t` (`fit_gate_direction`,
`thresh_r2`, `thresh_r3`, `fit_gate_op` in [`src/resgate/gate/fit.py`](src/resgate/gate/fit.py)) use
subcategory-centred within-class scatter (`Z_subcat`) and smooth Tikhonov regularisation at
`alpha = lambda_16` (`M1_ridge_lam16`).

| Gate \ Prompts (Gemma L8 / Phi L11 / Llama L10, **LIVE**) | Cyber (`675 CySecBench`) | Bio (`637 WMDP_BIO test`) | Non-target harmful (`CatQA TEST`) |
|---|---|---|---|
| **Cyber gate (`M1_ridge_lam16`)** | `526 / 549 / 551` = **77.9 / 81.3 / 81.6%** (`528 / 549 / 550` offline) | `5 / 2 / 33` = 0.78 / 0.31 / 5.18% | `16 / 22 / 17` = 6.8 / 9.4 / 7.3% (`16 / 21 / 17` offline) |
| **Bio gate (`B2_M1_ridge_lam16`)** | `3 / 0 / 2` = 0.44 / 0.00 / 0.30% | `633 / 630 / 632` = **99.4 / 98.9 / 99.2%** (`629 = 98.7%` at Phi `L12`) | `15 / 8 / 10` = 6.4 / 3.4 / 4.3% (`11 = 4.7%` at Phi `L12`) |

- **Held-out clean validation (`WMDP_BIO val`, 127 rows / 118 deduplicated):** **127 / 127 / 127 = 100.0%** (`95% CP: [97.1, 100.0]`) live across all three models.
- **294-prompt refusal agreement:** **294 of 294** agreement between the gate bit and the judged outcome on refused prompts (`115` fired & answered, `179` held & refused, `0` discordant), with **100%** `sha256` byte identity on withheld prompts.

---

## Part II — Rank-1 Ablation Pipeline (`resgate.ablation`)

### Install

```bash
git clone <this repo> && cd resgate
pip install -e .                 # core: torch, transformers, safetensors
pip install -e '.[eval]'         # + requests, for the three-way judge in eval/
pip install -e '.[shard]'        # + accelerate, if you need --devmap auto
```

Python >= 3.10. One console entry point, `resgate`, with four subcommands (`targets`, `capture`,
`direction`, `patch`).

### Quickstart (Unconditional Checkpoint Patching)

```bash
export MODEL=google/gemma-4-26B-A4B-it
export SRC=./gemma-pristine
export DST=./gemma-ablated
mkdir -p out

# 1. Enumerate target tensors from checkpoint headers (90 for Gemma-4-26B-A4B-it).
resgate targets "$SRC"

# 2. Capture residual stream at the last prompt token across all layers.
resgate capture --model "$MODEL" --prompts kits/ablation/minimal.json --out out/acts.pt

# 3. Compute per-layer directions + sep(l) weights.
resgate direction --acts out/acts.pt --out out/dir.pt --mode sep --keep 0

# 4. Apply the rank-1 PRE weight edit in place on a copy of SRC.
cp -r "$SRC" "$DST"
resgate patch --src "$SRC" --dst "$DST" --dirs out/dir.pt -K 1.5
```

See [`docs/ablation/METHOD.md`](docs/ablation/METHOD.md), [`docs/ablation/RESULTS.md`](docs/ablation/RESULTS.md),
and [`docs/ablation/PITFALLS.md`](docs/ablation/PITFALLS.md) for full details on `K` selection, kit
choice, and evaluation metrics.

---

## Intended use & Attribution

Intended for **safety research, red-teaming and interpretability work on open-weight models.** No
model weights are distributed by this repository; code is licensed under **AGPL-3.0-or-later**
([`LICENSE`](LICENSE)).

- **Foundational single-direction formulation:** Arditi et al. 2024, *Refusal in Language Models Is
  Mediated by a Single Direction*, [arXiv:2406.11717](https://arxiv.org/abs/2406.11717).
- **Orthogonalisation and rank-1 PRE row-normalised update rule:** follows Heretic
  ([`github.com/p-e-w/heretic`](https://github.com/p-e-w/heretic), `main.py:459-469`), verified to a
  weight cosine of **0.99985**.
- **This repository:** pre-band route-invariant runtime gating (`resgate.gate`), closed-form
  subcategory-centred Tikhonov ridge gate construction (`M1_ridge_lam16` in `resgate.gate.fit`),
  `sep(l)` per-layer band localisation and weighting (`resgate.ablation`), structural target
  enumeration across dense and fused 3-D MoE stacks, and live end-to-end verification.

See [`ATTRIBUTION.md`](ATTRIBUTION.md) and [`CITATION.cff`](CITATION.cff) for full citation details.
