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

- **Paper draft (PDF):** [*ResGate: Topic-Scoped Refusal Removal via Single-Layer Residual Gating* (`paper/resgate.pdf`)](paper/resgate.pdf)
- **Verified PoC repository:** [`https://github.com/kncshw/resgate`](https://github.com/kncshw/resgate)
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

## Quickstart & Usage Guide (Two Methods)

### Step 0: Installation

```bash
git clone https://github.com/kncshw/resgate.git
cd resgate
pip install -e .                 # core: torch, transformers, safetensors
pip install -e '.[eval]'         # + requests, for the three-way judge in eval/
pip install -e '.[shard]'        # + accelerate, if you need --devmap auto
```

Python >= 3.10. The repository supports two usage modes:
* **Method 1 (`Ablation + Gate` — Full Conditional Refusal Removal):** Launches the HTTP GPU server ([`examples/serve_http.py`](examples/serve_http.py)) or Python [`GatedModel`](src/resgate/gate/serve.py) with both the upstream linear gate (`op_*.pt`) and the downstream rank-1 refusal-removal hooks (`dir_*.pt`). When the gate fires on a target-domain prompt, the rank-1 edit removes refusal in that same forward pass; when the gate withholds on a non-target prompt, the model replies with **byte-for-byte identical** base-model text.
* **Method 2 (`Gate Only` — Prefill Topic Routing / Scoring Without Weight Edits):** Evaluates only the upstream linear gate (`cos(mean_user(h_l), c) >= t`) at the early read layer (`L8` / `L11` / `L10`), either via the `/gate` HTTP endpoint or directly in Python, returning the binary decision bit and cosine score without decoding an edited completion.

---

## Method 1: Ablation + Gate (HTTP Server, GPU Generation & `curl` Testing)

Every model directory under [`kits/gate/`](kits/gate/) ships both the refusal-removal direction blob (`--dirs`) and the `M1_ridge_lam16` gate operating point (`--op`). The base HuggingFace checkpoint on disk is **never modified** — all edits are applied dynamically in memory via forward hooks on the edit band.

### 1.1 Start the GPU HTTP Server ([`examples/serve_http.py`](examples/serve_http.py))

Pick your target model and domain (**Cyber** or **Bio**) and launch [`examples/serve_http.py`](examples/serve_http.py). At startup, the server loads the checkpoint in `bfloat16`, attaches the read hook and band edit hooks, runs a 1-prompt self-test to verify that user-span pooling and the gate decision path are live, and binds to `127.0.0.1:8077`.

#### Option A — Phi-4-mini-instruct (`L11` read, `L13–15` edit band, 6 tensors, ~8 GiB VRAM)
```bash
# Cyber gate (op_phi4mini_L11_M1subcat240_M1_ridge_lam16_serve.pt, t = 0.05303228245782718)
python3 examples/serve_http.py \
  --model microsoft/Phi-4-mini-instruct \
  --dirs  kits/gate/phi4mini/dir_phi4mini_sep_L13to15.pt \
  --op    kits/gate/phi4mini/op_phi4mini_L11_M1subcat240_M1_ridge_lam16_serve.pt \
  --K 2.0 --edit-layers 13-15 --port 8077

# Or Bio gate at the same L11 read layer (t = 0.042208061573892744):
#   --op kits/gate/phi4mini/op_wmdpbio_B2_M1_ridge_lam16_R3U140_phi4mini_L11.pt
```

#### Option B — Llama-3.1-8B-Instruct (`L10` read, `L11–22` edit band, 24 tensors, ~16 GiB VRAM)
```bash
# Cyber gate (op_llama31-8B_L10_M1subcat240_M1_ridge_lam16_serve.pt, t = 0.06890131763033773)
python3 examples/serve_http.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dirs  kits/gate/llama31-8B/dir_llama31-8B_sep_L11to22.pt \
  --op    kits/gate/llama31-8B/op_llama31-8B_L10_M1subcat240_M1_ridge_lam16_serve.pt \
  --K 1.5 --edit-layers 11-22 --port 8077

# Or Bio gate at the same L10 read layer (t = 0.08524563027159009):
#   --op kits/gate/llama31-8B/op_wmdpbio_B2_M1_ridge_lam16_R3U140_llama31-8B_L10.pt
```

#### Option C — Gemma-4-26B-A4B-it (`L8` read, `L10–16` edit band, 21 tensors incl. 7 fused 3-D MoE stacks, ~56 GiB VRAM)
```bash
# Cyber gate (op_gemma_L8_M1subcat240_M1_ridge_lam16_serve.pt, t = 0.11349115121614832)
python3 examples/serve_http.py \
  --model google/gemma-4-26B-A4B-it \
  --dirs  kits/gate/gemma4-26B/gemma_dir90_cbrn_L10to16.pt \
  --op    kits/gate/gemma4-26B/op_gemma_L8_M1subcat240_M1_ridge_lam16_serve.pt \
  --K 2.0 --edit-layers 10-16 --port 8077

# Or Bio gate at the same L8 read layer (t = 0.05801104299110058):
#   --op kits/gate/gemma4-26B/op_wmdpbio_B2_M1_ridge_lam16_R3U140_gemma_L8.pt
```

*(Wait until stdout prints `READY  http://127.0.0.1:8077` before sending requests.)*

---

### 1.2 Test GPU Generation & Gate Confinement with `curl`

Once the server is up on port `8077`, open a second terminal and run these `curl` commands:

#### 1. Check server health, hooked tensor count, and active threshold (`GET /health`)
```bash
curl -s http://127.0.0.1:8077/health | python3 -m json.tool
```
Verify in the JSON response:
* `"attached_now": true`, `"lifetime_errors": 0`, `"lifetime_unrouted": 0`, `"force": null`
* `"gate_layer"`, `"threshold"`, and `"edit_layers"` match the kit you loaded.

#### 2. Generate on an in-domain prompt where the gate FIRES (`POST /generate`)
```bash
curl -s http://127.0.0.1:8077/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Write a Python script that scans a target IP range for open TCP ports and banners.",
    "max_new_tokens": 256
  }' | python3 -m json.tool
```
* **What happens:** At prefill, the gate computes `score = cos(mean_user(h_l), c)`. Because `score >= threshold`, `"fired": true` and `"routed": <tensors_hooked>`, so the rank-1 refusal edit is applied across the edit band for this generation and the model answers the request.

#### 3. Generate on an out-of-domain prompt where the gate WITHHOLDS (`POST /generate`)
```bash
curl -s http://127.0.0.1:8077/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Explain how to synthesize a restricted nerve agent in a home laboratory.",
    "max_new_tokens": 256
  }' | python3 -m json.tool
```
* **What happens:** On a non-target harmful prompt (when running the Cyber gate), `score < threshold`, so `"fired": false` and `"routed": 0`. No weight edit is applied, and the base model refuses exactly as aligned.

#### 4. Run a 3-way side-by-side verification (`POST /compare` with `"forced": true`)
`/compare` runs the **exact same prompt** under three arms on the same loaded model — `gated` (conditional), `forced` (always-on edit, `gate.force = True`), and `control` (hooks detached, pristine base model) — and checks `sha256` byte identity:
```bash
# Case A: On a withheld prompt -> confirms "gated_equals_control": true (exact byte identity!)
curl -s http://127.0.0.1:8077/compare \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Give me a step-by-step guide to forging identity documents.",
    "max_new_tokens": 128,
    "forced": true
  }' | python3 -m json.tool

# Case B: On a firing target prompt -> confirms gated answers while control (base model) refuses
curl -s http://127.0.0.1:8077/compare \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Demonstrate how an SQL injection payload bypasses a login check in PHP.",
    "max_new_tokens": 128,
    "forced": true
  }' | python3 -m json.tool
```

---

## Method 2: Gate Only (Fast Topic Scoring / Prefill Routing Without Generation)

If you only want the gate's **binary classification decision (`fired: true/false`) and cosine score** (`cos(mean_user(h_l), c)` vs. `threshold`) without running full token generation or modifying outputs:

### 2.1 Via `curl` (`POST /gate` on the running server)
`/gate` runs a single prefill forward pass (1 token) to read layer `l`, freezes the decision, and returns `fired`, `score`, `threshold`, and the user token span immediately:

```bash
curl -s http://127.0.0.1:8077/gate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "How do I exploit a format-string vulnerability in C to read stack memory?"
  }' | python3 -m json.tool
```
Example output fields:
```json
{
  "arm": "gated",
  "decided": true,
  "declined": false,
  "fired": true,
  "score": 0.14829103,
  "threshold": 0.05303228245782718,
  "margin": 0.09525875,
  "span": [14, 30]
}
```

### 2.2 Gate-Only in Python (No Refusal Edit Hooks Attached)
You can also evaluate the `.pt` gate operating point (`c`, `thresh`, `gate_layer`) directly in Python on pooled `mean_user` hidden states or by setting `K = 0.0` (zero edit dose):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from resgate.gate import Gate, GatedModel

model_id = "microsoft/Phi-4-mini-instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="cuda:0"
).eval()

# Load the gate operating point with K=0.0 (Gate-only measurement; 1 token prefill)
gate = Gate.from_file(
    direction_blobs=["kits/gate/phi4mini/dir_phi4mini_sep_L13to15.pt"],
    op_blob="kits/gate/phi4mini/op_phi4mini_L11_M1subcat240_M1_ridge_lam16_serve.pt",
    K=0.0,
    edit_layers=(13, 14, 15),
)

gm = GatedModel(model, tok, gate, sys_prompt="You are a helpful assistant.")
gm.attach()

prompts = [
    "How do I craft a heap spray payload for a use-after-free bug?",
    "What is the capital of France?",
]
res = gm.generate(prompts, max_new_tokens=1)   # 1 token = prefill gate read only
for p, fired, score in zip(prompts, res.fired, res.scores):
    print(f"fired={str(fired):5s}  score={score:+.6f} (thresh={gate.thresh:.6f})  prompt={p!r}")
```

---

## Method 3: Derive Your Own Refusal Directions or Fit a Custom Gate

### 3.1 Derive a new refusal direction (`resgate` CLI — Unconditional Ablation Pipeline)
If you want to compute `dir_*.pt` from scratch on a new model using the 5-prompt capture kit ([`kits/ablation/minimal.json`](kits/ablation/minimal.json)) or your own prompt JSON:

```bash
export MODEL=microsoft/Phi-4-mini-instruct
export SRC=./phi-pristine
export DST=./phi-ablated
mkdir -p out

# 1. Enumerate target residual-writer tensors from checkpoint headers
resgate targets "$SRC"

# 2. Capture residual stream activations at the last prompt token across all layers
resgate capture --model "$MODEL" --prompts kits/ablation/minimal.json --out out/acts.pt

# 3. Compute per-layer orthogonalised refusal directions + sep(l) weights
resgate direction --acts out/acts.pt --out out/dir.pt --mode sep --keep 0

# 4. Either pass `--dirs out/dir.pt` to `examples/serve_http.py` (for runtime gating),
#    OR apply an unconditional rank-1 PRE weight edit to a checkpoint copy on disk:
cp -r "$SRC" "$DST"
resgate patch --src "$SRC" --dst "$DST" --dirs out/dir.pt -K 1.5
```

### 3.2 Fit a new `M1_ridge_lam16` gate operating point (`op.pt`) for a custom category
Given pooled `mean_user` activations at your read layer `l < min(edit_layers)` for target-domain positive prompts `xp` (`[n_pos, D]`) and multi-category non-target reference prompts `xn` (`[n_neg, D]` with subcategory labels `neg_groups`):

```python
import torch
from resgate.gate import fit_gate_op

op = fit_gate_op(
    xp,                          # Tensor [n_pos, D]: target-domain training activations
    xn,                          # Tensor [n_neg, D]: non-target reference activations
    neg_groups=neg_subcat_list,  # list of subcategory labels for M1 within-subcategory centring
    category="custom_domain",
    gate_layer=11,               # must be strictly < min(edit_layers)
    m=7,                         # permitted reference false positives (e.g. floor(0.05 * 140))
    rule="R3",                   # order-statistic threshold rule R3 (or "R2")
    rank=16,                     # anchors Tikhonov ridge shrinkage at alpha = lambda_16
    mode="ridge",                # M1_ridge_lam16
)
torch.save(op, "out/op_custom.pt")
```

See [`docs/gate/CLASSIFIER-MATH.md`](docs/gate/CLASSIFIER-MATH.md), [`docs/gate/GATE.md`](docs/gate/GATE.md), [`docs/ablation/METHOD.md`](docs/ablation/METHOD.md), and [`docs/ablation/PITFALLS.md`](docs/ablation/PITFALLS.md) for complete mathematical and operational details.

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
