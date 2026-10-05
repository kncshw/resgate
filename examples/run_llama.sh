#!/usr/bin/env bash
#
# examples/run_llama.sh — the full four-step flow on Llama-3.1-8B-Instruct, a
# DENSE 64-target model, at the dose this repo recommends for that class:
# K = 1.25.
#
# This script is meant to be READ as much as run.
#
#   Usage:  MODEL=... SRC=... DST=... ./examples/run_llama.sh
#
# ---------------------------------------------------------------------------
# TWO HONEST WARNINGS, BEFORE THE CODE
# ---------------------------------------------------------------------------
# 1. NO BEHAVIOURAL RESULTS FOR LLAMA ARE PUBLISHED IN THIS REPO. Llama-3.1-8B
#    appears in docs/RESULTS.md only as an enumeration count (64 targets) and
#    the dose used. Every refusal/KL/three-way number in this repo is Gemma.
#    K=1.25 here is the dose that was used, not a dose validated by a published
#    ladder on this model. Run your own ladder (docs/PITFALLS.md).
#
# 2. NO META-PUBLISHED ASSET IS SHIPPED HERE, on licensing grounds. You fetch
#    the checkpoint yourself, under Meta's licence, and you are responsible for
#    complying with it — including its terms on modified versions.
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — override any of these from the environment.
# ---------------------------------------------------------------------------

# Model id for CAPTURE (step 2): hub id or local directory.
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

# Pristine checkpoint on disk, read by steps 1 and 4. Needs config.json and
# model.safetensors.index.json; the enumerator reads headers only.
SRC="${SRC:-./Llama-3.1-8B-Instruct}"

# Destination. Created below as a copy of $SRC.
DST="${DST:-./Llama-3.1-8B-Instruct-ablated}"

# The 5-prompt kit: 1 harmful + 4 harmless. See kits/README.md.
PROMPTS="${PROMPTS:-kits/minimal.json}"

# Scratch for the two intermediate blobs.
OUTDIR="${OUTDIR:-out/llama}"

# Dose. K = 1.25 is the recommended starting point for Llama-class 64-target
# models — and the dose used in the research behind this repo.
#
# Fewer targets is not a reason to raise K: K scales the per-layer weight
# w = K * sep_l, and the ceiling is per-layer, not per-model. The component
# along the direction is multiplied by (1 - w), so w=1 removes it, w=1.5 brings
# half of it back SIGN-FLIPPED, and w=2 is a pure reflection at full original
# magnitude. See docs/METHOD.md.
K="${K:-1.25}"

# Capture batch size. Pin it: it is part of the measurement.
BS="${BS:-8}"

# System prompt for the chat template. Capture under what you will serve under.
SYS="${SYS:-You are a helpful assistant.}"

mkdir -p "$OUTDIR"

# ---------------------------------------------------------------------------
# Step 1 — targets: derive the edit set from the checkpoint.
# ---------------------------------------------------------------------------
# Same two mechanical filters as on an MoE model, no architecture table:
# hidden_size in the OUTPUT position, and an output-projection name.
#
# LOOK AT: the printed TOTAL — 64 for Llama-3.1-8B-Instruct (32 layers x
# {o_proj, down_proj}), all 2-D, no expert stacks because the model is dense.
#
# Note the "rejected by role" count in the same line. Shape alone is not enough:
# Llama's q_proj is [4096, 4096] and would pass the shape filter, but it READS
# from the residual stream rather than writing to it. That is what the role
# filter is for.
echo "=== 1/4  targets  ($SRC) ==============================================="
tempx targets "$SRC"

# ---------------------------------------------------------------------------
# Step 2 — capture: residual stream at the last prompt token, every layer.
# ---------------------------------------------------------------------------
# An 8B model in bf16 fits comfortably on one 24GB device, so no sharding flags
# are needed here (unlike the Gemma example). If you do need them:
#     --devmap auto   (needs the [shard] extra: accelerate)
#
# LOOK AT: "32 layers, hidden 4096" and the group counts
# ({'harmful': 1, 'harmless': 4}).
echo "=== 2/4  capture  ($MODEL) ============================================="
tempx capture \
  --model   "$MODEL" \
  --prompts "$PROMPTS" \
  --out     "$OUTDIR/acts.pt" \
  --sys     "$SYS" \
  --bs      "$BS"

# Llama templates have no thinking mode, so --think is not used here. On a model
# that does (Qwen3.5/3.6), enable_thinking MUST match how you will serve: pass
# --think 0 to capture with it closed. Capturing in one mode and serving in the
# other measures a different computational state.

# ---------------------------------------------------------------------------
# Step 3 — direction: one direction per layer, plus the sep(l) weights.
# ---------------------------------------------------------------------------
#   d_l    = mean(harmful) - mean(harmless), then orthogonalised against the
#            unit-normalised harmless centroid  [Heretic's orthogonalize_direction]
#   sep(l) = gap / harmless-spread, normalised to mean 1   [ours]
#
# MODE=sep KEEP=0 are the canonical settings.
#
# LOOK AT TWO THINGS:
#   1. "after ortho max" must be ~0 (e-08 or smaller).
#   2. The sep(l) profile should show a band, not a flat line. Flat means sep is
#      carrying no information and you are effectively on --mode uniform.
echo "=== 3/4  direction ====================================================="
tempx direction \
  --acts "$OUTDIR/acts.pt" \
  --out  "$OUTDIR/dir.pt" \
  --mode sep \
  --keep 0

# ---------------------------------------------------------------------------
# Step 4 — patch: the rank-1 PRE row-normalised edit (Heretic's update rule).
# ---------------------------------------------------------------------------
#     n = rownorm(W);  dW = -w * outer(n (*) v, v @ (W/n))      with w = K * sep_l
#
# Written bf16 IN PLACE into $DST, read from $SRC. $DST must already be a full
# copy of $SRC.
if [ ! -d "$DST" ]; then
  echo "--- copying $SRC -> $DST (the patch edits DST in place) ---"
  cp -r "$SRC" "$DST"
fi

# LOOK AT: the printed max|dW| — the dose coordinate to compare on, linear in K.
# The published max|dW| figures in docs/RESULTS.md are GEMMA's; do not expect
# this run to reproduce them. Record your own and dose-match against it.
echo "=== 4/4  patch  (K=$K) ================================================="
tempx patch \
  --src  "$SRC" \
  --dst  "$DST" \
  --dirs "$OUTDIR/dir.pt" \
  -K     "$K"

echo
echo "Patched checkpoint: $DST"
echo
# ---------------------------------------------------------------------------
# Next: EVALUATE — and since nothing about Llama's behaviour is established in
# this repo, your own ladder (K = 1.0 / 1.25 / 1.5) is the measurement, not a
# confirmation. Generate responses once, score the SAME strings three ways.
# ---------------------------------------------------------------------------
cat <<EOF
Next: evaluate. A refusal count is not an ablation score — eval/README.md.
  python eval/keyword_rate.py responses.json --attribute
  python eval/kl.py --baseline "$SRC" --current "$DST" --batch-size $BS --out kl.json
  python eval/judge_threeway.py responses.json judged.json   # needs JUDGE_BASE_URL, JUDGE_MODEL

No Llama behavioural baseline is published here. Run K = 1.0 / 1.25 / 1.5 and
compare on max|dW|, not on K.
EOF
