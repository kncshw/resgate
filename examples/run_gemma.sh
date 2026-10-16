#!/usr/bin/env bash
#
# examples/run_gemma.sh — the full four-step flow on Gemma-4-26B-A4B-it, at the
# dose this repo recommends for 90-target Gemma-class models: K = 1.5.
#
# This script is meant to be READ as much as run. Each step says what it does,
# what to look at in its output, and what a wrong-looking number means.
#
#   Usage:  MODEL=... SRC=... DST=... ./examples/run_gemma.sh
#
# Nothing here downloads or distributes weights. $SRC is a checkpoint you
# already have, fetched from its publisher under its own licence.
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — override any of these from the environment.
# ---------------------------------------------------------------------------

# The model id used for CAPTURE (step 2). May be a hub id or a local directory.
MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"

# The pristine checkpoint on disk, read by steps 1 and 4. Must contain
# config.json and model.safetensors.index.json — the enumerator reads the
# safetensors headers directly and never loads the model.
SRC="${SRC:-./gemma-4-26B-A4B-it}"

# Where the patched checkpoint goes. Created below as a copy of $SRC.
DST="${DST:-./gemma-4-26B-A4B-it-ablated}"

# The 5-prompt kit: 1 harmful + 4 harmless. See kits/README.md.
PROMPTS="${PROMPTS:-kits/ablation/minimal.json}"

# Scratch for the two intermediate blobs.
OUTDIR="${OUTDIR:-out/gemma}"

# Dose. K = 1.5 is the recommended starting point for Gemma-class 90-target
# models. On this model K=1.5 and K=2.0 were IDENTICAL on every axis we
# measured — there is a broad plateau above 1.5, not a further rung.
#
# K is not a "more is better" knob. The applied per-layer weight is
# w = K * sep_l, and the component along the direction is multiplied by (1 - w):
# w=1 removes it, w=1.5 brings half of it back SIGN-FLIPPED, w=2 is a pure
# reflection at full original magnitude, beyond that it amplifies. Since sep_l
# is normalised to mean 1 but varies per layer, K=1.5 already puts the
# highest-sep layers near w=2. See docs/METHOD.md.
K="${K:-1.5}"

# Capture batch size. PIN THIS. On MoE models, temperature-0 output is not
# reproducible across batch sizes (docs/PITFALLS.md §4), so changing it changes
# your numbers.
BS="${BS:-8}"

# The system prompt used when rendering the chat template for capture. It is
# part of the measurement: capture under the prompt you will SERVE under.
SYS="${SYS:-You are a helpful assistant.}"

mkdir -p "$OUTDIR"

# ---------------------------------------------------------------------------
# Step 1 — targets: derive the edit set from the checkpoint.
# ---------------------------------------------------------------------------
# A tensor qualifies when hidden_size sits in its OUTPUT position (shape[0] for
# 2-D, shape[1] for 3-D fused routed-expert stacks) AND its name contains an
# output-projection token. Nothing about Gemma is hard-coded.
#
# LOOK AT: the printed TOTAL. Gemma-4-26B-A4B-it gives 90. And because this is
# an MoE checkpoint, confirm the ROUTED-EXPERT stacks are in the list (the 3-D
# shapes). Leaving the experts unedited is exactly the coverage failure behind
# the published build that deflects 77/100 — see docs/RESULTS.md §5.
echo "=== 1/4  targets  ($SRC) ==============================================="
resgate targets "$SRC"

# ---------------------------------------------------------------------------
# Step 2 — capture: residual stream at the last prompt token, every layer.
# ---------------------------------------------------------------------------
# Left-padded, so [:, -1, :] is the last REAL prompt token. The embedding output
# is dropped, so acts[l] is the output of layer l.
#
# This loads the model, so it is the memory-hungry step. For a model that does
# not fit on one device, add:
#     --devmap auto                 (needs the [shard] extra: accelerate)
#     --maxmem '{"0":"40GiB","cpu":"80GiB"}'
#
# LOOK AT: the layer/hidden line and the prompt-group counts
# ({'harmful': 1, 'harmless': 4}).
echo "=== 2/4  capture  ($MODEL) ============================================="
resgate capture \
  --model   "$MODEL" \
  --prompts "$PROMPTS" \
  --out     "$OUTDIR/acts.pt" \
  --sys     "$SYS" \
  --bs      "$BS"

# ---------------------------------------------------------------------------
# Step 3 — direction: one direction per layer, plus the sep(l) weights.
# ---------------------------------------------------------------------------
#   d_l    = mean(harmful) - mean(harmless), then orthogonalised against the
#            unit-normalised harmless centroid  [Heretic's orthogonalize_direction]
#   sep(l) = ||h_l - abar_l|| / mean_i||g_i,l - abar_l||   (gap / harmless spread),
#            normalised to mean 1 over kept layers         [ours]
#
# MODE=sep KEEP=0 are the canonical settings. KEEP=0 keeps every layer.
#
# LOOK AT TWO THINGS:
#   1. "after ortho max" must be ~0 (prints as e-08 or smaller). If it is not,
#      the orthogonalisation did not take and the direction is contaminated by
#      the harmless centroid.
#   2. The sep(l) bar chart should have STRUCTURE — a band of high-sep layers,
#      not a flat line. A flat profile means sep carries no information and you
#      are effectively running --mode uniform. The printed max/min ratio is the
#      quick read.
echo "=== 3/4  direction ====================================================="
resgate direction \
  --acts "$OUTDIR/acts.pt" \
  --out  "$OUTDIR/dir.pt" \
  --mode sep \
  --keep 0

# ---------------------------------------------------------------------------
# Step 4 — patch: the rank-1 PRE row-normalised edit.
# ---------------------------------------------------------------------------
#     n = rownorm(W);  dW = -w * outer(n (*) v, v @ (W/n))      with w = K * sep_l
#
# This update rule is Heretic's (see ATTRIBUTION.md). The patcher writes bf16
# IN PLACE into $DST's safetensors payload, so $DST must ALREADY be a full copy
# of $SRC — shards, index and config. It reads the originals from $SRC, so the
# edit is never applied twice even if you re-run this step.
if [ ! -d "$DST" ]; then
  echo "--- copying $SRC -> $DST (the patch edits DST in place) ---"
  cp -r "$SRC" "$DST"
fi

# LOOK AT: the printed max|dW|. That is the dose coordinate you compare builds
# on — NOT K (docs/PITFALLS.md §5). It is linear in K to +/-0.00007. For the
# CBRN kit on this model: 0.1808 at K=1.0, 0.2261 at K=1.25, 0.2713 at K=1.5.
# A different kit shifts the magnitude at equal K (the h006 kit gives 0.0840 at
# K=0.5 where the CBRN kit gives 0.0904), so compare kits on max|dW|.
echo "=== 4/4  patch  (K=$K) ================================================="
resgate patch \
  --src  "$SRC" \
  --dst  "$DST" \
  --dirs "$OUTDIR/dir.pt" \
  -K     "$K"

echo
echo "Patched checkpoint: $DST"
echo
# ---------------------------------------------------------------------------
# Next: EVALUATE. Do not read a refusal count as an ablation score.
# ---------------------------------------------------------------------------
# On our h006 build at this dose the keyword metric said 13 refusals, of which
# 11 fired on the bare word "illegal" inside answers that did the thing; the
# three-way judge put 2 in REFUSED. Generate responses once, then run all three
# instruments over THE SAME strings:
#
#   python eval/keyword_rate.py responses.json --attribute
#   python eval/kl.py --baseline "$SRC" --current "$DST" --batch-size 8 --out kl.json
#   JUDGE_BASE_URL=... JUDGE_MODEL=... python eval/judge_threeway.py responses.json judged.json
#
# See eval/README.md and docs/PITFALLS.md.
cat <<EOF
Next: evaluate. A refusal count is not an ablation score — eval/README.md.
  python eval/keyword_rate.py responses.json --attribute
  python eval/kl.py --baseline "$SRC" --current "$DST" --batch-size $BS --out kl.json
  python eval/judge_threeway.py responses.json judged.json   # needs JUDGE_BASE_URL, JUDGE_MODEL
EOF
