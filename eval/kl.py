#!/usr/bin/env python3
"""First-token KL divergence against a baseline model -- Heretic's `KLDivergence`, reproduced.

CREDIT
    The metric is Heretic's: github.com/p-e-w/heretic,
    src/heretic/scorers/kl_divergence.py (AGPL-3.0-or-later -- see the licence note in
    eval/keyword_rate.py; Heretic is NOT Apache-2.0). The computation below was transcribed
    from that file and from `Model.get_logits` in src/heretic/model.py.

WHAT IS COMPUTED
        baseline_logprobs = F.log_softmax(baseline_first_token_logits, dim=-1)
        logprobs          = F.log_softmax(current_first_token_logits,  dim=-1)
        kl = F.kl_div(logprobs, baseline_logprobs, reduction="batchmean", log_target=True)

    TWO PROPERTIES OF THIS NUMBER THAT ARE EASY TO MISREAD:

    1. DIRECTION. `F.kl_div(input, target)` computes KL(target || input). The arguments
       above put the CURRENT model in `input` and the BASELINE in `target`, so this is the
       forward KL(baseline || current) -- expectation taken under the baseline. It is not
       symmetric and it is not KL(current || baseline). It penalises the edited model for
       putting low probability where the baseline put high probability; it is comparatively
       forgiving of the edited model inventing new mass where the baseline had none.
       `reduction="batchmean"` divides by the number of prompts, so the result is mean nats
       per prompt.

    2. FIRST TOKEN ONLY. This is the distribution over the single next token after the
       prompt -- NOT a per-token average over a generated continuation. Only one token is
       generated (`max_new_tokens=1`) and only its logits are used. So it measures drift at
       the decision point where a refusal would begin, and says nothing directly about the
       other 99 tokens of a 100-token answer. Two models with the same first-token
       distribution can continue completely differently.

    Consequence, measured in this project rather than assumed: KL did not predict judged
    output quality. Our CBRN K=1.5 build has KL 0.3896 and was NOT distinguishable from the
    pristine model by a blind pairwise quality judge (p = 0.28). A published abliteration at
    roughly half that KL (0.2104) WAS judged worse than pristine (p = 0.0019). Report KL
    because Heretic reports it and it makes builds comparable -- do not read a low KL as
    evidence that quality survived, or a higher KL as evidence that it did not.

FAITHFULNESS DETAILS (all of these change the number)
    * Prompts go through the chat template with a system message, `add_generation_prompt=True`.
      Heretic's default system prompt is "You are a helpful assistant."
    * tokenizer.padding_side = "left", so the last position is the real final prompt token.
    * Greedy, `use_cache=False`, and RAW logits (`output_logits=True`), not the processed
      generation `scores`: processors can insert -inf for suppressed tokens, which makes the
      KL evaluate to NaN. This is Heretic's own stated reason.
    * Heretic's default KL prompt set is `mlabonne/harmless_alpaca` split `test[:100]`,
      column `text` -- HARMLESS prompts, not the harmful set used for the refusal count.
      Measuring drift on harmless input is the point: it asks what the edit cost on ordinary
      use. Do not substitute the harmful set and compare to a published figure.
    * Heretic also applies an auto-detected `response_prefix` (and a closed chain-of-thought
      block for thinking models) so that evaluation happens where responses start to differ.
      This script does NOT auto-detect it; pass `--response-prefix` if you need it. For a
      thinking model, leaving it empty measures the wrong token. This is the one place where
      a default here can silently diverge from Heretic's pipeline -- it is called out rather
      than hidden.
    * Batch composition matters for MoE models, because expert routing depends on the batch.
      Pin --batch-size across the arms you intend to compare.

MEMORY
    The two models are loaded one at a time: the baseline is loaded, its logprobs are kept,
    it is freed, then the current model is loaded. Only the (n_prompts x vocab) logprob
    tensor is held across the swap, on CPU by default.

USAGE
    kl.py --baseline <path-or-hf-id> --current <path-or-hf-id> \
          [--prompts prompts.json|prompts.txt] [--dataset mlabonne/harmless_alpaca] \
          [--split test[:100]] [--column text] \
          [--batch-size 8] [--dtype bfloat16] [--system "You are a helpful assistant."] \
          [--response-prefix ""] [--out kl.json]

    --prompts takes a JSON list of strings, a JSON list of objects (with --prompt-key), or a
    text file with one prompt per line. If omitted, `datasets` is used to pull the dataset.
"""

import argparse
import gc
import json
import os
import sys


def load_prompts(args):
    if args.prompts:
        if args.prompts.endswith(".json"):
            blob = json.load(open(args.prompts))
            if isinstance(blob, dict):
                for k in ("prompts", "rows", "records", "runs", "labels"):
                    if isinstance(blob.get(k), list):
                        blob = blob[k]
                        break
            if not isinstance(blob, list):
                sys.exit("--prompts JSON must be a list (or hold one under 'prompts'/'rows'/...)")
            if blob and isinstance(blob[0], dict):
                return [r[args.prompt_key] for r in blob]
            return [str(x) for x in blob]
        return [ln.rstrip("\n") for ln in open(args.prompts) if ln.strip()]

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Install `datasets`, or pass --prompts with a local file.")
    # Heretic resolves slice specs itself; here we lean on datasets' own slicing.
    ds = load_dataset(args.dataset, split=args.split)
    return [r[args.column] for r in ds]


def model_class(model_id, default):
    """Heretic's `get_model_class`: a config carrying `vision_config` needs the
    image-text-to-text class, not the causal-LM class. Gemma-4 is such a composite
    config, so assuming AutoModelForCausalLM fails on exactly the model this project's
    published numbers were measured on."""
    try:
        from transformers import AutoModelForImageTextToText, PretrainedConfig
        configs = PretrainedConfig.get_config_dict(model_id)
        if any("vision_config" in c for c in configs):
            print("  composite config detected -> AutoModelForImageTextToText", flush=True)
            return AutoModelForImageTextToText
    except Exception as e:
        print("  model-class probe failed (%s); using %s" % (e, default.__name__), flush=True)
    return default


def first_token_logprobs(model_id, prompts, args):
    """Log-softmax over the first generated token's raw logits, one row per prompt."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    # Left padding: the final position is then the real last prompt token for every row.
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"
    cls = model_class(model_id, AutoModelForCausalLM)
    # transformers renamed this kwarg; accept either so the script is not version-locked.
    try:
        model = cls.from_pretrained(model_id, dtype=dtype, device_map=args.device_map)
    except TypeError:
        model = cls.from_pretrained(model_id, torch_dtype=dtype, device_map=args.device_map)
    model.eval()

    chats = [[{"role": "system", "content": args.system},
              {"role": "user", "content": p}] for p in prompts]
    texts = tok.apply_chat_template(chats, add_generation_prompt=True, tokenize=False)
    if args.response_prefix:
        texts = [t + args.response_prefix for t in texts]

    out = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i:i + args.batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True,
                     return_token_type_ids=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=1,
                output_logits=True,
                return_dict_in_generate=True,
                use_cache=False,
                do_sample=False,              # greedy: deterministic
                pad_token_id=tok.pad_token_id,
            )
        # Raw logits for the first (only) generated token, NOT processed `scores`:
        # processors can write -inf and make the KL NaN. Shape (batch, vocab).
        logits = gen.logits[0].float().cpu()
        out.append(F.log_softmax(logits, dim=-1))
        del gen, inputs

    del model, tok
    gc.collect()
    try:
        import torch as _t
        if _t.cuda.is_available():
            _t.cuda.empty_cache()
    except Exception:
        pass
    return torch.cat(out, dim=0)


def main():
    ap = argparse.ArgumentParser(description="First-token KL divergence vs a baseline model.")
    ap.add_argument("--baseline", required=True, help="unmodified reference checkpoint")
    ap.add_argument("--current", required=True, help="edited checkpoint to score")
    ap.add_argument("--prompts", help="local JSON list / one-per-line text file")
    ap.add_argument("--prompt-key", default="text")
    ap.add_argument("--dataset", default="mlabonne/harmless_alpaca",
                    help="Heretic's default KL prompt set (HARMLESS prompts)")
    ap.add_argument("--split", default="test[:100]")
    ap.add_argument("--column", default="text")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="PIN this across arms you compare; MoE routing depends on the batch")
    ap.add_argument("--dtype", default="bfloat16",
                    help="bfloat16|float16|float32|auto ('auto' can silently pick fp32 for "
                         "composite configs with no top-level dtype)")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--system", default="You are a helpful assistant.",
                    help="Heretic's default system prompt")
    ap.add_argument("--response-prefix", default="",
                    help="NOT auto-detected here, unlike Heretic; required for thinking models")
    ap.add_argument("--out", help="write a JSON record of the measurement")
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F

    prompts = load_prompts(a)
    if not prompts:
        sys.exit("no prompts loaded")
    print("prompts: %d" % len(prompts), flush=True)
    if not a.response_prefix:
        print("note: --response-prefix is empty and is NOT auto-detected here. If this model "
              "emits a fixed opener or a chain-of-thought block, the first token is that "
              "opener for every prompt and the KL will be near zero for the wrong reason.",
              flush=True)

    print("loading baseline %s ..." % a.baseline, flush=True)
    baseline_logprobs = first_token_logprobs(a.baseline, prompts, a)
    print("  baseline logprobs %s" % (tuple(baseline_logprobs.shape),), flush=True)

    print("loading current %s ..." % a.current, flush=True)
    logprobs = first_token_logprobs(a.current, prompts, a)
    print("  current logprobs %s" % (tuple(logprobs.shape),), flush=True)

    if logprobs.shape != baseline_logprobs.shape:
        sys.exit("shape mismatch %s vs %s -- different tokenizer/vocab? KL is undefined here."
                 % (tuple(logprobs.shape), tuple(baseline_logprobs.shape)))

    # KL(baseline || current), mean nats per prompt. See WHAT IS COMPUTED above.
    kl = F.kl_div(logprobs, baseline_logprobs,
                  reduction="batchmean", log_target=True).item()

    print("\nKL divergence: %.4f" % kl)
    print("  direction: forward KL(baseline || current), mean nats per prompt")
    print("  scope:     FIRST GENERATED TOKEN ONLY -- not a continuation average")
    if kl != kl:  # NaN
        print("  NaN: check for -inf in the logits (use raw logits, not processed scores).")

    rec = {
        "metric": "first-token KL divergence (Heretic's KLDivergence, reproduced)",
        "credit": "github.com/p-e-w/heretic src/heretic/scorers/kl_divergence.py (AGPL-3.0-or-later)",
        "kl_divergence": kl,
        "direction": "KL(baseline || current) -- F.kl_div(current, baseline, log_target=True)",
        "scope": "first generated token only",
        "reduction": "batchmean (mean nats per prompt)",
        "baseline": a.baseline, "current": a.current,
        "n_prompts": len(prompts),
        "prompt_source": a.prompts or "%s split=%s column=%s" % (a.dataset, a.split, a.column),
        "batch_size": a.batch_size, "dtype": a.dtype,
        "system_prompt": a.system,
        "response_prefix": a.response_prefix,
        "response_prefix_autodetected": False,
        "caveat": "KL did not predict judged quality in this project; see the docstring.",
    }
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)
        print("WROTE %s" % a.out)


if __name__ == "__main__":
    main()
