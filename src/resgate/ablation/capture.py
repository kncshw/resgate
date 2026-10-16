"""Architecture-agnostic residual-stream capture via output_hidden_states.

CONVENTIONS (every downstream step depends on these):
  * acts[l] = OUTPUT of layer l.  transformers' hidden_states[0] is the embedding output and
    hidden_states[i] is the output of layer i-1, so we drop index 0 and keep [1:].
  * LEFT padding, so [:, -1, :] is the last REAL prompt token (Heretic does the same).
  * add_generation_prompt=True, system prompt from `sys_prompt`.
  * saved as {"nlast":1, "snapshot", "sys", "items":[{key, group, acts[L,1,D]}]}
    so p = P-1 = 0 selects the only position.

Prompt records are [{"key": str, "group": "harmful"|"harmless", "text": str}, ...].
"""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Sequence

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _think_kwargs(think: str | bool | None) -> dict:
    """enable_thinking must match the SERVING mode.

    Qwen3.5/3.6 templates default to enable_thinking=True, which ends the prompt INSIDE an
    open <think> block; serving with enable_thinking=False closes it. Capturing in one mode
    and applying the direction in the other measures a different computational state.
    None / "" keeps the template default (the original THINK="" behaviour).
    """
    if think is None or think == "": return {}
    if isinstance(think, bool): return {"enable_thinking": think}
    return {"enable_thinking": think not in ("0", "false", "False")}


def capture(
    model: str,
    prompts: str | os.PathLike | Sequence[dict],
    out: str | os.PathLike | None = None,
    sys_prompt: str = "You are a helpful assistant.",
    bs: int = 8,
    dev: str = "cuda",
    devmap: str | None = None,
    maxmem: str | dict | None = None,
    think: str | bool | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Capture the last-prompt-token residual at every layer, for every prompt.

    Returns the blob; also torch.save()s it to `out` when `out` is given.
    """
    def log(*a):
        if verbose: print(*a, flush=True)

    recs = list(prompts) if not isinstance(prompts, (str, os.PathLike)) else json.load(open(prompts))

    tok = AutoTokenizer.from_pretrained(model)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    kw: dict[str, Any] = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if devmap:                                   # only if accelerate is available
        kw["device_map"] = devmap
        if maxmem: kw["max_memory"] = json.loads(maxmem) if isinstance(maxmem, str) else maxmem
    log("loading %s ... (dev=%s, device_map=%s)" % (model, dev, kw.get("device_map")))
    m = AutoModelForCausalLM.from_pretrained(model, **kw)
    if "device_map" not in kw: m = m.to(dev)
    m.eval()
    cfg = AutoConfig.from_pretrained(model)
    tcfg = getattr(cfg, "text_config", cfg)
    log(f"  {tcfg.num_hidden_layers} layers, hidden {tcfg.hidden_size}")
    log(f"{len(recs)} prompts {dict(Counter(r['group'] for r in recs))}")

    _kw = _think_kwargs(think)
    def render(t: str) -> str:
        return tok.apply_chat_template(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": t}],
            add_generation_prompt=True, tokenize=False, **_kw)
    texts = [render(r["text"]) for r in recs]

    items: list[dict[str, Any]] = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(m.device)
            o = m(**enc, output_hidden_states=True)
            hs = o.hidden_states[1:]                                        # drop embedding output
            A = torch.stack([h[:, -1, :].float().cpu() for h in hs], dim=1)  # (B, L, D)
            for j, r in enumerate(recs[i:i + bs]):
                items.append({"key": r["key"], "group": r["group"],
                              "acts": A[j].unsqueeze(1).clone()})            # (L, 1, D)
            log(f"  {min(i+bs,len(texts))}/{len(texts)}")
            del o, hs, A

    blob = {"nlast": 1, "snapshot": os.path.basename(str(model).rstrip("/")),
            "sys": sys_prompt, "items": items}
    if out is not None:
        torch.save(blob, out)
        log(f"saved {len(items)} prompt captures -> {out}")
    log(f"  acts shape per item: {tuple(items[0]['acts'].shape)}  (L, 1, D)")
    return blob
