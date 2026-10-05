"""Derive the ablation target set from a checkpoint — no model card, no reference, no diff.

The method makes matrices unable to WRITE along v, so the target set is every weight whose
output is added to the residual stream. Any such matrix has hidden_size in its output
position; anything left unedited can simply re-introduce the direction.

Two filters, both mechanical:
  shape  2D [hidden, *]  or  3D [experts, hidden, *]        (MoE stores experts fused)
  role   the module is an OUTPUT projection

Shape alone is not enough: Llama's q_proj is [4096,4096] and would pass, but it READS from
the residual rather than writing to it. The role filter is a short architecture-general list
of output-projection names -- which is knowledge about transformers, not about any one model.

Known counts: Llama-3.1-8B-Instruct 64, Qwen3.6-35B-A3B 120, Gemma-4-26B-A4B-it 90.
"""
from __future__ import annotations

import collections
import json
import os
import re
import struct
import sys
from typing import Sequence

OUT_NAMES = ("o_proj", "out_proj", "down_proj", "wo", "dense_4h_to_h", "c_proj")
# mtp = multi-token-prediction / speculative head: not part of the served forward pass.
# vision_tower / visual: not on the text residual stream.
EXCLUDE_PARTS = ("mtp", "vision_tower", "visual")


def hidden_size(cfg: dict) -> int | None:
    """hidden_size from the top level, else from text_config (multimodal checkpoints)."""
    for k in ("hidden_size",):
        if k in cfg:
            return cfg[k]
    t = cfg.get("text_config") or {}
    return t.get("hidden_size")


def headers(d: str | os.PathLike, fn: str, cache: dict) -> dict:
    """safetensors header of shard `fn` inside `d`. `cache` is supplied by the caller."""
    if fn not in cache:
        with open(os.path.join(d, fn), "rb") as fh:
            hl = struct.unpack("<Q", fh.read(8))[0]
            cache[fn] = json.loads(fh.read(hl))
    return cache[fn]


def enumerate_targets(d: str | os.PathLike) -> tuple[int, list[tuple[str, tuple[int, ...]]], int, int]:
    """Target set of checkpoint directory `d`, read from safetensors headers only.

    Returns (hidden_size, [(tensor_name, shape)], n_rejected_by_shape, n_rejected_by_role).
    """
    cfg = json.load(open(os.path.join(d, "config.json")))
    H = hidden_size(cfg)
    wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    cache: dict = {}
    keep, rejected_shape, rejected_role = [], 0, 0
    for name, fn in wm.items():
        meta = headers(d, fn, cache).get(name)
        if not meta: continue
        sh = meta["shape"]
        writes = (len(sh) == 2 and sh[0] == H) or (len(sh) == 3 and sh[1] == H)
        if not writes: rejected_shape += 1; continue
        if not any(o in name for o in OUT_NAMES): rejected_role += 1; continue
        parts = name.split(".")
        if any(x in parts for x in EXCLUDE_PARTS): continue
        keep.append((name, tuple(sh)))
    return H, keep, rejected_shape, rejected_role


def format_report(d: str | os.PathLike, H: int, keep: Sequence[tuple[str, tuple[int, ...]]],
                  rs: int, rr: int) -> str:
    """The original script's stdout: one line per name pattern, then the total."""
    pat = collections.Counter(re.sub(r"\.\d+\.", ".N.", n) for n, _ in keep)
    out = ["=== %s   hidden=%d" % (os.path.basename(str(d).rstrip("/")), H)]
    for k, v in sorted(pat.items()):
        shp = next(s for n, s in keep if re.sub(r"\.\d+\.", ".N.", n) == k)
        out.append("   %4d  %-58s %s"
                   % (v, k.replace("model.language_model.", "").replace("model.", ""), shp))
    out.append("   TOTAL %d targets   (rejected: %d by shape, %d by role)" % (len(keep), rs, rr))
    return "\n".join(out)


def main(dirs: Sequence[str] | None = None) -> int:
    for d in (dirs if dirs is not None else sys.argv[1:]):
        H, keep, rs, rr = enumerate_targets(d)
        print(format_report(d, H, keep, rs, rr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
