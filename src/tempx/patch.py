"""Architecture-agnostic rank-1 PRE ablation patcher, driven by enumerate_targets.

Update rule (Heretic's -- github.com/p-e-w/heretic; verified by reproducing its published
weights to cos 0.99985. Not ours; the foundational method is Arditi et al. 2024,
arXiv:2406.11717):
    n = rownorm(W);  What = W/n
    dW = -w * (n (*) v) (v^T What)          with  w = K * sep_l

Targets are DERIVED from the checkpoint, not listed: every matrix whose output is added to
the residual stream. Handles 2D [hidden,*] and fused-MoE 3D [experts,hidden,*] alike.
Directions are keyed by layer; every target at layer l uses that layer's direction.
`dirs` may name several blobs, applied sequentially.

NOTE: this edits DST IN PLACE, reading the originals from SRC. DST must already be a copy
of the SRC checkpoint (shards, index and config), as the original script required.
"""
from __future__ import annotations

import collections
import json
import os
import re
import struct
from typing import Any, Sequence

import torch
from safetensors import safe_open

from .enumerate_targets import enumerate_targets

MLP_OUT_NAMES = ("down_proj", "dense_4h_to_h")


def layer_of(name: str) -> int | None:
    """Layer index from a tensor name, or None when it carries no layers.N. segment."""
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def key_of(name: str, l: int) -> str:
    """Direction key: '<layer>.mlp_out' for the MLP output projection, else '<layer>.attn_out'."""
    cls = "mlp_out" if any(o in name for o in MLP_OUT_NAMES) else "attn_out"
    return "%d.%s" % (l, cls)


def layout(p: str | os.PathLike) -> tuple[int, dict]:
    """(byte offset of the tensor data, safetensors header) of shard `p`."""
    with open(p, "rb") as fh:
        hl = struct.unpack("<Q", fh.read(8))[0]
        return 8 + hl, json.loads(fh.read(hl))


def pre_step(Wm: torch.Tensor, vv: torch.Tensor, ww: float) -> torch.Tensor:
    """One rank-1 PRE row-normalised step: Wm - ww * outer(n(*)vv, vv @ (Wm/n)).

    Heretic's update rule. `vv` must be unit-norm; `ww` is K * the layer weight.
    """
    n = Wm.norm(dim=1).clamp_min(1e-12)
    return Wm - ww * torch.outer(n * vv, vv @ (Wm / n.unsqueeze(1)))


def build_plan(src: str | os.PathLike, dirs: dict[str, Any]) -> list[tuple[str, str, int]]:
    """[(direction key, tensor name, n_dims)] for every derived target with a direction."""
    _, targets, _, _ = enumerate_targets(src)
    plan = []
    for name, shape in targets:
        l = layer_of(name)
        if l is None: continue
        k = key_of(name, l)
        if k not in dirs: continue
        plan.append((k, name, len(shape)))
    return plan


def load_dirs(dirs: str | Sequence[str | os.PathLike | dict]) -> list[dict[str, Any]]:
    """Direction blobs from a comma-separated path string, a path list, or loaded blobs."""
    if isinstance(dirs, (str, os.PathLike)):
        dirs = [x.strip() for x in str(dirs).split(",") if x.strip()]
    return [d if isinstance(d, dict) else torch.load(d, map_location="cpu") for d in dirs]


def patch(
    src: str | os.PathLike,
    dst: str | os.PathLike,
    dirs: str | Sequence[str | os.PathLike | dict],
    K: float = 1.0,
    verbose: bool = True,
) -> tuple[int, float]:
    """Apply the rank-1 PRE edit at dose K to every derived target, writing into `dst`.

    Returns (n_tensors_written, max|dW|). max|dW| is linear in K.
    """
    def log(*a):
        if verbose: print(*a, flush=True)

    _blobs = load_dirs(dirs)
    if not _blobs: raise ValueError("no direction blob given")
    dirs_d: dict[str, Any] = _blobs[0]["d"]; wts = _blobs[0]["w"]
    if len(_blobs) > 1: log("SEQUENTIAL: %d directions" % len(_blobs))

    wm = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    plan = build_plan(src, dirs_d)
    kinds = collections.Counter(re.sub(r"\.\d+\.", ".N.", n) for _, n, _ in plan)
    log("PRE-generic  K=%s  %d targets derived from the checkpoint" % (K, len(plan)))
    for k, v in sorted(kinds.items()):
        log("   %4d  %s" % (v, k.replace("model.language_model.", "").replace("model.", "")))

    byfile: dict[str, list[tuple[str, str, int]]] = {}
    for rec in plan: byfile.setdefault(wm[rec[1]], []).append(rec)
    tot = 0; mx = 0.0
    for fn, recs in sorted(byfile.items()):
        dstf = os.path.join(dst, fn)
        if not os.path.exists(dstf):
            raise FileNotFoundError(
                "%s is missing: DST must already be a copy of SRC (the patch is applied in "
                "place). Copy the checkpoint first, then patch the copy." % dstf)
        base, hdr = layout(dstf)
        fh = open(dstf, "r+b")
        with safe_open(os.path.join(src, fn), framework="pt") as sf:
            for k, name, nd in recs:
                v = dirs_d[k].float().flatten(); v = v / v.norm()
                steps = [(v, wts[k] * K)]
                for bb in _blobs[1:]:
                    if k in bb["d"]:
                        v2 = bb["d"][k].float().flatten(); v2 = v2 / v2.norm()
                        steps.append((v2, bb["w"][k] * K))
                W = sf.get_tensor(name).to(torch.float32); Wp = W
                for vv, ww in steps:
                    if nd == 3:
                        out = torch.empty_like(Wp)
                        for e in range(Wp.shape[0]): out[e] = pre_step(Wp[e], vv, ww)
                        Wp = out
                    else:
                        Wp = pre_step(Wp, vv, ww)
                mx = max(mx, float((Wp - W).abs().max()))
                Wp = Wp.to(torch.bfloat16)
                s, e = hdr[name]["data_offsets"]
                assert Wp.numel() * 2 == e - s, (name, Wp.numel() * 2, e - s)
                fh.seek(base + s); fh.write(Wp.contiguous().view(torch.uint8).numpy().tobytes())
                tot += 1; del W, Wp
        fh.flush(); os.fsync(fh.fileno()); fh.close()
    log("DONE %d tensors, max|dW| %.4f" % (tot, mx))
    return tot, mx
