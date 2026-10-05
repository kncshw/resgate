"""Per-layer directions AND per-layer weights, from the model's own activations only.

NOTHING here is imported from another model or from any ablated reference checkpoint.
  * direction  : difference of means, then Heretic's orthogonalize_direction
                 (method: Arditi et al. 2024, arXiv:2406.11717)
  * layer band : derived from THIS model's activations on the 5-prompt kit
  * strength   : a single global K, set by a damage-onset ladder (not here)

Blind per-layer separation statistic:
    sep(l) = ||h_l - abar_l|| / mean_i ||g_i,l - abar_l||
             (harmful gap) / (harmless spread)   -- the ratio cancels the depth-dependent
             growth of activation norm, so it is comparable across layers.
Layer weight w(l) = sep(l) normalised to mean 1 over kept layers; K scales it at patch time.

Architecture-agnostic despite its research-era filename (qwen_dirs_blind.py).
Canonical operating settings: mode="sep", keep=0.0.
"""
from __future__ import annotations

import os
import statistics as st
from typing import Any

import torch

torch.set_grad_enabled(False)


def unit(v: torch.Tensor) -> torch.Tensor:
    """v/||v||, with a zero-norm guard."""
    n = v.norm()
    return v / n if n > 1e-12 else v


def orthogonalize_direction(d: torch.Tensor, centroid_unit: torch.Tensor) -> torch.Tensor:
    """Heretic's orthogonalize_direction -- github.com/p-e-w/heretic, main.py:459-469.

    Not ours. Removes the component of `d` along the unit-normalised mean harmless
    residual, then renormalises.  `centroid_unit` must already be unit(abar_l).
    """
    return unit(d - torch.dot(d, centroid_unit) * centroid_unit)


def compute_directions(
    acts: str | os.PathLike | dict,
    out: str | os.PathLike | None = None,
    mode: str = "sep",
    keep: float = 0.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build the {"d","w","meta"} direction blob from a capture blob.

    `mode`: "sep" for sep(l) weighting, anything else for uniform weight 1.0.
    `keep`: drop layers whose raw weight is below this fraction of the mean.
    Only the FIRST harmful prompt (by sorted key) contributes the gap; the harmless
    prompts set the centroid and the spread.
    """
    def log(*a):
        if verbose: print(*a, flush=True)

    A = acts if isinstance(acts, dict) else torch.load(acts, map_location="cpu")
    G: dict[str, dict[str, torch.Tensor]] = {}
    for it in A["items"]:
        G.setdefault(it["group"], {})[it["key"]] = it["acts"]
    for grp in ("harmful", "harmless"):
        if grp not in G:
            raise ValueError("capture has no %r group; prompt records need group in "
                             "{'harmful','harmless'}" % grp)
    hk = sorted(G["harmful"]); gk = sorted(G["harmless"])
    # sep(l) divides by the harmless spread, which is identically 0 for a single harmless
    # prompt -- the statistic is undefined, not small. Refuse rather than emit zeros.
    if len(gk) < 2:
        raise ValueError(
            "sep(l) is undefined with %d harmless prompt(s): its denominator "
            "mean_i||g_i - abar_l|| is then exactly 0, not merely small. Use >=2 harmless "
            "prompts -- the blind kit uses 4." % len(gk))
    H = torch.stack([G["harmful"][k] for k in hk]).double()
    B = torch.stack([G["harmless"][k] for k in gk]).double()
    L, P, D = H.shape[1], H.shape[2], H.shape[3]; p = P - 1
    log("harmful %d  harmless %d   L=%d D=%d" % (len(hk), len(gk), L, D))

    sep: list[float] = []; dd: dict[int, torch.Tensor] = {}; pre: list[float] = []; post: list[float] = []
    for l in range(L):
        g  = B[:, l, p, :]
        ab = g.mean(0)
        spread = float(torch.stack([(g[i] - ab).norm() for i in range(g.shape[0])]).mean())
        gap    = float((H[0, l, p, :] - ab).norm())
        sep.append(gap / spread if spread > 1e-9 else 0.0)
        gd  = unit(ab)
        raw = unit(H[0, l, p, :] - ab)
        d   = orthogonalize_direction(raw, gd)
        pre.append(abs(float(torch.dot(raw, gd)))); post.append(abs(float(torch.dot(d, gd))))
        dd[l] = d.float()

    log("\n=== BLIND per-layer separation  sep(l) = gap / harmless-spread ===")
    mx = max(sep)
    for l in range(L):
        bar = "#" * int(round(40 * sep[l] / mx))
        log("  L%-3d %7.3f  %s" % (l, sep[l], bar))
    log("\n  sep: mean %.3f  min %.3f (L%d)  max %.3f (L%d)  ratio %.1fx"
        % (st.mean(sep), min(sep), sep.index(min(sep)), mx, sep.index(mx), mx / max(min(sep), 1e-9)))
    log("  |cos(raw,centroid)| mean %.4f max %.4f ; after ortho max %.2e"
        % (st.mean(pre), max(pre), max(post)))

    w_raw = [s if mode == "sep" else 1.0 for s in sep]
    kept  = [l for l in range(L) if w_raw[l] >= keep * st.mean(w_raw)]
    m = st.mean([w_raw[l] for l in kept])
    out_d: dict[str, torch.Tensor] = {}; out_w: dict[str, float] = {}
    for l in kept:
        for c in ("attn_out", "mlp_out"):
            out_d["%d.%s" % (l, c)] = dd[l]
            out_w["%d.%s" % (l, c)] = w_raw[l] / m
    blob = {"d": out_d, "w": out_w,
            "meta": {"mode": mode, "keep_thresh": keep, "layers": kept,
                     "sep": sep, "source": "self-reliant: no reference model"}}
    if out is not None:
        torch.save(blob, out)
    log("\n  MODE=%s KEEP=%.2f -> %d layers kept, %d tensors" % (mode, keep, len(kept), len(out_d)))
    log("  per-layer weight (normalised, before K): min %.3f  max %.3f"
        % (min(out_w.values()), max(out_w.values())))
    return blob
