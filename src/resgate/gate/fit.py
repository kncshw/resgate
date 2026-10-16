"""Closed-form `M1_ridge_lam16` (`M1 + M3`) gate direction and leak-constrained threshold fitting.

Implements:
  * M1 (Subcategory-centred within-class scatter `Z_subcat`): centres each non-target reference row
    at its own subcategory centroid `mu_ref^(s)` rather than the pooled negative centroid `mu_ref`,
    preventing between-category offsets from consuming the leading eigenvectors of the scatter matrix.
  * M3 (Smooth Tikhonov shrinkage `T_ridge` at `alpha = lambda_16`):
    `c = unit(alpha (Sigma_subcat + alpha I)^-1 d)` with `alpha = lambda_16`, eliminating the
    discontinuous spectral step of hard `k=5` clipping (`B0`).
  * Leak-constrained threshold rules `R2` (`thresh_r2`) and `R3` / `R3U140` (`thresh_r3`).
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch


def _eig_scatter_gram(Z: torch.Tensor, dof: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact positive eigensystem (lam, U) of Sigma = Z.T @ Z / dof via the N x N Gram matrix.

    Since N <= 240 << D (2048..4096), we diagonalise G = Z @ Z.T in float64 with `eigh`
    (ascending eigenvalues, flipped to descending), retain modes with ev > 1e-12, and lift
    orthonormal eigenvectors into D-dimensional residual space via U = Z.T @ V / sqrt(ev).
    """
    Z64 = Z.detach().to(torch.float64)
    G = Z64 @ Z64.T
    G = 0.5 * (G + G.T)
    ev, V = torch.linalg.eigh(G)
    ev = torch.flip(ev, dims=[0])
    V = torch.flip(V, dims=[1])
    keep = ev > 1e-12
    ev = ev[keep]
    V = V[:, keep]
    lam = ev / float(dof)
    U = (Z64.T @ V) / torch.sqrt(ev).unsqueeze(0)
    return lam, U


def fit_gate_direction(
    xp: torch.Tensor,
    xn: torch.Tensor,
    neg_groups: Sequence[Any] | None = None,
    *,
    rank: int = 16,
    mode: str = "ridge",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the closed-form gate direction `c` (`M1_ridge_lam16` by default).

    Parameters
    ----------
    xp : Tensor [n_pos, D]
        Pooled read-layer activations for positive target-category training prompts.
    xn : Tensor [n_neg, D]
        Pooled read-layer activations for multi-category non-target reference prompts.
    neg_groups : sequence of length n_neg, optional
        Subcategory label per row of `xn` (e.g., 14 non-target harmful categories). When supplied
        (the M1 formulation), each reference row is centred at its own subcategory centroid
        `mu_ref^(s)` rather than the pooled negative centroid `mu_ref`, preventing between-category
        offsets from dominating the top eigenvectors of the scatter matrix (dof = n_pos + n_neg - 1 - S).
        When None, falls back to two-block centring (dof = n_pos + n_neg - 2).
    rank : int, default 16
        Eigenvalue index anchoring the regulariser: for `mode="ridge"`, `alpha = lam[rank - 1]`
        (so `rank=16` selects `alpha = lambda_16`, i.e. `M1_ridge_lam16`; `rank=6` selects
        `alpha = lambda_6`, i.e. `M1_ridge_lam6`). For `mode="clip"`, `k = rank - 1` leading
        eigenvalues are clipped to the floor `lam[rank - 1]` (`rank=6` -> `k=5` clipped at `lambda_6`).
    mode : {"ridge", "clip", "raw"}, default "ridge"
        - `"ridge"` (M3): smooth Tikhonov shrinkage `T_ridge d = alpha (Sigma + alpha I)^-1 d`
          with `alpha = lam[rank - 1]`.
        - `"clip"`: hard spectral clipping of the top `k = rank - 1` modes to `lam[rank - 1]`.
        - `"raw"`: unwhitened `unit(d)`.

    Returns
    -------
    (c, info) : (Tensor[D] float32 unit vector, dict of spectral diagnostics)
    """
    if mode not in ("ridge", "clip", "raw"):
        raise ValueError("mode must be 'ridge', 'clip', or 'raw', not %r" % (mode,))
    xp64 = xp.detach().to(torch.float64)
    xn64 = xn.detach().to(torch.float64)
    xp64 = xp64 / xp64.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    xn64 = xn64 / xn64.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    n_pos, n_neg = int(xp64.shape[0]), int(xn64.shape[0])
    mu_pos = xp64.mean(dim=0)
    mu_neg = xn64.mean(dim=0)
    d = mu_pos - mu_neg
    unit_d = d / d.norm().clamp_min(1e-12)

    if neg_groups is not None:
        if len(neg_groups) != n_neg:
            raise ValueError("len(neg_groups)=%d != n_neg=%d" % (len(neg_groups), n_neg))
        uniq_g = sorted({str(g) for g in neg_groups})
        n_subcats = len(uniq_g)
        dof = n_pos + n_neg - 1 - n_subcats
        if dof <= 0:
            raise ValueError("non-positive degrees of freedom %d" % dof)
        z_blocks = [xp64 - mu_pos]
        g_strs = [str(g) for g in neg_groups]
        for g in uniq_g:
            idx = [i for i, x in enumerate(g_strs) if x == g]
            xg = xn64[idx]
            z_blocks.append(xg - xg.mean(dim=0))
        Z = torch.cat(z_blocks, dim=0)
        scatter_kind = "M1_subcat"
    else:
        n_subcats = 1
        dof = n_pos + n_neg - 2
        if dof <= 0:
            raise ValueError("non-positive degrees of freedom %d" % dof)
        Z = torch.cat([xp64 - mu_pos, xn64 - mu_neg], dim=0)
        scatter_kind = "global_2class"

    lam, U = _eig_scatter_gram(Z, dof)
    idx_anchor = min(max(int(rank) - 1, 0), int(lam.numel()) - 1)
    alpha = float(lam[idx_anchor].item())

    if mode == "raw" or rank <= 0:
        w = d
    elif mode == "ridge":
        coef = U.T @ d
        shrink = lam / (lam + alpha)
        w = d - U @ (shrink * coef)
    else:  # mode == "clip"
        kk = idx_anchor
        Uk = U[:, :kk]
        fac = alpha / lam[:kk]
        coef = Uk.T @ d
        w = d - Uk @ coef + Uk @ (fac * coef)

    c64 = w / w.norm().clamp_min(1e-12)
    cos_to_d = float((c64 @ unit_d).item())
    info = {
        "scatter": scatter_kind,
        "mode": mode,
        "rank": int(rank),
        "dof": int(dof),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_neg_subcats": int(n_subcats),
        "alpha": alpha,
        "lam_1": float(lam[0].item()),
        "lam_anchor": alpha,
        "cos_c_unit_d": cos_to_d,
    }
    return c64.to(torch.float32), info


def thresh_r2(pos_scores: torch.Tensor, neg_scores: torch.Tensor, m: int) -> float:
    """In-fit positive-grid threshold rule R2: min {s in pos_scores : #{neg_scores >= s} <= m}."""
    ps = torch.sort(pos_scores.detach().to(torch.float64).flatten()).values
    ns = neg_scores.detach().to(torch.float64).flatten()
    for s in ps:
        if int((ns >= s).sum().item()) <= int(m):
            return float(s.item())
    return float(math.nextafter(float(ns.max().item()), math.inf))


def thresh_r3(neg_scores: torch.Tensor, m: int, eps: float = 0.0) -> float:
    """Order-statistic reference threshold rule R3 / R3U140: nextafter(r_{(m+1)} + eps, +inf)."""
    ns = torch.sort(neg_scores.detach().to(torch.float64).flatten(), descending=True).values
    if int(m) <= 0:
        base = float(ns[0].item()) + float(eps)
    elif int(m) >= int(ns.numel()):
        return float("-inf")
    else:
        base = float(ns[int(m)].item()) + float(eps)
    return float(math.nextafter(base, math.inf))


def fit_gate_op(
    xp: torch.Tensor,
    xn: torch.Tensor,
    neg_groups: Sequence[Any] | None = None,
    *,
    category: str = "cyber_offense",
    gate_layer: int = 8,
    pool: str = "mean_user",
    m: int = 11,
    rule: str = "R3",
    eps: float = 0.0,
    rank: int = 16,
    mode: str = "ridge",
    meta_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit a complete `Gate.from_file`-compatible operating-point dict (`M1_ridge_lam16` default)."""
    c, info = fit_gate_direction(xp, xn, neg_groups, rank=rank, mode=mode)
    c64 = c.to(torch.float64)
    xp_u = xp.detach().to(torch.float64)
    xp_u = xp_u / xp_u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    xn_u = xn.detach().to(torch.float64)
    xn_u = xn_u / xn_u.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    s_pos = xp_u @ c64
    s_neg = xn_u @ c64
    if rule == "R2":
        t = thresh_r2(s_pos, s_neg, m)
    elif rule in ("R3", "R3U140"):
        t = thresh_r3(s_neg, m, eps=eps)
    else:
        raise ValueError("rule must be 'R2' or 'R3'/'R3U140', not %r" % (rule,))

    deliv = int((s_pos >= t).sum().item())
    lk = int((s_neg >= t).sum().item())
    arm_tag = "M1_%s_lam%d" % (mode, rank) if neg_groups is not None else "%s_lam%d" % (mode, rank)
    meta: dict[str, Any] = {
        "category": str(category),
        "pool": str(pool),
        "gate_layer": int(gate_layer),
        "op": "%s_%s_m%d" % (arm_tag, rule, int(m)),
        "delivered": deliv,
        "n_target": int(xp.shape[0]),
        "leak": lk,
        "n_nontarget": int(xn.shape[0]),
        "baseline_leak": None,
        "benign_fpr": None,
        "fit_info": info,
    }
    if meta_extra:
        meta.update(meta_extra)
    return {"c": c, "thresh": float(t), "meta": meta}
