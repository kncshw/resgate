"""tempx -- the sep(l) abliteration method: enumerate, capture, direction, patch.

Four steps, no search, no reference model, no gradients:

    from tempx import enumerate_targets          # stdlib only, no torch needed
    from tempx.capture   import capture          # residual at the last prompt token
    from tempx.direction import compute_directions
    from tempx.patch     import patch

    H, targets, _, _ = enumerate_targets("./Gemma-4-26B-A4B-it")   # 90 targets
    acts = capture("google/gemma-4-26B-A4B-it", "kits/h006.json", "out/acts.pt")
    dirs = compute_directions(acts, "out/dir.pt")                  # mode="sep", keep=0.0
    patch("./pristine", "./ablated", "out/dir.pt", K=1.5)          # dst is a copy of src

The submodules are not re-exported here: `tempx.patch` is the module, `tempx.patch.patch`
the function, and importing tempx costs no torch import.

Attribution: `orthogonalize_direction` and the rank-1 PRE row-normalised update rule are
Heretic's (github.com/p-e-w/heretic, AGPL-3.0-or-later; main.py:459-469 for the former), verified by
reproducing Heretic's published weights to cos 0.99985. The foundational method is Arditi et
al. 2024, "Refusal in Language Models Is Mediated by a Single Direction", arXiv:2406.11717 --
difference-of-means direction plus rank-1 weight edit.
Ours: the sep(l) per-layer weighting, the blind 5-prompt kit (1 harmful + 4 harmless),
structural target enumeration applied across dense and MoE, the three-way
REFUSED/DEFLECTED/COMPLIED evaluation, and the dose/coverage findings.

Canonical operating settings: mode="sep", keep=0.0.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .enumerate_targets import OUT_NAMES, enumerate_targets, format_report

__all__ = ["__version__", "OUT_NAMES", "enumerate_targets", "format_report"]
