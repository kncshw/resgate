"""resgate -- organized into two sub-packages:

  * `resgate.ablation` (`enumerate_targets`, `capture`, `direction`, `patch`):
    the unconditional rank-1 PRE abliteration pipeline and `sep(l)` band localisation.
  * `resgate.gate` (`Gate`, `fit_gate_direction`, `thresh_r2`, `thresh_r3`, `fit_gate_op`, `serve`):
    the pre-band route-invariant runtime gate and closed-form `M1_ridge_lam16` (`M1 + M3`)
    subcategory-centred Tikhonov ridge classifier.

Top-level imports (`resgate.patch`, `resgate.gate`, `resgate.serve`, `resgate.direction`,
`resgate.capture`, `resgate.enumerate_targets`) remain aliased for full backward compatibility.
"""
from __future__ import annotations

__version__ = "0.2.0"

from .ablation.enumerate_targets import OUT_NAMES, enumerate_targets, format_report

__all__ = ["__version__", "OUT_NAMES", "enumerate_targets", "format_report"]
