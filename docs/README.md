# Documentation Index (`docs/`)

Documentation is organized into two sub-folders matching `src/resgate/` and `kits/`:

## 1. [`docs/gate/`](gate/) — Runtime Gate & `M1_ridge_lam16` Classifier
- [`gate/CLASSIFIER-MATH.md`](gate/CLASSIFIER-MATH.md) — Mathematical specification of the closed-form `M1_ridge_lam16` (`M1 + M3`: subcategory-centred within-class scatter + Tikhonov ridge shrinkage at $\alpha = \lambda_{16}$) gate direction, leak-constrained threshold rules (`R2`, `R3` / `R3U140`), and live GPU tables across Gemma (L8), Phi (L11), and Llama (L10) for Cyber and Bio gates.
- [`gate/GATE.md`](gate/GATE.md) — Runtime hook architecture, route invariance (`max(gate_layers) < min(edit_layers)`), `mean_user` span pooling, fail-safe design, and 3-D fused MoE stack reconstruction.
- [`gate/CLASSIFIER-MATH-PHI4MINI.md`](gate/CLASSIFIER-MATH-PHI4MINI.md) — Phi-4-mini read-layer selection and protocol comparison record.
- [`gate/CLASSIFIER.md`](gate/CLASSIFIER.md) — Experimental archive of earlier classifier arms.

## 2. [`docs/ablation/`](ablation/) — Rank-1 Refusal Direction & Weight Edit
- [`ablation/METHOD.md`](ablation/METHOD.md) — Rank-1 `PRE` row-normalised weight update, `sep(l)` per-layer effect-size weighting, and contiguous band localisation.
- [`ablation/RESULTS.md`](ablation/RESULTS.md) — Unconditional ablation evaluation tables, dose ladder, and scope bounds.
- [`ablation/PITFALLS.md`](ablation/PITFALLS.md) — Nine measurement pitfalls (keyword refusal metrics, fused 3-D MoE coverage, batch padding, dose matching on `max|dW|`, and evaluation protocol leakage).
