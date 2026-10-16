# CLASSIFIER-MATH — the `M1_ridge_lam16` gate specification

**Scope.** This file specifies the closed-form gate classifier shipped in `resgate.gate`
(`fit_gate_direction`, `thresh_r2`, `thresh_r3`, `fit_gate_op`) and the operating-point `.pt`
files in `kits/*/`: what `c` and `t` are, how they are computed in closed form from activations,
and what they measure in live serving on an NVIDIA GB10 (`vllm-node-tf5:latest`, `bfloat16`).

For the runtime hook mechanism — the read site, the fail-safe property, and the operating-point
file format — see `docs/GATE.md` and `src/resgate/gate.py`.

---

## 1. The object

One direction, one threshold, one bit:

    score(x) = < x / ||x||_2 , c >          a cosine, not an unnormalised projection
    fire     <=>  score(x) >= t

`Gate.decide` computes the pooled user-span hidden state `x = pool(h, mask, "mean_user", spans)`
at a single early decoder layer strictly below the edit band (`L8` on Gemma, `L11` on Phi, `L10`
on Llama) and compares `cos(x, c)` against `t`. Because the edit band (`L10–16`, `L13–15`,
`L11–22`) lies strictly downstream of the read layer, `x` is identical whether or not the edit
fires (route invariance).

---

## 2. Notation

    X_pos     n_pos = 100 positive training rows, pooled mean_user at read layer l, unit-normalised
    X_ref     n_neg = 140 reference rows across S = 14 non-target harmful subcategories (10 per subcategory),
              pooled mean_user at read layer l, unit-normalised
    mu_pos    mean(X_pos)
    mu_ref    mean(X_ref)                                  global reference centroid
    mu_ref^s  mean(X_ref^(s))                              centroid of reference subcategory s in {1..S}
    d         mu_pos - mu_ref                              raw difference-in-means vector
    N         n_pos + n_neg = 240
    K         S + 1 = 15                                   total number of groups (1 positive + 14 reference)

---

## 3. The direction (`M1 + M3`: subcategory-centred scatter + Tikhonov ridge shrinkage)

### 3.1 Why global two-class scatter (`B0`) leaks between-category variance into the top modes (`M1`)

When the negative reference pool `X_ref` spans `S = 14` distinct harmful subcategories, the
two-class scatter centred only at the pooled negative centroid `mu_ref` decomposes as:

    Sigma_global = Sigma_subcat + Sigma_between

where `Sigma_between` is the rank-(S - 1) covariance of the subcategory centroids `mu_ref^(s)`
around `mu_ref`. Under global centring, the top principal axes of `Sigma_global` are dominated
by between-category offsets (`Sigma_between`) rather than within-category prompt variation,
crowd out genuine within-class nuisance directions, and partially suppress signal aligned with
individual non-target categories.

**M1 (Subcategory-centred within-class scatter):** We centre each reference row at its own
subcategory mean `mu_ref^(s)`:

    Z_subcat = [ X_pos - mu_pos ; X_ref^(1) - mu_ref^(1) ; ... ; X_ref^(S) - mu_ref^(S) ]   (240 x D)
    Sigma_subcat = Z_subcat^T Z_subcat / (N - K) = Z_subcat^T Z_subcat / 225

### 3.2 Exact eigensystem via the 240 x 240 Gram matrix

Because `rank(Sigma_subcat) <= N - K = 225 << D`, the `D x D` matrix is never materialised:

    G      = Z_subcat Z_subcat^T            240 x 240 symmetric positive semi-definite
    ev, V  = eigh(G)                        in float64, sorted descending, keeping ev > 1e-12
    lam_i  = ev_i / (N - K)
    u_i    = Z_subcat^T v_i / sqrt(ev_i)    orthonormal eigenvectors in R^D

### 3.3 Smooth Tikhonov (ridge) shrinkage anchored to `lambda_16` (`M3`, `M1_ridge_lam16`)

Instead of hard spectral clipping at a low rank `k = 5` (which leaves modes `6..15` unwhitened
and introduces a sharp spectral step at `k`), we apply smooth Tikhonov regularisation anchored
to the 16th eigenvalue of `Sigma_subcat` (`alpha = lam_16`):

    T_ridge d = alpha (Sigma_subcat + alpha I)^(-1) d
              = d - sum_{i=1}^r [ lam_i / (lam_i + alpha) ] (u_i^T d) u_i,     alpha = lam_16
    c         = T_ridge d / ||T_ridge d||_2

A mode with `lam_i >> alpha` is scaled by `alpha / lam_i`, `lam_16` itself is scaled by `1/2`,
modes with `lam_i << alpha` pass with weight approaching `1`, and the `(D - r)`-dimensional null
space is preserved with weight `1`. Both the `M1` centring and the `M3` ridge regularisation are
closed-form linear algebra with zero learned parameters (`resgate.gate.fit_gate_direction`).

---

## 4. Leak-constrained threshold rules (`R2` and `R3` / `R3U140`)

Given a nominal leak budget `L` on the 140 reference rows (`m = floor(L * 140)` permitted
reference fires):

- **`R2` (`thresh_r2`):** Scans the 100 positive `TRAIN` scores ascending and selects the
  smallest positive score `s` with `#{X_ref scores >= s} <= m`.
- **`R3` / `R3U140` (`thresh_r3`):** Places `t` directly at the `(m + 1)`-th largest reference
  score plus one IEEE-754 float64 ULP:
  `t = nextafter(r_{(m+1)}, +inf)` where `r_{(1)} >= r_{(2)} >= ... >= r_{(140)}` are the sorted
  reference scores. This guarantees `#{X_ref scores >= t} <= m` by construction without suffering
  from discrete positive-grid gaps in the tail.

---

## 5. Shipped operating points and live GPU measurements

All three architectures use a single shared read layer across both the **cyber** and **bio**
gates (`L8` on Gemma-4-26B-A4B-it, `L11` on Phi-4-mini-instruct, `L10` on Llama-3.1-8B-Instruct),
strictly below each model's edit band (`L10–16`, `L13–15`, `L11–22`). Every number below was
measured **live** through `serve_http.py` (`POST /gate`, `/compare`, `/generate`) on an NVIDIA
GB10 (`121 GiB` unified memory).

### 5.1 Complete 2x2 live firing table (Gemma L8 / Phi L11 / Llama L10)

| Gate \ Prompts | Cyber (`675 CySecBench`) | Bio (`637 WMDP_BIO test`) | Non-target harmful (`CatQA TEST`, 234 / Phi 235) |
|---|---|---|---|
| **Cyber gate (`M1_ridge_lam16`)** | `526 / 549 / 551` = **77.9 / 81.3 / 81.6%** live (`528 / 549 / 550` offline) | `5 / 2 / 33` = 0.78 / 0.31 / 5.18% | `16 / 22 / 17` = 6.8 / 9.4 / 7.3% live (`16 / 21 / 17` offline) |
| **Bio gate (`B2_M1_ridge_lam16`)** | `3 / 0 / 2` = 0.44 / 0.00 / 0.30% | `633 / 630 / 632` = **99.4 / 98.9 / 99.2%** live & offline (`629 = 98.7%` at Phi `L12`) | `15 / 8 / 10` = 6.4 / 3.4 / 4.3% (`11 = 4.7%` at Phi `L12`) |

### 5.2 Exact shipped `.pt` operating-point thresholds (`op["thresh"]`)

| Model | Gate | `.pt` file in `kits/gate/<model>/` | Read layer | `op["thresh"]` (`t`) | Threshold rule | `meta["delivered"]` (offline) | Live `/gate` delivered |
|---|---|---|---|---|---|---|---|
| **Gemma-4-26B-A4B-it** | Cyber | `op_gemma_L8_M1subcat240_M1_ridge_lam16_serve.pt` | `L8` | `0.11349115121614832` | `iso_leakage` | `528 / 675` | `526 / 675` |
| **Gemma-4-26B-A4B-it** | Bio | `op_wmdpbio_B2_M1_ridge_lam16_R3U140_gemma_L8.pt` | `L8` | `0.05801104299110058` | `R3U140` ($m=6$) | `633 / 637` | `633 / 637` |
| **Phi-4-mini-instruct** | Cyber | `op_phi4mini_L11_M1subcat240_M1_ridge_lam16_serve.pt` | `L11` | `0.05303228245782718` | `iso_leakage` | `549 / 675` | `549 / 675` |
| **Phi-4-mini-instruct** | Bio | `op_wmdpbio_B2_M1_ridge_lam16_R3U140_phi4mini_L11.pt` | `L11` | `0.042208061573892744` | `R3U140` ($m=6$) | `630 / 637` | `630 / 637` |
| **Phi-4-mini-instruct** | Bio (`L12` comparator) | `op_wmdpbio_B2_M1_ridge_lam16_R3U140_phi4mini_L12.pt` | `L12` | `0.041939858967326334` | `R3U140` ($m=6$) | `629 / 637` | `629 / 637` |
| **Llama-3.1-8B-Instruct** | Cyber | `op_llama31-8B_L10_M1subcat240_M1_ridge_lam16_serve.pt` | `L10` | `0.06890131763033773` | `in_fit_R3` ($m=17$) | `550 / 675` | `551 / 675` |
| **Llama-3.1-8B-Instruct** | Bio | `op_wmdpbio_B2_M1_ridge_lam16_R3U140_llama31-8B_L10.pt` | `L10` | `0.08524563027159009` | `R3U140` ($m=6$) | `632 / 637` | `632 / 637` |

### 5.3 Additional live evaluation surfaces

- **Cyber gate on `WMDP_CYBER test` (637 rows):** `602 / 506 / 600` = **94.5 / 79.4 / 94.2%**
- **Cyber gate on 64-prompt fresh cyber generation sample:** **55 / 64 (85.9%)** judged `COMPLIED` on all three models (`10/10` on benign control); **294 of 294** agreement between gate bit and judged outcome on refused prompts (`115` fired & answered, `179` held & refused, `0` discordant); **100%** `sha256` byte identity across all withheld prompts and all 113 live `/compare` + `/generate` checks.
- **Bio gate on held-out `WMDP_BIO val` (127 rows, 118 deduplicated; zero overlap with train/fit/selection):** `127 / 127 / 127` = **100.0 / 100.0 / 100.0%** (`95% CP: [97.1, 100.0]`)
- **Bio gate on `WMDP_CYBER test` (637 rows):** `0 / 0 / 0` = **0.00%**
- **Bio gate on external `SOSBench Bio` (498 rows):** `480 / 473 / 443` = **96.4 / 95.0 / 89.0%** (`477 = 95.8%` at Phi `L12`)
- **Bio gate on `SOSBench Bio` keyword-free subset (310 rows, zero WMDP training stems):** `297 / 288 / 267` = **95.8 / 92.9 / 86.1%** (`291 = 93.9%` at Phi `L12`)
- **Live P10 replay determinism:** **24 / 24** bit-identical IEEE-754 hex scores (`max|Delta| = 0.0`) on every run across both sweeps.
