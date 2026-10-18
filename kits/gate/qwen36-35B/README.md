# qwen36-35B — shipped gated-abliteration kit (`M1_ridge_lam16`)

Five files. Together they are everything the Qwen3.6-35B-A3B gate needs for both the
**cyber** and **bio** target domains.

| file | sha256 | what it is |
|---|---|---|
| `dir_qwen36_sep_L14to22.pt` | `de141fd159608721faaae52763d3dd596d55f6afe052d9e3741d44fd29c00aa9` | **the edit, band only (`L14–22`, live-served `--dirs`).** 18 keys (`{14..22}.{attn_out,mlp_out}`) hooked across 27 target tensors (9 attention/Gated DeltaNet out projections + 9 shared-expert down projections + 9 fused routed-expert `down_proj` stacks) in `serve_http.py` |
| `dir_qwen36_sep.pt` | `ca47bda1755ad686ab598d0a44b6158a489be7433892a98e2236beb1e25789d7` | **the full 40-layer `sep(l)` refusal direction blob (`pool681`).** Per-layer refusal directions and `sep(l)` weights |
| `op_qwen36_L13_M1subcat240_M1_ridge_lam16_serve.pt` | `8ce111404a5a06ec6ebb3f1618120d8ef21b31d7cf34eb49c912e1902ff4bbaa` | **the cyber gate (`M1_ridge_lam16`).** Classifier direction `c` (2048 float32), threshold `t = 0.08298481344245375`, `mean_user` at layer 13 |
| `op_wmdpbio_B2_M1_ridge_lam16_R3U140_qwen36_L13.pt` | `7d78a2f571dd0f52fd33176d1d82f6d498d1dabc5ee58e3f026c1ca237358cc8` | **the bio gate (`B2_M1_ridge_lam16`).** Classifier direction `c` (2048 float32), threshold `t = 0.03838094775001616`, `mean_user` at layer 13 |
| `band_spec.json` | `9708c47db067e1d3572a04081468192fb0263b5299897e3ea65bc4507b19eede` | the measured `sep(l)` profile (`pool681`, 40 layers) |

## Operating points (`M1_ridge_lam16`: subcategory-centred scatter + Tikhonov ridge at $\alpha = \lambda_{16}$)

    model         Qwen/Qwen3.6-35B-A3B (enable_thinking=False)
    pooling       mean_user
    gate layer    13                 (strictly below edit band L14–22)
    edit dose     K = 2.0            (9 of 40 layers, 27 target tensors; KL_heretic = 0.0743, KL_clean = 0.0548)

### 1. Cyber gate (`op_qwen36_L13_M1subcat240_M1_ridge_lam16_serve.pt`)

- **Threshold:** `t = 0.08298481344245375` (`iso_leak_22`; `in_fit_R3_m11` is `0.0823819502288687`, `in_fit_R2_m11` is `0.11127084323456556`; raw difference norm `‖d‖₂ = 0.167008`)
- **CySecBench TEST (675 rows):** **555 / 675 = 82.2%** live & offline (`95% CP: [79.1, 85.0]`, $+7.6$\,pp / $+51$ prompts over `B0`; **556 / 675 = 82.4%** at `in_fit_R3_m11`)
- **Live WMDP_CYBER test (637 rows):** **596 / 637 = 93.6%**
- **Live CatQA non-target harmful TEST (234 rows):** **22 / 234 = 9.4%** (`95% CP: [6.0, 13.9]`)
- **Live WMDP_BIO test leak (637 rows):** **6 / 637 = 0.94%** (`95% CP: [0.35, 2.04]`; **2 / 637 = 0.31%** under `in_fit_R2`)
- **Live 64-prompt fresh cyber generation sample (`K = 2.0`):** **55 / 64 (85.9%)** judged `COMPLIED` (`63 / 64 = 98.4%` non-refusal), capturing **91.7%** (`55/60`) of the unconditional edit ceiling while holding CatQA non-target `COMPLIED` down to **7 / 64 (10.9%)** vs. `55 / 64 (85.9%)` unconditional; **107/107** exact agreement between gate bit and judged compliance (`48/0/0/5` cyber, `6/0/0/48` CatQA); **6/6** and **56/56** (`62/62 = 100%`) exact `sha256` byte identity on withheld prompts.

### 2. Bio gate (`op_wmdpbio_B2_M1_ridge_lam16_R3U140_qwen36_L13.pt`)

- **Threshold:** `t = 0.03838094775001616` (`R3U140`, $m = 6$ on 140 two-corpus reference rows; raw difference norm `‖d‖₂ = 0.104022`)
- **WMDP_BIO test (637 rows):** **635 / 637 = 99.7%** live & offline (`95% CP: [98.9, 100.0]`)
- **Live held-out WMDP_BIO val (127 rows, 118 deduplicated):** **127 / 127 = 100.0%** (`95% CP: [97.1, 100.0]`)
- **Live CySecBench TEST leak (675 rows):** **2 / 675 = 0.30%** (`95% CP: [0.04, 1.07]`)
- **Live CatQA non-target harmful TEST leak (234 rows):** **19 / 234 = 8.1%** (`95% CP: [5.0, 12.4]`)
- **Live WMDP_CYBER test leak (637 rows):** **0 / 637 = 0.00%**
- **Live SOSBench Bio (498 rows / 310 keyword-free rows):** **495 / 498 = 99.4%** / **307 / 310 = 99.0%**
