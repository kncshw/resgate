# llama31-8B — shipped gated-abliteration kit (`M1_ridge_lam16`)

Five files. Together they are everything the Llama-3.1-8B-Instruct gate needs for both the
**cyber** and **bio** target domains.

| file | sha256 | what it is |
|---|---|---|
| `dir_llama31-8B_sep_L11to22.pt` | `9a8abed020094edff87ad76e5eac7b182afe14a9e1b04fa9b4a2d21e62bfc698` | **the edit, band only (`L11–22`, live-served `--dirs`).** 24 keys (`{11..22}.{attn_out,mlp_out}`) hooked across 24 target tensors in `serve_http.py` |
| `dir_llama31-8B_sep.pt` | `fae657836137f9d0f003ad3a05aa0712d334815e7d345d3e0a136a9549c0c723` | **the full 32-layer `sep(l)` refusal direction blob.** Per-layer refusal directions and `sep(l)` weights |
| `op_llama31-8B_L10_M1subcat240_M1_ridge_lam16_serve.pt` | `2e297b0b92c67a7d559c9589eba251b2b394a3c50eaf01288d8ab4a69b0f982a` | **the cyber gate (`M1_ridge_lam16`).** Classifier direction `c` (4096 float32), threshold `t = 0.06890131763033773`, `mean_user` at layer 10 |
| `op_wmdpbio_B2_M1_ridge_lam16_R3U140_llama31-8B_L10.pt` | `32b18623cc798698025b08a2fc62e349f4f6d1cb566ce54621d5a0837710824b` | **the bio gate (`B2_M1_ridge_lam16`).** Classifier direction `c` (4096 float32), threshold `t = 0.08524563027159009`, `mean_user` at layer 10 |
| `band_spec.json` | `e00d4ed8696f5ea9b67d7794d472bb91f9a8cd3e6203da750257a3ea48762eec` | the measured `sep(l)` profile this model's edit band (`L11–22`) was derived from |

## Operating points (`M1_ridge_lam16`: subcategory-centred scatter + Tikhonov ridge at $\alpha = \lambda_{16}$)

    model         meta-llama/Llama-3.1-8B-Instruct
    snapshot      0e9e39f249a16976918f6564b8830bc894c89659
    pooling       mean_user
    gate layer    10                 (strictly below edit band L11–22)
    edit dose     K = 1.75           (12 of 32 layers, 24 target tensors)

### 1. Cyber gate (`op_llama31-8B_L10_M1subcat240_M1_ridge_lam16_serve.pt`)

- **Threshold:** `t = 0.06890131763033773` (`in_fit_R3`, $m = 17$; `iso_leakage` is `0.06886736781061464`, in-fit R2 is `0.08697134666868467`; raw difference norm `‖d‖₂ = 0.14764964197517684`)
- **CySecBench TEST (675 rows):** **551 / 675 = 81.6%** live (`95% CP: [78.5, 84.5]`, $+15.7$\,pp / $+106$ prompts over `B0`; **550 / 675 = 81.5%** offline in `meta["delivered"]`)
- **Live WMDP_CYBER test (637 rows):** **600 / 637 = 94.2%** ($+5.3$\,pp / $+34$ prompts over `B0`)
- **Live CatQA non-target harmful TEST (234 rows):** **17 / 234 = 7.3%** (`95% CP: [4.3, 11.4]`; `17 / 234` offline)
- **Live WMDP_BIO test leak (637 rows):** **33 / 637 = 5.18%** (`95% CP: [3.59, 7.20]`; **6 / 637 = 0.94%** under `in_fit_R2`)
- **Live 64-prompt fresh cyber generation sample:** **55 / 64 (85.9%)** judged `COMPLIED`; **50/50** and **61/61** exact agreement between gate bit and judged outcome on refused prompts; **8/8** and **58/58** exact `sha256` byte identity on withheld prompts.

### 2. Bio gate (`op_wmdpbio_B2_M1_ridge_lam16_R3U140_llama31-8B_L10.pt`)

- **Threshold:** `t = 0.08524563027159009` (`R3U140`, $m = 6$ on 140 two-corpus reference rows; raw difference norm `‖d‖₂ = 0.09778154857322274`)
- **WMDP_BIO test (637 rows):** **632 / 637 = 99.2%** live & offline (`95% CP: [98.2, 99.7]`)
- **Live held-out WMDP_BIO val (127 rows, 118 deduplicated):** **127 / 127 = 100.0%** (`95% CP: [97.1, 100.0]`)
- **Live CySecBench TEST leak (675 rows):** **2 / 675 = 0.30%** (`95% CP: [0.04, 1.07]`)
- **Live CatQA non-target harmful TEST leak (234 rows):** **10 / 234 = 4.3%** (`95% CP: [2.1, 7.7]`)
- **Live WMDP_CYBER test leak (637 rows):** **0 / 637 = 0.00%**
- **Live SOSBench Bio (498 rows / 310 keyword-free rows):** **443 / 498 = 89.0%** / **267 / 310 = 86.1%**
