# gemma4-26B — shipped gated-abliteration kit (`M1_ridge_lam16`)

Five files. Together they are everything the Gemma-4-26B-A4B-it gate needs for both the
**cyber** and **bio** target domains.

| file | sha256 | what it is |
|---|---|---|
| `gemma_dir90_cbrn_L10to16.pt` | `1ca509415f3b39089bc19156c230a49af2577a552d993d2aae6fc045d2abbd09` | **the edit, band only (`L10–16`, live-served `--dirs`).** 14 keys (`{10..16}.{attn_out,mlp_out}`) hooked across 21 target tensors in `serve_http.py` |
| `dir_gemma4-26B_sep.pt` | `15cb39c3bf3619d36cf76ad1c4590713fe58f0cdde921796e342390578a4f3f1` | **the full 30-layer `sep(l)` refusal direction blob.** Layout: `{'d': {'<l>.attn_out'\|'<l>.mlp_out': unit vector}, 'w': {same keys: float}, 'meta': {...}}` |
| `op_gemma_L8_M1subcat240_M1_ridge_lam16_serve.pt` | `0efe1b07f72ea5ffc2baf03a1009b5fc3f295ba34902e9bc74fd32a3194fb773` | **the cyber gate (`M1_ridge_lam16`).** Classifier direction `c` (2816 float32), threshold `t = 0.11349115121614832`, `mean_user` at layer 8, and fit metadata |
| `op_wmdpbio_B2_M1_ridge_lam16_R3U140_gemma_L8.pt` | `602b49ba68c4b94c9bb04e94f33a0ec0aa7416ec0ac165481bac4843890e16ae` | **the bio gate (`B2_M1_ridge_lam16`).** Classifier direction `c` (2816 float32), threshold `t = 0.05801104299110058`, `mean_user` at layer 8, and fit metadata |
| `band_spec.json` | `4d2478f625931eeca3e733b87ae3e8e61a1ed0aa8694dd055fb331cc2739fb05` | the measured `sep(l)` profile this model's edit band (`L10–16`) was derived from |

## Operating points (`M1_ridge_lam16`: subcategory-centred scatter + Tikhonov ridge at $\alpha = \lambda_{16}$)

    model         google/gemma-4-26B-A4B-it
    snapshot      7d4c97e54145f8ffd1a4dd1b4986a5015a517842
    pooling       mean_user
    gate layer    8                  (strictly below edit band L10–16)
    edit dose     K = 2.0            (7 of 30 layers, 21 target tensors: 14 Linear + 7 fused 3-D MoE)

### 1. Cyber gate (`op_gemma_L8_M1subcat240_M1_ridge_lam16_serve.pt`)

- **Threshold:** `t = 0.11349115121614832` (`iso_leakage`; in-fit R2 is `0.1222858130483601`, in-fit R3 is `0.10213419723255007`; raw difference norm `‖d‖₂ = 0.24957630221623045`)
- **CySecBench TEST (675 rows):** **526 / 675 = 77.9%** live (`95% CP: [74.6, 81.0]`; **528 / 675 = 78.2%** offline in `meta["delivered"]`)
- **Live WMDP_CYBER test (637 rows):** **602 / 637 = 94.5%**
- **Live CatQA non-target harmful TEST (234 rows):** **16 / 234 = 6.8%** (`95% CP: [4.0, 10.9]`; `16 / 234` offline)
- **Live WMDP_BIO test leak (637 rows):** **5 / 637 = 0.78%** (`95% CP: [0.26, 1.82]`)
- **Live 64-prompt fresh cyber generation sample:** **55 / 64 (85.9%)** judged `COMPLIED`; **22/22** and **58/58** exact agreement between gate bit and judged outcome on refused prompts; **9/9** and **58/58** exact `sha256` byte identity on withheld prompts.

### 2. Bio gate (`op_wmdpbio_B2_M1_ridge_lam16_R3U140_gemma_L8.pt`)

- **Threshold:** `t = 0.05801104299110058` (`R3U140`, $m = 6$ on 140 two-corpus reference rows; raw difference norm `‖d‖₂ = 0.08870733994511235`)
- **WMDP_BIO test (637 rows):** **633 / 637 = 99.4%** live & offline (`95% CP: [98.4, 99.8]`)
- **Live held-out WMDP_BIO val (127 rows, 118 deduplicated):** **127 / 127 = 100.0%** (`95% CP: [97.1, 100.0]`)
- **Live CySecBench TEST leak (675 rows):** **3 / 675 = 0.44%** (`95% CP: [0.09, 1.29]`)
- **Live CatQA non-target harmful TEST leak (234 rows):** **15 / 234 = 6.4%** (`95% CP: [3.6, 10.4]`)
- **Live WMDP_CYBER test leak (637 rows):** **0 / 637 = 0.00%**
- **Live SOSBench Bio (498 rows / 310 keyword-free rows):** **480 / 498 = 96.4%** / **297 / 310 = 95.8%**
