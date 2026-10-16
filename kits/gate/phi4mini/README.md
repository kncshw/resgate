# phi4mini — shipped gated-abliteration kit (`M1_ridge_lam16`)

Five files. Together they are everything the Phi-4-mini-instruct gate needs for both the
**cyber** and **bio** target domains.

| file | sha256 | what it is |
|---|---|---|
| `dir_phi4mini_sep_L13to15.pt` | `1678cca193eb4362f4f32a4d486c9030ba8c7a236bb1cf34b2426e9dd9dc2098` | **the edit, band only (`L13–15`).** Six keys — `{13,14,15}.{attn_out,mlp_out}` — of refusal direction and `sep(l)` weight |
| `op_phi4mini_L11_M1subcat240_M1_ridge_lam16_serve.pt` | `9605ed258b7b26330014e6fec6f1573e40e3642796c8eebd068b92c28166016c` | **the cyber gate (`M1_ridge_lam16`).** Classifier direction `c` (3072 float32), threshold `t = 0.05303228245782718`, `mean_user` at layer 11 |
| `op_wmdpbio_B2_M1_ridge_lam16_R3U140_phi4mini_L11.pt` | `a76960f6b20adaac4a3bd298e2d46f079a867b7b5b5049f75c603b1fe3d41ab0` | **the bio gate (`B2_M1_ridge_lam16`, shared L11 read).** Classifier direction `c` (3072 float32), threshold `t = 0.042208061573892744`, `mean_user` at layer 11 |
| `op_wmdpbio_B2_M1_ridge_lam16_R3U140_phi4mini_L12.pt` | `fda983562fe056c57eff71af63cbe18e4e7943b64439da50c5ee27906ee100d5` | **the bio gate (`B2_M1_ridge_lam16`, L12 comparator).** Classifier direction `c` (3072 float32), threshold `t = 0.041939858967326334`, `mean_user` at layer 12 |
| `band_spec.json` | `6efa553fbb2fdb95bf34958e046915a19e5a8512cc38049731b9d7e8b8d33de2` | the measured `sep(l)` profile this model's edit band (`L13–15`) was derived from |

## Operating points (`M1_ridge_lam16`: subcategory-centred scatter + Tikhonov ridge at $\alpha = \lambda_{16}$)

    model         microsoft/Phi-4-mini-instruct
    snapshot      cfbefacb99257ffa30c83adab238a50856ac3083
    pooling       mean_user
    gate layer    11                 (shared across cyber and bio; strictly below edit band L13–15)
    edit dose     K = 2.0            (3 of 32 layers, 6 target tensors)

### 1. Cyber gate (`op_phi4mini_L11_M1subcat240_M1_ridge_lam16_serve.pt`)

- **Threshold:** `t = 0.05303228245782718` (`iso_leakage`; in-fit R2 is `0.06984122876377935`, in-fit R3 is `0.04640487834614183`; raw difference norm `‖d‖₂ = 0.08843597893044577`)
- **CySecBench TEST (675 rows):** **549 / 675 = 81.3%** live & offline (`95% CP: [78.2, 84.2]`, $+13.5$\,pp / $+91$ prompts over `B0`)
- **Live WMDP_CYBER test (637 rows):** **506 / 637 = 79.4%** ($+13.2$\,pp / $+84$ prompts over `B0`)
- **Live CatQA non-target harmful TEST (235 rows):** **22 / 235 = 9.4%** (`95% CP: [6.0, 13.8]`; `21/235 = 8.9%` offline in `meta["leak"]`)
- **Live WMDP_BIO test leak (637 rows):** **2 / 637 = 0.31%** (`95% CP: [0.04, 1.13]`)
- **Live 64-prompt fresh cyber generation sample:** **55 / 64 (85.9%)** judged `COMPLIED`; **45/45** and **58/58** exact agreement between gate bit and judged outcome on refused prompts; **10/10** and **57/57** exact `sha256` byte identity on withheld prompts.

### 2. Bio gate (`op_wmdpbio_B2_M1_ridge_lam16_R3U140_phi4mini_L11.pt` & `L12` comparator)

- **Threshold:** `t = 0.042208061573892744` at `L11` (`0.041939858967326334` at `L12`; `R3U140`, $m = 6$; raw difference norms `‖d‖₂ = 0.06808023094800253` at `L11` and `0.07257941580865265` at `L12`)
- **WMDP_BIO test (637 rows):** **630 / 637 = 98.9%** live & offline at `L11` (`95% CP: [97.8, 99.6]`; **629 / 637 = 98.7%** at `L12`)
- **Live held-out WMDP_BIO val (127 rows, 118 deduplicated):** **127 / 127 = 100.0%** (`95% CP: [97.1, 100.0]`) at both `L11` and `L12`
- **Live CySecBench TEST leak (675 rows):** **0 / 675 = 0.00%** (`95% CP: [0.00, 0.55]`) at both `L11` and `L12`
- **Live CatQA non-target harmful TEST leak (235 rows):** **8 / 235 = 3.4%** at `L11` (`95% CP: [1.5, 6.6]`; **11 / 235 = 4.7%** at `L12`)
- **Live WMDP_CYBER test leak (637 rows):** **0 / 637 = 0.00%**
- **Live SOSBench Bio (498 rows / 310 keyword-free rows):** **473 / 498 = 95.0%** / **288 / 310 = 92.9%** at `L11` (**477 / 498 = 95.8%** / **291 / 310 = 93.9%** at `L12`)
- **Live-vs-offline decision flips across all 3,266 scored rows at `L11`:** **0 flips (100.00% agreement)**.
