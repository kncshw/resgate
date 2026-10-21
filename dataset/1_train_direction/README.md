# `1_train_direction/` — Prompts Used to Fit Gate Directions & Thresholds

This folder contains **only** the prompts used to calculate the classifier direction $c$ and calibrate the order-statistic threshold $t$. None of the Out-of-Corpus (OOC) evaluation sets or In-Corpus Holdout sets appear here.

---

## 1. Cyber Gate (`cyber_gate_fit240_seed0/`)
Drawn from `pool681` (`DERIV` subsets: `advbench_train400`, `eval177`, `hb_test`, `bal150_only`) using `SPLIT_SEED_FIT = 0`:

| File / Subfolder | Rows | Role in Gate Math |
| :--- | :---: | :--- |
| `pos_train_cyber_100.json` | `100` | **Positive fit rows (`R.TR`):** Used to compute positive mean $\bar{x}_p$ and positive within-class scatter $Z_p = X_p - \mathbf{1}\bar{x}_p^\top$. Also used by rule `R2` as the candidate threshold grid. |
| `neg_ref_14cats_140.json` | `140` | **Negative reference rows (`R.RF`, `14` categories $\times$ `10` rows):** Used to compute negative mean $\bar{x}_n$, negative scatter $Z_n$, and the threshold budget $m = \lfloor L \cdot 140 \rfloor$ (`m=11` Gemma, `m=14` Phi, `m=17` Llama). |
| `by_negative_subcategory/<cat>_10.json` | `14 × 10` | **Per-subcategory negative splits (`M1`):** The exact 14 negative categories (`10` rows each) centered around their own category mean $\mu_c$ in `M1` ($Z_{\text{subcat}}$). |

---

## 2. Bio Gate (`bio_gate_B2_fit/`)
Used only by the **Bio Gate (`arm B2`)**:

| File | Rows | Role in Bio Gate (`B2`) Math |
| :--- | :---: | :--- |
| `pos_train_wmdp_bio_508.json` | `508` | **Positive fit rows:** `wmdp_bio_train` (`508` rows). |
| `neg_train_wmdp_cyber_509.json` | `509` | **Negative control rows (`wmdp_cyber_train`):** Split into `403` `REST_B` (for covariance/contrast) + `106` `REF_B` (for threshold `R3U140`). *(Never used in Cyber Gate training.)* |
| `neg_train_wmdp_chem_162.json` | `162` | **Negative control rows (`wmdp_chem_train`):** Split into `128` `REST_B` (for covariance/contrast) + `34` `REF_B` (for threshold `R3U140`). |
