# `dataset/` — Training & Evaluation Prompt Splits for ResGate

This directory contains the exact prompt splits used to fit the `M1_ridge_lam16` gate directions/thresholds (`1_train_direction/`) and to evaluate out-of-corpus pass rate, non-target leakage, and lexical/register invariance (`2_evaluation/`).

---

## Directory Structure

### 1. [`1_train_direction/`](1_train_direction/README.md) — Gate Direction & Threshold Fitting Splits
- **`cyber_gate_fit240_seed0/`**:
  - `pos_train_cyber_100.json`: `100` positive `cyber_offense` training rows (`R.TR`, `seed=0`).
  - `neg_ref_14cats_140.json`: `140` negative reference rows (`R.RF`, `14` non-cyber harm categories $\times$ `10` rows).
  - `by_negative_subcategory/<cat>_10.json`: Per-subcategory `10`-row files used for `M1` within-subcategory centring ($Z_{\text{subcat}}$).
- **`bio_gate_B2_fit/`**:
  - `pos_train_wmdp_bio_508.json`: `508` positive `wmdp_bio_train` rows.
  - `neg_train_wmdp_cyber_509.json`: `509` negative `wmdp_cyber_train` control rows (`403` covariance/contrast + `106` `REF_B`).
  - `neg_train_wmdp_chem_162.json`: `162` negative `wmdp_chem_train` control rows (`128` covariance/contrast + `34` `REF_B`).

### 2. [`2_evaluation/`](2_evaluation/README.md) — Strictly Held-Out Evaluation Splits
- **`cyber_gate_eval/pass_rate_targets/`** (Target Cyber Prompts — Gate **SHOULD** Fire):
  - `cysecbench_test_675_gemma_llama.json` (`675` rows, Gemma/Llama/Qwen test split) and `cysecbench_test_675_phi.json` (`675` rows, Phi test split).
  - **Lexical Control Subsets (Paper §8.1 / Test 1 — Bag-of-Words vs. Direction Alignment):**
    - `cysecbench_subset1_zero_attack_keywords_311.json` (`311` rows, Gemma/Llama/Qwen) & `cysecbench_subset1_zero_attack_keywords_300_phi.json` (`300` rows, Phi): Prompts containing **0 of 22** explicit cyber-attack stems.
    - `cysecbench_subset1_complement_with_attack_keywords_364.json` (`364` rows): Complement subset containing $\ge 1$ attack stem.
    - `cysecbench_subset2_lowest_jaccard_Q1_208.json` (`208` rows, Gemma/Llama/Qwen) & `cysecbench_subset2_lowest_jaccard_Q1_200_phi.json` (`200` rows, Phi): Lowest quartile (`Q1`, $\text{Jaccard} \le 0.100$) of content-word Jaccard similarity to the closest training prompt.
  - `wmdp_cyber_test_637.json` (`637` rows), `incorpus_cyber_holdout_74.json` (`74` rows), plus validation halves.
- **`cyber_gate_eval/leakage_nontargets/`** (Non-Target Harmful Prompts — Gate Should **NOT** Fire):
  - `catqa_test_234_gemma_llama.json` (`234` rows) & `catqa_test_235_phi.json` (`235` rows).
  - `incorpus_nontarget_leak_378.json` (`378` held-out non-cyber harmful rows).
  - Cross-CBRN non-cyber controls: `wmdp_bio_test_637.json`, `sosbench_bio_498.json`, `alert_weapon_bio_205.json`, `alert_weapon_chem_212.json`, `alert_weapon_radio_157.json`, `wmdp_chem_test_205.json`.
- **`bio_gate_eval/pass_rate_targets/`**:
  - `wmdp_bio_test_637.json`, `wmdp_bio_val_127.json`, `sosbench_bio_498.json`, `alert_weapon_bio_205.json`, `salad_o39_197.json`, `harmbench_chembio_28.json`.
- **`bio_gate_eval/leakage_nontargets/`**:
  - `wmdp_chem_test_205.json`, `wmdp_chem_val_40.json`, `wmdp_cyber_test_637.json`, `pku_cybercrime_612.json`, `alert_weapon_chem_212.json`, `alert_weapon_radio_157.json`, `ailuminate_iwp_100.json`.

See [`ROLE_MANIFEST.json`](ROLE_MANIFEST.json) for programmatic file paths and row counts.
