# `2_evaluation/` — Prompts Used Strictly for Evaluation (Pass Rate vs. Leakage)

Every file in this folder is **strictly held out** from direction fitting and threshold calibration (`seed = 0`).

---

## 1. Cyber Gate Evaluation (`cyber_gate_eval/`)

### A. `pass_rate_targets/` — Target Cyber Prompts (Gate **SHOULD** Fire)
| File | Rows | Split Type | Description |
| :--- | :---: | :--- | :--- |
| `cysecbench_test_675_gemma_llama.json` | `675` | **Primary OOC TEST** | Confirm (`TEST`) half of CySecBench (`1350` rows) under the frozen Gemma/Llama/Qwen `split.npz` (`seed=20261007`). |
| `cysecbench_test_675_phi.json` | `675` | **Primary OOC TEST** | Confirm (`TEST`) half of CySecBench (`1350` rows) under the frozen Phi `split.npz` (`seed=20261006`). |
| `cysecbench_subset1_zero_attack_keywords_311.json` | `311` | **Lexical Control (Test 1)** | Subset of `cysecbench_test_675_gemma_llama.json` containing **0 of 22** explicit cyber-attack stems. |
| `cysecbench_subset1_zero_attack_keywords_300_phi.json` | `300` | **Lexical Control (Test 1)** | Subset of `cysecbench_test_675_phi.json` containing **0 of 22** explicit cyber-attack stems. |
| `cysecbench_subset1_complement_with_attack_keywords_364.json` | `364` | **Lexical Control (Test 1)** | Complement of Subset 1 (`675 - 311 = 364` rows containing $\ge 1$ of the 22 cyber-attack stems). |
| `cysecbench_subset2_lowest_jaccard_Q1_208.json` | `208` | **Lexical Control (Test 1)** | Lowest quartile (`Q1`, $\text{Jaccard} \le 0.100$) of content-word Jaccard similarity to the closest training prompt (`Gemma/Llama/Qwen` split). |
| `cysecbench_subset2_lowest_jaccard_Q1_200_phi.json` | `200` | **Lexical Control (Test 1)** | Lowest quartile (`Q1`, $\text{Jaccard} \le 0.100$) of content-word Jaccard similarity to the closest training prompt (`Phi` split). |
| `wmdp_cyber_test_637.json` | `637` | **Primary OOC TEST** | Cross-collection WMDP Cyber test split (`wmd_cyber_harmful_test.json`). |
| `incorpus_cyber_holdout_74.json` | `74` | **In-Corpus Holdout** | The remaining `74` `cyber_offense` prompts in `pool681` (`174 - 100 = 74`, `R.TE`) held out at `seed = 0`. |
| `cysecbench_val_675_gemma_llama.json` | `675` | **OOC VAL Half** | Validation (`SEL`) half of CySecBench for Gemma/Llama (unused in `M1`/`M3` fit). |
| `cysecbench_val_675_phi.json` | `675` | **OOC VAL Half** | Validation (`SEL`) half of CySecBench for Phi (unused in `M1`/`M3` fit). |
| `cysecbench_full_1350.json` | `1350` | **OOC Full** | Union of `VAL` (`675`) + `TEST` (`675`). |
| `wmdp_cyber_train_509_unseen_by_cyber_gate.json` | `509` | **Extra OOC Cyber** | `wmdp_cyber_train` (used only as negative control for the Bio Gate; `100%` unseen by the Cyber Gate). |
| `wmdp_cyber_val_127.json` | `127` | **Extra OOC Cyber** | `wmdp_cyber_val` (`100%` unseen by both gates). |

### B. `leakage_nontargets/` — Non-Target Harmful Prompts (Gate Should **NOT** Fire)
| File | Rows | Split Type | Description |
| :--- | :---: | :--- | :--- |
| `catqa_test_234_gemma_llama.json` | `234` | **Primary OOC Leak** | Confirm (`TEST`) half of CatQA (`469` rows across 10 non-cyber harm categories) for Gemma/Llama. |
| `catqa_test_235_phi.json` | `235` | **Primary OOC Leak** | Confirm (`TEST`) half of CatQA (`469` rows) for Phi. |
| `wmdp_bio_test_637.json` | `637` | **Off-Target OOC Leak** | WMDP Bio test split — measures cross-CBRN bio leakage of the Cyber Gate. |
| `sosbench_bio_498.json` | `498` | **Off-Target OOC Leak** | SOSBench biology (`498` rows). |
| `alert_weapon_bio_205.json` | `205` | **Off-Target OOC Leak** | ALERT biological weapons (`205` rows). |
| `alert_weapon_chem_212.json` | `212` | **Off-Target OOC Leak** | ALERT chemical weapons (`212` rows). |
| `alert_weapon_radio_157.json` | `157` | **Off-Target OOC Leak** | ALERT radiological weapons (`157` rows). |
| `wmdp_chem_test_205.json` | `205` | **Off-Target OOC Leak** | WMDP Chem test (`205` rows). |
| `incorpus_nontarget_leak_378.json` | `378` | **In-Corpus Leak** | The remaining `378` non-target harmful rows in `pool681` (`518 - 140 = 378`, `R.LK`) held out at `seed = 0`. |
| `catqa_val_235_gemma_llama.json` | `235` | **OOC VAL Leak** | Validation half of CatQA for Gemma/Llama. |
| `catqa_val_234_phi.json` | `234` | **OOC VAL Leak** | Validation half of CatQA for Phi. |
| `catqa_full_469.json` | `469` | **OOC Full Leak** | All `469` CatQA non-target harmful rows. |

### C. `dropped_benchmarks/` — Excluded from Cyber Pass-Rate Testing
| File | Rows | Reason for Exclusion |
| :--- | :---: | :--- |
| `pku_cybercrime_612_DROPPED_general_crime.json` | `612` | Originally sourced as a non-bio leak control for the Bio Gate. Excluded from Cyber Gate pass-rate evaluation because PKU-SafeRLHF's `"Cybercrime"` tag includes general crime, online scams, fraud, privacy violations, and piracy (which belong to the Cyber Gate's 14 negative `REF` categories). |

---

## 2. Bio Gate Evaluation (`bio_gate_eval/`)
* `pass_rate_targets/`: `wmdp_bio_test_637.json`, `wmdp_bio_val_127.json`, `sosbench_bio_498.json`, `alert_weapon_bio_205.json`, `salad_o39_197.json`, `harmbench_chembio_28.json`.
* `leakage_nontargets/`: `wmdp_chem_test_205.json`, `wmdp_chem_val_40.json`, `wmdp_cyber_test_637.json`, `pku_cybercrime_612.json`, `alert_weapon_chem_212.json`, `alert_weapon_radio_157.json`, `ailuminate_iwp_100.json`.
