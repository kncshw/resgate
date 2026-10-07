# phi4mini — the shipped gated-abliteration kit

Three files. Together they are everything the Phi gate needs that is not code.
Without them a clone of this repo can describe the gate but cannot run it.

| file | sha256 | what it is |
|---|---|---|
| `op_phi4mini_k5_L0.05_mean_user_L12.pt` | `66401d6aa01ce65ffa1e22ff2b9d3f9cc25bba68c9de0959d46e37c75b7d1381` | **the gate.** Classifier direction `c` (3072 float32), threshold `t`, pooling, gate layer, and the fit metadata |
| `dir_phi4mini_sep_L13to15.pt` | `1678cca193eb4362f4f32a4d486c9030ba8c7a236bb1cf34b2426e9dd9dc2098` | **the edit, band only.** Six keys — `{13,14,15}.{attn_out,mlp_out}` — of refusal direction and `sep(l)` weight. Layout: `{'d': {'<l>.attn_out'\|'<l>.mlp_out': unit vector}, 'w': {same keys: float}, 'meta': {...}}` |
| `band_spec.json` | `6efa553fbb2fdb95bf34958e046915a19e5a8512cc38049731b9d7e8b8d33de2` | the measured `sep(l)` profile this model's band was derived from |

## The operating point

    model         microsoft/Phi-4-mini-instruct
    snapshot      cfbefacb99257ffa30c83adab238a50856ac3083
    category      cyber_offense
    pooling       mean_user          <- a threshold is in the units of its pooling
    gate layer    12
    threshold     0.05713314461835679
    mode          trunc_whiten_k5_eigclip
    rule          R2_L0.05  /  op = phi4mini_k5_L0.05_frozen_fit240_L12
    k             5          (truncation rank of the whitening — NOT the dose, NOT a layer)
    L             0.05       (leak budget fraction; m = floor(0.05 x 140) = 7)

Target **`K = 2.0`**, edit layers **13–15**. Layers 0–12 are never modified, which is why
`h12` is identical in both arms and one forward pass can serve a conditional edit.

`docs/CLASSIFIER-MATH-PHI4MINI.md` is the spec: every constant above is derived there, with
the spectrum, the inherited `k = 5`, the leak-constrained threshold rule and the full
in-corpus / out-of-corpus numbers table.

## `k`, `K`, `L` and `L12` are four different things

A reader has already confused `k = 5` with "layer 5" and with "5 of 14 categories".

| token | what it is | value |
|---|---|---|
| `k` | truncation rank of the whitening, a count of eigendirections | 5 |
| `K` | abliteration **dose**, a multiplier on a weight edit | 2.0 |
| `L` | the leak **budget fraction** in the threshold rule | 0.05 |
| `L12` | the **read layer INDEX** | 12 |

`L12` and `L5` in this record are layer indices, not counts. The gate at layer 12 is still a
`k = 5` build.

## Read the counts correctly

The op file's meta carries `delivered: 65`, `n_target: 74`, `leak: 19`, `n_nontarget: 378`.
**Those field names are wrong and the file says so itself** in `counts_note`: *"EVERY count in
this meta is a GATE-FIRING count (score >= thresh) computed offline on stored activations. No
model was loaded, no text was generated, nothing was judged."*

So read them as: **65 of 74 held-out cyber prompts FIRED THE GATE**, and **19 of 378 held-out
non-cyber prompts fired it**. Whether the model then delivered harmful content is a separate
measurement and is not in this file. `baseline_leak` is `None` because the pristine model's own
willingness was never measured, so **no figure here can be apportioned between the edit and the
model**.

The misnaming is **enforced by our own code**: `tempx/src/tempx/gate.py:130` lists `delivered`
and `leak` in `META_FIELDS`, and `Gate.from_file` raises `ValueError` on a missing field. A file
with honest names would not load. Recorded as a defect in
`docs/CLASSIFIER-MATH-PHI4MINI.md` §6.

## The numbers, in one place

Every count is a gate firing. **In corpus** means disjoint rows of the *same* 1426-prompt
capture, which is the optimistic set.

| | coverage | leak | AUC |
|---|---|---|---|
| in corpus, split seed 0 | 65/74 = 0.878 `[0.7816, 0.9429]` | 19/378 = 0.0503 `[0.0305, 0.0774]` | 0.9819 |
| in corpus, 100-seed median [IQR] | 69/74 = 0.932 `[67, 70]` | 25/378 = 0.0661 `[19, 31]` | 0.9843 `[0.9812, 0.9875]` |
| **out of corpus, frozen** | **511/1350 = 0.3785** `[0.3526, 0.4050]` | **27/469 = 0.0576** `[0.0383, 0.0827]` | **0.9083** |

**The coverage intervals are disjoint and AUC fell too**, so the direction degrades out of
corpus, not merely the threshold's calibration. Leak does not get worse — its intervals overlap.

Benign false positives are **2/554 = 0.0036** at seed 0 and **5.0 `[2.0, 7.0]` of 554** across
100 seeds, **in corpus only**. There is **no out-of-corpus benign arm**: all 3176 fresh rows are
harmful-intent. The out-of-corpus benign FPR is **NOT MEASURED**.

**Never quote the aggregate leak alone.** Pooled over 100 in-corpus seeds, `fraud_financial`
carries **63.8%** of all fires by volume while the two highest *rates* are small and
cyber-adjacent — `critical_infrastructure` 24.5% and `surveillance_stalking` 15.7% — and five of
fourteen categories never fire. Out of corpus, Fraud/Deception leads on both at 9/49 = 0.1837.
Full tables in the spec, §7.4.

## The read layer was reverted, and that matters

This kit ships **layer 12**. An earlier build the same day shipped **layer 5**, selected by an
in-corpus argmax over 13 layers on 100 resplits of one corpus. A fresh out-of-corpus capture
reversed the sign at every point tested, and the two intervals do not overlap — in corpus
`[+2.72, +4.16]` leaks in L12's favour against out of corpus `[-21, -1]` in L5's. **What is dead
is the claim that L5 is better; at L5's own operating point the interval crosses zero and the two
layers are indistinguishable.** `docs/CLASSIFIER-MATH-PHI4MINI.md` §5 is the full record, and the
L5 op file is retained there unmodified as the artefact.

**The margin is ONE layer.** `min(edit) - read = 13 - 12 = 1`. Widen the band downward to include
L12 and this operating point is **invalid** and must be **refitted** at a shallower layer — not
re-thresholded, refitted, because `c` is layer-specific. The withdrawn L5 read had eight layers of
margin; that was its one genuine advantage and this revert gives it up knowingly.

## Two things in `band_spec.json` that are NOT the shipped values

`band_spec.json` is the **band derivation record**, not the gate configuration. Two of its fields
will mislead a reader who takes it as config:

- **`gate_layers: [0]`** is the band_spec's own *"shallowest layer whose mean one-vs-rest category
  AUC >= 0.80"* recommendation. **It is not the shipped read layer.** The shipped read is **12**,
  and it comes from the op file, which `Gate.from_file` reads it out of.
- **`direction_blob: "dir_phi4mini_sep.pt"`** names the **full 32-layer** blob (64 keys, 808 KB).
  This kit ships the **band-only subset**, `dir_phi4mini_sep_L13to15.pt` — 6 of those 64 keys,
  `d` and `w` copied **bit-for-bit** (`d_and_w_unmodified: true`, source md5
  `96d906bdbef543c2e4ca3e0a75b1d10c`). The other 58 keys are **absent**, so `tempx.patch` skips
  them via `if k not in dirs: continue`: **6 keys → 6 of 64 tensors patched.** Same edit, smaller
  file.

What `band_spec.json` *is* good for: the `sep(l)` and `w(l)` profiles the band was cut from
(`w(l) >= 0.9*max(w)`, hull taken for contiguity), the refusal peak at **L14**, the max:min
sharpness of **2.43x**, and the shuffled-label control. Its `NOTE` is the point: a fixed band is
unsupported because max:min sharpness spans 1.5x–3.6x across models.

## Why `.gitignore` has an exception

The repo ignores `*.pt`, because captures and checkpoints do not belong in git. These three
files are the deliverable, so `!kits/**/*.pt` re-admits them.
