# gemma4-26B — the shipped gated-abliteration kit

Three files. Together they are everything the gate needs that is not code.
Without them a clone of this repo can describe the gate but cannot run it.

| file | sha256 | what it is |
|---|---|---|
| `dir_gemma4-26B_sep.pt` | `15cb39c3bf3619d36cf76ad1c4590713fe58f0cdde921796e342390578a4f3f1` | **the edit.** Per-layer refusal directions and the `sep(l)` weights. Layout: `{'d': {'<l>.attn_out'\|'<l>.mlp_out': unit vector}, 'w': {same keys: float}, 'meta': {...}}` |
| `op_k5_L0.05_mean_user_L8.pt` | `fbc0f534520196e979a09a4b4e06a2fd1152b2e3ca76000162fa3ad760c27045` | **the gate.** Classifier direction `c`, threshold `t`, pooling, gate layer, and the fit metadata |
| `band_spec.json` | `4d2478f625931eeca3e733b87ae3e8e61a1ed0aa8694dd055fb331cc2739fb05` | the measured profile this model's band was derived from |

## The operating point

    category      cyber_offense
    pooling       mean_user          <- a threshold is in the units of its pooling
    gate layer    8
    threshold     0.140376451
    mode          trunc_whiten_k5_eigclip
    rule          k5_L0.05_frozen_train240

Target `K = 2.0`, edit layers **10-16**. Layers 0-9 are never modified, which is why
`h8` is identical in both arms and one forward pass can serve a conditional edit.

`docs/CLASSIFIER-MATH.md` is the spec: every constant above is derived there, with the
spectrum, the `k = 5` choice and the leak-constrained threshold rule written out.

## Read the counts correctly

The op file's meta carries `delivered: 878`, `n_target: 1350`, `leak: 16`,
`n_nontarget: 469`. **Those field names are wrong and the file says so itself** in
`counts_note`: *"EVERY count in this meta is a GATE-FIRING count (score >= thresh), not
a judged compliance count. No generation arm produced them."*

So read them as: 878 of 1350 out-of-corpus cyber prompts **fired the gate**, and 16 of
469 non-cyber prompts fired it. Whether the model then delivered harmful content is a
separate measurement and is not in this file.

The coverage difference is significant (McNemar p = 8.6e-24). **The leak difference is
not** (p = 1.0) and must not be claimed.

## Why `.gitignore` has an exception

The repo ignores `*.pt`, because captures and checkpoints do not belong in git. These
three files are 705 KB and they are the deliverable, so `!kits/**/*.pt` re-admits them.
