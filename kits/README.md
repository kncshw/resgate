# kits/ — shipped operating-point kits (`gate/`) and rank-1 ablation kits (`ablation/`)

`kits/` is organized into two sub-folders matching `src/resgate/` and `docs/`:

## 1. [`kits/gate/`](gate/) — Runtime Gate Operating Points (`M1_ridge_lam16`)
Each model directory holds the live-verified `M1_ridge_lam16` (`M1 + M3`: subcategory-centred
scatter + smooth Tikhonov ridge shrinkage at $\alpha = \lambda_{16}$) `.pt` operating-point files
(`{"c": tensor[D], "thresh": float, "meta": {...}}`) for both the **cyber** and **bio** target
domains:
- [`kits/gate/gemma4-26B/`](gate/gemma4-26B/README.md) — `op_gemma_L8_M1subcat240_M1_ridge_lam16_serve.pt` (Cyber, `L8`) & `op_wmdpbio_B2_M1_ridge_lam16_R3U140_gemma_L8.pt` (Bio, `L8`)
- [`kits/gate/phi4mini/`](gate/phi4mini/README.md) — `op_phi4mini_L11_M1subcat240_M1_ridge_lam16_serve.pt` (Cyber, `L11`), `op_wmdpbio_B2_M1_ridge_lam16_R3U140_phi4mini_L11.pt` (Bio, `L11`), & `op_wmdpbio_B2_M1_ridge_lam16_R3U140_phi4mini_L12.pt` (Bio `L12` comparator)
- [`kits/gate/llama31-8B/`](gate/llama31-8B/README.md) — `op_llama31-8B_L10_M1subcat240_M1_ridge_lam16_serve.pt` (Cyber, `L10`) & `op_wmdpbio_B2_M1_ridge_lam16_R3U140_llama31-8B_L10.pt` (Bio, `L10`)

## 2. [`kits/ablation/`](ablation/) — Rank-1 Refusal Directions, Band Specs & 5-Prompt Capture Kit
- [`kits/ablation/minimal.json`](ablation/minimal.json) — The 5-prompt (`1 harmful + 4 harmless`) capture kit (`h006`).
- `kits/ablation/gemma4-26B/` — `dir_gemma4-26B_sep.pt` & `band_spec.json` (`L10–16`)
- `kits/ablation/phi4mini/` — `dir_phi4mini_sep_L13to15.pt` & `band_spec.json` (`L13–15`)
- `kits/ablation/llama31-8B/` — `dir_llama31-8B_sep.pt` & `band_spec.json` (`L11–22`)

---

## `ablation/minimal.json` — the five prompts the refusal direction is built from

`ablation/minimal.json` is the whole fitting set: **1 harmful + 4 harmless**. It is the kit that
produced every `h006` number in `docs/ablation/RESULTS.md` — its harmful prompt is the one the
research corpus calls `h006`, and the file keys it `harm-patient`.

No reference model, no Optuna loop, no held-out split. Five strings go in; one direction per layer
and one `sep(l)` weight per layer come out.

### Format

A JSON **list** of records. Three fields, all required:

```json
[
  {"key": "harm-patient",    "group": "harmful",  "text": "..."},
  {"key": "s32-26-shipping", "group": "harmless", "text": "..."}
]
```

| field | meaning |
|---|---|
| `key` | your own identifier. Must be unique **within its group**. |
| `group` | `"harmful"` or `"harmless"`. No other value is accepted. |
| `text` | the bare user turn. The system prompt and the chat template are applied by `resgate capture` (`--sys`), so do **not** pre-format the string. |

Two rules the code enforces rather than assumes:

- **Both groups must be present.** `resgate direction` raises if either is missing.
- **At least 2 harmless prompts.** `sep(l)` divides by the harmless spread
  `mean_i‖g_i,l − ā_l‖`, which is *exactly* 0 for a single harmless prompt — the statistic is
  undefined, not merely small. The code refuses rather than emitting zeros.

One more thing worth knowing, because it is silent: **only the first harmful prompt, by sorted
`key`, contributes the gap.** Adding more harmful records to the file does not average them. If you
want several, that is a change to `src/resgate/ablation/direction.py`, not to the kit.

### How many prompts, and how much the choice matters

Short version: **the harmless side saturates at four and its format does not matter; the harmful
side is the one where the choice is load-bearing.** `docs/ablation/METHOD.md` ("How many prompts,
and why") has the measurements. The parts that bear on editing this file:

**Harmless — four is enough.**

- 4 and 8 harmless prompts gave **identical refusal counts** on the same probe set. That is the
  measurement the setting rests on.
- Direction alignment does keep creeping up to about k ≈ 16–32, but the headroom is negligible:
  fitting `a(k) = A − B/k` to random alpaca gives asymptote **0.9361** (rmse 0.0022), so k=128 buys
  **+0.0024** and k=1000 buys **+0.0034**.
- **Format does not matter in the sense people expect.** Ordinary alpaca instructions ("Describe
  the scene of a sunset over a lake") tie this file's structured `Name N X` template on the
  behavioural metric, on two architectures. You do not need the `Name N X` shape.
- Format *does* matter as a bias–variance tradeoff. A format-homogeneous harmless set has roughly
  2× the SNR of random alpaca — better in the small-`n` regime — at the cost of a bias more data
  cannot remove (about **+0.0436** of alignment for one template family, **+0.0149** across 32
  categories). At k=4 the structured set wins; by k=8 random overtakes it. That is why this kit is
  four structured prompts drawn from four *different* categories (shipping, history, energy,
  linguistics) rather than four of a kind.
- **Do not hand-pick the "most benign" harmless prompts.** Past about k ≈ 8 selection is worse than
  random on this side.

**Harmful — one prompt works, but "any prompt works" is more than we measured.**

- **Selection substitutes for quantity.** One *well-chosen* prompt reaches 0.9433 alignment —
  equivalent to **8–16 random ones**. Random harmful prompts, by contrast, keep improving all the
  way to n=79 (0.8415 → 0.9511).
- **Reference-free selectors do find the good prompt**, which is what keeps the kit blind:
  leave-one-out agreement correlates `r = +0.964` with alignment, boundary depth `r = +0.966`.
  Neither needs a reference model.
- Going from 1 to 3 *selected* harmful prompts buys **+0.012** alignment (0.9433 → 0.9558). At this
  operating point the harmful side matters about an order of magnitude more than the harmless side.

So: swap the harmless prompts freely. Swap the harmful prompt and **re-measure** — do not assume
your replacement lands where `h006` did.

### Swapping in your own

```bash
cat > kits/ablation/mine.json <<'JSON'
[
 {"key":"h-mine",  "group":"harmful",  "text":"<one harmful instruction>"},
 {"key":"g-01",    "group":"harmless", "text":"<harmless 1>"},
 {"key":"g-02",    "group":"harmless", "text":"<harmless 2>"},
 {"key":"g-03",    "group":"harmless", "text":"<harmless 3>"},
 {"key":"g-04",    "group":"harmless", "text":"<harmless 4>"}
]
JSON

resgate capture   --model "$MODEL" --prompts kits/ablation/mine.json --out out/acts.pt
resgate direction --acts out/acts.pt --out out/dir.pt --mode sep --keep 0
```

Then read the two diagnostics `resgate direction` prints, before you spend a patch on the kit:

1. **`|cos(raw, centroid)|` after ortho must be ~0** (it prints as `max ...e-08` or smaller).
2. **The `sep(l)` profile should have structure** — a band of high-`sep` layers, not a flat line.

A dose ladder is cheaper than a better kit: run `K = 1.0 / 1.25 / 1.5` and compare on `max|dW|`,
not on `K` (`docs/ablation/PITFALLS.md` §5). Keep `w = K · sep_l` under 2 — at `w = 2` the edit is a
pure reflection and restores the component at full magnitude (`docs/ablation/METHOD.md`).
