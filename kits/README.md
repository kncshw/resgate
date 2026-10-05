# kits/ — the five prompts the direction is built from

`minimal.json` is the whole fitting set: **1 harmful + 4 harmless**. It is the kit that produced
every `h006` number in `docs/RESULTS.md` — its harmful prompt is the one the research corpus calls
`h006`, and the file keys it `harm-patient`.

No reference model, no Optuna loop, no held-out split. Five strings go in; one direction per layer
and one `sep(l)` weight per layer come out.

## Format

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
| `text` | the bare user turn. The system prompt and the chat template are applied by `tempx capture` (`--sys`), so do **not** pre-format the string. |

Two rules the code enforces rather than assumes:

- **Both groups must be present.** `tempx direction` raises if either is missing.
- **At least 2 harmless prompts.** `sep(l)` divides by the harmless spread
  `mean_i‖g_i,l − ā_l‖`, which is *exactly* 0 for a single harmless prompt — the statistic is
  undefined, not merely small. The code refuses rather than emitting zeros.

One more thing worth knowing, because it is silent: **only the first harmful prompt, by sorted
`key`, contributes the gap.** Adding more harmful records to the file does not average them. If you
want several, that is a change to `src/tempx/direction.py`, not to the kit.

## How many prompts, and how much the choice matters

Short version: **the harmless side saturates at four and its format does not matter; the harmful
side is the one where the choice is load-bearing.** `docs/METHOD.md` ("How many prompts, and why")
has the measurements. The parts that bear on editing this file:

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

An earlier version of this project claimed one arbitrary harmful prompt is enough. **That claim was
too strong and is retracted.** What the measurements actually support:

- **Selection substitutes for quantity.** One *well-chosen* prompt reaches 0.9433 alignment —
  equivalent to **8–16 random ones**. Random harmful prompts, by contrast, keep improving all the
  way to n=79 (0.8415 → 0.9511). So the honest sample-efficiency figure on this side is about
  eight, not one.
- **Reference-free selectors do find the good prompt**, which is what keeps the kit blind:
  leave-one-out agreement correlates `r = +0.964` with alignment, boundary depth `r = +0.966`.
  Neither needs a reference model.
- **Arbitrary single prompts can also work — on the evidence of two draws.** The two harmful
  prompts this project shipped on (`h006`, in this file, and `h042`) were *not* reference-selected;
  one was picked by index. They gave statistically indistinguishable results (14/177 vs 15/177,
  p = 1.000). **Two draws are not a distribution.** The published prior on this is a lottery effect,
  not insensitivity: Dunefsky & Cohan ([arXiv:2502.18862](https://arxiv.org/abs/2502.18862)) report
  96.9% HarmBench ASR from a single example *and* that "choice of the single example is crucial".
- Going from 1 to 3 *selected* harmful prompts buys **+0.012** alignment (0.9433 → 0.9558). At this
  operating point the harmful side matters about an order of magnitude more than the harmless side.

So: swap the harmless prompts freely. Swap the harmful prompt and **re-measure** — do not assume
your replacement lands where `h006` did.

**And the one place the kit demonstrably changes the outcome:** a different harmful prompt can be
*better*, not merely equivalent. A domain-matched kit measured significantly more delivery than
`h006` at full dose (COMPLIED 85 vs 71, McNemar exact p = 0.0043) while being indistinguishable
from it at half dose (every McNemar p ≥ 0.72). See `docs/RESULTS.md` §2. The effect was only
visible at full dose, which is a warning about how you compare kits as much as about which kit to
use.

## Swapping in your own

```bash
cat > kits/mine.json <<'JSON'
[
 {"key":"h-mine",  "group":"harmful",  "text":"<one harmful instruction>"},
 {"key":"g-01",    "group":"harmless", "text":"<harmless 1>"},
 {"key":"g-02",    "group":"harmless", "text":"<harmless 2>"},
 {"key":"g-03",    "group":"harmless", "text":"<harmless 3>"},
 {"key":"g-04",    "group":"harmless", "text":"<harmless 4>"}
]
JSON

tempx capture   --model "$MODEL" --prompts kits/mine.json --out out/acts.pt
tempx direction --acts out/acts.pt --out out/dir.pt --mode sep --keep 0
```

Then read the two diagnostics `tempx direction` prints, before you spend a patch on the kit:

1. **`|cos(raw, centroid)|` after ortho must be ~0** (it prints as `max ...e-08` or smaller). If it
   is not, the orthogonalisation did not take and the direction is contaminated by the harmless
   centroid.
2. **The `sep(l)` profile should have structure** — a band of high-`sep` layers, not a flat line.
   A flat profile means `sep` is carrying no information and you are effectively running
   `--mode uniform`. The printed `ratio` (max/min) is the quick read.

A dose ladder is cheaper than a better kit: run `K = 1.0 / 1.25 / 1.5` and compare on `max|dW|`,
not on `K` (`docs/PITFALLS.md` §5). Keep `w = K · sep_l` under 2 — at `w = 2` the edit is a pure
reflection and restores the component at full magnitude (`docs/METHOD.md`, "What the update rule
actually does").

## What is not in here

- **No bio/chem/CBRN prompt text.** The domain-matched kit referenced above and in
  `docs/RESULTS.md` is not distributed, so its rows are honest measurements that **cannot be
  reproduced from this repo as shipped**. `minimal.json` can.
- **No harmful-behaviour corpus.** `minimal.json` holds exactly one harmful string; this repo is
  not a source of them. The evaluation set used in `docs/RESULTS.md` is the public
  `mlabonne/harmful_behaviors`, which you fetch yourself.
- **No prompt text from the private research corpora** beyond the one `h006` example in this file.
