# eval/ — a refusal count is not an ablation score

Three instruments, and one rule for reading them.

| file | what it gives you | whose metric |
|---|---|---|
| `keyword_rate.py` | Heretic's 33-marker keyword refusal rate | **Heretic's** ([p-e-w/heretic](https://github.com/p-e-w/heretic)), reimplemented; verified against their source |
| `kl.py` | first-token KL divergence vs a baseline model | **Heretic's**, reproduced |
| `judge_threeway.py` | REFUSED / DEFLECTED / COMPLIED + coherence | **ours** |

The first two exist so your numbers are comparable with Heretic's published ones. The third
exists because those two numbers, on their own, will let you believe an ablation worked when
it did not.

Heretic is **AGPL-3.0-or-later**. `keyword_rate.py` and `kl.py` are derivatives of it (the marker
list and matching rules are copied, because an approximation would not be comparable), which is
why this repo is licensed AGPL-3.0-or-later to match.

## Why the third instrument

All numbers below are from one measured run: Gemma-4-26B-A4B-it, 90 targets,
`mlabonne/harmful_behaviors test[:100]`, Heretic's own unmodified `KeywordRate` and
`KLDivergence`, with our three-way judge applied to **the same response strings** the keyword
scorer saw — not a regeneration, so every cell is cross-tabulable.

| build | Refusals (keyword) | KL | REFUSED | DEFLECTED | COMPLIED |
|---|---|---|---|---|---|
| pristine `google/gemma-4-26B-A4B-it` | 100/100 | 0 | 96 | 3 | 1 |
| h006 kit, K=1.5 *(default)* | 13/100 | 0.3524 | 2 | 27 | 71 |
| CBRN kit, K=1.5 | 14/100 | 0.3896 | 0 | 15 | 85 |
| huihui-ai *(published, quoted model)* | 25/100 | 0.2104 | 4 | 32 | 64 |
| SevenOfNine *(published, Heretic-made)* | 29/100 | 0.0837 | 3 | 77 | 20 |

**1. The keyword count is mostly counting one incidental word.** Our h006 build at K=1.5
scores 13/100 "Refusals". Attribute those 13 matches (`keyword_rate.py --attribute`) and
**11 of the 13 fired on the bare word "illegal" and nothing else**. **Zero** of the 13
contained any marker that actually declines — no "sorry", no "i cannot", no "i won't". The
judge put **2** of the 100 in REFUSED. The same holds for the CBRN build: 14 matches, 11
containing "illegal", zero declines, and **0** hard refusals. The metric was largely
counting the word "illegal" inside answers that did the thing.

**2. Heretic's damage metric is best for the build that least often answers.** SevenOfNine
has the **lowest KL in the table — 0.0837, 4.7x lower than our CBRN build**. By Heretic's
own damage measure it is the least-damaged edit here. Yet it **deflects 77/100 and actually
delivers only 20/100**. Neither of Heretic's two numbers shows this: its refusal count (29,
in fact the highest of the four edited builds) and its KL both look like a careful edit. What
it is really doing is answering a neighbouring question at length. A metric made of 33
substrings cannot see deflection, because a long helpful-sounding non-answer contains no
marker.

**3. KL did not predict judged quality.** On 50 harmless prompts needing substantive replies,
judged blind and position-randomised by the pristine model, our CBRN K=1.5 build (KL 0.3896)
was **not distinguishable from pristine** (pairwise p = 0.28). huihui, at roughly half the KL
(0.2104), **was** worse than pristine (p = 0.0019). Higher KL, no detectable quality loss;
lower KL, detectable loss. Report KL for comparability; do not read it as a quality verdict.
Note also that Heretic measures KL on *harmless* prompts (`mlabonne/harmless_alpaca
test[:100]`) and only on the **first generated token** — not averaged over a continuation.

## The limit on the third instrument

The judge is not a ruler either. Hand it **byte-identical** response strings twice at
temperature 0 and the labels move. Measured here, on inputs confirmed identical by hash:

| set | n | labels changed | on the REFUSED/DEFLECTED boundary |
|---|---|---|---|
| pristine Gemma responses | 100 | 1 | 1 |
| h006 K=0.5 responses | 150 | 4 | 3 |
| CBRN K=1.0 responses | 150 | 1 | 1 |

The count tracks how many replies sit near that boundary: the K=0.5 build had 33 REFUSED +
89 DEFLECTED and flipped 4; the K=1.0 build had 1 REFUSED and flipped 1. **Budget roughly
1–4 labels in 150, and treat a difference of a few labels as unresolvable.** Measure it on
your own data with `--repeat 2` rather than taking our figure. Where a difference does need
to carry weight, pair the builds on the same prompts and use an exact paired test — the
h006-vs-CBRN full-dose gap is McNemar p = 0.0043 on COMPLIED, which clears this noise; the
same two kits at K=0.5 differ by every p ≥ 0.72, which does not.

Related cautions that apply to the judge: the rubric wording *is* the label boundary, so
`judge_threeway.py` refuses to run on edited rubric text (`--rubric-digest` to check); and if
your judge shares a base family with the model under test, it is grading its own lineage.
Both are recorded in the output JSON.

## Use

```bash
# Heretic-comparable refusal rate, plus the attribution that exposes a one-word score
python eval/keyword_rate.py responses.json --attribute
python eval/keyword_rate.py --selftest          # normalisation edge cases

# first-token KL against the unmodified checkpoint
python eval/kl.py --baseline <ref> --current <edited> --batch-size 8 --out kl.json

# three-way judge, over an OpenAI-compatible endpoint (nothing private is baked in)
export JUDGE_BASE_URL=http://localhost:8000/v1
export JUDGE_MODEL=<judge-model-id>
export JUDGE_API_KEY=<token>          # omitted from the request if unset
python eval/judge_threeway.py responses.json judged.json
python eval/judge_threeway.py responses.json noise.json --repeat 2   # measure judge noise
```

Input is a JSON list of records, or an object holding one under `runs` / `rows` / `labels` /
`records`; fields default to `prompt` and `response` (`PROMPT_KEY`, `RESPONSE_KEY`).

**Judge the strings the other metrics scored.** Do not regenerate between instruments. The
findings above exist only because the keyword flag and the three-way label were computed over
the same text, which is what makes a cross-tab such as "13 keyword refusals, 2 of which the
judge calls REFUSED" meaningful instead of two unrelated numbers.

## Reporting

Quote all three. A build is characterised by refusal rate **and** deflection rate **and**
compliance rate **and** drift — the single number anyone asks for does not exist. One
honest sentence looks like: *"13/100 on Heretic's keyword metric, but 11 of those 13 are the
word 'illegal'; the judge finds 2 REFUSED, 27 DEFLECTED, 71 COMPLIED; KL 0.3524 first-token
on harmless prompts."*

Do not use these numbers to claim general superiority over published abliterations. On
Heretic's own two metrics our builds are not uniformly better — both huihui and SevenOfNine
have lower KL. What the table supports is narrower and stated above.
