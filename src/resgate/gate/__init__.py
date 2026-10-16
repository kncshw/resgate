"""Selective ablation at runtime: the same rank-1 PRE edit, applied only when the gate fires.

`patch.py` writes an edited checkpoint whose refusal is removed for EVERY category at once --
which is what every abliterated model does. This module applies the identical edit as a forward
hook, under a per-sequence condition read from an EARLY layer, so refusal is removed for ONE
designated category and left intact for the rest. One category unlocked instead of fifteen.
Nothing is written to disk and no second copy of any weight is made.

The equivalence is derived, not fitted. For an `nn.Linear` forward y = x @ W.T and the PRE step
of `patch.pre_step`, W' = W - w * outer(n (*) v, v @ (W/n)) with n = rownorm(W):

    y' = x @ W'.T  =  y - w * (x @ u)[..., None] * (n (*) v)        u = v @ (W / n)

So per target tensor there are exactly two precomputed vectors -- `u` of shape [in_features] and
w * (n (*) v) of shape [out_features] -- and the hook is one dot product and one scaled add,
O(in + out) per token instead of O(in * out). `tests/test_gate.py` asserts this against
`pre_step`'s own materialised weights; it is the load-bearing claim of the module, so if that
test fails nothing else here means anything.

Five properties, each of which is a trap:

  * The read is strictly UPSTREAM of the edit -- `max(gate_layers) < min(edit_layers)`, asserted
    at construction. The layers the gate reads are therefore identical in the edited and the
    unedited model, so the decision cannot be a function of its own effect.
  * The decision is PER SEQUENCE. The hook multiplies the correction by a [B] 0/1 mask; a row
    that did not fire is left exactly as the unedited module produced it, never the whole batch.
    A fused routed-expert stack is fed a flattened token list instead of a batch, so there the
    mask is EXPANDED to tokens -- and only where the expansion is derivable and checked, which
    is `_token_rows`; where it is not, that stack declines and counts itself in `unrouted`.
  * The decision is taken ONCE at prefill and frozen for the rest of generation. A mean over a
    growing sequence drifts, and a gate recomputed per token can flip mid-answer.
  * The read pools the USER-CONTENT SPAN, and that is a measured requirement, not a preference.
    `mean_all` -- every non-pad position, scaffold included -- reads category at 0.9394 AUC on
    Gemma-4-26B-A4B-it and at 0.5593-0.6959 on four other models, because a scaffold position
    carries an activation 11x-31x the user span's and nearly constant across prompts, so the mean
    is dominated by a near-constant (on Llama it is near-ORTHOGONAL to the user content, cos
    -0.047). `mean_user` reads 0.89-0.93 at the same layer on all five. The default is therefore
    `mean_user`; see docs/GATE.md and
    ablation/NOTES-2026-10-05-cross-model-ablation-vs-classify.md.
  * Pooling is MASK-WEIGHTED. `capture.py` pads LEFT, so an unweighted mean averages pad rows in,
    and the span's own offset is measured from the right.
  * Every failure lands on NO EDIT: no decision yet, `force=False`, no direction, no threshold, a
    user span that could not be located, a row layout the per-sequence mask cannot be applied to
    or expanded over, a fused stack whose apply site did not prove out, or any exception inside
    the gate. The pristine weights then answer, which means the model
    refuses. There is no path the other way -- in particular an unlocatable span does NOT fall
    back to `mean_all`, because that fallback is the defect, not the mitigation. Note the scope:
    this is the gate DECLINING when it cannot tell. It is not protection against the gate deciding
    confidently and wrongly, and measured twice on the build the published operating points come
    from, nothing else is either -- when the gate fires, the band-edited model delivers, with no
    residual refusal behind it (docs/GATE.md, "There is no second line of defence"). The threshold
    is the only thing handling that class of error.

Fused 3-D MoE expert stacks are reached by a SECOND apply site, and it is the one part of this
module whose correctness is not guaranteed by its fail-safes:

  * A fused stack is a 3-D parameter inside somebody's routing code, so there is no `nn.Linear`
    to hook, and the input to its down projection is a local variable inside a module-level
    function the class dispatches to at call time -- in transformers 5.5 that is
    `integrations.moe.grouped_mm_experts_forward`, selected from `config._experts_implementation`
    and NOT the experts class's own `forward` body. No forward hook can see that local. What a
    hook CAN see is the stack's own module boundary: the [T,H] token list and the [T,K] routing
    it was called with. So the down projection's input is RECOMPUTED there, from the stack's own
    up weight through the stack's own `_apply_gate`, once per routed (token, expert) pair; each
    pair's expert then contributes its own (u, w (n (*) v)), which `precompute` already produces
    for a whole stack in one broadcast. The recomputation pays the up projection a second time in
    the band's layers, and reaching the other 7 tensors is what that buys.
  * That recomputation is a claim about code this module does not own, and a WRONG one would be a
    confident wrong edit rather than no edit -- the one failure mode the rest of this file's
    design does not cover. So `attach` proves it per stack against the stack's own forward, on a
    probe, before it registers any hook, and declines the stack -- NOT REACHED, exactly as this
    module behaved before the fused path existed -- when the proof misses or when the stack does
    not carry the layout flags the arithmetic depends on. `_fused_recon_error` is that proof and
    `attach` logs the number it got.
  * Why it is worth the trouble. On this project's primary MoE checkpoint the band's 21 target
    tensors are 14 `nn.Linear` (`self_attn.o_proj` and `mlp.down_proj`, seven each) and 7 fused
    (`layers.N.experts.down_proj`) -- and in float64 those 7 hold 95.50% of the band's squared
    Frobenius edit energy and 57.02% of its per-token ROUTED edit. A runtime path that reached
    only the 14 was applying 42.98% of the intended per-token edit: published-abliteration
    coverage, not this project's, which is the asymmetry `docs/PITFALLS.md` 2 is about. A dense
    backbone has no fused stacks and nothing here applies to it -- 14 of 14.

One coverage gap is left, and it leaves the edit SMALLER than `patch()`'s, i.e. closer to
refusing, so it is a coverage gap and not a safety gap: ONE DIRECTION BLOB ONLY. A sequential
second blob needs the second step's `u` computed from the already-edited W, which is the one
thing a hook does not have. Patch offline for that.

And two constraints on where this can be used, neither of them a coverage gap.

It cannot locate the user span by itself. That needs the tokenizer and the prompt text, and a
forward pass carries neither, so the caller computes it with `user_span()` and hands it over with
`set_spans()` before each prefill. Not supplied is not assumed: the gate declines.

And the frozen
decision is per-Gate state, so ONE forward pass at a time per instance. Two concurrent forwards
through the same Gate would interleave their decisions, and a sequence could then be answered
under another sequence's decision -- which is the failure this whole module exists to prevent.
A batch is fine, and is the supported way to serve several sequences at once: the decision is a
[B] vector and the hook masks by row. Threads are not.

The rank-1 PRE row-normalised update rule follows Heretic (github.com/p-e-w/heretic) and Arditi et
al. 2024, arXiv:2406.11717; see `patch.py`. The pre-band runtime gate and its closed-form
subcategory-centred ridge classifier (`M1_ridge_lam16`: `fit_gate_direction`, `thresh_r2`,
`thresh_r3`, `fit_gate_op`) are specified in `docs/CLASSIFIER-MATH.md`.
"""
from __future__ import annotations

import inspect
import math
import os
import re
from typing import Any, Sequence

import torch

from ..ablation.enumerate_targets import EXCLUDE_PARTS, OUT_NAMES
from ..ablation.patch import key_of, layer_of, load_dirs

# Layer 8 and band 10-16 are Gemma-4-26B-A4B-it's, and only the READ generalises as measured:
# category AUC at L8 on pooling mean_user is 0.89-0.93 on all five models in
# ablation/NOTES-2026-10-05-cross-model-ablation-vs-classify.md, and L8 sits strictly below every
# one of their refusal bands. The BAND does not generalise -- those five span 7% to 25% of depth --
# so derive it per model from that model's own sep(l) profile and pass it in.
GATE_LAYER = 8
EDIT_LAYERS = tuple(range(10, 17))
# mean_user FIRST because it is the default: it is the only pooling measured to read the request on
# more than one model. See the module docstring and docs/GATE.md.
POOLS = ("mean_user", "mean_all", "last")
# An operating point without its provenance is uncomparable (docs/PITFALLS.md 3), so every one of
# these is required rather than defaulted: which category, which read the direction was built at,
# which point on the curve, and what that point measured.
META_FIELDS = ("category", "pool", "gate_layer", "op", "delivered", "n_target",
               "leak", "n_nontarget", "baseline_leak", "benign_fpr")
# The fused routed-expert path recomputes the input to a stack's down projection and PROVES that
# recomputation against the stack's own forward before it hooks anything. This is the bar it has
# to clear, as ||reconstructed - actual||_F / ||actual||_F. It is set between two MEASURED
# clusters rather than picked, both at the served dimensions (hidden 2816, intermediate 704,
# bfloat16, 8 to 256 probe tokens):
#   a correct reconstruction, rounding only   0.0 against `grouped_mm_experts_forward`, which is
#                                             the served dispatch, and at worst 6.7e-3 against
#                                             `batched_mm_experts_forward`, a genuinely different
#                                             kernel for the same algebra
#   a reconstruction MISSING A TERM           0.52 to 1.07 (a stack with a shared expert)
# So 0.1 sits ~15x above what rounding costs and ~5x below the smallest wrong answer seen. It is
# the Frobenius ratio and not max|.| because max|.| on that same batched_mm measurement is
# 1.05e-2 -- a couple of bfloat16 ulps on one element of a 2816-wide output -- which would leave
# any usable bar only 2x of headroom over rounding. A convention error is not subtle at either
# measure: a swapped gate/up half, a dropped routing weight, an off-by-one expert index or an
# unmodelled extra term all put it at O(1).
FUSED_RECON_TOL = 0.1


def user_span(
    tok,
    text: str,
    sys_prompt: str = "You are a helpful assistant.",
    think: str | bool | None = None,
) -> tuple[int, int, int, int] | None:
    """(a, u, b, tot) token counts around the USER-CONTENT span, or None if not located exactly.

    `capture_hf_v2.py`'s arithmetic, unchanged, because a gate read has to land on the same span
    the category direction was derived from: render the chat template, require the user text to
    appear exactly once, split there, and require the three token counts to sum to the whole.
    Every count uses add_special_tokens=False, as the capture does. It was exact for 1426/1426
    items on all five models measured, so it is not a pooling that silently degrades.

    None means NOT LOCATED. Pass it through to `Gate.set_spans` anyway: the gate then declines to
    decide for that sequence, which is no edit. It does NOT fall back to `mean_all`. The capture
    script does fall back, which is right for an offline measurement that records every fallback
    and wrong for a gate, whose fallback would be a silent read of a near-constant.
    """
    from ..ablation.capture import _think_kwargs       # lazy: `import resgate.gate` stays torch-only

    f = tok.apply_chat_template(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}],
        add_generation_prompt=True, tokenize=False, **_think_kwargs(think))
    if f.count(text) != 1:
        return None
    pre, _, suf = f.partition(text)

    def nt(x: str) -> int:
        return len(tok(x, add_special_tokens=False)["input_ids"])

    a, u, b, tot = nt(pre), nt(text), nt(suf), nt(f)
    return (a, u, b, tot) if a + u + b == tot else None


def _user_mask(mask: torch.Tensor, spans: Sequence[tuple[int, int, int, int] | None] | None
               ) -> torch.Tensor:
    """The [B,S] 0/1 mask selecting each row's user-content span and nothing else.

    LEFT padding, so the real tokens end at [-1] and the span starts at `off + a` with
    `off = S - n_real` -- `capture_hf_v2.py`'s own index arithmetic, including its per-item
    assertion that the span's own total equals the row's real-token count. Every way this can fail
    raises, and the caller turns that into "no decision", which is "no edit".
    """
    B, S = mask.shape
    if spans is None:
        raise ValueError("mean_user needs the user-content spans: call Gate.set_spans() with one "
                         "user_span() per sequence before the forward. No span, no read, no edit.")
    if len(spans) != B:
        raise ValueError("got %d span(s) for a batch of %d" % (len(spans), B))
    m = torch.zeros(B, S, dtype=torch.float32, device=mask.device)
    for j, sp in enumerate(spans):
        if sp is None:
            raise ValueError("user span was not located for row %d" % j)
        a, u, _, tot = sp
        n_real = int(mask[j].sum())
        if tot != n_real:
            raise ValueError("span total %d != %d real tokens in row %d: these spans are not this "
                             "batch's" % (tot, n_real, j))
        if u <= 0 or a < 0 or a + u > n_real:
            raise ValueError("span (a=%d, u=%d) does not fit %d real tokens in row %d"
                             % (a, u, n_real, j))
        off = S - n_real                                      # LEFT padding
        m[j, off + a: off + a + u] = 1.0
    return m


def pool(h: torch.Tensor, mask: torch.Tensor | None = None, how: str = "mean_user",
         spans: Sequence[tuple[int, int, int, int] | None] | None = None) -> torch.Tensor:
    """Pool [B,S,D] hidden states to [B,D], in float32, over the positions `how` names.

    `mask` is the 2-D [B,S] padding mask the model root is called with (1 = real token); None
    means every position is real, which is what `attention_mask=None` means to transformers.
    Padding is LEFT (`capture.py`), so `"last"` is `[:, -1, :]` -- the last real token of every
    row -- and any mean has to be weighted or it averages the pad rows in.

      mean_user  the USER-CONTENT span only: no system prompt, no chat scaffold, no padding.
                 The default, and the only one measured to read the request on more than one
                 model. Needs `spans`.
      mean_all   every non-pad position, scaffold included. Validated on Gemma and does NOT
                 transfer: on four other models a scaffold position carries an activation 11x-31x
                 the user span's and nearly constant across prompts, so this mean is dominated by
                 a near-constant and the category read falls to 0.56-0.70 AUC.
      last       the final real position -- a fixed template token. What the repo's own `capture`
                 and `direction` steps use for the REFUSAL direction, which is a different
                 measurement from this one.

    A mask that is not the [B,S] padding mask raises: a 4-D causal mask cannot tell a pad from a
    real token, and scoring a pooled vector that includes pads is a wrong read, not a small one.
    """
    if how not in POOLS:
        raise ValueError("pool must be one of %s, not %r" % (POOLS, how))
    if how == "last":
        return h[:, -1, :].float()
    if mask is None:
        mask = h.new_ones(h.shape[:2])
    if tuple(mask.shape) != tuple(h.shape[:2]):
        raise ValueError("mask %s is not the [B,S] padding mask for hidden states %s"
                         % (tuple(mask.shape), tuple(h.shape)))
    m = (mask.float() if how == "mean_all" else _user_mask(mask, spans)).unsqueeze(-1)
    return (h.float() * m).sum(1) / m.sum(1).clamp_min(1e-9)


def precompute(W: torch.Tensor, v: torch.Tensor, w: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The two vectors the hook needs: (u, w * (n (*) v)) with n = rownorm(W), u = v @ (W/n).

    2-D [out,in] and fused 3-D [experts,out,in] alike, by the same broadcast `pre_step` uses.
    The dose is folded into the second vector, so the hook is one dot product and one scaled add.
    """
    n = W.norm(dim=-1).clamp_min(1e-12)
    return v @ (W / n.unsqueeze(-1)), w * (n * v)


def _fused_x(mod: torch.nn.Module, h: torch.Tensor, e: int) -> torch.Tensor:
    """The input to a fused stack's down projection, for rows `h` all routed to expert `e`.

    Taken from the stack itself -- its own up weight, its own `_apply_gate` or `act_fn` -- so the
    only thing assumed here is the LAYOUT: [experts, out, in], multiplied against `[e].T`, which
    is `nn.Linear`'s convention and `transformers.integrations.moe._grouped_linear`'s
    `is_transposed=False`. `_fused_refusal` reads that flag off the module rather than hoping,
    and `_fused_recon_error` proves the whole reconstruction against the stack's own forward.
    """
    W = mod.gate_up_proj if mod.has_gate else mod.up_proj
    g = h @ W[e].transpose(-2, -1)
    return mod._apply_gate(g) if mod.has_gate else mod.act_fn(g)


def fused_correction(mod: torch.nn.Module, h: torch.Tensor, idx: torch.Tensor,
                     wts: torch.Tensor, u: torch.Tensor, wnv: torch.Tensor,
                     m: torch.Tensor) -> torch.Tensor:
    """sum_k w[t,k] * (x[t,k] @ u_e) * (n (*) v)_e per token: `_edit_hook`'s edit on a fused stack.

    A fused stack's rows are routed (token, expert) PAIRS, and each pair's expert has its own
    (u, w (n (*) v)) -- which is why one direction key covers all 128 experts of a layer and the
    correction still differs per expert. `m` is the [T] 0/1 expansion of the per-sequence
    decision, folded into the scalar exactly as the 2-D hook folds its [B] mask, so a token that
    did not fire contributes a hard zero and its row of the output is the unedited stack's, bit
    for bit.

    The pair loop runs over the experts the FIRED tokens routed to, and over their rows only, so
    a row that did not fire is never put through `_fused_x` at all and its scalar stays the zero
    it was initialised to. That is bit-identity by construction rather than by multiplying by
    zero -- the multiply stays, as the second guard and as the statement of what the mask is --
    and on a selective gate, where most sequences do not fire, it is also most of the work.

    Two costs, both deliberate. Driving the loop from Python is one device sync per stack per
    forward; and `_fused_x` pays the up projection a second time in the band's layers. Reaching
    the other 7 tensors is what they buy. Neither has been measured on the served checkpoint.

    The accumulation is a sum over the K routing slots, NOT an `index_add_` over duplicate token
    indices: that would make the served arm differ from itself run to run, because its CUDA
    atomics reorder low-precision adds. `grouped_mm_experts_forward` avoids the same thing for
    the same reason, and in its own comment.
    """
    T, K = idx.shape
    s = h.new_zeros(T, K)
    on = m.bool()
    for e in idx[on].unique().tolist():
        sel = (idx == e) & on.unsqueeze(-1)
        rows = sel.nonzero(as_tuple=True)[0]
        x = _fused_x(mod, h[rows], e)
        s[sel] = (x * u[e].to(x.dtype)).sum(-1).to(s.dtype)
    s = s * wts.to(s.dtype) * m.to(s.dtype).unsqueeze(-1)
    corr = h.new_zeros(T, wnv.shape[-1])
    for k in range(K):
        corr += s[:, k, None] * wnv[idx[:, k]].to(corr.dtype)
    return corr


def _fused_recon_error(mod: torch.nn.Module, n_tok: int = 16) -> float:
    """||reconstruction - the stack's own forward||_F / ||its own forward||_F. The positive control.

    `fused_correction` subtracts a rank-1 term from an output it did not compute, so it is an edit
    of this stack only if `_fused_x` really is the input to this stack's down projection. That is
    a claim about somebody else's routing code, and in transformers 5.5 the body that runs is not
    even the experts class's own `forward` -- the class is wrapped by
    `integrations.moe.use_experts_implementation`, whose `forward` dispatches on
    `config._experts_implementation` at call time. So the claim is MEASURED, per stack, on a
    probe of `n_tok` tokens routed to the first and last expert in equal numbers -- equal so the
    grouped kernel's group offsets stay aligned -- and measured BEFORE a single hook exists.

    This is the one claim fail-closed does not cover: a wrong reconstruction would be a confident
    WRONG edit, not a smaller one. `attach` therefore declines any stack whose error exceeds
    `FUSED_RECON_TOL` and reports it NOT REACHED, which is this module's behaviour before the
    fused path existed. READ THE NUMBER IT LOGS -- it is the proof, not a formality.
    """
    W = mod.down_proj
    up = mod.gate_up_proj if mod.has_gate else mod.up_proj
    es = [0, W.shape[0] - 1] if W.shape[0] > 1 else [0]
    g = torch.Generator().manual_seed(0)
    h = torch.randn(n_tok, up.shape[-1], generator=g).to(W.device, W.dtype)
    idx = torch.tensor([es] * n_tok, device=W.device)
    wts = torch.tensor([[0.6, 0.4][:len(es)]] * n_tok, device=W.device, dtype=W.dtype)
    with torch.no_grad():
        ref = mod(h, idx, wts)
        mine = torch.zeros_like(ref)
        for e in es:
            sel = idx == e
            rows = sel.nonzero(as_tuple=True)[0]
            y = _fused_x(mod, h[rows], e) @ W[e].transpose(-2, -1)
            mine.index_add_(0, rows, (y * wts[sel].to(y.dtype).unsqueeze(-1)).to(mine.dtype))
    den = float(ref.float().pow(2).sum()) ** 0.5
    return float((mine.float() - ref.float()).pow(2).sum()) ** 0.5 / max(den, 1e-30)


def _fused_refusal(mod: torch.nn.Module, W: torch.Tensor, nv: int) -> str | None:
    """Why this fused stack cannot be driven, or None when it can. Refuse rather than assume.

    Every branch is something the correction depends on and cannot read off the parameter: which
    way round the stack is stored, whether the up projection is gated, whether a per-expert bias
    rides along. `use_experts_implementation` sets all three as instance attributes, so they are
    READ; a 3-D parameter that does not carry them is somebody else's routing code and is left to
    `patch()` with its name and the reason printed.
    """
    for a in ("has_gate", "has_bias", "is_transposed", "num_experts"):
        if not hasattr(mod, a):
            return "%s has no .%s, so it is not a transformers fused expert stack" % (
                type(mod).__name__, a)
    if mod.is_transposed:
        return ("is_transposed=True stores the stack [experts,in,out]; precompute, pre_step and "
                "the patched checkpoints all read [experts,out,in]")
    if mod.has_bias:
        return ("has_bias=True: the correction itself is bias-independent, but the "
                "reconstruction that proves the apply site would have to model the bias, and an "
                "unproven apply site is not one")
    if int(mod.num_experts) != W.shape[0]:
        return "num_experts %d is not the stack's %d experts" % (mod.num_experts, W.shape[0])
    if nv != W.shape[1]:
        return "direction %d != out dim %d" % (nv, W.shape[1])
    nm = "gate_up_proj" if mod.has_gate else "up_proj"
    up = getattr(mod, nm, None)
    if not isinstance(up, torch.Tensor) or up.dim() != 3:
        return "no 3-D %s to recompute the down projection's input from" % nm
    if tuple(up.shape[:2]) != (W.shape[0], (2 if mod.has_gate else 1) * W.shape[2]):
        return "%s %s does not feed a [%d,%d,%d] down projection" % (
            nm, tuple(up.shape), W.shape[0], W.shape[1], W.shape[2])
    if not callable(getattr(mod, "_apply_gate" if mod.has_gate else "act_fn", None)):
        return "no %s to take the gate from" % ("_apply_gate" if mod.has_gate else "act_fn")
    return None


def _precompute_fused(W: torch.Tensor, v: torch.Tensor, w: float, dt: torch.dtype,
                      chunk: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """`precompute` over a fused [experts,out,in] stack, `chunk` experts at a time.

    The same arithmetic -- the broadcast is per expert and mixes none of them -- but `precompute`
    needs a float32 cast of W and then W/n beside it, and a stack is 1.0 GB of bf16 on this
    project's primary checkpoint, so doing it whole costs ~2 GB transient per layer. `attach`
    runs again on every `/compare` call, so that transient is paid repeatedly on a box whose
    119.6 GiB is one pool shared with the host. Chunked it is ~130 MB, and what is KEPT is
    u [experts,in] and w (n (*) v) [experts,out], under 1 MB a layer.
    """
    us, wnvs = [], []
    for i in range(0, W.shape[0], chunk):
        u, wnv = precompute(W[i:i + chunk].to(dt), v, w)
        us.append(u.to(W.dtype)); wnvs.append(wnv.to(W.dtype))
        del u, wnv
    return torch.cat(us), torch.cat(wnvs)


def _is_prefill(kwargs: dict) -> bool:
    """True when this forward starts a sequence, i.e. the KV cache is empty.

    `generate` calls the model once per token, so "decide at prefill" has to be recognised rather
    than assumed. A Cache of length 0, a legacy empty tuple and no cache at all all count.
    Generating with `use_cache=False` re-feeds the whole sequence every step and so has no
    prefill to recognise: every step then looks like one and the decision is retaken. Generate
    with a cache (the default) if you want the freeze.
    """
    pkv = kwargs.get("past_key_values")
    if pkv is None: return True
    n = getattr(pkv, "get_seq_length", None)
    return n() == 0 if callable(n) else not pkv


class Gate:
    """One category unlocked instead of all of them: the PRE edit behind a per-sequence read.

    Construct from a direction blob (`patch.py`'s `{"d","w"}`), a category direction `c` in the
    read layer's residual space, a threshold on cos(pooled, c), and a band spec; then
    `attach(model)`, and `detach()` when done. `from_file` is the loading path: the direction and
    the threshold come out of an operating-point file that carries its own provenance, never out
    of this source.

    The read is ONE layer, because an operating-point file carries one direction and a cosine
    against another layer's direction measures nothing; `gate_layers` is a tuple so the band
    assertion is the general `max(read) < min(edit)`.

    A threshold belongs to a pooling and a layer: it is only meaningful against scores produced
    the same way. Prefer `from_file`, which takes both from the operating point's own provenance,
    over constructing with a threshold from one read and the default `pooling` of another.

    Observable state: `fired` is the frozen per-sequence decision ([B] bool; None means no
    decision, which every hook reads as "do not edit"), `scores` the cosines it came from,
    `last_error` the last exception the gate swallowed, `errors` and `unrouted` their counts.
    `force` overrides the read -- None scores normally, True always edits (the parity test needs
    it), False never edits, which is the disabled state.
    """

    def __init__(
        self,
        dirs: str | Sequence[str | os.PathLike | dict],
        c: torch.Tensor | None,
        thresh: float | None,
        gate_layer: int = GATE_LAYER,
        edit_layers: Sequence[int] = EDIT_LAYERS,
        K: float = 1.0,
        pooling: str = "mean_user",
        meta: dict[str, Any] | None = None,
    ) -> None:
        blobs = load_dirs(dirs)
        if not blobs: raise ValueError("no direction blob given")
        if len(blobs) > 1:
            raise ValueError(
                "the gate takes ONE direction blob: a sequential second step needs its u "
                "computed from the already-edited W, which a hook never sees. Use patch() "
                "offline for sequential directions.")
        self.d: dict[str, torch.Tensor] = blobs[0]["d"]
        miss = [k for k in self.d if k not in blobs[0]["w"]]
        if miss:
            raise ValueError("blob has no per-layer weight for %d direction key(s), e.g. %r"
                             % (len(miss), miss[0]))
        self.K = float(K)
        self.w = {k: float(blobs[0]["w"][k]) * self.K for k in self.d}
        self.c = None if c is None else c.detach().float().flatten()
        self.thresh = None if thresh is None else float(thresh)
        if pooling not in POOLS:
            raise ValueError("pooling must be one of %s, not %r" % (POOLS, pooling))
        self.pooling = pooling
        self.gate_layers = (int(gate_layer),)
        self.edit_layers = tuple(sorted(int(l) for l in edit_layers))
        if not self.edit_layers: raise ValueError("edit_layers is empty")
        if max(self.gate_layers) >= min(self.edit_layers):
            raise ValueError(
                "the read must be strictly upstream of the edit: read %s against edit band "
                "%d-%d. A gate reading a layer the edit touches is reading its own effect, and "
                "the route-invariance argument for a single forward pass is then gone."
                % (list(self.gate_layers), self.edit_layers[0], self.edit_layers[-1]))
        self.meta = dict(meta or {})
        self.force: bool | None = None
        self.fired: torch.Tensor | None = None
        self.scores: torch.Tensor | None = None
        self.last_error: str | None = None
        self.errors = 0
        self.unrouted = 0
        self._live = False
        self._decided = False
        self._spans: Sequence[tuple[int, int, int, int] | None] | None = None
        self._mask: torch.Tensor | None = None
        self._mask_known = False
        self._sig: inspect.Signature | None = None
        self._has_mask = False
        self._handles: list = []

    # --- loading -----------------------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        dirs: str | Sequence[str | os.PathLike | dict],
        op: str | os.PathLike | dict,
        K: float = 1.0,
        edit_layers: Sequence[int] = EDIT_LAYERS,
    ) -> "Gate":
        """Gate from a direction blob and an operating-point file. Nothing here is hardcoded.

        The file is `{"c": tensor[D], "thresh": float, "meta": {...}}` and its meta must carry
        every field in `META_FIELDS`: the category, the read the direction was built at (`pool`
        and `gate_layer`), the operating point's name, and what that point measured
        (`delivered`/`n_target`, `leak`/`n_nontarget`, the pristine `baseline_leak`, the
        `benign_fpr`). A missing field is refused rather than defaulted -- an operating point
        whose provenance is not recorded is not an operating point.

        The read layer AND the pooling come from the file, so the direction and the read it was
        built with cannot drift apart; the edit band is the caller's.

        That the file wins over this module's default is not a convenience. A threshold is in the
        units of the pooling that produced the scores it was fitted on, and the published cyber
        operating points are `mean_all` at layer 8 while the default read is `mean_user`: scoring
        one and thresholding against the other is a units error, not a small bias. See
        `docs/GATE.md`, "These thresholds are mean_all L8 numbers".
        """
        o = op if isinstance(op, dict) else torch.load(op, map_location="cpu")
        m = o.get("meta") or {}
        gone = [f for f in META_FIELDS if f not in m]
        if gone:
            raise ValueError("operating point is missing provenance field(s): %s"
                             % ", ".join(gone))
        if o.get("c") is None or o.get("thresh") is None:
            raise ValueError("operating point needs both 'c' (the category direction) "
                             "and 'thresh'")
        return cls(dirs, o["c"], o["thresh"], gate_layer=int(m["gate_layer"]),
                   edit_layers=edit_layers, K=K, pooling=str(m["pool"]), meta=m)

    # --- the read ----------------------------------------------------------------------

    def reset(self) -> None:
        """Forget the frozen decision. A prefill does this; so can the caller.

        It does NOT clear the spans: a prefill resets through here BEFORE the read runs, so
        clearing them here would throw away exactly what the read is about to need. The read
        clears them once it has consumed them.

        It DOES clear `_decided`, which is what re-arms the read for the next prefill. The
        read sets that flag in a `finally`, which runs after the exception handler calls this,
        so the ordering holds: a raising read still ends up decided.
        """
        self.fired = None
        self.scores = None
        self._live = False
        self._decided = False

    def set_spans(self, spans: Sequence[tuple[int, int, int, int] | None]) -> None:
        """Hand the next prefill its user-content spans, one `user_span()` per sequence, in order.

        `mean_user` cannot be computed from a forward pass alone -- locating the span needs the
        tokenizer and the prompt text, neither of which a hook can see -- so the caller supplies
        it. The read CONSUMES these: after it has decided they are cleared, so a second prefill
        that forgets to set them declines rather than reusing the first one's.
        """
        self._spans = list(spans)

    def score(self, p: torch.Tensor) -> torch.Tensor:
        """cos(pooled, c) per sequence: [B,D] -> [B].

        A cosine and not a raw projection, for the reason `sep(l)` is a ratio: residual norms
        grow with depth and vary per prompt, so an unnormalised dot product is not comparable
        across prompts and a threshold on it is not an operating point.
        """
        c = self.c.to(p.device)
        c = c / c.norm().clamp_min(1e-12)
        return (p / p.norm(dim=-1, keepdim=True).clamp_min(1e-12)) @ c

    def decide(self, h: torch.Tensor, mask: torch.Tensor | None,
               spans: Sequence[tuple[int, int, int, int] | None] | None = None
               ) -> torch.Tensor | None:
        """Per-sequence decision from the read layer's output: [B,S,D] -> [B] bool, or None.

        None is "no decision", which is what every edit hook treats as "do not edit".
        """
        if self.force is not None:
            self.scores = None
            return torch.full((h.shape[0],), bool(self.force), dtype=torch.bool, device=h.device)
        if self.c is None or self.thresh is None:
            return None
        s = self.score(pool(h, mask, self.pooling, spans))
        self.scores = s
        return s >= self.thresh

    def _root_pre_hook(self, mod, args, kwargs):
        """Capture the padding mask and spot a prefill, which is the only thing that re-decides.

        The call is bound against the forward signature rather than read out of `kwargs`, because
        `attention_mask` passed positionally is still the padding mask and missing it would pool
        over pads. A signature that does not name `attention_mask` leaves the mask UNKNOWN, which
        is not the same as "no padding": `_read_hook` then declines to decide.
        """
        self._mask, self._mask_known = None, False
        try:
            try:
                got = self._sig.bind_partial(*args, **kwargs).arguments
            except TypeError:                                   # an opaque *args/**kwargs forward
                got = dict(kwargs)
            self._mask = got.get("attention_mask")
            self._mask_known = self._has_mask or "attention_mask" in got
            if _is_prefill(got): self.reset()
        except Exception as e:                                  # fail-safe: no read -> no edit
            self._note(e)
            self._mask, self._mask_known = None, False
            self.reset()
        return None

    def _read_hook(self, mod, args, out):
        """Decide once from the read layer's OUTPUT, then leave the decision alone.

        `capture.py`'s convention is that acts[l] is the output of layer l (transformers'
        `hidden_states` drops index 0), so a forward hook on `layers[l]` reads exactly what the
        direction was built from -- the off-by-one here would be silent.
        """
        # Frozen: decode changes nothing. `_decided` and not just `fired`, because a read that
        # DECLINED is as final as one that succeeded -- it leaves `fired` None, so keying the
        # freeze on `fired` alone let every decode token retry a read that cannot work. Each
        # retry overwrote `last_error` with a downstream shape complaint, hiding the real cause,
        # and counted itself in `errors`, so one fault read as one-per-token. Worse, `pool` does
        # not raise for every pooling: "last" returns h[:, -1, :] unchecked, so a retry could
        # set `fired` from a GENERATED token. Decline once, stay declined.
        if self._decided or self.fired is not None: return None
        if self.force is None and self.pooling != "last" and not self._mask_known:
            # Not an error in the model, but it is one here: both means need the mask -- one to
            # avoid averaging pads in, the other to find where the real tokens start -- and
            # pooling a possibly padded batch without it is a wrong read, not an approximate one.
            # "last" needs no mask, left padding putting the last real token at [-1] for every
            # row, and `force` skips the read altogether, so it skips this with it.
            self._note(RuntimeError("attention_mask could not be read from the forward call"))
            self._decided = True
            return None
        try:
            h = out[0] if isinstance(out, tuple) else out
            f = self.decide(h, self._mask, self._spans)
            if f is None: return None
            self.fired = f
            self._live = bool(f.any())                      # one sync per prefill, not per token
        except Exception as e:                                  # fail-safe: no read -> no edit
            self._note(e)
            self.reset()
        finally:
            self._spans = None                              # consumed, whatever the outcome
            self._decided = True                            # decided, declined or raised alike
        return None

    def _note(self, e: Exception) -> None:
        self.errors += 1
        self.last_error = repr(e)

    # --- the edit ----------------------------------------------------------------------

    def _edit_hook(self, u: torch.Tensor, wnv: torch.Tensor):
        """y - w * (x @ u)[..., None] * (n (*) v), on the rows that fired and no others."""
        def hook(mod, args, out):
            if self.fired is None or not self._live: return None     # no decision -> no edit
            try:
                if not args or not isinstance(out, torch.Tensor): return None
                f = self.fired
                if out.dim() < 2 or out.shape[0] != f.shape[0]:
                    # The leading dim is not the batch -- a flattened expert token list, say.
                    # A per-sequence mask cannot be applied to rows that are not sequences, and
                    # applying it to the whole batch is the one thing this must never do.
                    self.unrouted += 1
                    return None
                x = args[0]
                t = x @ u.to(x.dtype)
                m = f.to(t.dtype).view(-1, *([1] * (t.dim() - 1)))
                return out - (t * m).unsqueeze(-1) * wnv.to(out.dtype)
            except Exception as e:                                   # fail-safe: no edit
                self._note(e)
                return None
        return hook

    def _token_rows(self, n_tok: int) -> torch.Tensor | None:
        """The [T] per-token expansion of the [B] decision, or None -- counted -- if unsound.

        A fused stack is handed a FLATTENED token list, so the per-sequence mask has to be
        expanded rather than applied. The only layout that expansion is sound for is the one the
        decoder layer produces: a row-major `[B,S,H] -> [B*S,H]` reshape of a rectangular batch,
        where token t belongs to sequence t // (T // B). So that is DERIVED and CHECKED -- T must
        divide by B, and when the padding mask is known the per-sequence length has to be the
        mask's own width or 1, a decode step. Anything else declines and counts itself, exactly
        as `_edit_hook` does for a row layout that is not a batch: mis-expanding this mask would
        answer one sequence under another sequence's decision, which is the failure this whole
        module exists to prevent, and it would be silent.
        """
        f = self.fired
        B = f.shape[0]
        if B == 0 or n_tok % B:
            self.unrouted += 1
            return None
        n = n_tok // B
        m = self._mask
        if n != 1 and isinstance(m, torch.Tensor) and m.dim() == 2 and n != m.shape[1]:
            self.unrouted += 1
            return None
        return f if n == 1 else f.repeat_interleave(n)

    def _fused_hook(self, u: torch.Tensor, wnv: torch.Tensor):
        """The same correction on a fused stack: per routed (token, expert) pair, same fail-safe."""
        def hook(mod, args, kwargs, out):
            if self.fired is None or not self._live: return None     # no decision -> no edit
            try:
                h = args[0] if len(args) > 0 else kwargs.get("hidden_states")
                idx = args[1] if len(args) > 1 else kwargs.get("top_k_index")
                wts = args[2] if len(args) > 2 else kwargs.get("top_k_weights")
                if not all(isinstance(t, torch.Tensor) for t in (out, h, idx, wts)):
                    raise RuntimeError(
                        "a fused stack's forward did not carry (hidden_states, top_k_index, "
                        "top_k_weights) and a tensor output, so there is no routing to edit by")
                if ((h.dim(), idx.dim(), out.dim()) != (2, 2, 2) or idx.shape != wts.shape
                        or out.shape[0] != h.shape[0] or idx.shape[0] != h.shape[0]):
                    raise RuntimeError(
                        "a fused stack's token list is not [T,H] with [T,K] routing and a [T,*] "
                        "output: %s, %s, %s"
                        % (tuple(h.shape), tuple(idx.shape), tuple(out.shape)))
                m = self._token_rows(h.shape[0])
                if m is None: return None            # counted: the mask cannot be expanded
                return out - fused_correction(mod, h, idx, wts, u, wnv, m).to(out.dtype)
            except Exception as e:                                   # fail-safe: no edit
                self._note(e)
                return None
        return hook

    # --- attach / detach ---------------------------------------------------------------

    @staticmethod
    def _is_target(name: str) -> bool:
        """`enumerate_targets`' role filter by name: an output projection, not an excluded tower."""
        return (any(o in name for o in OUT_NAMES)
                and not any(x in name.split(".") for x in EXCLUDE_PARTS))

    def _key(self, name: str) -> str | None:
        """The direction key of a target in the edit band, or None if it is not one."""
        if not self._is_target(name): return None
        l = layer_of(name)
        if l is None or l not in self.edit_layers: return None
        k = key_of(name, l)
        return k if k in self.d else None

    def _read_module(self, model: torch.nn.Module) -> tuple[str, torch.nn.Module]:
        """The decoder layer whose output the gate reads: `...layers.<gate_layer>`.

        An ambiguous or missing read site is refused. A gate reading the wrong depth is a gate
        measuring something else, and it would not announce itself.
        """
        l = self.gate_layers[-1]
        pat = re.compile(r"(?:^|\.)layers\.%d$" % l)
        hits = [(n, m) for n, m in model.named_modules()
                if pat.search(n) and not any(x in n.split(".") for x in EXCLUDE_PARTS)]
        if len(hits) != 1:
            raise ValueError("expected exactly one `layers.%d` module to read, found %d%s"
                             % (l, len(hits), ": %s" % [n for n, _ in hits] if hits else ""))
        return hits[0]

    def attach(self, model: torch.nn.Module, verbose: bool = True) -> tuple[int, int]:
        """Hook the read layer and every target in the band. Returns (hooked, not reached).

        Two passes, and they are separate on purpose. The first is the `nn.Linear` pass, which is
        every 2-D target and is untouched by the second: everything this module has ever measured
        was measured through it. The second is the fused routed-expert pass, which drives a 3-D
        stack through its own module boundary and is the only way to reach a parameter that has
        no `nn.Linear` of its own -- and it declines, loudly, rather than guess (`_fused_refusal`,
        `_fused_recon_error`).

        The plan is built in full before a single hook is registered, so a failure here leaves
        the model exactly as it was -- unedited, which is the safe side. `not reached` counts the
        direction-keyed target parameters in the band that no hook covers.

        READ THAT NUMBER. It is the coverage check `docs/PITFALLS.md` 2 says to run on your own
        checkpoint. 0 means this is the same edit `patch()` would write. Anything else is a
        weaker edit at the same K, and on an MoE the shortfall is not incidental -- 95.5% of the
        band's squared-Frobenius edit energy and 57.0% of its per-token routed edit sit in the
        fused stacks on this project's primary checkpoint, so a run that reaches only the 14 is
        applying 43.0% of the intended per-token edit.
        """
        def log(*a):
            if verbose: print(*a, flush=True)

        if self._handles: raise RuntimeError("already attached; detach() first")
        read_name, read_mod = self._read_module(model)
        sig = inspect.signature(model.forward)
        has_mask = "attention_mask" in sig.parameters
        plan: list[tuple[str, torch.nn.Module, torch.Tensor, torch.Tensor]] = []
        bad: list[str] = []
        for name, mod in model.named_modules():
            if not isinstance(mod, torch.nn.Linear): continue
            k = self._key(name)
            if k is None: continue
            v = self.d[k].detach().flatten()
            if v.numel() != mod.out_features:                   # shape mismatch -> no hook
                bad.append("%s: direction %d != out_features %d"
                           % (name, v.numel(), mod.out_features))
                continue
            # float32 or the weight's own dtype, whichever is wider: patch() computes the edit in
            # float32 from a bf16 checkpoint, and precomputing in bf16 instead would put the two
            # paths a long way apart for no reason. Normalising v has to happen in the same dtype.
            dt = torch.promote_types(mod.weight.dtype, torch.float32)
            W = mod.weight.detach().to(dt)
            v = v.to(W.device, dt)
            u, wnv = precompute(W, v / v.norm().clamp_min(1e-12), self.w[k])
            plan.append((name, mod, u.to(mod.weight.dtype), wnv.to(mod.weight.dtype)))

        # The fused pass. Keyed off the PARAMETER, because that is the thing with a direction and
        # the thing the coverage count is about; the module that owns it is where the hook goes.
        fused: list[tuple[str, torch.nn.Module, torch.Tensor, torch.Tensor]] = []
        why: dict[str, str] = {}
        for pname, p in model.named_parameters():
            if p.dim() != 3: continue
            k = self._key(pname)
            if k is None: continue
            mod = model.get_submodule(pname.rsplit(".", 1)[0])
            no = _fused_refusal(mod, p.detach(), self.d[k].numel())
            if no is None:
                # `attach` stays total. The proof calls somebody else's forward with synthetic
                # inputs, and a stack that cannot take them is a coverage question rather than a
                # server failure, so the exception becomes the reason and the stack is declined.
                try:
                    err = _fused_recon_error(mod)
                except Exception as e:
                    no = "its own forward raised on the proof probe: %r" % (e,)
                else:
                    if err > FUSED_RECON_TOL:
                        no = ("its own forward is not the %.2e reconstruction this would edit "
                              "(tolerance %.0e), so the edit is unproven here"
                              % (err, FUSED_RECON_TOL))
            if no is not None:
                why[pname] = no
                continue
            dt = torch.promote_types(p.dtype, torch.float32)
            v = self.d[k].detach().flatten().to(p.device, dt)
            u, wnv = _precompute_fused(p.detach(), v / v.norm().clamp_min(1e-12), self.w[k], dt)
            fused.append((pname, mod, u, wnv))
            why[pname] = "reconstruction %.2e relative" % err

        hooked = ({"%s.weight" % name for name, _, _, _ in plan}
                  | {pname for pname, _, _, _ in fused})
        want = {n for n, p in model.named_parameters()
                if p.dim() in (2, 3) and self._key(n) is not None}
        unreached = sorted(want - hooked)

        self._sig, self._has_mask = sig, has_mask
        self._handles.append(
            model.register_forward_pre_hook(self._root_pre_hook, with_kwargs=True))
        self._handles.append(read_mod.register_forward_hook(self._read_hook))
        for _, mod, u, wnv in plan:
            self._handles.append(mod.register_forward_hook(self._edit_hook(u, wnv)))
        for _, mod, u, wnv in fused:
            self._handles.append(
                mod.register_forward_hook(self._fused_hook(u, wnv), with_kwargs=True))

        log("GATE  category=%s  read %s %s  thresh %s  edit L%d-%d  K=%.2f"
            % (self.meta.get("category", "?"), read_name, self.pooling,
               "NONE (never fires)" if self.thresh is None else "%.4f" % self.thresh,
               self.edit_layers[0], self.edit_layers[-1], self.K))
        log("   %d tensors hooked, %d direction-keyed target(s) NOT reached"
            % (len(plan) + len(fused), len(unreached)))
        for b in bad: log("   SHAPE MISMATCH, not hooked  %s" % b)
        for n, _, _, _ in fused: log("   FUSED, driven through its own forward  %s  %s"
                                     % (n, why[n]))
        for n in unreached:
            log("   NOT REACHED  %s%s" % (n, "  %s" % why[n] if n in why else ""))
        if self.pooling != "last" and not has_mask:
            log("   forward() does not name attention_mask: %s pooling has no mask to weight by, "
                "so the gate will NOT fire. Nothing is edited." % self.pooling)
        return len(plan) + len(fused), len(unreached)

    def detach(self) -> None:
        """Remove every hook and forget the decision. Idempotent; the model is then pristine."""
        for h in self._handles: h.remove()
        self._handles = []
        self._mask, self._mask_known = None, False
        self._sig, self._has_mask = None, False
        self._spans = None
        self.reset()


from .fit import (  # noqa: E402
    _eig_scatter_gram,
    fit_gate_direction,
    fit_gate_op,
    thresh_r2,
    thresh_r3,
)

