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
    user span that could not be located, a row layout the per-sequence mask cannot be applied to,
    or any exception inside the gate. The pristine weights then answer, which means the model
    refuses. There is no path the other way -- in particular an unlocatable span does NOT fall
    back to `mean_all`, because that fallback is the defect, not the mitigation. Note the scope:
    this is the gate DECLINING when it cannot tell. It is not protection against the gate deciding
    confidently and wrongly, and measured twice on the build the published operating points come
    from, nothing else is either -- when the gate fires, the band-edited model delivers, with no
    residual refusal behind it (docs/GATE.md, "There is no second line of defence"). The threshold
    is the only thing handling that class of error.

Two coverage gaps, both of which leave the edit SMALLER than `patch()`'s, i.e. closer to refusing,
and so are coverage gaps rather than safety gaps:

  * Fused 3-D MoE expert stacks are not `nn.Linear` modules -- a parameter inside somebody's
    routing code has no forward to hook -- so `attach` counts and names them instead of editing
    them. On this project's primary MoE checkpoint the band's 21 target tensors split 14 hookable
    (`self_attn.o_proj` and `mlp.down_proj`, seven each) and 7 not (`layers.N.experts.down_proj`).
    Those 14 are exactly the set both published abliterations of that checkpoint edit, and the 7
    are the ones this project added: the runtime path reproduces published-standard coverage and
    misses precisely this project's coverage improvement. See `docs/PITFALLS.md` 2 for the counts
    and `docs/GATE.md` for what follows. A dense backbone has no gap -- 14 of 14.
  * One direction blob only. A sequential second blob needs the second step's `u` computed from
    the already-edited W, which is the one thing a hook does not have. Patch offline for that.
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

The update rule is Heretic's (github.com/p-e-w/heretic) and the foundational method is Arditi et
al. 2024, arXiv:2406.11717; see `patch.py`. The restriction -- gating it on an early read -- is
the only new part, and `docs/GATE.md` states what was measured about it and what was not.
"""
from __future__ import annotations

import inspect
import os
import re
from typing import Any, Sequence

import torch

from .enumerate_targets import EXCLUDE_PARTS, OUT_NAMES
from .patch import key_of, layer_of, load_dirs

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
    from .capture import _think_kwargs                 # lazy: `import tempx.gate` stays torch-only

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
        """
        self.fired = None
        self.scores = None
        self._live = False

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
        if self.fired is not None: return None                  # frozen: decode changes nothing
        if self.force is None and self.pooling != "last" and not self._mask_known:
            # Not an error in the model, but it is one here: both means need the mask -- one to
            # avoid averaging pads in, the other to find where the real tokens start -- and
            # pooling a possibly padded batch without it is a wrong read, not an approximate one.
            # "last" needs no mask, left padding putting the last real token at [-1] for every
            # row, and `force` skips the read altogether, so it skips this with it.
            self._note(RuntimeError("attention_mask could not be read from the forward call"))
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
        """Hook the read layer and every 2-D target in the band. Returns (hooked, not reached).

        The plan is built in full before a single hook is registered, so a failure here leaves
        the model exactly as it was -- unedited, which is the safe side. `not reached` counts the
        direction-keyed target parameters in the band that no hook covers: fused 3-D expert
        stacks above all, which are not `nn.Linear` modules.

        READ THAT NUMBER. It is the coverage check `docs/PITFALLS.md` 2 says to run on your own
        checkpoint. 0 means this is the same edit `patch()` would write. Anything else is a
        weaker edit at the same K, and on an MoE the shortfall is not incidental -- it is the
        routed experts, which is the part published abliterations of these checkpoints also miss.
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

        hooked = {"%s.weight" % name for name, _, _, _ in plan}
        want = {n for n, p in model.named_parameters()
                if p.dim() in (2, 3) and self._key(n) is not None}
        unreached = sorted(want - hooked)

        self._sig, self._has_mask = sig, has_mask
        self._handles.append(
            model.register_forward_pre_hook(self._root_pre_hook, with_kwargs=True))
        self._handles.append(read_mod.register_forward_hook(self._read_hook))
        for _, mod, u, wnv in plan:
            self._handles.append(mod.register_forward_hook(self._edit_hook(u, wnv)))

        log("GATE  category=%s  read %s %s  thresh %s  edit L%d-%d  K=%.2f"
            % (self.meta.get("category", "?"), read_name, self.pooling,
               "NONE (never fires)" if self.thresh is None else "%.4f" % self.thresh,
               self.edit_layers[0], self.edit_layers[-1], self.K))
        log("   %d tensors hooked, %d direction-keyed target(s) NOT reached"
            % (len(plan), len(unreached)))
        for b in bad: log("   SHAPE MISMATCH, not hooked  %s" % b)
        for n in unreached: log("   NOT REACHED  %s" % n)
        if self.pooling != "last" and not has_mask:
            log("   forward() does not name attention_mask: %s pooling has no mask to weight by, "
                "so the gate will NOT fire. Nothing is edited." % self.pooling)
        return len(plan), len(unreached)

    def detach(self) -> None:
        """Remove every hook and forget the decision. Idempotent; the model is then pristine."""
        for h in self._handles: h.remove()
        self._handles = []
        self._mask, self._mask_known = None, False
        self._sig, self._has_mask = None, False
        self._spans = None
        self.reset()
