"""Serving the gate: a pristine checkpoint, its hooks, and a generate() that cannot forget a span.

`gate.py` stops at the hooks, because a hook needs a model somebody else loaded, and nothing else
in tempx loads one -- the four CLI subcommands are offline artefact transforms, which is why
`docs/GATE.md` says the gate is a library entry point and not a fifth subcommand. This module is
the missing half: load the pristine checkpoint, attach the gate to it, generate, and report what
the gate did. No weight is modified and nothing is written to disk; `close()` leaves the model
bit-for-bit as it was loaded.

It is one class because both constraints in `ablation/HANDOVER_1005/SERVING.md` are properties of
the CALL SITE, and a library cannot enforce either from inside a forward pass:

  * THE CALLER SUPPLIES THE USER SPAN. A forward pass carries neither the tokenizer nor the prompt
    text, so `generate` renders the chat template, locates the span with `user_span` and calls
    `set_spans` -- in that order, on every batch, with no public path that skips it. Forgetting it
    is fail-safe (the gate declines, so the model refuses) and therefore a SILENT coverage bug: a
    gate that never fires looks like a well-behaved one. That is why `_batch` is private and
    `generate` is the only way in.
  * ONE FORWARD AT A TIME PER Gate INSTANCE. The frozen decision is instance state, so two threads
    through one gate could answer one sequence under another sequence's decision -- the failure the
    gate exists to prevent. A batch is fine and is the supported way to serve several sequences at
    once; threads are not. The lock here is NON-BLOCKING: a second thread gets an exception naming
    the constraint rather than a queue, because queuing would make a safety constraint look like a
    throughput one.

What it reports, and why each field exists. `docs/GATE.md`: "a gate that silently never fires and
a gate that is working look identical from outside." Three things look identical in a transcript
and are different failures -- the gate DECLINED (it could not tell, so nothing was edited), the
gate did not FIRE (it decided, below threshold), and the gate fired and the model refused anyway.
So every `Reply` carries the decision (`fired`, `score`, `routed`) and separately whether a
decision was taken at all (`decided`, `declined`), and every `Reply` in a batch carries that
batch's `GateHealth`: the gate's swallowed errors, its `unrouted` count, and how many user spans
could not be located. Conflating those three is how a coverage bug hides.

Three things this deliberately does not do:

  * No continuous batching, no paged attention, no CUDA graphs. SERVING.md's decision is HF
    `transformers` plus a loop, because it is the only backend where the whole mechanism works
    today, and what it costs is throughput only -- not correctness and not capability. A batch
    runs until its longest sequence finishes.
  * No prompt cache across calls. `generate`'s own KV cache lives for one call, which is what
    makes `gate._is_prefill` work. A cross-request prefix cache would be a correctness trap:
    the KV for layers >= 10 is decision-dependent, so a gated and an ungated request sharing a
    prefix must not share it (`HANDOVER_1005/METHOD.md`). Nothing here caches across calls, and
    anything that adds one has to fork the cache key on the decision.
  * No refusal label. Whether a reply is a refusal is a judge's call: a decline-marker metric was
    measured at 1.7% precision on this project's edited models (0 of 121 marker hits across 600
    edited responses), so keyword labelling of an edited arm is invalid, not merely noisy. `Reply`
    therefore carries the generated TEXT and `capped`, which says whether the token cap cut it off
    -- 83.8% of edited replies hit the cap in the measured e2e run, so "short" is not "refused".
    Labelling is the caller's job and it needs a judge.
"""
from __future__ import annotations

import contextlib
import os
import threading
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch

from .gate import EDIT_LAYERS, META_FIELDS, Gate, user_span


@dataclass(frozen=True)
class GateHealth:
    """What the gate did on ONE batch, and what it swallowed doing it.

    Shared by every `Reply` in that batch, because the gate's state is per batch: one read, one
    decision vector, and one set of spans that stands the whole batch's read down if any of them
    is missing. `errors` and `unrouted` are this batch's increments, not the gate's running
    totals -- those stay on the gate. `last_error` is the last exception the gate swallowed at any
    time, so it belongs to this batch only when `errors` is non-zero.
    """

    n: int
    attached: bool
    fired: int
    decided: bool
    unlocatable: int
    errors: int
    unrouted: int
    last_error: str | None


@dataclass(frozen=True)
class Reply:
    """One prompt, its completion, and the route that produced it.

    `routed` is "edited" when the gate fired for this sequence and "pristine" otherwise, and
    `decided` separates the two ways of being pristine: the gate decided and this sequence was
    below the threshold (`decided=True`), or the gate could not decide at all (`declined`).
    `score` is the cosine the decision came from, and is None exactly when no decision was taken.
    `span` is this prompt's `user_span` result; None means it was not located, which stands down
    the read for the whole batch and is therefore visible in `health.unlocatable` as well.
    """

    prompt: str
    text: str
    fired: bool
    score: float | None
    routed: str
    decided: bool
    capped: bool
    n_new: int
    span: tuple[int, int, int, int] | None
    health: GateHealth

    @property
    def declined(self) -> bool:
        """The gate was attached and produced no decision: not an answer about this prompt."""
        return self.health.attached and not self.decided


class GatedModel:
    """Pristine checkpoint + the gate's hooks. One instance owns one forward at a time.

    `load` is the loading path; the constructor takes an already-loaded model, tokenizer and
    `Gate` so this is testable without a checkpoint. Constructing ATTACHES, so there is no window
    in which this object exists and the gate does not; `close()` detaches and the model is the
    pristine checkpoint again. Use it as a context manager.

    `sys_prompt` is part of the read, not a presentation choice: the span arithmetic and the
    category direction were both built with exactly one system prompt (`band_spec.json`
    `capture.sys`), and changing it moves the span and invalidates the direction.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tok: Any,
        gate: Gate,
        sys_prompt: str = "You are a helpful assistant.",
        think: str | bool | None = None,
        verbose: bool = True,
    ) -> None:
        from .capture import _think_kwargs           # lazy, as gate.user_span imports it

        if getattr(tok, "padding_side", None) != "left":
            raise ValueError(
                "the tokenizer must pad LEFT: `pool` measures the user span's offset from the "
                "right and weights the mean by the padding mask, both of which are left-padding "
                "arithmetic (gate.pool, capture.py). Set tok.padding_side = 'left'.")
        self.model = model
        self.tok = tok
        self.gate = gate
        self.sys_prompt = sys_prompt
        self.think = think
        self.verbose = verbose
        self._think_kw = _think_kwargs(think)
        # Only `mean_user` reads through the spans (`gate.pool` ignores them for `mean_all` and
        # `last`), so only `mean_user` can be stood down by a missing one. They are computed and
        # reported either way -- an unlocatable span is worth seeing at any pooling -- but it is
        # a hard stop only where it actually suppresses the read.
        self._needs_spans = gate.pooling == "mean_user"
        # Re-entrant on purpose: the constraint is between THREADS, and `detached()` holds this
        # across its own generates. A non-blocking acquire still refuses a second thread.
        self._lock = threading.RLock()
        self._stop = _stop_ids(model, tok)
        self.coverage = gate.attach(model, verbose=verbose)
        self._attached = True
        self._log_op()
        self._span_probe()

    # --- loading -----------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        model_path: str | os.PathLike,
        dirs: str | Sequence[str | os.PathLike | dict],
        op: str | os.PathLike | dict,
        K: float = 2.0,
        edit_layers: Sequence[int] = EDIT_LAYERS,
        sys_prompt: str = "You are a helpful assistant.",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        device_map: str | None = None,
        think: str | bool | None = None,
        verbose: bool = True,
    ) -> "GatedModel":
        """Load the PRISTINE checkpoint and attach a gate built from an operating-point file.

        `model_path` must be the unedited checkpoint. Serving an already-abliterated one would
        apply the edit twice and the gate would then be deciding nothing: a pre-patched checkpoint
        has refusal removed for every category whether the gate fires or not.

        `K` and `edit_layers` are the caller's; `pool` and `gate_layer` come out of the operating
        point via `Gate.from_file` and are never overridden here, because a threshold is in the
        units of the pooling that produced it. The defaults are Gemma-4-26B-A4B-it's measured band
        (10-16 at K=2.0) and that band does NOT transfer -- derive it from the target model's own
        `sep(l)` profile. An operating point marked `VOID` is refused rather than served.

        `dtype` is bfloat16 because that is what the measurements ran in. Parity with the
        materialised weight edit is a float32 statement; at bf16 the two paths differ by about one
        bf16 ulp and neither is the more faithful one (`docs/GATE.md`, "Parity").

        On a unified-memory box pass `device_map=device` (needs accelerate): the plain
        `.to(device)` path materialises the checkpoint in host memory and then copies it, which
        peaks at about twice the checkpoint size out of one shared pool.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        def log(*a):
            if verbose: print(*a, flush=True)

        # The gate first: it is two small files and a CPU precompute, and every way it can be
        # refused -- a missing provenance field, a sequential blob, a read inside the band, a void
        # point -- should fail in a second rather than after a multi-minute checkpoint load.
        gate = Gate.from_file(dirs, op, K=K, edit_layers=edit_layers)
        if gate.meta.get("VOID"):
            raise ValueError(
                "this operating point is marked VOID in its own meta and must not be served: %s. "
                "A void point is one whose sweep failed its control (ablation/kits/write_ops.py), "
                "and it ships under a VOID_ filename for the same reason."
                % (gate.meta.get("void_reason") or "no reason recorded"))

        tok = AutoTokenizer.from_pretrained(model_path)
        tok.padding_side = "left"
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        kw: dict[str, Any] = dict(dtype=dtype, low_cpu_mem_usage=True)
        if device_map: kw["device_map"] = device_map
        log("loading PRISTINE %s ... (dev=%s, device_map=%s, dtype=%s)"
            % (model_path, device, kw.get("device_map"), dtype))
        model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
        if "device_map" not in kw: model = model.to(device)
        model.eval().requires_grad_(False)                     # served weights; no edit on disk
        return cls(model, tok, gate, sys_prompt=sys_prompt, think=think, verbose=verbose)

    # --- what the run log has to record ------------------------------------------------

    def _log(self, *a) -> None:
        if self.verbose: print(*a, flush=True)

    def _log_op(self) -> None:
        """Print the operating point's provenance, so a run log says which point produced it.

        An operating point without its provenance is uncomparable (`docs/PITFALLS.md` 3), and the
        run is the one place the two can still be tied together. `META_FIELDS` first, in order,
        then whatever else the file carries that fits on a line -- the meta also holds per-category
        breakdowns and paragraph-long notes, which belong in the file and not in a log.
        """
        m = self.gate.meta
        self._log("OP    " + "  ".join("%s=%s" % (f, m.get(f)) for f in META_FIELDS))
        extra = ["%s=%s" % (k, m[k]) for k in sorted(m)
                 if k not in META_FIELDS and isinstance(m[k], (str, int, float, bool))
                 and len(str(m[k])) <= 64]
        if extra: self._log("OP    " + "  ".join(extra))
        self._log("OP    the rest of this point's meta (per-category leak, intervals, notes) is "
                  "in the file, not here")

    def _span_probe(self) -> None:
        """Locate one span at load time, so the run log shows the arithmetic working.

        `user_span` returning None for everything means the render and the tokenizer disagree --
        a different system prompt, a template with no system role, a `think` mode that does not
        match the capture's. That costs a batch to discover and a line to show. The hard stop is
        in `_batch`, which raises when a whole batch is unlocatable AND the pooling reads spans.
        """
        sp = user_span(self.tok, "How do I write a port scanner?", sys_prompt=self.sys_prompt,
                       think=self.think)
        self._log("SPAN  probe %s  pool=%s (%s)  sys=%r"
                  % ("(a,u,b,tot)=%s" % (sp,) if sp else "NOT LOCATED", self.gate.pooling,
                     "the read pools these spans" if self._needs_spans
                     else "this pooling does not read the spans", self.sys_prompt))

    # --- generation --------------------------------------------------------------------

    def _render(self, text: str) -> str:
        """The chat template, rendered exactly as `user_span` renders it to count tokens.

        These two renders must stay identical; `user_span` renders internally and returns counts,
        not text, so the duplication is unavoidable. What catches a drift is `_batch`'s check that
        each span's own total equals the row's real-token count -- the same assertion
        `capture_hf_v2.py` and `gate._user_mask` make, moved early so it names the cause.
        """
        return self.tok.apply_chat_template(
            [{"role": "system", "content": self.sys_prompt}, {"role": "user", "content": text}],
            add_generation_prompt=True, tokenize=False, **self._think_kw)

    def generate(self, prompts: Sequence[str], max_new_tokens: int = 256, batch_size: int = 4,
                 **gen_kwargs) -> list[Reply]:
        """One `Reply` per prompt, in order. Render, locate, hand over, forward, read, decode.

        That order is the contract and nothing outside can reorder it: the spans are computed from
        the same text that is tokenized, handed over immediately before the forward that consumes
        them, and `fired`/`scores` are read out before the next prefill can reset them.
        `batch_size` is a batch in the gate's sense -- one read, one `[B]` decision, one row per
        prompt -- not several forwards.

        Greedy by default, because the gated arm and the pristine control arm have to be
        comparable. Three generation settings are REFUSED rather than served, because each one
        turns the gate off silently: see the error texts.
        """
        if isinstance(prompts, str):
            raise TypeError("prompts is a sequence of prompts; one string is not a batch of one")
        ps = list(prompts)
        if not ps: return []
        if gen_kwargs.get("use_cache") is False:
            raise ValueError(
                "use_cache=False re-feeds the whole sequence every step, so there is no prefill "
                "to recognise and the decision is retaken per token: a mean over a growing "
                "sequence drifts and the gate can flip mid-answer. Generate with the cache.")
        if int(gen_kwargs.get("num_beams", 1)) != 1 \
                or int(gen_kwargs.get("num_return_sequences", 1)) != 1:
            raise ValueError(
                "beam search and num_return_sequences > 1 expand the batch BEFORE the prefill, so "
                "the B spans no longer match the B*n rows the read sees and the gate declines for "
                "the whole batch -- fail-safe and silent. One sequence per prompt.")
        gen_kwargs.setdefault("do_sample", False)
        gen_kwargs.setdefault("pad_token_id", self.tok.pad_token_id)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(
                "one forward at a time per Gate instance: another thread is inside generate() on "
                "this GatedModel. The frozen decision is instance state, so two concurrent "
                "forwards would interleave their decisions and a sequence could be answered under "
                "another sequence's. Batch instead, or use one GatedModel per thread.")
        try:
            out: list[Reply] = []
            for i in range(0, len(ps), batch_size):
                out.extend(self._batch(ps[i:i + batch_size], max_new_tokens, gen_kwargs))
            return out
        finally:
            self._lock.release()

    def _batch(self, chunk: list[str], max_new_tokens: int, gen_kwargs: dict) -> list[Reply]:
        """One forward's worth: the six steps, in order, and private so none can be skipped."""
        tok = self.tok
        if tok.padding_side != "left":
            raise RuntimeError("tok.padding_side became %r: the pooling is left-padding "
                               "arithmetic and would read pad rows" % tok.padding_side)

        texts = [self._render(p) for p in chunk]
        spans = [user_span(tok, p, sys_prompt=self.sys_prompt, think=self.think) for p in chunk]
        lost = [j for j, sp in enumerate(spans) if sp is None]
        reads_spans = self._attached and self._needs_spans     # whose read they actually feed
        if lost and len(lost) == len(chunk) and reads_spans:
            raise RuntimeError(
                "the user span was not located for ANY of the %d prompt(s) in this batch, so the "
                "gate would decline every one of them and the whole run would read as a gate that "
                "never fires. That is a configuration failure, not a prompt property: the system "
                "prompt, the chat template or the `think` mode does not match the one the "
                "direction was captured with (sys=%r). The capture had 0 fallbacks in 1426 items "
                "on this model." % (len(chunk), self.sys_prompt))

        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        mask = enc["attention_mask"]
        if mask.dim() != 2:
            raise RuntimeError(
                "the tokenizer returned a %d-D attention mask; `pool` needs the [B,S] padding "
                "mask and raises on anything else, because a causal mask cannot tell a pad from a "
                "real token" % mask.dim())
        for j, (sp, n_real) in enumerate(zip(spans, mask.sum(1).tolist())):
            if sp is not None and sp[3] != int(n_real):
                raise RuntimeError(
                    "row %d: the span totals %d token(s) and this batch encodes %d of them. The "
                    "render in _render and the one inside user_span have drifted apart, or the "
                    "tokenizer was called with different flags -- both must be "
                    "add_special_tokens=False over the same rendered template."
                    % (j, sp[3], int(n_real)))
        enc = enc.to(self.model.device)

        e0, u0 = self.gate.errors, self.gate.unrouted
        if self._attached: self.gate.set_spans(spans)          # consumed by the next prefill
        with torch.no_grad():
            out = self.model.generate(input_ids=enc["input_ids"],
                                      attention_mask=enc["attention_mask"],
                                      max_new_tokens=max_new_tokens, **gen_kwargs)
        fired, scores = self.gate.fired, self.gate.scores      # before the next prefill resets

        decided = fired is not None
        fl = [bool(x) for x in fired.tolist()] if decided else [False] * len(chunk)
        sl = None if scores is None else [float(x) for x in scores.tolist()]
        if len(fl) != len(chunk) or (sl is not None and len(sl) != len(chunk)):
            raise RuntimeError(
                "the gate decided for %d row(s) and this batch has %d prompt(s): the rows the "
                "read saw are not the sequences that were submitted, so no decision here belongs "
                "to any prompt" % (len(fl), len(chunk)))
        health = GateHealth(n=len(chunk), attached=self._attached, fired=int(sum(fl)),
                            decided=decided, unlocatable=len(lost),
                            errors=self.gate.errors - e0, unrouted=self.gate.unrouted - u0,
                            last_error=self.gate.last_error)

        S = enc["input_ids"].shape[1]
        new = out[:, S:]
        replies = []
        for j, p in enumerate(chunk):
            ids = new[j].tolist()
            hit = next((i for i, t in enumerate(ids) if t in self._stop), None)
            n_new = len(ids) if hit is None else hit + 1
            replies.append(Reply(
                prompt=p, text=tok.decode(new[j, :n_new], skip_special_tokens=True),
                fired=fl[j], score=None if sl is None else sl[j],
                routed="edited" if fl[j] else "pristine", decided=decided,
                capped=hit is None, n_new=n_new, span=spans[j], health=health))

        self._log("BATCH n=%d  %s  fired %d  errors +%d  unrouted +%d  capped %d"
                  % (health.n, "gate attached" if health.attached else "PRISTINE ARM (no hooks)",
                     health.fired, health.errors, health.unrouted,
                     sum(r.capped for r in replies)))
        if health.attached and not health.decided:
            self._log("   GATE DECLINED for this batch: nothing was edited, so every reply here "
                      "is the pristine model's and this is NOT a measurement of the gate not "
                      "firing. last_error=%s" % health.last_error)
        if health.unlocatable:
            self._log("   %d of %d user span(s) NOT LOCATED: %s"
                      % (health.unlocatable, health.n,
                         "one unlocatable span stands down the whole batch's read, since one read "
                         "produces one decision vector" if reads_spans else
                         "nothing read them in this arm (pool=%s, gate %s) -- but the span "
                         "arithmetic failing at all is worth chasing"
                         % (self.gate.pooling, "attached" if self._attached else "detached")))
        if health.errors or health.unrouted:
            self._log("   gate swallowed %d error(s) and skipped %d unroutable module output(s); "
                      "both land on NO EDIT. last_error=%s"
                      % (health.errors, health.unrouted, health.last_error))
        return replies

    # --- the control arm ---------------------------------------------------------------

    @contextlib.contextmanager
    def detached(self) -> Iterator["GatedModel"]:
        """The same prompts through the pristine model, hooks off: the control arm.

        `detach()` restores the pristine forward bit-for-bit (`docs/GATE.md`, "Parity": forced off
        is `torch.equal`, and detached has no hooks at all), so what comes out of here is the
        unedited checkpoint's own answer -- which is the only thing that makes a gated answer mean
        anything. `generate` still works inside, still computes spans so `Reply.span` is populated,
        and reports `health.attached=False` so the two arms cannot be confused in a log.

        Re-attaching on the way out asserts the same coverage counts it loaded with: a band that
        reached a different number of tensors the second time is not the same arm.
        """
        if not self._attached:
            raise RuntimeError("already detached; detached() does not nest")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("another thread is inside generate() on this GatedModel: detaching "
                               "under a live forward would change the model mid-pass")
        self.gate.detach()
        self._attached = False
        self._log("DETACHED: hooks off, pristine forward. Replies from here are the control arm.")
        try:
            yield self
        finally:
            got = self.gate.attach(self.model, verbose=self.verbose)
            self._attached = True
            self._lock.release()
            if got != self.coverage:
                raise RuntimeError(
                    "re-attaching reached %s, not the %s this instance loaded with: the two arms "
                    "did not run against the same edit" % (got, self.coverage))

    # --- close -------------------------------------------------------------------------

    def close(self) -> None:
        """Detach the gate. The model is then the pristine checkpoint again, bit-for-bit."""
        if self._attached:
            self.gate.detach()
            self._attached = False
            self._log("CLOSED: hooks removed, model pristine.")

    def __enter__(self) -> "GatedModel":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _stop_ids(model: torch.nn.Module, tok: Any) -> set[int]:
    """Every id that ends a sequence, so `capped` can mean "hit the cap" and not "stopped".

    `capped` is the difference between a short answer and a cut-off one, and 83.8% of edited
    replies hit the cap in the measured e2e run (`HANDOVER_1005/FINDINGS.md` 1) -- so a reply that
    ends without one of these did not finish, and reading it as a refusal would be wrong. A
    caller who overrides `eos_token_id` on the generate call is measured against the model's own
    ids, not theirs.
    """
    cfg = getattr(model, "generation_config", None)
    e = getattr(cfg, "eos_token_id", None)
    ids = list(e) if isinstance(e, (list, tuple)) else [e]
    ids.append(getattr(tok, "eos_token_id", None))
    return {int(i) for i in ids if i is not None}
