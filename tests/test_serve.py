"""The serving wrapper: the span cannot be forgotten, the lock is real, and detached is identical.

`test_gate.py` tests the gate's arithmetic and its decision machinery against a toy stack. This
file tests the two things `ablation/HANDOVER_1005/SERVING.md` says a SERVER must respect, which are
properties of the call site and not of the hook:

  * THE CALLER SUPPLIES THE USER SPAN, so `generate` has to render, locate and hand over -- in that
    order, on every batch, with no public path that skips it. Forgetting is fail-safe and therefore
    SILENT: a gate that never fires looks exactly like a gate that is working. So the tests here
    assert the spans were set for every prompt and in prompt order, that an unlocatable span lands
    on `routed="pristine"` rather than a quiet `mean_all` read, and that a batch where every span
    fails raises instead of returning a batch of pristine replies.
  * ONE FORWARD AT A TIME PER Gate INSTANCE, enforced rather than documented. The lock is driven
    from a real second thread, with the first thread held open inside its own forward pass.

A real `LlamaForCausalLM` at toy width, config-built with random weights and nothing downloaded --
`test_gate.py`'s `_llama`, because `generate()` is the thing under test and a hand-rolled fake
`generate` would test the fake. The tokenizer is `test_gate.py`'s `FakeTok` extended with the two
calls a server makes: a batch encoding (left padded, so the span arithmetic is exercised for real)
and `decode`. It tokenises on whitespace, so a token count is a word count and `user_span`'s
arithmetic is exact by construction, and its ids are a pure function of the word -- no incremental
vocabulary -- so the gated and the ungated arm encode the same text to the same ids and the
bit-identity assertion means something.

No refusal labelling anywhere in this file, and `Reply` carries no field for one: a decline-marker
metric measured 1.7% precision on this project's edited models (0 of 121 marker hits across 600
edited responses), so keyword labelling of an edited arm is invalid rather than noisy. What is
asserted is the ROUTE and the RECORD -- fired, decided, declined, routed, capped -- which is what
a caller needs to hand a judge.
"""
from __future__ import annotations

import inspect
import threading

import pytest
import torch

from tempx.gate import EDIT_LAYERS, GATE_LAYER, Gate, pool, user_span
from tempx.serve import GateHealth, GatedModel, Reply, _stop_ids

H, NL = 8, 17
BAND = EDIT_LAYERS                                   # layers 10-16
# The system prompt is part of the read, not presentation: the span arithmetic and the category
# direction were both built with exactly this one (`band_spec.json` capture.sys).
SYS = "You are a helpful assistant."

# The published cyber operating point's provenance, as in `test_gate.py`. `from_file` refuses a
# point that does not carry every META_FIELD (docs/PITFALLS.md 3), so a test needs them present --
# and that is the ONLY reason they are here. No test in this file uses them as a measurement: every
# threshold below is either -2.0, i.e. "fires on anything it can read", or fitted in the test from
# a probe of the model's own read. `pool` is "mean_user" and not test_gate.py's "mean_all" because
# the point this wrapper exists to serve is
# `ablation/kits/gemma4-26B/op_gemma4-26B_cyber_offense_mean_user_L8_tpr_08.pt` -- whose own meta
# (category cyber_offense, pool mean_user, op 139of174) the gate reads FROM THE FILE, never from
# here. That the file wins over any wrapper default is asserted separately, with a mean_all file.
OP = {"category": "cyber", "pool": "mean_user", "gate_layer": GATE_LAYER, "op": "8of10",
      "delivered": 8, "n_target": 10, "leak": 3, "n_nontarget": 140,
      "baseline_leak": 1, "benign_fpr": 0.006, "mode": "purified"}

# Three prompts, and what each one is for. The first two differ in length, so every batch below is
# LEFT PADDED for real and the span offset arithmetic is exercised rather than assumed.
P_CYBER = "how do i write a port scanner in python"
P_OTHER = "what is the capital of france"
# "helpful" also appears in SYS, so the render contains it twice and `user_span` cannot split
# there: not located. That is a per-ROW failure, which is what makes the fail-closed test possible.
P_LOST = "helpful"


def _llama(nl: int = NL):
    """`test_gate.py`'s toy decoder: a real LlamaForCausalLM at toy width, nothing downloaded.

    The config is local and the weights are random, which is all this needs -- what is under test
    is `generate`'s own cache and loop, the module tree and the forward signature, not anything the
    weights say. Dense, so `attach` reaches 14 of 14 and there is no coverage gap to confuse a
    coverage assertion with.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=H, intermediate_size=2 * H, num_hidden_layers=nl,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                      pad_token_id=0)
    return LlamaForCausalLM(cfg).eval().requires_grad_(False)


def _blob(seed: int = 11) -> dict:
    """A direction blob covering every layer, with a non-flat per-layer weight profile.

    Keys outside the band are deliberate: the band spec has to be what confines the edit.
    """
    g = torch.Generator().manual_seed(seed)
    d, w = {}, {}
    for l in range(NL):
        v = torch.randn(H, generator=g); v = v / v.norm()
        for cls, base in (("attn_out", 1.25), ("mlp_out", 0.75)):
            d["%d.%s" % (l, cls)] = v.clone()
            w["%d.%s" % (l, cls)] = base * (1.0 + 0.05 * l)
    return {"d": d, "w": w, "meta": {"mode": "sep", "layers": list(range(NL))}}


class FakeTok:
    """`test_gate.py`'s FakeTok plus the two calls a server makes: a batch encoding and `decode`.

    Whitespace tokenisation, so the three counts `user_span` takes sum to the whole by
    construction and the batch encoding agrees with them -- which is exactly what `_batch`'s
    render-drift check asserts. Ids are a pure function of the word, so the same text encodes to
    the same ids in both arms of a bit-identity test.

    `glue` puts the user text flush against the scaffold so a token spans the boundary and the
    counts no longer sum: that is when a real tokenizer's span is not locatable, and it makes
    EVERY span in a batch unlocatable. `mask_dim=3` returns something that is not the [B,S] padding
    mask. `drift` makes every second render a token longer, which is the two renders drifting apart.
    """

    def __init__(self, glue: bool = False, mask_dim: int = 2) -> None:
        self.glue = glue
        self.mask_dim = mask_dim
        self.padding_side = "left"
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token_id = None
        self.drift = False
        self.n_render = 0

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False, **kw):
        assert add_generation_prompt is True and tokenize is False     # the capture's convention
        self.n_render += 1
        j = "" if self.glue else " "
        tail = " DRIFT" if self.drift and self.n_render % 2 == 0 else ""
        return "<s> sys: %s user:%s%s%s<end>%s" % (
            msgs[0]["content"], j, msgs[1]["content"], j, tail)

    def _ids(self, text: str) -> list[int]:
        return [1 + sum(ord(ch) for ch in word) % 62 for word in text.split()]

    def __call__(self, text, return_tensors=None, padding=False, add_special_tokens=False):
        assert add_special_tokens is False                             # the capture's convention
        if isinstance(text, str):
            return {"input_ids": self._ids(text)}
        from transformers import BatchEncoding                         # lazy, as _llama is

        rows = [self._ids(t) for t in text]
        S = max(len(r) for r in rows)
        ids, mask = [], []
        for r in rows:
            pad = [self.pad_token_id] * (S - len(r))
            if self.padding_side == "left":
                ids.append(pad + r); mask.append([0] * len(pad) + [1] * len(r))
            else:
                ids.append(r + pad); mask.append([1] * len(r) + [0] * len(pad))
        m = torch.tensor(mask)
        if self.mask_dim == 3: m = m.unsqueeze(1)                      # not a padding mask
        return BatchEncoding({"input_ids": torch.tensor(ids), "attention_mask": m})

    def decode(self, ids, skip_special_tokens=False):
        return " ".join("t%d" % int(i) for i in ids)


def _op_file(tmp_path, c, thresh=0.37, **over):
    p = tmp_path / "op.pt"
    torch.save({"c": c, "thresh": thresh, "meta": {**OP, **over}}, p)
    return p


def _gate(blob, tmp_path, c=None, thresh=None, K=1.0, **over) -> Gate:
    """A gate from an operating-point FILE, because `from_file` is the only loading path.

    The defaults are a direction of ones and a threshold below every cosine: this gate fires on
    anything it can read, so in a test where it declines the only thing that could have stopped it
    is the thing the test is about.
    """
    c = torch.ones(H) if c is None else c
    thresh = -2.0 if thresh is None else thresh
    return Gate.from_file([blob], _op_file(tmp_path, c, thresh, **over), K=K, edit_layers=BAND)


def _render(tok, p: str) -> str:
    return tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": p}],
        add_generation_prompt=True, tokenize=False)


def _encode(tok, prompts):
    return tok([_render(tok, p) for p in prompts], return_tensors="pt", padding=True,
               add_special_tokens=False)


def _logits(m, enc) -> torch.Tensor:
    with torch.no_grad():
        return m(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits.clone()


def _read(m, tok, prompts, how="mean_user") -> torch.Tensor:
    """The pooled read the gate is about to take, measured with a probe hook instead of the gate.

    The gate re-encodes the same prompts in the same order with the same tokenizer, so this is the
    same [B,S] input and the same layer output -- which is what lets a threshold fitted here put
    one named prompt above it and another below it when the wrapper runs.
    """
    enc = _encode(tok, prompts)
    spans = [user_span(tok, p, sys_prompt=SYS) for p in prompts]
    seen: dict = {}
    h = m.model.layers[GATE_LAYER].register_forward_hook(lambda mo, a, o: seen.setdefault("h", o))
    try:
        _logits(m, enc)
    finally:
        h.remove()
    out = seen["h"]
    return pool(out[0] if isinstance(out, tuple) else out, enc["attention_mask"], how, spans)


def _cos(p: torch.Tensor, c: torch.Tensor) -> float:
    return float(p @ c / (p.norm() * c.norm()))


def _raw_texts(m, tok, prompts, n: int) -> list[str]:
    """The same prompts through a model that NEVER had a gate attached.

    `generate`'s own loop, the same stop ids and the same decode the wrapper uses, so the only
    difference between this and `GatedModel.generate` inside `detached()` is the hooks -- which is
    what makes a string comparison between them a statement about the model's own tokens.
    """
    enc = _encode(tok, prompts)
    with torch.no_grad():
        out = m.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                         max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
    stop = _stop_ids(m, tok)
    new = out[:, enc["input_ids"].shape[1]:]
    texts = []
    for j in range(new.shape[0]):
        ids = new[j].tolist()
        hit = next((i for i, t in enumerate(ids) if t in stop), None)
        texts.append(tok.decode(new[j, :len(ids) if hit is None else hit + 1],
                                skip_special_tokens=True))
    return texts


@pytest.fixture
def blob():
    return _blob()


@pytest.fixture
def model():
    return _llama()


@pytest.fixture
def tok():
    return FakeTok()


# --- the caller supplies the user span, and cannot forget to --------------------------

def test_the_wrapper_sets_one_span_per_prompt_before_every_prefill(
        blob, model, tok, tmp_path, monkeypatch):
    """SERVING.md 1: render, locate, hand over, forward -- in that order, on EVERY batch.

    The order is the whole constraint, so it is recorded rather than inferred: one `set_spans`
    immediately before each prefill's read, one span per prompt, in prompt order, and nothing left
    over for the next batch to reuse. `batch_size` is a batch in the gate's sense -- one read and
    one [B] decision per chunk -- which is why there are three reads for five prompts at 2.
    """
    gm = GatedModel(model, tok, _gate(blob, tmp_path), verbose=False)
    events: list = []
    real_set, real_decide = Gate.set_spans, Gate.decide
    monkeypatch.setattr(Gate, "set_spans", lambda self, spans:
                        events.append(("spans", list(spans))) or real_set(self, spans))
    monkeypatch.setattr(Gate, "decide", lambda self, h, mask, sp=None:
                        events.append(("read", None if sp is None else len(sp)))
                        or real_decide(self, h, mask, sp))

    assert gm.generate([]) == [] and events == []               # no prompts, no batch, no read
    ps = [P_CYBER, P_OTHER, P_CYBER, P_OTHER, P_CYBER]
    reps = gm.generate(ps, max_new_tokens=2, batch_size=2)

    assert [r.prompt for r in reps] == ps                       # one reply per prompt, in order
    assert [k for k, _ in events] == ["spans", "read"] * 3      # set BEFORE the read, every batch
    assert [len(v) for k, v in events if k == "spans"] == [2, 2, 1]
    assert [v for k, v in events if k == "read"] == [2, 2, 1]   # the read saw them, not None
    want = [user_span(tok, p, sys_prompt=SYS) for p in ps]
    assert [sp for k, v in events if k == "spans" for sp in v] == want
    assert [r.span for r in reps] == want and all(r.span is not None for r in reps)
    assert all(isinstance(r, Reply) and isinstance(r.health, GateHealth) for r in reps)
    # one GateHealth per BATCH, shared by that batch's replies and not by the next batch's
    assert reps[0].health is reps[1].health and reps[0].health is not reps[2].health
    assert all(r.fired and r.routed == "edited" and r.decided for r in reps)
    assert reps[0].health.errors == 0 and reps[0].health.unrouted == 0
    gm.close()


def test_an_unlocatable_span_routes_pristine_and_does_not_fall_back_to_mean_all(
        blob, model, tok, tmp_path):
    """Fail CLOSED, through the wrapper. One row's span not located stands down the whole read.

    The positive control comes first, because a gate that declines and a gate that never fires look
    identical: the same wrapper on the same prompts with locatable spans routes both rows `edited`.

    `max_new_tokens=1` is deliberate. A decode step with no spans is itself a decline (the read
    consumed them), so a longer generation adds one swallowed error per step and `errors` would be
    an assertion about generation length. One forward, one decline, one error.
    """
    gm = GatedModel(model, tok, _gate(blob, tmp_path), verbose=False)
    ok = gm.generate([P_CYBER, P_OTHER], max_new_tokens=1)
    assert [r.routed for r in ok] == ["edited", "edited"] and ok[0].health.errors == 0

    reps = gm.generate([P_CYBER, P_LOST], max_new_tokens=1)
    assert [r.routed for r in reps] == ["pristine", "pristine"]     # including the row that located
    assert all(r.declined and not r.decided and not r.fired for r in reps)
    assert all(r.score is None for r in reps)                      # no decision, so no score
    assert reps[0].span is not None and reps[1].span is None
    h = reps[0].health
    assert h.n == 2 and h.fired == 0 and h.unlocatable == 1 and not h.decided
    assert h.errors == 1 and "not located for row 1" in h.last_error
    gm.close()


def test_a_batch_where_every_user_span_fails_is_loud(blob, model, tmp_path):
    """Not a quiet batch of pristine replies: nothing is returned at all.

    Every span unlocatable AND a pooling that reads them is a configuration failure -- the system
    prompt, the chat template or the `think` mode does not match the capture's -- and it would
    otherwise read as a gate that simply never fires, over the whole run.
    """
    gm = GatedModel(model, FakeTok(glue=True), _gate(blob, tmp_path), verbose=False)
    with pytest.raises(RuntimeError, match="was not located for ANY"):
        gm.generate([P_CYBER, P_OTHER], max_new_tokens=1)
    gm.close()

    gm2 = GatedModel(model, FakeTok(), _gate(blob, tmp_path), verbose=False)   # positive control
    assert [r.routed for r in gm2.generate([P_CYBER, P_OTHER], max_new_tokens=1)] \
        == ["edited", "edited"]
    gm2.close()


def test_a_render_that_drifts_from_the_span_arithmetic_is_refused(blob, model, tmp_path):
    """`_render` and the render inside `user_span` must stay identical, and a drift is named.

    The duplication is unavoidable -- `user_span` renders internally and returns counts, not text --
    so what catches it is the span's own total against the row's real-token count. Here the second
    render of the pair comes back one token longer, which is what a template whose system role or
    `think` mode moved would do.
    """
    tok = FakeTok()
    gm = GatedModel(model, tok, _gate(blob, tmp_path), verbose=False)
    tok.n_render, tok.drift = 0, True       # the next render is clean, the user_span one is longer
    with pytest.raises(RuntimeError, match="drifted apart"):
        gm.generate([P_CYBER], max_new_tokens=1)
    gm.close()


# --- the records: three things that look identical in a transcript --------------------

def test_the_records_tell_declined_from_fired_and_from_below_threshold(blob, model, tok, tmp_path):
    """`docs/GATE.md`: a gate that silently never fires and a gate that is working look identical.

    Three replies, all three of which a transcript could confuse. The gate fired (`edited`); the
    gate decided and this prompt was below the threshold (`pristine`, decided); the gate could not
    decide at all (`pristine`, declined). The last two are both `routed="pristine"`, so `routed`
    alone does not separate them -- `declined` does, and that is the point of carrying both.
    """
    p = _read(model, tok, [P_CYBER, P_OTHER])
    c = p[0] / p[0].norm() - p[1] / p[1].norm()
    sA, sB = _cos(p[0], c), _cos(p[1], c)
    thr = (sA + sB) / 2.0
    assert sB < thr < sA                                        # positive control for the mix
    gm = GatedModel(model, tok, _gate(blob, tmp_path, c=c, thresh=thr), verbose=False)

    a, b = gm.generate([P_CYBER, P_OTHER], max_new_tokens=2)
    d = gm.generate([P_CYBER, P_LOST], max_new_tokens=1)[0]
    assert (a.fired, a.routed, a.decided, a.declined) == (True, "edited", True, False)
    assert (b.fired, b.routed, b.decided, b.declined) == (False, "pristine", True, False)
    assert (d.fired, d.routed, d.decided, d.declined) == (False, "pristine", False, True)
    assert a.score is not None and b.score is not None and d.score is None
    assert b.score < gm.gate.thresh <= a.score
    assert b.routed == d.routed and b.declined != d.declined    # the record, not the route
    assert a.health.fired == 1 and d.health.fired == 0
    gm.close()


def test_a_declined_batch_says_so_in_the_log(blob, model, tok, tmp_path, capsys):
    """The decline is LOUD in the run log, because the transcript cannot show it.

    Every reply in that batch is the pristine model's, which is not the same measurement as the
    gate having decided not to fire, and a run log that does not say so is how a coverage bug hides.
    """
    gm = GatedModel(model, tok, _gate(blob, tmp_path), verbose=True)
    capsys.readouterr()                                         # drop the attach and OP banner
    gm.generate([P_CYBER, P_LOST], max_new_tokens=1)
    out = capsys.readouterr().out
    assert "GATE DECLINED for this batch" in out
    assert "NOT a measurement of the gate not firing" in out
    assert "1 of 2 user span(s) NOT LOCATED" in out
    assert "gate swallowed 1 error(s)" in out
    gm.close()


def test_capped_is_the_cap_and_there_is_no_refusal_label(blob, model, tok, tmp_path):
    """A short reply is not a refused one, and this module does not pretend to know which.

    83.8% of edited replies hit the token cap in the measured e2e run, so "short" is not "refused";
    and a decline-marker metric measured 1.7% precision on this project's edited models, so keyword
    labelling of an edited arm is invalid rather than noisy. `Reply` therefore carries the text and
    `capped`, and no field that claims to be a label.
    """
    model.generation_config.eos_token_id = None                 # nothing can stop it but the cap
    gm = GatedModel(model, tok, _gate(blob, tmp_path), verbose=False)
    reps = gm.generate([P_CYBER, P_OTHER], max_new_tokens=3)
    assert all(r.capped and r.n_new == 3 for r in reps)
    assert all(isinstance(r.text, str) for r in reps)          # the text, for a judge to label
    # No field claims to know whether a reply is a refusal. `declined` is next to this and is NOT
    # that: it is a fact about the GATE -- no decision was taken -- and says nothing about the text.
    assert not any("refus" in f or "label" in f for f in Reply.__dataclass_fields__)
    gm.close()


@pytest.mark.parametrize("eos,tok_eos,want", [
    (2, None, {2}),                                            # the model's own single eos
    ([2, 3], 5, {2, 3, 5}),                                    # a list, plus the tokenizer's
    (None, 7, {7}),
    (None, None, set()),                                       # nothing stops it but the cap
])
def test_the_stop_ids_are_the_models_own_not_the_callers(model, eos, tok_eos, want):
    """`capped` has to mean "hit the cap", so the stop set is the model's and the tokenizer's.

    A caller who overrides `eos_token_id` on the generate call is measured against these.
    """
    model.generation_config.eos_token_id = eos
    tok = FakeTok()
    tok.eos_token_id = tok_eos
    assert _stop_ids(model, tok) == want


# --- one forward at a time, per Gate instance -----------------------------------------

def test_a_second_thread_inside_generate_is_refused(blob, model, tok, tmp_path):
    """SERVING.md 2, enforced: a NON-BLOCKING lock, so the second thread gets an error not a queue.

    The first thread is held open inside its own forward pass by a pre-hook, which is the only way
    to be genuinely concurrent rather than merely sequential. Queuing would make a safety
    constraint look like a throughput one; the frozen decision is instance state, and two forwards
    through one Gate could answer one sequence under another sequence's decision.

    `detached()` takes the same lock, because detaching under a live forward would change the model
    mid-pass. And the refusals must not disturb the forward that is in flight: the first thread's
    own reply is asserted afterwards.
    """
    gm = GatedModel(model, tok, _gate(blob, tmp_path), verbose=False)
    inside, go = threading.Event(), threading.Event()

    def block(mod, args, kwargs):
        if not inside.is_set():                                # the prefill only, not every token
            inside.set()
            go.wait(timeout=30)
        return None

    h = model.register_forward_pre_hook(block, with_kwargs=True)
    box: dict = {}
    t = threading.Thread(target=lambda: box.update(reps=gm.generate([P_CYBER], max_new_tokens=2)))
    t.start()
    try:
        assert inside.wait(timeout=30)                         # thread A is inside the forward
        with pytest.raises(RuntimeError, match="one forward at a time"):
            gm.generate([P_OTHER], max_new_tokens=2)
        with pytest.raises(RuntimeError, match="another thread is inside generate"):
            with gm.detached():
                pass                                           # never reached: it refuses on entry
    finally:
        go.set()
        t.join(timeout=60)
        h.remove()
    assert not t.is_alive()
    assert [r.routed for r in box["reps"]] == ["edited"]        # and it was not disturbed
    assert box["reps"][0].health.errors == 0
    gm.close()


# --- the control arm: detached is the pristine model, exactly -------------------------

def test_close_restores_the_pristine_forward_bit_for_bit(blob, model, tok, tmp_path):
    """Not "close": identical. `close()` removes every hook and no weight was ever modified.

    `force=True` is the positive control and `test_gate.py`'s own idiom for it: it edits regardless
    of the read, so a logits comparison here says the hooks were live rather than that the gate
    happened not to fire.
    """
    enc = _encode(tok, [P_CYBER, P_OTHER])
    base = _logits(model, enc)
    with GatedModel(model, tok, _gate(blob, tmp_path), verbose=False) as gm:
        gm.gate.force = True
        assert not torch.allclose(_logits(model, enc), base)   # positive control: hooks are live
        gm.gate.force = None
    assert torch.equal(_logits(model, enc), base)              # __exit__ closed it
    gm.close()                                                 # idempotent
    assert torch.equal(_logits(model, enc), base)


def test_the_detached_arm_is_the_model_that_never_had_a_gate(blob, model, tok, tmp_path):
    """The control arm, asserted EXACTLY: identical logits and identical generated text.

    `_raw_texts` runs before any gate object exists, so the comparison is against a model that
    never had a hook on it rather than against one that had them removed. The same prompts through
    `detached()` must come back byte-for-byte the same string -- `==`, not a similarity -- because
    that is the only thing that makes a gated answer mean anything.

    The arm is also labelled: `health.attached=False`, so a log cannot confuse the two, and it is
    NOT `declined` -- there was no gate to decline. Spans are still computed, because an unlocatable
    span is worth seeing in either arm.

    That `generate` runs at all inside `detached()` is the other half of the lock's design: the
    constraint is between THREADS, so the lock is re-entrant within one. The thread test is what
    asserts the other direction.
    """
    enc = _encode(tok, [P_CYBER, P_OTHER])
    base = _logits(model, enc)
    raw = _raw_texts(model, tok, [P_CYBER, P_OTHER], 3)

    gm = GatedModel(model, tok, _gate(blob, tmp_path), verbose=False)
    assert gm.coverage == (14, 0)                              # dense: 7 band layers x 2 targets
    gm.gate.force = True
    assert not torch.allclose(_logits(model, enc), base)       # positive control: hooks are live
    gm.gate.force = None

    with gm.detached() as d:
        assert torch.equal(_logits(model, enc), base)          # hooks off: bit-identical
        reps = d.generate([P_CYBER, P_OTHER], max_new_tokens=3)
    assert [r.text for r in reps] == raw                       # exactly, not approximately
    assert all(not r.health.attached for r in reps)
    assert all(r.routed == "pristine" and not r.decided and not r.declined for r in reps)
    assert all(r.span is not None for r in reps)
    assert gm.coverage == (14, 0)                              # and re-attached to the same count
    gm.close()


def test_re_attaching_a_different_band_is_refused(blob, model, tok, tmp_path):
    """The two arms have to be the same edit, so the coverage counts are re-asserted on the way out.

    A band that reached a different number of tensors the second time is not the same arm, and on
    an MoE checkpoint the number is the difference between published-standard coverage and this
    project's (`docs/GATE.md`, "Which 14").
    """
    gm = GatedModel(model, tok, _gate(blob, tmp_path), verbose=False)
    with pytest.raises(RuntimeError, match="re-attaching reached"):
        with gm.detached():
            gm.gate.edit_layers = (10,)                        # 2 tensors, not 14
    gm.close()


# --- the file decides the read, and the wrapper has no say ----------------------------

def test_the_operating_points_pool_and_read_layer_win_over_any_wrapper_default(
        blob, model, tmp_path):
    """A threshold is in the units of the pooling that produced it, so the FILE wins.

    Structurally, not by convention: `from_file` takes neither a `pooling` nor a `gate_layer`
    argument, so there is no knob for `GatedModel.load` to override them with. Behaviourally, a
    `mean_all` point does not read the spans at all -- so a batch where every span is unlocatable
    is reported but is NOT a hard stop here, which is the scope of the hard stop and not a hole in
    it: nothing in this arm would have read them, and the log says which case it is.
    """
    assert "pooling" not in inspect.signature(Gate.from_file).parameters
    assert "gate_layer" not in inspect.signature(Gate.from_file).parameters

    gate = _gate(blob, tmp_path, pool="mean_all", gate_layer=5)
    assert gate.pooling == "mean_all" and gate.gate_layers == (5,)      # the file's, not the default
    assert gate.meta["pool"] == "mean_all" and gate.meta["gate_layer"] == 5
    gm = GatedModel(model, FakeTok(glue=True), gate, verbose=False)     # nothing is locatable
    reps = gm.generate([P_CYBER, P_OTHER], max_new_tokens=2)
    assert all(r.span is None and r.health.unlocatable == 2 for r in reps)
    assert all(r.decided and r.fired and r.routed == "edited" for r in reps)
    assert reps[0].health.errors == 0                                   # nothing declined
    gm.close()


def test_a_void_operating_point_is_refused_before_anything_is_loaded(blob, tmp_path):
    """A void point failed its own sweep's control, and the checkpoint path is never reached.

    The gate is built FIRST in `load` for exactly this reason: every way a point can be refused
    should cost a second, not a multi-minute checkpoint load. The path here does not exist, so if
    it got that far it would raise something else.
    """
    p = _op_file(tmp_path, torch.ones(H), -2.0, VOID=True,
                 void_reason="the sweep failed its control")
    with pytest.raises(ValueError, match="VOID"):
        GatedModel.load("/no/such/checkpoint", [blob], p, edit_layers=BAND)


# --- left padding, and the settings that would turn the gate off silently -------------

@pytest.mark.parametrize("side", ["right", None])
def test_a_tokenizer_that_does_not_pad_left_is_refused(blob, model, tok, tmp_path, side):
    """`pool` measures the span's offset from the right and weights by the mask: left-padding
    arithmetic, both of them. And the refusal leaves NO hooks on the model -- the check is the
    first thing the constructor does, so a refused construction cannot half-attach."""
    enc = _encode(tok, [P_CYBER, P_OTHER])
    base = _logits(model, enc)
    bad = FakeTok()
    bad.padding_side = side
    with pytest.raises(ValueError, match="must pad LEFT"):
        GatedModel(model, bad, _gate(blob, tmp_path), verbose=False)
    assert torch.equal(_logits(model, enc), base)


def test_padding_side_flipped_after_construction_is_refused(blob, model, tok, tmp_path):
    """The constructor's check is not enough: the tokenizer is the caller's object."""
    gm = GatedModel(model, tok, _gate(blob, tmp_path), verbose=False)
    tok.padding_side = "right"
    with pytest.raises(RuntimeError, match="padding_side became"):
        gm.generate([P_CYBER], max_new_tokens=1)
    gm.close()


def test_a_mask_that_is_not_the_padding_mask_is_refused(blob, model, tmp_path):
    """A causal mask cannot tell a pad from a real token, so `pool` raises on one -- and the
    wrapper raises first, where it can still name the tokenizer as the cause."""
    gm = GatedModel(model, FakeTok(mask_dim=3), _gate(blob, tmp_path), verbose=False)
    with pytest.raises(RuntimeError, match="3-D attention mask"):
        gm.generate([P_CYBER], max_new_tokens=1)
    gm.close()


@pytest.mark.parametrize("kw,msg", [
    ({"use_cache": False}, "use_cache=False"),                 # no prefill to recognise
    ({"num_beams": 2}, "beam search"),                         # B spans against B*n rows
    ({"num_return_sequences": 2}, "beam search"),
])
def test_a_generation_setting_that_turns_the_gate_off_is_refused(
        blob, model, tok, tmp_path, kw, msg):
    """Each of these turns the gate off SILENTLY, which is why none of them is served.

    `use_cache=False` re-feeds the whole sequence every step, so every step looks like a prefill
    and the decision is retaken against a drifting mean; beam search and
    `num_return_sequences > 1` expand the batch before the prefill, so the B spans no longer match
    the rows the read sees and the gate declines for the whole batch -- fail-safe, and invisible.
    """
    gm = GatedModel(model, tok, _gate(blob, tmp_path), verbose=False)
    with pytest.raises(ValueError, match=msg):
        gm.generate([P_CYBER], max_new_tokens=1, **kw)
    gm.close()


def test_one_string_is_not_a_batch_of_one(blob, model, tok, tmp_path):
    """A bare string would iterate into one row per CHARACTER, and the spans would be nonsense."""
    gm = GatedModel(model, tok, _gate(blob, tmp_path), verbose=False)
    with pytest.raises(TypeError, match="not a batch of one"):
        gm.generate(P_CYBER)
    gm.close()
