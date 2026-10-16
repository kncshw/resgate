"""The runtime gate: the edit it applies must be `pre_step`'s, and everything else is a trap.

Parity is the test that decides whether this module is trustworthy. For the same direction blob
and dose K, a module whose weights `patch.pre_step` has actually edited and the UNEDITED module
with the hook forced on must produce the same output; forced off, the hooked module must produce
the unedited output EXACTLY. Nothing else here means anything if that fails -- and the answer to a
failure is to report it, not to loosen the tolerance.

The rest of the file is the restriction: per sequence, decided once at prefill, mask-weighted
under left padding, and no edit on any failure whatsoever.

No real model and no checkpoint: a small random `nn.Linear` stack on CPU at fixed seeds, named the
way a checkpoint names things so `layer_of` and `key_of` do their real work, over the real band --
layers 10-16, which is 14 direction keys and 21 target tensors, 14 of them 2-D and 7 fused 3-D.
"""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from resgate.enumerate_targets import OUT_NAMES
from resgate.gate import (EDIT_LAYERS, FUSED_RECON_TOL, GATE_LAYER, META_FIELDS, Gate,
                        fused_correction, pool, precompute, user_span)
from resgate.patch import key_of, layer_of, pre_step

H, F, E, NL = 8, 16, 2, 17
BAND = EDIT_LAYERS                                   # layers 10-16
S = 5
# cyber at 8-of-10 delivered, from docs/GATE.md. The numbers are provenance, not thresholds the
# code reads: the point of them being in the file is that the file says which point it is.
# `pool` is "mean_all" and NOT the module default, because that is what the published cyber sweep
# was measured with -- mean_all at layer 8 with a purified direction, per three artifacts named in
# docs/GATE.md. A threshold is in the units of the pooling that produced it, so the file has to say
# which, and the gate has to obey the file over its own default. `mode` is an extra provenance key:
# META_FIELDS is a minimum, and anything else in meta is carried through to `gate.meta`.
OP = {"category": "cyber", "pool": "mean_all", "gate_layer": GATE_LAYER, "op": "8of10",
      "delivered": 8, "n_target": 10, "leak": 3, "n_nontarget": 140,
      "baseline_leak": 1, "benign_fpr": 0.006, "mode": "purified"}


def _lin(i: int, o: int, g: torch.Generator, dtype=torch.float32) -> nn.Linear:
    m = nn.Linear(i, o, bias=False, dtype=dtype)
    with torch.no_grad():
        m.weight.copy_(torch.randn(o, i, generator=g))
    return m.requires_grad_(False)


class Blk(nn.Module):
    """A stand-in decoder layer: two residual writes, one read, one fused expert stack.

    POSITION-WISE on purpose -- there is no attention mixing anywhere in this file -- so a row's
    hidden states do not depend on what else is in the batch or on how much it was padded. That is
    what lets the left-padding test isolate the POOLING and the mixed-batch test isolate the MASK:
    if either comes out unequal it is the gate's arithmetic, not the model's.
    """

    def __init__(self, g: torch.Generator, flat: bool = False, dtype=torch.float32) -> None:
        super().__init__()
        self.flat = flat
        self.self_attn = nn.Module()
        self.self_attn.o_proj = _lin(H, H, g, dtype)               # 2-D target -> attn_out
        self.mlp = nn.Module()
        self.mlp.up_proj = _lin(H, F, g, dtype)                    # reads: never a target
        self.mlp.down_proj = _lin(F, H, g, dtype)                  # 2-D target -> mlp_out
        # A fused routed-expert stack, declared and never used in the forward: it is here so the
        # coverage report has the thing a hook cannot reach. Its arithmetic is checked directly.
        # The name is the one the project's primary MoE checkpoint actually uses --
        # `layers.N.experts.down_proj`, with NO `mlp.` segment. `key_of` maps it to mlp_out on the
        # `down_proj` token alone, so both that shape and `mlp.experts.down_proj` land on the
        # layer's direction; this file uses the real one.
        self.experts = nn.Module()
        self.experts.down_proj = nn.Parameter(torch.randn(E, H, F, generator=g).to(dtype))

    def forward(self, h, **kw):
        h = h + self.self_attn.o_proj(h)
        t = torch.tanh(self.mlp.up_proj(h))
        if self.flat:                                 # rows are tokens, not sequences
            return h + self.mlp.down_proj(t.reshape(-1, F)).view_as(h)
        return h + self.mlp.down_proj(t)


class Stack(nn.Module):
    """Model root: `layers`, and a forward naming the two kwargs the gate binds against."""

    def __init__(self, g: torch.Generator, nl: int = NL, flat_at=(), dtype=torch.float32) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Blk(g, l in flat_at, dtype) for l in range(nl)])
        self.requires_grad_(False)                   # served weights; and gate.py sets no global

    def forward(self, h, attention_mask=None, past_key_values=None):
        for l in self.layers: h = l(h)
        return h


class Cache:
    """The one thing `_is_prefill` asks a KV cache. Length 0 is a prefill, anything else decode."""

    def __init__(self, n: int) -> None:
        self.n = n

    def get_seq_length(self) -> int:
        return self.n


def _blob(seed: int = 11) -> dict:
    """A direction blob covering EVERY layer, with a non-flat per-layer weight profile.

    Keys outside the band are deliberate: the band spec has to be what confines the edit, not the
    blob happening to stop at layer 10. Both classes at a layer share that layer's direction and
    differ in weight, which is exactly what `direction.compute_directions` emits.
    """
    g = torch.Generator().manual_seed(seed)
    d, w = {}, {}
    for l in range(NL):
        v = torch.randn(H, generator=g); v = v / v.norm()
        for cls, base in (("attn_out", 1.25), ("mlp_out", 0.75)):
            d["%d.%s" % (l, cls)] = v.clone()
            w["%d.%s" % (l, cls)] = base * (1.0 + 0.05 * l)        # a key mix-up changes the answer
    return {"d": d, "w": w, "meta": {"mode": "sep", "layers": list(range(NL))}}


def _x(b: int, s: int, seed: int) -> torch.Tensor:
    return torch.randn(b, s, H, generator=torch.Generator().manual_seed(seed))


def _targets(model: nn.Module, band=BAND) -> list[str]:
    """Every direction-keyed target PARAMETER in the band, 2-D and fused 3-D alike."""
    return [n for n, p in model.named_parameters()
            if p.dim() in (2, 3) and any(o in n for o in OUT_NAMES) and layer_of(n) in band]


def _materialise(model: nn.Module, gate: Gate, skip=(),
                 dtype=torch.float32) -> tuple[nn.Module, int]:
    """A deep copy of `model` with every band target's weight replaced by `pre_step`'s output.

    This is the trusted path -- the same function `patch()` writes to disk with. Copy BEFORE
    attaching: a hooked module's hooks hold a reference to the gate. `dtype` is float32 because
    that is what `patch()` computes in; the float64 parity arm raises it on both sides at once.
    """
    ref = copy.deepcopy(model)
    n = 0
    for name, mod in ref.named_modules():
        if not isinstance(mod, nn.Linear) or name in skip: continue
        if not any(o in name for o in OUT_NAMES): continue
        l = layer_of(name)
        if l is None or l not in gate.edit_layers: continue
        k = key_of(name, l)
        if k not in gate.d: continue
        v = gate.d[k].to(dtype).flatten(); v = v / v.norm()
        with torch.no_grad():
            mod.weight.copy_(pre_step(mod.weight.detach().to(dtype), v, gate.w[k]))
        n += 1
    return ref, n


def _pooled(model: nn.Module, x, mask=None, how="mean_all", spans=None) -> torch.Tensor:
    """The pooled read the gate would take, measured with a probe hook instead of the gate."""
    seen: dict = {}
    h = model.layers[GATE_LAYER].register_forward_hook(lambda m, a, o: seen.setdefault("h", o))
    try:
        model(x, attention_mask=mask)
    finally:
        h.remove()
    return pool(seen["h"], mask, how, spans)


def _cos(p: torch.Tensor, c: torch.Tensor) -> float:
    return float(p @ c / (p.norm() * c.norm()))


def _two_sided(model, xA, xB, mA=None, mB=None):
    """A category direction and threshold with xA above it and xB below. The positive control.

    c = unit(pA) - unit(pB) makes cos(pA,c) - cos(pB,c) = ||c|| > 0 by construction, so this does
    not depend on a lucky seed -- and the caller still asserts the ordering it was handed.
    """
    pA, pB = _pooled(model, xA, mA)[0], _pooled(model, xB, mB)[0]
    c = pA / pA.norm() - pB / pB.norm()
    sA, sB = _cos(pA, c), _cos(pB, c)
    return c, (sA + sB) / 2.0, sA, sB


def _tok_pair(model, seedA=21, seedB=22):
    """Two single tokens, the prompts made of them repeated, and a direction and threshold that put
    A above the threshold and B below it -- as PROMPTS and as single DECODE tokens both.

    Nothing in this file mixes across positions, so the pooled read of a token repeated S times is
    the read of that token. That is what makes "the decode step's own read is on the other side of
    the threshold" true by construction rather than by a lucky seed.
    """
    tokA, tokB = _x(1, 1, seedA), _x(1, 1, seedB)
    xA, xB = tokA.repeat(1, S, 1), tokB.repeat(1, S, 1)
    c, thr, sA, sB = _two_sided(model, xA, xB, torch.ones(1, S), torch.ones(1, S))
    return tokA, tokB, xA, xB, c, thr, sA, sB


def _ref_and_gate(model, blob, c=None, thresh=None, K=1.0, force=None, skip=(),
                  dtype=torch.float32, pooling="mean_all", **kw):
    """(gate, materialised reference, n edited) from ONE spec, in the only safe order.

    The reference copy is taken before any hook exists, and both sides are driven by the same
    gate object -- so they cannot end up at different K, which is a mistake this file made once.

    `pooling` is pinned to `mean_all` rather than taking the module default, because these tests
    are about the DECISION MACHINERY -- per-sequence masking, the freeze, the fail-safe paths --
    and `mean_all` needs no span plumbing. A gate declining here therefore has exactly one
    possible cause, which is the point of the test. The default pooling, `mean_user`, has its own
    section at the end of this file.
    """
    gate = Gate([blob], c, thresh, K=K, pooling=pooling, **kw)
    gate.force = force
    ref, n = _materialise(model, gate, skip=skip, dtype=dtype)
    gate.attach(model, verbose=False)
    return gate, ref, n


def _forced(model, blob, K=1.0, on=True, **kw) -> Gate:
    return _ref_and_gate(model, blob, K=K, force=on, **kw)[0]


@pytest.fixture
def blob():
    return _blob()


@pytest.fixture
def model():
    return Stack(torch.Generator().manual_seed(7))


# --- the equivalence, at float64 where it is exact ------------------------------------

@pytest.mark.parametrize("shape", [(H, H), (H, F), (16, 32)])
@pytest.mark.parametrize("w", [0.0, 0.4, 1.0, 1.9])
def test_precompute_reconstructs_pre_step(shape, w):
    """W - outer(w*(n (*) v), u) IS pre_step's W'. Equal matrices, so equal outputs for any x."""
    g = torch.Generator().manual_seed(3)
    W = torch.randn(*shape, generator=g, dtype=torch.float64)
    v = torch.randn(shape[0], generator=g, dtype=torch.float64); v = v / v.norm()
    u, wnv = precompute(W, v, w)
    assert u.shape == (shape[1],) and wnv.shape == (shape[0],)
    assert torch.allclose(W - torch.outer(wnv, u), pre_step(W, v, w), atol=1e-14, rtol=0)


@pytest.mark.parametrize("w", [0.4, 1.0, 1.9])
def test_precompute_matches_pre_step_per_fused_expert(w):
    """A fused [E,out,in] stack: the same two lines, one (u, wnv) pair per expert slice."""
    g = torch.Generator().manual_seed(5)
    W = torch.randn(E, H, F, generator=g, dtype=torch.float64)
    v = torch.randn(H, generator=g, dtype=torch.float64); v = v / v.norm()
    u, wnv = precompute(W, v, w)
    assert u.shape == (E, F) and wnv.shape == (E, H)
    for e in range(E):
        assert torch.allclose(W[e] - torch.outer(wnv[e], u[e]), pre_step(W[e], v, w),
                              atol=1e-14, rtol=0), e


@pytest.mark.parametrize("dtype,rel", [(torch.float32, 1e-6), (torch.float64, 1e-14)])
@pytest.mark.parametrize("w", [0.4, 1.25, 1.9])
def test_the_hook_order_of_operations_is_one_rounding(dtype, rel, w):
    """The hook never builds W'; it computes y - w*(x @ u)*(n (*) v). Same answer, other order.

    One Linear, so nothing accumulates and this is the arithmetic on its own. Measured worst:
    1.8e-7 relative at float32 (about 1.5 float32 eps) and 2.3e-16 at float64 (about 1 eps).
    """
    g = torch.Generator().manual_seed(9)
    W = torch.randn(H, F, generator=g).to(dtype)
    v = torch.randn(H, generator=g).to(dtype); v = v / v.norm()
    x = torch.randn(64, F, generator=g).to(dtype)
    u, wnv = precompute(W, v, w)
    hook = x @ W.T - (x @ u).unsqueeze(-1) * wnv
    ref = x @ pre_step(W, v, w).T
    err, scale = float((hook - ref).abs().max()), float(ref.abs().max())
    assert err <= rel * scale, (err, scale, err / scale)


# --- parity through a real forward, over the real band --------------------------------

@pytest.mark.parametrize("dtype,rel", [(torch.float32, 1e-5), (torch.float64, 1e-12)])
@pytest.mark.parametrize("K", [0.5, 1.0, 1.25, 1.5])
def test_parity_forced_on_matches_the_materialised_edit(blob, dtype, rel, K):
    """THE test: the hooked unedited stack against pre_step's own weights, over all 14 2-D keys.

    Measured worst over these sixteen cases: 1.32e-6 relative at float32, 1.5e-15 at float64.
    NINE orders between the two arms is the statement that the gap is rounding and not algebra --
    which is why the float64 arm is here at all, since float32 is what anyone will actually serve.
    The bounds carry ~8x and ~700x headroom over those measurements. If one of them ever needs
    loosening, that is a finding to report, not a constant to edit.
    """
    model = Stack(torch.Generator().manual_seed(7), dtype=dtype)
    x, m = _x(2, S, 17).to(dtype), torch.ones(2, S, dtype=dtype)
    y0 = model(x, attention_mask=m)
    gate, ref, n = _ref_and_gate(model, blob, K=K, force=True, dtype=dtype)
    assert n == 14                                             # 7 band layers x 2 2-D targets
    y_ref, y = ref(x, attention_mask=m), model(x, attention_mask=m)
    err, scale = float((y - y_ref).abs().max()), float(y_ref.abs().max())
    assert err <= rel * scale, (err, scale, err / scale)
    assert float((y_ref - y0).abs().max()) > 1e3 * max(err, 1e-300)   # not a no-op being compared
    assert gate.fired.tolist() == [True, True]


def test_forced_off_is_bit_identical_to_the_unedited_module(blob):
    """Not "close": identical. A gate that is off must not perturb the model at all."""
    model = Stack(torch.Generator().manual_seed(7))
    x, m = _x(2, S, 17), torch.ones(2, S)
    y0 = model(x, attention_mask=m)
    gate = _forced(model, blob, on=False)
    assert torch.equal(model(x, attention_mask=m), y0)
    assert not bool(gate.fired.any())


def test_the_band_is_14_keys_and_21_tensors(blob, model):
    """layers 10-16 -> 14 direction keys -> 21 target tensors, of which a hook reaches 14.

    The 14 are `self_attn.o_proj` and `mlp.down_proj`, seven each; the 7 it misses are the fused
    `experts.down_proj` stacks. That split is not arbitrary -- see docs/GATE.md, "Which 14".
    """
    names = _targets(model)
    assert len(names) == 21
    assert len({key_of(n, layer_of(n)) for n in names}) == 14
    gate = Gate([blob], None, None)
    hooked, unreached = gate.attach(model, verbose=False)
    assert (hooked, unreached) == (14, 7)


def test_the_fused_stacks_a_hook_cannot_reach_are_named_not_hidden(blob, model, capsys):
    """The coverage gap is reported by name. docs/PITFALLS.md 2 is about not noticing this one."""
    Gate([blob], None, None).attach(model, verbose=True)
    out = capsys.readouterr().out
    assert "14 tensors hooked, 7 direction-keyed target(s) NOT reached" in out
    for l in BAND:
        assert "layers.%d.experts.down_proj" % l in out


def test_the_fused_stacks_still_have_an_exact_edit(blob, model):
    """What a hook cannot reach, `patch()` can: each expert slice has its own exact (u, wnv).

    So the gap is in the hook machinery, not in the arithmetic -- which is why an MoE checkpoint
    wanting full coverage needs the offline patch, and why `attach` returns that count.
    """
    gate = Gate([blob], None, None)
    n = 0
    for name, p in model.named_parameters():
        if p.dim() != 3 or layer_of(name) not in BAND: continue
        k = key_of(name, layer_of(name))
        v = gate.d[k].float().flatten(); v = v / v.norm()
        W = p.detach().double()
        u, wnv = precompute(W, v.double(), gate.w[k])
        for e in range(E):
            assert torch.allclose(W[e] - torch.outer(wnv[e], u[e]),
                                  pre_step(W[e], v.double(), gate.w[k]), atol=1e-13, rtol=0)
        n += 1
    assert n == 7


@pytest.mark.parametrize("band,want", [((10,), 2), ((10, 11), 4), ((16,), 2), (BAND, 14)])
def test_the_band_spec_is_what_confines_the_edit(blob, model, band, want):
    """The blob carries all 17 layers, so the only thing holding the edit to the band is the band."""
    assert Gate([blob], None, None, edit_layers=band).attach(model, verbose=False)[0] == want


def test_bfloat16_parity_is_bfloat16_rounding(blob):
    """At bf16 the two paths differ, and inherently: the offline path rounds the WEIGHT, the hook
    rounds the CORRECTION. Measured on one target Linear: max|diff| ~0.02 against |y| ~8, i.e.
    about 2 bf16 ulps. Parity is a float32 statement; this is the bf16 caveat, measured.
    """
    g = torch.Generator().manual_seed(7)
    mod = _lin(F, H, g, torch.bfloat16)
    v = blob["d"]["10.mlp_out"].float(); v = v / v.norm()
    w = blob["w"]["10.mlp_out"]
    x = torch.randn(4, F, generator=g).to(torch.bfloat16)
    u, wnv = precompute(mod.weight.detach().float(), v, w)
    hook = mod(x) - (x @ u.to(x.dtype)).unsqueeze(-1) * wnv.to(x.dtype)
    ref = x @ pre_step(mod.weight.detach().float(), v, w).to(torch.bfloat16).T
    err, scale = float((hook - ref).abs().max()), float(ref.abs().max())
    assert err <= 0.02 * scale, (err, scale)
    assert err > 0.0                                           # it is not exact, and says so


# --- the read is upstream of the edit -------------------------------------------------

@pytest.mark.parametrize("bad", [10, 11, 16, 20])
def test_the_read_must_be_strictly_upstream_of_the_edit(blob, bad):
    with pytest.raises(ValueError, match="strictly upstream"):
        Gate([blob], None, None, gate_layer=bad, edit_layers=BAND)


@pytest.mark.parametrize("ok", [0, 8, 9])
def test_a_read_below_the_band_is_accepted(blob, ok):
    assert Gate([blob], None, None, gate_layer=ok, edit_layers=BAND).gate_layers == (ok,)


def test_an_ambiguous_or_missing_read_site_is_refused(blob):
    shallow = Stack(torch.Generator().manual_seed(7), nl=5)
    with pytest.raises(ValueError, match="exactly one"):
        Gate([blob], None, None).attach(shallow, verbose=False)

    two = Stack(torch.Generator().manual_seed(7))
    two.other = Stack(torch.Generator().manual_seed(8))        # a second `layers.8` in the tree
    with pytest.raises(ValueError, match="exactly one"):
        Gate([blob], None, None).attach(two, verbose=False)


# --- per sequence, not per batch ------------------------------------------------------

def test_mixed_batch_each_row_gets_its_own_decision(blob):
    """One firing row and one not, in one forward. Each row must equal its own solo run."""
    model = Stack(torch.Generator().manual_seed(7))
    xA, xB = _x(1, S, 21), _x(1, S, 22)
    c, thr, sA, sB = _two_sided(model, xA, xB)
    assert sB < thr < sA                                       # positive control for the mix
    x, m = torch.cat([xA, xB]), torch.ones(2, S)
    y0 = model(x, attention_mask=m)
    gate, ref, _ = _ref_and_gate(model, blob, c, thr)
    y_edit = ref(x, attention_mask=m)
    y = model(x, attention_mask=m)
    assert gate.fired.tolist() == [True, False]
    assert torch.equal(y[1], y0[1])                            # did not fire: untouched, exactly
    assert not torch.allclose(y[0], y0[0])                     # fired: and it mattered
    assert torch.allclose(y[0], y_edit[0], atol=1e-5 * float(y_edit.abs().max()), rtol=0)

    yA = model(xA, attention_mask=torch.ones(1, S))            # the same rows, run alone
    assert gate.fired.tolist() == [True]
    yB = model(xB, attention_mask=torch.ones(1, S))
    assert gate.fired.tolist() == [False]
    assert torch.equal(y[0], yA[0]) and torch.equal(y[1], yB[0])


def test_rows_that_are_not_sequences_are_not_edited(blob):
    """A down_proj fed a flattened token list: a per-sequence mask cannot be applied to rows that
    are not sequences, so that tensor is skipped and counted -- never applied to the whole batch.
    """
    model = Stack(torch.Generator().manual_seed(7), flat_at=(10, 11))
    skip = {"layers.10.mlp.down_proj", "layers.11.mlp.down_proj"}
    gate, ref, n = _ref_and_gate(model, blob, force=True, skip=skip)
    assert n == 12                                             # the other 12 are still edited
    x, m = _x(2, S, 17), torch.ones(2, S)
    y_ref = ref(x, attention_mask=m)
    y = model(x, attention_mask=m)
    assert gate.unrouted == 2                                  # one per flattened tensor
    assert torch.allclose(y, y_ref, atol=1e-5 * float(y_ref.abs().max()), rtol=0)


# --- decided once at prefill, then frozen ---------------------------------------------


def test_the_decision_is_frozen_across_decode_steps(blob):
    """Prefill on a firing prompt, then decode tokens whose OWN read is below the threshold.

    A gate that re-read would flip to not-firing mid-answer, which is the whole point of the
    freeze; that it would is asserted here, not assumed.
    """
    model = Stack(torch.Generator().manual_seed(7))
    tokA, tokB, xA, xB, c, thr, sA, sB = _tok_pair(model)
    one = torch.ones(1, 1)
    assert _cos(_pooled(model, tokB, one)[0], c) < thr < sA
    y_dec_0 = model(tokB, attention_mask=one, past_key_values=Cache(S))

    gate, ref, _ = _ref_and_gate(model, blob, c, thr)
    y_dec_edit = ref(tokB, attention_mask=one, past_key_values=Cache(S))
    model(xA, attention_mask=torch.ones(1, S))                 # prefill: the cache is empty
    assert gate.fired.tolist() == [True]
    scores = gate.scores.clone()
    for _ in range(3):
        y = model(tokB, attention_mask=one, past_key_values=Cache(S))
        assert gate.fired.tolist() == [True]                   # frozen ON
        assert torch.equal(gate.scores, scores)                # and not re-scored at all
        assert torch.allclose(y, y_dec_edit, atol=1e-5 * float(y_dec_edit.abs().max()), rtol=0)
        assert not torch.allclose(y, y_dec_0)



def test_a_frozen_no_stays_no(blob):
    """The same freeze in the safe direction: a prompt that did not unlock cannot unlock later."""
    model = Stack(torch.Generator().manual_seed(7))
    tokA, tokB, xA, xB, c, thr, sA, sB = _tok_pair(model)
    one = torch.ones(1, 1)
    assert _cos(_pooled(model, tokA, one)[0], c) > thr > sB
    y_dec_0 = model(tokA, attention_mask=one, past_key_values=Cache(S))

    gate, _, _ = _ref_and_gate(model, blob, c, thr)
    model(xB, attention_mask=torch.ones(1, S))
    assert gate.fired.tolist() == [False]
    for _ in range(3):
        y = model(tokA, attention_mask=one, past_key_values=Cache(S))
        assert gate.fired.tolist() == [False]
        assert torch.equal(y, y_dec_0)                         # bit-identical, every step



def test_a_new_prefill_re_decides(blob):
    """The freeze is per sequence: an empty cache is a new prompt and so a new decision."""
    model = Stack(torch.Generator().manual_seed(7))
    _, _, xA, xB, c, thr, _, _ = _tok_pair(model)
    gate, _, _ = _ref_and_gate(model, blob, c, thr)
    for x, want in ((xA, True), (xB, False), (xA, True)):
        model(x, attention_mask=torch.ones(1, S), past_key_values=Cache(0))
        assert gate.fired.tolist() == [want]


# --- left padding ---------------------------------------------------------------------

def test_pool_ignores_left_pads():
    """Hand-computed: two pad rows carrying 9s, then [1,2] [3,4] [5,6]. The mean is [3,4]."""
    h = torch.tensor([[[9., 9.], [9., 9.], [1., 2.], [3., 4.], [5., 6.]]])
    m = torch.tensor([[0., 0., 1., 1., 1.]])
    assert torch.allclose(pool(h, m, "mean_all"), torch.tensor([[3., 4.]]))
    assert torch.allclose(pool(h, None, "mean_all"), h.mean(1))  # None: every position is real
    assert torch.allclose(pool(h, m, "last"), torch.tensor([[5., 6.]]))
    assert not torch.allclose(pool(h, m, "mean_all"),
                              pool(h, None, "mean_all"))        # the pads really would move it



def test_a_padded_row_reads_the_same_as_the_same_row_alone(blob):
    """Unequal lengths, LEFT padded, with pads that are not zero. Two statements.

    (1) The read does not depend on what sits in the pad slots at all. Same batch shape, so the
        real positions are bit-identical and `torch.equal` is the right assertion.
    (2) It equals the short row run alone, to float32 rounding. A different batch shape is a
        different GEMM, so that one cannot be bit-exact -- and the UNWEIGHTED mean of the same
        batch misses by ~1e4 times the rounding gap, which is what makes (2) a measurement.
    """
    model = Stack(torch.Generator().manual_seed(7))
    short, long = _x(1, 3, 31), _x(1, S, 32)
    m = torch.tensor([[0.] * (S - 3) + [1.] * 3, [1.] * S])

    def batch(pad):
        return torch.cat([torch.cat([pad, short], 1), long], 0)

    p = _pooled(model, batch(7.0 * _x(1, S - 3, 33)), m)[0]
    assert torch.equal(p, _pooled(model, batch(-31.0 * _x(1, S - 3, 34)), m)[0])

    solo = _pooled(model, short, torch.ones(1, 3))[0]
    gap = float((p - solo).abs().max())
    assert gap <= 1e-5 * float(solo.abs().max()), gap
    unweighted = _pooled(model, batch(7.0 * _x(1, S - 3, 33)), None)[0]
    assert float((unweighted - solo).abs().max()) > 1e3 * gap

    c, thr, _, _ = _two_sided(model, short, long, torch.ones(1, 3), torch.ones(1, S))
    gate, _, _ = _ref_and_gate(model, blob, c, thr)            # mean_all: this tests the weighting
    model(batch(7.0 * _x(1, S - 3, 33)), attention_mask=m)
    padded = gate.scores.clone()
    model(short, attention_mask=torch.ones(1, 3))
    assert torch.allclose(gate.scores, padded[:1], atol=1e-6, rtol=0)



def test_a_mask_that_is_not_the_padding_mask_blocks_the_read(blob):
    """A 4-D causal mask cannot tell a pad from a token, so there is no read -- and no edit."""
    with pytest.raises(ValueError, match="padding mask"):
        pool(torch.zeros(1, 3, H), torch.zeros(1, 1, 3, 3))

    model = Stack(torch.Generator().manual_seed(7))
    x, m = _x(1, S, 21), torch.ones(1, S)
    y0 = model(x, attention_mask=m)
    # A threshold below every cosine: this gate fires on anything it can read, so what stops it
    # below is the mask and nothing else.
    gate, _, _ = _ref_and_gate(model, blob, torch.ones(H), -2.0)
    assert not torch.allclose(model(x, attention_mask=m), y0)      # positive control: it fires
    assert gate.fired.tolist() == [True]

    assert torch.equal(model(x, attention_mask=torch.zeros(1, 1, S, S)), y0)
    assert gate.fired is None and gate.errors == 1 and "padding mask" in gate.last_error



def test_a_forward_that_does_not_name_attention_mask_cannot_fire(blob, capsys):
    """An unknown mask is not "no padding". It is no read, which is no edit -- and it says so."""
    class Bare(Stack):
        def forward(self, h):                                  # no attention_mask parameter
            for l in self.layers: h = l(h)
            return h

    model = Bare(torch.Generator().manual_seed(7))
    x = _x(1, S, 21)
    y0 = model(x)
    gate = Gate([blob], torch.ones(H), -2.0, pooling="mean_all")   # fires on anything readable
    gate.attach(model, verbose=True)
    assert "does not name attention_mask" in capsys.readouterr().out
    assert torch.equal(model(x), y0)
    assert gate.fired is None and gate.errors == 1


# --- fail-safe: every failure lands on no edit ----------------------------------------

def test_an_exception_in_scoring_leaves_the_output_bit_identical(blob, monkeypatch):
    """The positive control comes first: with scoring working, the gate DOES change the output."""
    model = Stack(torch.Generator().manual_seed(7))
    xA, xB = _x(1, S, 21), _x(1, S, 22)
    c, thr, _, _ = _two_sided(model, xA, xB)
    y0 = model(xA, attention_mask=torch.ones(1, S))
    gate = Gate([blob], c, thr, pooling="mean_all")
    gate.attach(model, verbose=False)
    assert not torch.allclose(model(xA, attention_mask=torch.ones(1, S)), y0)

    monkeypatch.setattr(Gate, "score", lambda self, p: 1 / 0)
    assert torch.equal(model(xA, attention_mask=torch.ones(1, S)), y0)
    assert gate.fired is None and gate.scores is None
    assert "ZeroDivisionError" in gate.last_error and gate.errors == 1


@pytest.mark.parametrize("has_c,has_thresh", [(False, True), (True, False), (False, False)])
def test_no_direction_or_no_threshold_never_fires(blob, has_c, has_thresh):
    model = Stack(torch.Generator().manual_seed(7))
    x, m = _x(2, S, 17), torch.ones(2, S)
    y0 = model(x, attention_mask=m)
    gate = Gate([blob], torch.ones(H) if has_c else None, 0.0 if has_thresh else None,
                pooling="mean_all")
    gate.attach(model, verbose=False)
    assert torch.equal(model(x, attention_mask=m), y0)         # a threshold of 0.0 would fire
    assert gate.fired is None


def test_no_decision_yet_means_no_edit(blob):
    """A target module called on its own, with no forward through the root. Even forced on."""
    model = Stack(torch.Generator().manual_seed(7))
    mod = model.layers[10].self_attn.o_proj
    z = _x(1, S, 44)
    y0 = mod(z)
    gate = _forced(model, blob)
    assert gate.fired is None
    assert torch.equal(mod(z), y0)


def test_a_direction_of_the_wrong_width_is_not_hooked(model, capsys):
    """A shape mismatch is a missing hook, not a crash and not a wrong edit."""
    bad = _blob()
    bad["d"]["12.attn_out"] = torch.ones(H + 3)
    hooked, unreached = Gate([bad], None, None).attach(model, verbose=True)
    assert (hooked, unreached) == (13, 8)
    assert "SHAPE MISMATCH" in capsys.readouterr().out


def test_detach_restores_the_pristine_forward(blob):
    model = Stack(torch.Generator().manual_seed(7))
    x, m = _x(2, S, 17), torch.ones(2, S)
    y0 = model(x, attention_mask=m)
    gate = _forced(model, blob)
    assert not torch.allclose(model(x, attention_mask=m), y0)
    gate.detach()
    assert torch.equal(model(x, attention_mask=m), y0)
    gate.detach()                                              # idempotent
    assert torch.equal(model(x, attention_mask=m), y0)
    gate.attach(model, verbose=False)                          # and re-attachable
    assert not torch.allclose(model(x, attention_mask=m), y0)


def test_attaching_twice_is_refused(blob, model):
    gate = Gate([blob], None, None)
    gate.attach(model, verbose=False)
    with pytest.raises(RuntimeError, match="already attached"):
        gate.attach(model, verbose=False)


# --- the blob and the operating-point file --------------------------------------------

def test_sequential_blobs_are_refused(blob):
    with pytest.raises(ValueError, match="ONE direction blob"):
        Gate([blob, blob], None, None)
    with pytest.raises(ValueError, match="no direction blob"):
        Gate([], None, None)


def test_a_blob_missing_a_per_layer_weight_is_refused(blob):
    del blob["w"]["12.mlp_out"]
    with pytest.raises(ValueError, match="no per-layer weight"):
        Gate([blob], None, None)


def test_the_score_is_a_cosine_so_the_read_is_scale_free(blob):
    """Residual norms grow with depth and vary per prompt (docs/METHOD.md), so a threshold on a
    raw projection would be a threshold on activation magnitude. Scaling must not move it."""
    gate = Gate([blob], torch.randn(H, generator=torch.Generator().manual_seed(2)), 0.5)
    p = _x(1, 3, 5)[0]
    assert torch.allclose(gate.score(p), gate.score(7.5 * p), atol=1e-6)
    assert float(gate.score(p).abs().max()) <= 1.0


def _op_file(tmp_path, c, thresh=0.37, **over):
    p = tmp_path / "op.pt"
    torch.save({"c": c, "thresh": thresh, "meta": {**OP, **over}}, p)
    return p


def test_from_file_carries_the_operating_point_through(blob, tmp_path):
    """Direction, threshold, pooling and read layer come out of the file. None of them are here."""
    c = torch.randn(H, generator=torch.Generator().manual_seed(41))
    gate = Gate.from_file([blob], _op_file(tmp_path, c), K=1.25)
    assert gate.thresh == 0.37 and gate.gate_layers == (GATE_LAYER,)
    assert gate.pooling == "mean_all"                          # the file's, not the default
    assert gate.meta["mode"] == "purified"                     # extra provenance carried through
    assert gate.meta["category"] == "cyber" and gate.meta["leak"] == 3
    assert gate.meta["leak"] - gate.meta["baseline_leak"] == 2          # the excess, as published
    assert gate.K == 1.25
    assert gate.w["10.attn_out"] == pytest.approx(1.25 * blob["w"]["10.attn_out"])
    assert torch.allclose(gate.c, c)


@pytest.mark.parametrize("field", META_FIELDS)
def test_from_file_refuses_an_operating_point_missing_its_provenance(blob, tmp_path, field):
    """An operating point whose provenance is not recorded is not an operating point."""
    p = tmp_path / "op.pt"
    torch.save({"c": torch.ones(H), "thresh": 0.3,
                "meta": {k: v for k, v in OP.items() if k != field}}, p)
    with pytest.raises(ValueError, match=field):
        Gate.from_file([blob], p)


@pytest.mark.parametrize("payload", [{"thresh": 0.3}, {"c": torch.ones(H)}, {}])
def test_from_file_refuses_a_missing_direction_or_threshold(blob, tmp_path, payload):
    p = tmp_path / "op.pt"
    torch.save({**payload, "meta": dict(OP)}, p)
    with pytest.raises(ValueError, match="needs both"):
        Gate.from_file([blob], p)


def test_the_read_layer_in_the_file_is_checked_against_the_band(blob, tmp_path):
    """A file built at layer 12 against the default band is refused, not read at the wrong depth."""
    with pytest.raises(ValueError, match="strictly upstream"):
        Gate.from_file([blob], _op_file(tmp_path, torch.ones(H), gate_layer=12))


def test_an_unknown_pooling_in_the_file_is_refused(blob, tmp_path):
    with pytest.raises(ValueError, match="pooling must be one of"):
        Gate.from_file([blob], _op_file(tmp_path, torch.ones(H), pool="mean_of_everything"))


# --- against a real decoder: config-built, random weights, nothing downloaded ----------

def _llama(nl: int = NL):
    """A real LlamaForCausalLM at toy width. No network: the config is local and the weights
    are random, which is all this needs -- what is under test is the module tree, the forward
    signature, the layer-output convention and `generate`, not anything the weights say.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=H, intermediate_size=2 * H, num_hidden_layers=nl,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                      pad_token_id=0)
    return LlamaForCausalLM(cfg).eval().requires_grad_(False)


def _wide_blob(d: int = H) -> dict:
    g = torch.Generator().manual_seed(11)
    blob: dict = {"d": {}, "w": {}}
    for l in range(NL):
        v = torch.randn(d, generator=g); v = v / v.norm()
        for cls in ("attn_out", "mlp_out"):
            blob["d"]["%d.%s" % (l, cls)] = v.clone()
            blob["w"]["%d.%s" % (l, cls)] = 1.0
    return blob


def test_the_read_site_is_the_capture_convention_on_a_real_decoder():
    """acts[l] is the OUTPUT of layer l, i.e. `hidden_states[l+1]` (capture.py drops index 0).

    A forward hook on `model.layers[8]` must land on exactly that. This off-by-one would be
    completely silent -- the gate would score the wrong depth and still produce numbers.
    """
    m = _llama()
    assert [n for n, _ in m.named_modules() if n.endswith("layers.%d" % GATE_LAYER)] \
        == ["model.layers.%d" % GATE_LAYER]

    seen: dict = {}
    hk = m.model.layers[GATE_LAYER].register_forward_hook(lambda mo, a, o: seen.setdefault("h", o))
    try:
        out = m(input_ids=torch.randint(0, 64, (2, 6)), attention_mask=torch.ones(2, 6).long(),
                output_hidden_states=True)
    finally:
        hk.remove()
    h = seen["h"][0] if isinstance(seen["h"], tuple) else seen["h"]
    assert torch.equal(h, out.hidden_states[GATE_LAYER + 1])
    assert not torch.equal(h, out.hidden_states[GATE_LAYER])


def test_a_real_decoder_exposes_the_band_and_the_kwargs_the_gate_binds():
    """14 targets in the band on a dense checkpoint, and a forward naming both kwargs it reads."""
    import inspect

    m = _llama()
    params = inspect.signature(m.forward).parameters
    assert "attention_mask" in params and "past_key_values" in params
    gate = Gate([_wide_blob()], None, None)
    assert gate.attach(m, verbose=False) == (14, 0)             # dense: no fused stacks to miss
    gate.detach()


def test_the_freeze_survives_a_real_generate(monkeypatch):
    """One decision per `generate` call, not one per token -- through the real Cache and loop."""
    m = _llama()
    ids, msk = torch.randint(0, 64, (2, 6)), torch.ones(2, 6).long()
    gate = Gate([_wide_blob()], torch.ones(H), -2.0, pooling="mean_all")   # fires on anything
    base = m(input_ids=ids, attention_mask=msk).logits.clone()
    gate.attach(m, verbose=False)
    assert not torch.allclose(m(input_ids=ids, attention_mask=msk).logits, base)

    calls: list = []
    real = Gate.decide
    monkeypatch.setattr(Gate, "decide",
                        lambda self, h, mask, spans=None:
                        calls.append(1) or real(self, h, mask, spans))
    m.generate(input_ids=ids, attention_mask=msk, max_new_tokens=6, do_sample=False)
    assert len(calls) == 1 and gate.fired.tolist() == [True, True]
    m.generate(input_ids=ids[:1], attention_mask=msk[:1], max_new_tokens=3, do_sample=False)
    assert len(calls) == 2 and gate.fired.tolist() == [True]    # a new prompt, a new decision
    assert gate.errors == 0 and gate.unrouted == 0

    gate.detach()
    assert torch.equal(m(input_ids=ids, attention_mask=msk).logits, base)


def test_the_split_is_o_proj_plus_down_proj_against_the_fused_stacks(blob, model):
    """Name the two halves, because which 14 are reachable is the whole practical limitation.

    The reachable half is exactly what published abliterations of this project's primary MoE
    checkpoint edit; the unreachable half is the routed-expert coverage this project added
    (docs/PITFALLS.md 2). A change that quietly moved a tensor across this line would change what
    the module is, so the line is asserted by name and not just by count.
    """
    gate = Gate([blob], None, None)
    hooked, unreached = gate.attach(model, verbose=False)
    names = set(_targets(model))
    fused = {n for n, p in model.named_parameters() if n in names and p.dim() == 3}
    assert len(fused) == 7 and all(n.endswith(".experts.down_proj") for n in fused)
    twod = names - fused
    assert len(twod) == 14
    assert {n.split("layers.")[1].split(".", 1)[1] for n in twod} == {
        "self_attn.o_proj.weight", "mlp.down_proj.weight"}
    assert (hooked, unreached) == (len(twod), len(fused))


class Multimodal(nn.Module):
    """A tree shaped like the project's primary checkpoint's index.

    Two things it reproduces that a plain `Stack` cannot. The language model's fused stacks sit at
    `language_model.layers.N.experts.down_proj`; and a vision tower carries its own
    `layers.N.self_attn.o_proj.weight`, which matches the band by layer and by role and must be
    excluded by name -- `enumerate_targets.EXCLUDE_PARTS` -- or the gate would edit a vision
    encoder and find two candidate read sites at layer 8.
    """

    def __init__(self, g: torch.Generator) -> None:
        super().__init__()
        self.language_model = Stack(g)
        self.vision_tower = nn.Module()
        self.vision_tower.vision_model = nn.Module()
        self.vision_tower.vision_model.encoder = Stack(g)

    def forward(self, h, attention_mask=None, past_key_values=None):
        return self.language_model(h, attention_mask=attention_mask,
                                   past_key_values=past_key_values)


def test_a_vision_tower_is_excluded_from_both_the_read_and_the_edit(blob, capsys):
    """The tower's `layers.8` would make the read site ambiguous and its o_projs would be edited.

    `attach` succeeding at all is the proof on the read side: it refuses anything but exactly one
    candidate, so a single `(14, 7)` here says the tower was excluded from both.
    """
    m = Multimodal(torch.Generator().manual_seed(7))
    assert sum(1 for n, _ in m.named_modules() if n.endswith("layers.%d" % GATE_LAYER)) == 2
    gate = Gate([blob], None, None)
    assert gate.attach(m, verbose=True) == (14, 7)
    out = capsys.readouterr().out
    assert "read language_model.layers.%d" % GATE_LAYER in out
    assert "vision_tower" not in out
    for l in BAND:
        assert "language_model.layers.%d.experts.down_proj" % l in out
    gate.detach()


# --- the read pools the USER SPAN, which is a measured requirement ---------------------
#
# ablation/NOTES-2026-10-05-cross-model-ablation-vs-classify.md 3.1: on four of five models a chat
# scaffold position carries an activation 11x-31x the user span's and nearly constant across
# prompts, so `mean_all` is dominated by a near-constant and the category read falls to 0.56-0.70
# AUC against 0.89-0.93 for `mean_user` at the same layer. These tests encode the mechanism, with
# the magnitudes taken from that range, so the default cannot quietly go back.

SPAN = (1, 2, 1, 4)                    # a=1 prefix, u=2 user, b=1 suffix, tot=4 real tokens
PAD_M = torch.tensor([[0.] + [1.] * 4] * 2)          # [pad][pre][usr][usr][suf], LEFT padded


def _scaffold_batch(scale: float, seed: int = 51, pad: float = 9.0):
    """[2,5,H] hidden states: two rows differing ONLY in their user span, sharing a scaffold
    position `scale` times a constant vector. `scale` is the measured norm ratio.
    """
    g = torch.Generator().manual_seed(seed)
    const = torch.randn(H, generator=g)
    x = torch.zeros(2, 5, H)
    x[:, 0, :] = pad * torch.randn(H, generator=g)   # a pad slot: nothing may depend on it
    x[:, 1, :] = scale * const                       # the scaffold prefix, shared by both rows
    for j in range(2):
        u = torch.randn(H, generator=g)
        x[j, 2, :] = u
        x[j, 3, :] = u
    x[:, 4, :] = const                               # the scaffold suffix, shared
    return x, PAD_M.clone(), [SPAN, SPAN]


def _margin(p: torch.Tensor) -> float:
    """The cosine margin a midpoint threshold would have between two pooled reads."""
    c = p[0] / p[0].norm() - p[1] / p[1].norm()
    return _cos(p[0], c) - _cos(p[1], c)


@pytest.mark.parametrize("scale", [11.0, 31.0])
def test_a_scaffold_position_dominates_mean_all_and_not_mean_user(scale):
    """Two prompts, one scaffold constant. As the constant grows, `mean_all` makes them the same
    vector and `mean_user` does not move at all.

    Measured at pool level, so the numbers are exact: cos between the two rows' `mean_all` reads
    goes 0.82 -> 0.986 (11x) -> 0.998 (31x), while their `mean_user` reads are bit-identical to
    the 1x case at every scale.
    """
    base, m, spans = _scaffold_batch(1.0)
    big, _, _ = _scaffold_batch(scale)
    u_base, u_big = pool(base, m, "mean_user", spans), pool(big, m, "mean_user", spans)
    a_base, a_big = pool(base, m, "mean_all"), pool(big, m, "mean_all")

    assert torch.equal(u_base, u_big)                          # the scaffold is simply not read
    assert _cos(a_big[0], a_big[1]) > _cos(a_base[0], a_base[1])
    assert _cos(a_big[0], a_big[1]) > 0.98                     # the two prompts, now one vector
    assert abs(_cos(u_big[0], u_big[1])) < 0.9                 # still told apart
    assert float(a_big.norm(dim=-1).max() / u_big.norm(dim=-1).max()) > 2.0


@pytest.mark.parametrize("how,spans", [("mean_user", [SPAN, SPAN]), ("mean_all", None)])
def test_pad_content_is_never_read(how, spans):
    """Neither mean reads a pad slot, and `torch.equal` is the right assertion for that."""
    x, m, _ = _scaffold_batch(11.0, pad=9.0)
    y = x.clone()
    y[:, 0, :] = -77.0                                         # whatever a pad token embeds to
    assert torch.equal(pool(x, m, how, spans), pool(y, m, how, spans))


@pytest.mark.parametrize("scale", [11.0, 31.0])
def test_mean_user_reads_the_request_where_mean_all_reads_a_near_constant(scale):
    """The same mechanism end to end, at the gate's own read layer rather than at pool level.

    The decisive quantity is the MARGIN a threshold would have between the two prompts: measured
    1.51 under `mean_user` at every scale against 0.60 / 0.17 / 0.06 under `mean_all` at 1x / 11x /
    31x, i.e. the ratio grows 2.5x -> 9.0x -> 23.8x. `mean_all` does not become unable to separate
    them, it loses the signal-to-noise to do it reliably -- which is what an AUC near 0.56 is.
    """
    model = Stack(torch.Generator().manual_seed(7))
    x1, m, spans = _scaffold_batch(1.0)
    xs, _, _ = _scaffold_batch(scale)
    u = _pooled(model, xs, m, "mean_user", spans)
    assert torch.equal(u, _pooled(model, x1, m, "mean_user", spans))
    a1 = _pooled(model, x1, m, "mean_all")
    abig = _pooled(model, xs, m, "mean_all")
    assert _cos(abig[0], abig[1]) > 0.98 > _cos(a1[0], a1[1])
    assert _margin(u) / max(_margin(abig), 1e-12) > 5.0
    assert _margin(abig) < _margin(a1)                         # and it degrades with the scale


def test_the_gate_decides_from_the_user_span(blob):
    """The default pooling, end to end: one row fires on its request, the other does not."""
    model = Stack(torch.Generator().manual_seed(7))
    x, m, spans = _scaffold_batch(31.0)
    u = _pooled(model, x, m, "mean_user", spans)
    c = u[0] / u[0].norm() - u[1] / u[1].norm()
    sA, sB = _cos(u[0], c), _cos(u[1], c)
    thr = (sA + sB) / 2.0
    assert sB < thr < sA
    y0 = model(x, attention_mask=m)

    gate, _, _ = _ref_and_gate(model, blob, c, thr, pooling="mean_user")
    gate.set_spans(spans)
    y = model(x, attention_mask=m)
    assert gate.fired.tolist() == [True, False]
    assert torch.equal(y[1], y0[1])                            # the other request: untouched
    assert not torch.allclose(y[0], y0[0])


def test_mean_user_without_spans_declines_to_decide(blob):
    """Fail CLOSED. No span, no read, no edit -- and no quiet fall back to `mean_all`."""
    model = Stack(torch.Generator().manual_seed(7))
    x, m, spans = _scaffold_batch(31.0)
    y0 = model(x, attention_mask=m)
    # a threshold below every cosine: this gate fires on anything it can read, so the only thing
    # that can stop it is the missing span.
    gate, _, _ = _ref_and_gate(model, blob, torch.ones(H), -2.0, pooling="mean_user")
    assert torch.equal(model(x, attention_mask=m), y0)
    assert gate.fired is None and "set_spans" in gate.last_error

    gate.set_spans(spans)                                      # positive control: with them, it edits
    assert not torch.allclose(model(x, attention_mask=m), y0)
    assert gate.fired.tolist() == [True, True]


def test_an_unlocatable_span_declines_for_the_whole_batch(blob):
    """`user_span` returning None for one row stands the WHOLE read down. That is the conservative
    direction: one read, one decision vector, and no row gets an edit on a guess."""
    model = Stack(torch.Generator().manual_seed(7))
    x, m, _ = _scaffold_batch(31.0)
    y0 = model(x, attention_mask=m)
    gate, _, _ = _ref_and_gate(model, blob, torch.ones(H), -2.0, pooling="mean_user")
    gate.set_spans([SPAN, None])
    assert torch.equal(model(x, attention_mask=m), y0)
    assert gate.fired is None and "not located for row 1" in gate.last_error


def test_spans_are_consumed_by_the_read(blob):
    """A second prefill that forgets `set_spans` declines rather than reusing the first one's."""
    model = Stack(torch.Generator().manual_seed(7))
    x, m, spans = _scaffold_batch(31.0)
    y0 = model(x, attention_mask=m)
    gate, _, _ = _ref_and_gate(model, blob, torch.ones(H), -2.0, pooling="mean_user")
    gate.set_spans(spans)
    assert not torch.allclose(model(x, attention_mask=m), y0)
    assert torch.equal(model(x, attention_mask=m), y0)         # same batch, no spans: declines
    assert gate.fired is None and "set_spans" in gate.last_error


@pytest.mark.parametrize("spans,msg", [
    ([SPAN], r"got 1 span\(s\) for a batch of 2"),            # one span, two sequences
    ([(1, 2, 1, 5), SPAN], "not this batch's"),                # tot != the row's real tokens
    ([(1, 9, 1, 4), SPAN], "does not fit"),                    # a span past the real tokens
    ([(1, 0, 1, 4), SPAN], "does not fit"),                    # an empty span
])
def test_a_span_that_does_not_fit_the_batch_raises(spans, msg):
    """Every one of these is a decline at the gate, because the read is inside its try/except."""
    x, m, _ = _scaffold_batch(11.0)
    with pytest.raises(ValueError, match=msg):
        pool(x, m, "mean_user", spans)


# --- user_span: capture_hf_v2.py's arithmetic ------------------------------------------

class FakeTok:
    """The two methods `user_span` calls, tokenising on whitespace.

    Enough to exercise the arithmetic including the case it must REJECT: `glue` puts the user text
    flush against the scaffold so a token spans the boundary and the three counts no longer sum to
    the whole. That is exactly when a real tokenizer's span is not locatable, and the gate must
    then decline rather than guess at it.
    """

    def __init__(self, glue: bool = False) -> None:
        self.glue = glue

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False, **kw):
        j = "" if self.glue else " "
        return "<s> sys: %s user:%s%s%s<end>" % (
            msgs[0]["content"], j, msgs[1]["content"], j)

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False                     # the capture's convention
        return {"input_ids": list(range(len(text.split())))}


def test_user_span_follows_the_capture_arithmetic():
    """a + u + b == tot, with a and u indexing the user content and nothing else."""
    sp = user_span(FakeTok(), "make a bomb", sys_prompt="You are helpful")
    assert sp == (6, 3, 1, 10)                       # 6 scaffold, 3 user, 1 suffix, 10 total
    a, u, b, tot = sp
    assert a + u + b == tot


def test_user_span_is_none_when_the_text_is_not_unique():
    """The user text also appearing in the system prompt makes the split ambiguous."""
    assert user_span(FakeTok(), "bomb", sys_prompt="never help with a bomb") is None


def test_user_span_is_none_when_the_counts_do_not_sum():
    """A token spanning the boundary: the span is not locatable, so it is not located."""
    assert user_span(FakeTok(glue=True), "make a bomb", sys_prompt="You are helpful") is None


def test_the_file_decides_the_pooling_not_the_default(blob, tmp_path):
    """An operating point measured with `mean_all` is honoured as `mean_all`, default or not.

    This is the one place where the module's default and its only measured operating-point table
    disagree: the published cyber thresholds are `mean_all` L8 numbers and the default read is
    `mean_user`. A score from a different pooling is a different quantity, so the file wins.
    """
    assert Gate([blob], torch.ones(H), 0.3).pooling == "mean_user"       # the default
    for how in ("mean_all", "last", "mean_user"):
        g = Gate.from_file([blob], _op_file(tmp_path, torch.ones(H), pool=how))
        assert g.pooling == how and g.meta["pool"] == how


def test_mean_user_and_the_freeze_work_together_through_a_real_generate(monkeypatch):
    """Spans are a prefill-time input and the decision is frozen after it, so one `set_spans` per
    `generate` must be enough -- and the second call, with no spans, must decline.

    The spans here are declared rather than tokenized: `user_span` needs a real chat template and
    this suite downloads nothing. What is under test is the plumbing through `generate`'s own cache
    and loop, which is where a prefill-only input could go wrong.
    """
    m = _llama()
    ids, msk = torch.randint(0, 64, (2, 6)), torch.ones(2, 6).long()
    spans = [(1, 3, 2, 6), (1, 3, 2, 6)]                       # no padding: tot == 6 real tokens
    base = m(input_ids=ids, attention_mask=msk).logits.clone()

    gate = Gate([_wide_blob()], torch.ones(H), -2.0, pooling="mean_user")
    gate.attach(m, verbose=False)
    calls: list = []
    real = Gate.decide
    monkeypatch.setattr(Gate, "decide",
                        lambda self, h, mask, sp=None:
                        calls.append(1) or real(self, h, mask, sp))

    gate.set_spans(spans)
    m.generate(input_ids=ids, attention_mask=msk, max_new_tokens=5, do_sample=False)
    assert len(calls) == 1 and gate.fired.tolist() == [True, True]
    assert gate.errors == 0

    m.generate(input_ids=ids, attention_mask=msk, max_new_tokens=2, do_sample=False)
    assert gate.fired is None and "set_spans" in gate.last_error   # consumed, so it declines
    gate.detach()
    assert torch.equal(m(input_ids=ids, attention_mask=msk).logits, base)


# --- a declined read is as final as a successful one ----------------------------------
#
# The freeze used to key on `fired is not None`, which only holds after a read SUCCEEDS. A
# read that declined left `fired` None, so every decode token retried it. Three consequences,
# and the third is a safety hole rather than a diagnostics one:
#
#   1. `last_error` ended up holding whatever the LAST retry raised -- on decode that is a
#      [B,S]-versus-[B,1] shape complaint -- which buries the real cause.
#   2. `errors` counted one per decode token, so one fault read as N+1 faults.
#   3. `pool` does not raise for every pooling. With "last" it returns `h[:, -1, :]` with no
#      mask and no spans, so a retry on a decode step SUCCEEDS and sets `fired` from a
#      GENERATED token -- a decision about text the model just wrote, not about the request.
#
# `_decided` is the fix: set on every exit path of the read, cleared only by `reset()`, which
# a prefill calls. These tests pin all three consequences.


def test_a_declined_read_does_not_retry_on_every_decode_token(monkeypatch):
    """One decline is one error and one `last_error`, not one per generated token."""
    m = _llama()
    ids, msk = torch.randint(0, 64, (2, 6)), torch.ones(2, 6).long()

    gate = Gate([_wide_blob()], torch.ones(H), -2.0, pooling="mean_user")
    gate.attach(m, verbose=False)

    gate.set_spans([(1, 3, 2, 6), (1, 3, 2, 6)])
    m.generate(input_ids=ids, attention_mask=msk, max_new_tokens=5, do_sample=False)
    assert gate.fired.tolist() == [True, True] and gate.errors == 0

    # Second generate, spans consumed: the read must decline ONCE and stay declined.
    m.generate(input_ids=ids, attention_mask=msk, max_new_tokens=9, do_sample=False)
    assert gate.fired is None                                # fail-closed, as before
    assert gate.errors == 1, "one decline must be one error, not one per decode token"
    assert "set_spans" in gate.last_error, "last_error must name the CAUSE, not a decode symptom"
    gate.detach()


def test_a_declined_read_cannot_fire_from_a_generated_token_under_last_pooling(monkeypatch):
    """The safety half: with `last`, a retry on a decode step would read generated text.

    `pool(h, mask, "last")` returns `h[:, -1, :]` before it checks anything, so unlike
    `mean_user` it cannot raise its way out of a decode-step read. If the prefill read
    declines and the freeze does not hold, the FIRST decode step decides the request from
    the token the model just produced. That is not a smaller read, it is the wrong one.
    """
    m = _llama()
    ids, msk = torch.randint(0, 64, (2, 6)), torch.ones(2, 6).long()

    gate = Gate([_wide_blob()], torch.ones(H), -2.0, pooling="last")   # thresh -2: would fire
    gate.attach(m, verbose=False)

    calls: list = []
    real = Gate.decide

    def once(self, h, mask, sp=None):
        calls.append(tuple(h.shape))
        if len(calls) == 1:
            raise RuntimeError("synthetic prefill failure")            # decline the prefill
        return real(self, h, mask, sp)

    monkeypatch.setattr(Gate, "decide", once)
    m.generate(input_ids=ids, attention_mask=msk, max_new_tokens=5, do_sample=False)

    assert len(calls) == 1, "the read ran again after declining: %r" % (calls,)
    assert gate.fired is None, "fired was set from a generated token"
    assert gate._live is False
    assert gate.errors == 1
    gate.detach()


def test_a_prefill_re_arms_the_read_after_a_decline(monkeypatch):
    """`_decided` must not be sticky across prefills, or the gate would never fire again."""
    m = _llama()
    ids, msk = torch.randint(0, 64, (2, 6)), torch.ones(2, 6).long()
    spans = [(1, 3, 2, 6), (1, 3, 2, 6)]

    gate = Gate([_wide_blob()], torch.ones(H), -2.0, pooling="mean_user")
    gate.attach(m, verbose=False)

    m.generate(input_ids=ids, attention_mask=msk, max_new_tokens=3, do_sample=False)
    assert gate.fired is None and gate.errors == 1          # declined: no spans

    gate.set_spans(spans)                                    # a later caller gets it right
    m.generate(input_ids=ids, attention_mask=msk, max_new_tokens=3, do_sample=False)
    assert gate.fired.tolist() == [True, True], "a decline poisoned every later prefill"
    assert gate.errors == 1                                  # no new errors
    gate.detach()


# --- the fused routed-expert stacks: the other 7 of 21 ---------------------------------
#
# `Blk` declares a fused stack and never routes through it, which is what the coverage counts
# above are measured against and is still the right stub for them -- a bare 3-D Parameter with no
# routing code around it is exactly the thing the fused path must decline. This section adds one
# that IS routed, shaped and attributed the way `transformers.integrations.moe`'s
# `use_experts_implementation` shapes and attributes `Gemma4TextExperts`: a [experts, out, in]
# down projection, a [experts, 2*in, hidden] gate_up, `_apply_gate` as `_default_apply_gate` spells
# it, and the three layout flags the fused path READS off the module rather than assuming
# (`has_gate`, `has_bias`, `is_transposed`). Its forward takes the real contract -- a flattened
# [T,H] token list plus [T,K] routing -- which is why the down projection's input is a local that
# no forward hook can see, and has to be recomputed from the module boundary instead.
#
# Still no real model and nothing downloaded. What is being tested is the apply site, and the
# served checkpoint's own forward is proved against the reconstruction at `attach` time, on the
# box, by `gate._fused_recon_error` -- not here.

NE, TK = 4, 2                                        # experts per stack, routing slots per token


class MoEExperts(nn.Module):
    """A fused routed-expert stack: one 3-D down projection, driven by a token list.

    `shared` adds a dense term every token gets routed or not. Nothing in the fused path models
    it, so a stack carrying one has to FAIL the reconstruction proof and be declined -- even
    though the routed part of the correction would still have been right. That is the
    conservative side, and it is the side an unproven apply site has to fall on.
    """

    def __init__(self, g: torch.Generator, dtype=torch.float32, shared: bool = False) -> None:
        super().__init__()
        self.num_experts = NE
        self.has_gate, self.has_bias, self.is_transposed = True, False, False
        self.gate_up_proj = nn.Parameter((torch.randn(NE, 2 * F, H, generator=g) * 0.3).to(dtype))
        self.down_proj = nn.Parameter((torch.randn(NE, H, F, generator=g) * 0.3).to(dtype))
        self.act_fn = nn.GELU()
        self.shared = nn.Parameter(
            (torch.randn(H, H, generator=g) * 0.3).to(dtype)) if shared else None

    def _apply_gate(self, gate_up_out: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up_out.chunk(2, dim=-1)
        return self.act_fn(gate) * up

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        for e in range(self.num_experts):
            sel = top_k_index == e
            rows = sel.nonzero(as_tuple=True)[0]
            if rows.numel() == 0: continue
            x = self._apply_gate(hidden_states[rows] @ self.gate_up_proj[e].transpose(-2, -1))
            y = (x @ self.down_proj[e].transpose(-2, -1)) * top_k_weights[sel].unsqueeze(-1)
            out.index_add_(0, rows, y)
        if self.shared is not None: out = out + hidden_states @ self.shared.T
        return out


def _rms(h: torch.Tensor) -> torch.Tensor:
    """Pre-block normalisation, as a real decoder layer has one, over the last dim only."""
    return h / h.norm(dim=-1, keepdim=True).clamp_min(1e-12) * (H ** 0.5)


class MoEBlk(nn.Module):
    """`Blk`'s three residual writes with a routed MoE block as the third, over a TOKEN LIST.

    PRE-NORMALISED, unlike `Blk`, and that is numerics and not taste: a fused stack's gating is
    quadratic in its input, so `Blk`'s unnormalised residual -- which grows ~4x a layer and is
    harmless on its own -- squares that growth and overflows float32 well before layer 17. The
    norm is per token over the last dim only, so this stays POSITION-WISE like the rest of this
    file: a row's output does not depend on what else is in the batch or on how much it was
    padded, which is what lets the mixed-batch test isolate the mask. It also leaves the three
    branches the same order of magnitude, so a fused edit is actually visible in the output
    rather than buried under an exponentially larger residual.

    The flatten is the real one -- a row-major `[B,S,H] -> [B*S,H]` reshape -- because that
    layout is the entire reason `Gate._token_rows` can map a token back to its sequence.

    `ragged` breaks that on purpose: the stack is then called with a token list that is not B
    equal sequences, which no per-sequence mask can be expanded over.
    """

    def __init__(self, g: torch.Generator, dtype=torch.float32, shared: bool = False,
                 ragged: bool = False) -> None:
        super().__init__()
        self.ragged = ragged
        self.self_attn = nn.Module()
        self.self_attn.o_proj = _lin(H, H, g, dtype)               # 2-D target -> attn_out
        self.mlp = nn.Module()
        self.mlp.up_proj = _lin(H, F, g, dtype)                    # reads: never a target
        self.mlp.down_proj = _lin(F, H, g, dtype)                  # 2-D target -> mlp_out
        self.router = _lin(H, NE, g, dtype)                        # "router" matches no OUT_NAME
        self.experts = MoEExperts(g, dtype, shared)                # 3-D target -> mlp_out

    def forward(self, h, **kw):
        h = h + 0.5 * self.self_attn.o_proj(_rms(h))
        h = h + 0.5 * self.mlp.down_proj(torch.tanh(self.mlp.up_proj(_rms(h))))
        B, Sq, _ = h.shape
        flat = _rms(h).reshape(-1, H)                 # row-major: token t is sequence t // Sq
        w, idx = torch.topk(self.router(flat).softmax(-1), TK, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        if self.ragged:
            part = self.experts(flat[:-1], idx[:-1], w[:-1])
            out = torch.cat([part, torch.zeros_like(part[:1])])
        else:
            out = self.experts(flat, idx, w)
        return h + 0.5 * out.view(B, Sq, H)


class MoEStack(nn.Module):
    """Model root with a routed MoE block at every layer: the 21-of-21 shape."""

    def __init__(self, g: torch.Generator, nl: int = NL, dtype=torch.float32, **kw) -> None:
        super().__init__()
        self.layers = nn.ModuleList([MoEBlk(g, dtype, **kw) for _ in range(nl)])
        self.requires_grad_(False)                   # served weights; and gate.py sets no global

    def forward(self, h, attention_mask=None, past_key_values=None):
        for l in self.layers: h = l(h)
        return h


def _moe(seed: int = 7, **kw) -> MoEStack:
    return MoEStack(torch.Generator().manual_seed(seed), **kw)


def _declined(model: MoEStack) -> MoEStack:
    """The same model with every fused stack refused, and its FORWARD untouched.

    `is_transposed` is read by `_fused_refusal` and by nothing in the stub's forward, so this is
    the 14-of-21 arm of the same weights -- which is what makes "the fused pass changed nothing
    about the 2-D pass" a comparison and not an assertion.
    """
    for l in model.layers: l.experts.is_transposed = True
    return model


def _materialise_moe(model: nn.Module, gate: Gate,
                     dtype=torch.float32) -> tuple[nn.Module, int, int]:
    """`_materialise`, plus `patch()`'s own per-expert loop over the fused stacks. 21 of 21."""
    ref, n2 = _materialise(model, gate, dtype=dtype)
    n3 = 0
    for pname, p in ref.named_parameters():
        if p.dim() != 3: continue
        # The role filter `enumerate_targets` and `Gate._is_target` apply, and it earns its keep
        # here: `experts.gate_up_proj` is 3-D, is at a band layer, and `key_of` would hand it the
        # layer's ATTN_OUT direction. It writes to nothing but the stack's own gate, so it is not
        # a target at all, and patching it would make this reference a different model.
        if not any(o in pname for o in OUT_NAMES): continue
        l = layer_of(pname)
        if l is None or l not in gate.edit_layers: continue
        k = key_of(pname, l)
        if k not in gate.d: continue
        v = gate.d[k].to(dtype).flatten(); v = v / v.norm()
        with torch.no_grad():
            p.copy_(torch.stack([pre_step(p[e].detach().to(dtype), v, gate.w[k])
                                 for e in range(p.shape[0])]))
        n3 += 1
    return ref, n2, n3


def _moe_ref_and_gate(model, blob, c=None, thresh=None, K=1.0, force=None,
                      dtype=torch.float32) -> tuple[Gate, nn.Module]:
    """(gate, materialised 21-of-21 reference), reference copied before any hook exists."""
    gate = Gate([blob], c, thresh, K=K, pooling="mean_all")
    gate.force = force
    ref, n2, n3 = _materialise_moe(model, gate, dtype=dtype)
    assert (n2, n3) == (14, 7)
    assert gate.attach(model, verbose=False) == (21, 0)
    return gate, ref


def _routing(model: nn.Module, x, msk) -> list:
    """Every layer's [T,K] expert choice for one forward. The parity arms have to AGREE on it.

    The two arms differ by rounding, and a topk that flipped on that rounding would make them
    different models -- a parity failure that is not an arithmetic failure, and the one way this
    section could fail for a reason that is not the thing under test. So it is checked first.
    """
    seen: list = []
    hs = [l.experts.register_forward_hook(
        lambda mod, a, o, seen=seen: seen.append(a[1].tolist())) for l in model.layers]
    try:
        model(x, attention_mask=msk)
    finally:
        for h in hs: h.remove()
    return seen


def test_fused_correction_is_the_materialised_edit_per_routed_pair(blob):
    """The arithmetic alone, at float64: the correction IS routing those tokens through pre_step.

    No model, no hooks, no gate. `pre_step` is applied to every expert slice -- `patch()`'s own
    loop -- and the difference the stack's forward then shows has to be what `fused_correction`
    computed without touching a weight. The mask is mixed, so this also pins that a zero row
    contributes exactly nothing rather than nearly nothing.
    """
    g = torch.Generator().manual_seed(13)
    mod = MoEExperts(g, dtype=torch.float64)
    v = blob["d"]["10.mlp_out"].double().flatten(); v = v / v.norm()
    w = blob["w"]["10.mlp_out"]
    u, wnv = precompute(mod.down_proj.detach(), v, w)
    assert u.shape == (NE, F) and wnv.shape == (NE, H)
    h = torch.randn(6, H, generator=g, dtype=torch.float64)
    idx = torch.randint(0, NE, (6, TK), generator=g)
    wt = torch.rand(6, TK, generator=g, dtype=torch.float64)
    m = torch.tensor([1., 0., 1., 1., 0., 1.], dtype=torch.float64)
    got = fused_correction(mod, h, idx, wt, u, wnv, m)
    y0 = mod(h, idx, wt)
    assert got.shape == (6, H)
    with torch.no_grad():
        mod.down_proj.copy_(torch.stack([pre_step(mod.down_proj[e].detach(), v, w)
                                         for e in range(NE)]))
    want = (y0 - mod(h, idx, wt)) * m.unsqueeze(-1)
    err, scale = float((got - want).abs().max()), float(want.abs().max())
    assert err <= 1e-12 * scale, (err, scale, err / scale)
    assert torch.equal(got[1], torch.zeros(H, dtype=torch.float64))
    assert torch.equal(got[4], torch.zeros(H, dtype=torch.float64))


def test_a_row_that_did_not_fire_never_reaches_the_gate_recomputation(blob, monkeypatch):
    """Not "contributes zero": not computed. The pair loop is restricted to the fired rows.

    So a non-fired row's output is the unedited stack's by construction, which is a stronger
    statement than multiplying its correction by zero -- and on a selective gate, where most
    sequences do not fire, it is also where most of the work goes.
    """
    import resgate.gate as G
    g = torch.Generator().manual_seed(13)
    mod = MoEExperts(g, dtype=torch.float64)
    v = blob["d"]["10.mlp_out"].double().flatten(); v = v / v.norm()
    u, wnv = precompute(mod.down_proj.detach(), v, blob["w"]["10.mlp_out"])
    h = torch.randn(6, H, generator=g, dtype=torch.float64)
    idx = torch.randint(0, NE, (6, TK), generator=g)
    wt = torch.rand(6, TK, generator=g, dtype=torch.float64)
    rows: list = []
    real = G._fused_x
    monkeypatch.setattr(G, "_fused_x",
                        lambda mo, hh, e: rows.append(hh.shape[0]) or real(mo, hh, e))
    G.fused_correction(mod, h, idx, wt, u, wnv,
                       torch.tensor([1., 0., 1., 1., 0., 1.], dtype=torch.float64))
    assert sum(rows) == 4 * TK                                 # 4 fired rows x TK slots, no more
    rows.clear()
    G.fused_correction(mod, h, idx, wt, u, wnv, torch.zeros(6, dtype=torch.float64))
    assert rows == []                                          # nothing fired: nothing computed


def test_a_routed_stack_closes_the_band_to_21_of_21(blob, capsys):
    """The count this whole path exists for, and it names which 7 and what proved them."""
    gate = Gate([blob], None, None)
    assert gate.attach(_moe(), verbose=True) == (21, 0)
    out = capsys.readouterr().out
    assert "21 tensors hooked, 0 direction-keyed target(s) NOT reached" in out
    assert "NOT REACHED" not in out
    for l in BAND:
        assert ("FUSED, driven through its own forward  layers.%d.experts.down_proj" % l) in out
    assert "reconstruction" in out and "relative" in out


@pytest.mark.parametrize("dtype,rel", [(torch.float32, 1e-5), (torch.float64, 1e-12)])
@pytest.mark.parametrize("K", [0.5, 1.0, 1.5])
def test_parity_on_all_21_matches_the_materialised_edit(blob, dtype, rel, K):
    """THE test, extended to the other 7: the hooked unedited model against pre_step's weights.

    `test_parity_forced_on_matches_the_materialised_edit` is this statement for the 14 that are
    `nn.Linear`. This one patches all 21 -- the fused ones through `patch()`'s per-expert loop --
    and the hook has to reproduce it through a routing permutation it never sees. Same bounds as
    the 2-D arm, because it is the same two lines of arithmetic.
    """
    m = _moe(dtype=dtype)
    x, msk = _x(2, S, 17).to(dtype), torch.ones(2, S, dtype=dtype)
    y0 = m(x, attention_mask=msk)
    gate, ref = _moe_ref_and_gate(m, blob, K=K, force=True, dtype=dtype)
    assert _routing(ref, x, msk) == _routing(m, x, msk)
    y_ref, y = ref(x, attention_mask=msk), m(x, attention_mask=msk)
    err, scale = float((y - y_ref).abs().max()), float(y_ref.abs().max())
    assert err <= rel * scale, (err, scale, err / scale)
    assert float((y_ref - y0).abs().max()) > 1e3 * max(err, 1e-300)   # not a no-op comparison
    assert gate.fired.tolist() == [True, True] and gate.unrouted == 0 and gate.errors == 0


def test_the_fused_pass_does_not_change_one_bit_of_the_2_d_pass(blob):
    """Proved, not asserted: with the fused stacks declined, every 2-D hook is BIT-IDENTICAL.

    Everything this project has measured ran through the 2-D pass, so the fused pass is only
    admissible if it is purely additive to it. Both arms are the same weights at the same K with
    the same frozen decision; the modules are then called directly on the same input, which is
    the 2-D correction on its own with the MoE block's downstream effect out of the way.
    """
    a, b = _moe(), _declined(_moe())
    ga, gb = Gate([blob], None, None), Gate([blob], None, None)
    ga.force = gb.force = True
    assert ga.attach(a, verbose=False) == (21, 0)
    assert gb.attach(b, verbose=False) == (14, 7)
    x, msk = _x(2, S, 17), torch.ones(2, S)
    a(x, attention_mask=msk); b(x, attention_mask=msk)             # freeze a decision in both
    assert ga.fired.tolist() == gb.fired.tolist() == [True, True]
    g, n = torch.Generator().manual_seed(31), 0
    for l in BAND:
        for sub in ("self_attn.o_proj", "mlp.down_proj"):
            ma = a.get_submodule("layers.%d.%s" % (l, sub))
            mb = b.get_submodule("layers.%d.%s" % (l, sub))
            assert torch.equal(ma.weight, mb.weight)               # same weights to begin with
            xx = torch.randn(2, ma.in_features, generator=g)
            assert torch.equal(ma(xx), mb(xx)), (l, sub)
            n += 1
    assert n == 14


def test_a_fused_stack_edits_only_the_sequences_that_fired(blob):
    """One firing sequence and one not, through a flattened token list that mixes their tokens."""
    m = _moe()
    xA, xB = _x(1, S, 21), _x(1, S, 22)
    c, thr, sA, sB = _two_sided(m, xA, xB)
    assert sB < thr < sA                                       # positive control for the mix
    x, msk = torch.cat([xA, xB]), torch.ones(2, S)
    y0 = m(x, attention_mask=msk)
    gate, ref = _moe_ref_and_gate(m, blob, c, thr)
    y_edit = ref(x, attention_mask=msk)
    y = m(x, attention_mask=msk)
    assert gate.fired.tolist() == [True, False]
    assert torch.equal(y[1], y0[1])                            # did not fire: untouched, exactly
    assert not torch.allclose(y[0], y0[0])                     # fired: and it mattered
    assert torch.allclose(y[0], y_edit[0], atol=1e-5 * float(y_edit.abs().max()), rtol=0)
    assert gate.unrouted == 0 and gate.errors == 0


def test_no_decision_means_no_fused_edit(blob):
    """`fired is None` is what a declining read leaves behind, and it has to reach the stacks too.

    A fused path that edited on a missing decision would be the worst bug in this file: the whole
    design is that no decision means pristine weights, which means the model refuses.
    """
    m = _moe()
    x, msk = _x(2, S, 17), torch.ones(2, S)
    y0 = m(x, attention_mask=msk)
    gate = Gate([blob], None, None)                            # no direction, no threshold
    assert gate.attach(m, verbose=False) == (21, 0)
    assert torch.equal(m(x, attention_mask=msk), y0)
    assert gate.fired is None and gate.errors == 0
    gate.force = False
    assert torch.equal(m(x, attention_mask=msk), y0)
    assert not bool(gate.fired.any())


def test_an_exception_on_the_fused_path_is_exactly_the_14_of_21_model(blob, monkeypatch):
    """A raising correction is NO edit of that stack, counted -- and nothing else changes.

    The comparison arm is the same weights with the fused stacks declined, so "fail-safe" here is
    not "close to unedited": it is bit-for-bit the model this module served before the fused path
    existed.
    """
    a, b = _moe(), _declined(_moe())
    ga, gb = Gate([blob], None, None), Gate([blob], None, None)
    ga.force = gb.force = True
    assert ga.attach(a, verbose=False) == (21, 0)
    assert gb.attach(b, verbose=False) == (14, 7)
    monkeypatch.setattr("resgate.gate.fused_correction",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope")))
    x, msk = _x(2, S, 17), torch.ones(2, S)
    assert torch.equal(a(x, attention_mask=msk), b(x, attention_mask=msk))
    assert ga.errors == 7 and "nope" in ga.last_error
    assert gb.errors == 0


def test_a_token_list_the_decision_cannot_be_expanded_over_is_declined_and_counted(blob):
    """Not B equal sequences: decline, count it, and leave that stack exactly unedited.

    `_edit_hook` already refuses a leading dim that is not the batch. This is the same refusal
    one level further in -- the leading dim IS tokens here, by design, and the question is
    whether they map back to sequences. When they do not, editing the whole list would answer one
    sequence under another sequence's decision.
    """
    a, b = _moe(ragged=True), _declined(_moe(ragged=True))
    ga, gb = Gate([blob], None, None), Gate([blob], None, None)
    ga.force = gb.force = True
    assert ga.attach(a, verbose=False) == (21, 0)
    assert gb.attach(b, verbose=False) == (14, 7)
    x, msk = _x(2, S, 17), torch.ones(2, S)                    # B=2, S=5 -> 9 rows, not 2 x k
    assert torch.equal(a(x, attention_mask=msk), b(x, attention_mask=msk))
    assert ga.unrouted == 7 and ga.errors == 0                 # one per fused stack, not an error
    assert gb.unrouted == 0


@pytest.mark.parametrize("n_tok,want", [(10, 5), (2, 1), (9, None), (20, None), (11, None)])
def test_the_token_expansion_is_derived_and_checked(blob, n_tok, want):
    """The expansion rule itself: T must divide B, and the per-sequence length must be the mask's.

    B=2 and a [2,5] padding mask, so 10 tokens is a prefill (5 each), 2 is a decode step (1 each),
    and 9, 11 and 20 are none of the layouts this mask describes. The last is the dangerous one --
    20 divides by 2 cleanly and would expand to 10 tokens a sequence against a mask that says 5.
    """
    m = _moe()
    gate = Gate([blob], None, None)
    gate.force = True
    gate.attach(m, verbose=False)
    m(_x(2, S, 17), attention_mask=torch.ones(2, S))
    assert gate.unrouted == 0                                   # the forward's own layout is fine
    got = gate._token_rows(n_tok)
    if want is None:
        assert got is None and gate.unrouted == 1
    else:
        assert got.tolist() == [True] * n_tok
        assert got.shape[0] == want * 2 and gate.unrouted == 0


def test_detach_restores_the_pristine_forward_with_fused_stacks(blob):
    """A detached model is byte-identical to one that never had a gate, fused stacks included.

    The fused path adds a second kind of hook and writes no weight anywhere, so `detach`'s one
    job does not change -- which is the reason the apply site is a hook and not a replaced
    forward. Asserted on the hook dicts as well as the output, because "restores" is a claim
    about the module and not only about one input.
    """
    m = _moe()
    x, msk = _x(2, S, 17), torch.ones(2, S)
    y0 = m(x, attention_mask=msk)
    gate = Gate([blob], None, None)
    gate.force = True
    assert gate.attach(m, verbose=False) == (21, 0)
    assert not torch.allclose(m(x, attention_mask=msk), y0)
    gate.detach()
    assert torch.equal(m(x, attention_mask=msk), y0)
    assert not any(mod._forward_hooks or mod._forward_pre_hooks for mod in m.modules())
    gate.detach()                                              # idempotent
    assert torch.equal(m(x, attention_mask=msk), y0)
    assert gate.attach(m, verbose=False) == (21, 0)            # and re-attachable to the same
    assert not torch.allclose(m(x, attention_mask=msk), y0)
    gate.detach()


def test_a_stack_whose_forward_the_reconstruction_misses_is_declined_by_name(blob, capsys):
    """The positive control doing its job, which is the one thing fail-closed cannot do for it.

    A shared expert is additive, so the ROUTED correction would still have been right and this
    decline costs coverage for nothing. It is still the only admissible answer: the reconstruction
    is missing a term, so the apply site is unproven, and an unproven apply site would be a
    confident wrong edit rather than a smaller one.
    """
    gate = Gate([blob], None, None)
    assert gate.attach(_moe(shared=True), verbose=True) == (14, 7)
    out = capsys.readouterr().out
    assert "14 tensors hooked, 7 direction-keyed target(s) NOT reached" in out
    for l in BAND:
        assert ("NOT REACHED  layers.%d.experts.down_proj" % l) in out
    assert "is not the" in out and "reconstruction" in out
    assert "tolerance %.0e" % FUSED_RECON_TOL in out


@pytest.mark.parametrize("attr,val,msg", [
    ("is_transposed", True, "is_transposed=True"),
    ("has_bias", True, "has_bias=True"),
    ("num_experts", NE + 1, "num_experts 5 is not the stack's 4"),
])
def test_a_fused_stack_that_does_not_carry_the_layout_is_refused_by_name(blob, capsys,
                                                                        attr, val, msg):
    """Read the flags off the module, refuse what they do not say. Never guess the layout.

    Each of these changes what the correction's arithmetic would have to be, and none of them is
    visible in the parameter's shape. A 3-D stack is reached because the module said how it is
    stored, or it is not reached.
    """
    m = _moe()
    for l in BAND: setattr(m.layers[l].experts, attr, val)
    assert Gate([blob], None, None).attach(m, verbose=True) == (14, 7)
    assert msg in capsys.readouterr().out


def test_a_stack_whose_forward_raises_at_the_probe_is_declined_not_propagated(blob, monkeypatch,
                                                                              capsys):
    """`attach` stays total. The proof calls somebody else's forward with synthetic inputs, and a
    stack that cannot take them is a coverage question and not a server failure: it declines and
    names the exception, which is where every other failure in this file lands too. `attach`
    raising instead would take a long-lived server's re-attach down over a coverage detail.
    """
    monkeypatch.setattr("resgate.gate._fused_recon_error",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("probe blew up")))
    assert Gate([blob], None, None).attach(_moe(), verbose=True) == (14, 7)
    out = capsys.readouterr().out
    assert "raised on the proof probe" in out and "probe blew up" in out


def test_a_bare_3_d_parameter_is_left_to_patch(blob, model, capsys):
    """`Blk`'s fused stack is a Parameter with no routing code around it, and the reason says so.

    This is what the 14-of-21 counts in this file's coverage tests are measured against, and it
    is the honest answer for a 3-D tensor whose routing the gate cannot see: `patch()` can edit
    it offline, a hook cannot.
    """
    assert Gate([blob], None, None).attach(model, verbose=True) == (14, 7)
    assert "has no .has_gate" in capsys.readouterr().out


# --- M1 + M3 closed-form gate fitting & shipped kits -----------------------------------

def test_fit_gate_direction_m1_ridge_matches_full_covariance_solve():
    """Verify Gram-route `fit_gate_direction` (`M1_ridge_lam16`) matches full D x D ridge solve."""
    from resgate.gate import fit_gate_direction, fit_gate_op, thresh_r2, thresh_r3

    g = torch.Generator().manual_seed(42)
    n_pos, n_neg, D = 20, 28, 32
    xp = torch.randn(n_pos, D, generator=g, dtype=torch.float64)
    xn = torch.randn(n_neg, D, generator=g, dtype=torch.float64)
    neg_groups = ["cat_%d" % (i // 2) for i in range(n_neg)]  # 14 subcategories

    c, info = fit_gate_direction(xp, xn, neg_groups, rank=16, mode="ridge")
    assert c.shape == (D,) and c.dtype == torch.float32
    assert abs(float(c.norm().item()) - 1.0) < 1e-6
    assert info["scatter"] == "M1_subcat"
    assert info["dof"] == n_pos + n_neg - 1 - 14

    # Independent full D x D matrix check in float64
    xp_u = xp / xp.norm(dim=-1, keepdim=True)
    xn_u = xn / xn.norm(dim=-1, keepdim=True)
    d = xp_u.mean(0) - xn_u.mean(0)
    z_list = [xp_u - xp_u.mean(0)]
    for s in sorted(set(neg_groups)):
        idx = [i for i, x in enumerate(neg_groups) if x == s]
        xg = xn_u[idx]
        z_list.append(xg - xg.mean(0))
    Z = torch.cat(z_list, dim=0)
    Sigma = (Z.T @ Z) / float(info["dof"])
    alpha = info["alpha"]
    w_full = alpha * torch.linalg.solve(Sigma + alpha * torch.eye(D, dtype=torch.float64), d)
    c_full = (w_full / w_full.norm()).to(torch.float32)
    assert torch.allclose(c, c_full, atol=1e-6)

    # Check fit_gate_op and Gate.from_file round-trip
    op = fit_gate_op(xp, xn, neg_groups, category="cyber_offense", gate_layer=GATE_LAYER,
                     m=3, rule="R3", rank=16, mode="ridge")
    gate = Gate.from_file([_blob()], op)
    assert gate.thresh == op["thresh"]
    assert int(((xn_u @ c.to(torch.float64)) >= gate.thresh).sum().item()) <= 3
    t_r2 = thresh_r2(xp_u @ c.to(torch.float64), xn_u @ c.to(torch.float64), 3)
    assert int(((xn_u @ c.to(torch.float64)) >= t_r2).sum().item()) <= 3
    t_r3 = thresh_r3(xn_u @ c.to(torch.float64), 3)
    assert t_r3 == op["thresh"]


def test_shipped_m1_ridge_lam16_kits_load_cleanly():
    """Verify every shipped M1_ridge_lam16 kit in `kits/{ablation,gate}/*/` satisfies Gate.from_file
    and that its documented sha256 and threshold in `kits/gate/<model>/README.md` match the file."""
    import hashlib
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "kits"
    pairs = [
        ("gemma4-26B", "gemma_dir90_cbrn_L10to16.pt", "op_gemma_L8_M1subcat240_M1_ridge_lam16_serve.pt", range(10, 17)),
        ("gemma4-26B", "dir_gemma4-26B_sep.pt", "op_gemma_L8_M1subcat240_M1_ridge_lam16_serve.pt", range(10, 17)),
        ("gemma4-26B", "gemma_dir90_cbrn_L10to16.pt", "op_wmdpbio_B2_M1_ridge_lam16_R3U140_gemma_L8.pt", range(10, 17)),
        ("phi4mini", "dir_phi4mini_sep_L13to15.pt", "op_phi4mini_L11_M1subcat240_M1_ridge_lam16_serve.pt", range(13, 16)),
        ("phi4mini", "dir_phi4mini_sep_L13to15.pt", "op_wmdpbio_B2_M1_ridge_lam16_R3U140_phi4mini_L11.pt", range(13, 16)),
        ("phi4mini", "dir_phi4mini_sep_L13to15.pt", "op_wmdpbio_B2_M1_ridge_lam16_R3U140_phi4mini_L12.pt", range(13, 16)),
        ("llama31-8B", "dir_llama31-8B_sep_L11to22.pt", "op_llama31-8B_L10_M1subcat240_M1_ridge_lam16_serve.pt", range(11, 23)),
        ("llama31-8B", "dir_llama31-8B_sep.pt", "op_llama31-8B_L10_M1subcat240_M1_ridge_lam16_serve.pt", range(11, 23)),
        ("llama31-8B", "dir_llama31-8B_sep_L11to22.pt", "op_wmdpbio_B2_M1_ridge_lam16_R3U140_llama31-8B_L10.pt", range(11, 23)),
    ]
    for sub, dir_fn, op_fn, band in pairs:
        dir_path = root / "gate" / sub / dir_fn
        op_path = root / "gate" / sub / op_fn
        readme_text = (root / "gate" / sub / "README.md").read_text(encoding="utf-8")
        g = Gate.from_file([dir_path], op_path, edit_layers=tuple(band))
        assert max(g.gate_layers) < min(g.edit_layers)
        assert g.pooling == "mean_user"
        assert g.c is not None and g.thresh is not None
        assert repr(float(g.thresh)) in readme_text
        assert hashlib.sha256(dir_path.read_bytes()).hexdigest() in readme_text
        assert hashlib.sha256(op_path.read_bytes()).hexdigest() in readme_text




