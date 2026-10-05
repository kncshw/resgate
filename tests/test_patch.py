"""The rank-1 PRE row-normalised edit: the properties that matter, on a tiny checkpoint.

Update rule (Heretic's; the foundational method is Arditi et al. 2024, arXiv:2406.11717):

    n = rownorm(W);   What = W / n
    dW = -w * outer(n (*) v, v @ What)          w = K * sep_l

With W = N What (N = diag(n), What unit rows), that is Wp = N (I - w v v^T) What, so in the
row-normalised frame the component along v is scaled by exactly (1 - w). Each test below is
one consequence of that identity.
"""
from __future__ import annotations

import os
import shutil

import pytest
import torch
from safetensors import safe_open

from tempx.enumerate_targets import enumerate_targets
from tempx.patch import build_plan, key_of, layer_of, patch, pre_step
from conftest import SHARD, SHARD2

H = 8
TARGETS = {
    "model.layers.0.self_attn.o_proj.weight": (H, H),
    "model.layers.0.mlp.down_proj.weight": (H, 16),
    "model.layers.0.mlp.experts.down_proj": (2, H, 16),     # fused MoE stack
}
UNTOUCHED = {
    "model.layers.0.self_attn.q_proj.weight": (H, H),       # reads, never writes
    "model.layers.0.mlp.up_proj.weight": (16, H),           # wrong shape orientation
    "model.layers.1.self_attn.o_proj.weight": (H, H),       # a target with NO direction
}
SHARDS = {
    "model.layers.0.self_attn.o_proj.weight": SHARD,
    "model.layers.0.mlp.down_proj.weight": SHARD,
    "model.layers.0.self_attn.q_proj.weight": SHARD,
    "model.layers.0.mlp.up_proj.weight": SHARD,
    "model.layers.1.self_attn.o_proj.weight": SHARD,
    "model.layers.0.mlp.experts.down_proj": SHARD2,         # second shard, second file loop
}
W_ATTN, W_MLP = 1.25, 0.75          # a non-trivial sep(l) profile, so K is not the only factor


def _rand(shape, g):
    return torch.randn(*shape, generator=g, dtype=torch.float32).to(torch.bfloat16)


@pytest.fixture
def src(tmp_path, write_ckpt):
    g = torch.Generator().manual_seed(7)
    tensors = {n: _rand(s, g) for n, s in {**TARGETS, **UNTOUCHED}.items()}
    return write_ckpt(tmp_path / "src", hidden=H, tensors=tensors, shards=SHARDS)


@pytest.fixture
def dirs():
    g = torch.Generator().manual_seed(11)
    v0 = torch.randn(H, generator=g); v0 = v0 / v0.norm()
    v1 = torch.randn(H, generator=g); v1 = v1 / v1.norm()
    return [{"d": {"0.attn_out": v0, "0.mlp_out": v1},
             "w": {"0.attn_out": W_ATTN, "0.mlp_out": W_MLP},
             "meta": {"mode": "sep", "layers": [0]}}]


def _copy(src, dst):
    shutil.copytree(src, dst)
    return dst


def read(d, name):
    with safe_open(os.path.join(d, SHARDS[name]), framework="pt") as f:
        return f.get_tensor(name)


def shard_bytes(d):
    return {fn: open(os.path.join(d, fn), "rb").read() for fn in (SHARD, SHARD2)}


# --- the identity, at float64 where it is exact --------------------------------------

@pytest.mark.parametrize("shape", [(H, H), (H, 16), (16, 32)])
@pytest.mark.parametrize("w", [0.4, 1.0, 1.9])
def test_delta_is_rank_one(shape, w):
    """Rank is measured by SPECTRAL ENERGY, S[0]**2 / sum(S**2) -- not S[0] / sum(S),
    which is a different quantity and does not reach 1 for a rank-1 matrix."""
    g = torch.Generator().manual_seed(3)
    W = torch.randn(*shape, generator=g, dtype=torch.float64)
    v = torch.randn(shape[0], generator=g, dtype=torch.float64); v = v / v.norm()
    dW = pre_step(W, v, w) - W
    S = torch.linalg.svdvals(dW)
    assert float(S[0] ** 2 / (S ** 2).sum()) > 1 - 1e-12
    assert float(S[1]) / float(S[0]) < 1e-7            # the second singular value is noise


@pytest.mark.parametrize("w", [0.0, 0.5, 1.0, 1.5, 2.0])
def test_residual_along_v_is_one_minus_w(w):
    g = torch.Generator().manual_seed(5)
    W = torch.randn(H, 16, generator=g, dtype=torch.float64)
    v = torch.randn(H, generator=g, dtype=torch.float64); v = v / v.norm()
    n = W.norm(dim=1).unsqueeze(1)
    before = v @ (W / n)
    after = v @ (pre_step(W, v, w) / n)                # the frame is the ORIGINAL rownorm
    assert float(after.norm() / before.norm()) == pytest.approx(abs(1 - w), abs=1e-12)


def test_w_one_removes_the_component_entirely():
    g = torch.Generator().manual_seed(5)
    W = torch.randn(H, 16, generator=g, dtype=torch.float64)
    v = torch.randn(H, generator=g, dtype=torch.float64); v = v / v.norm()
    n = W.norm(dim=1).unsqueeze(1)
    assert float((v @ (pre_step(W, v, 1.0) / n)).abs().max()) < 1e-14


def test_w_two_is_a_reflection():
    """Magnitude preserved, sign flipped: w=2 does not remove the direction, it inverts it."""
    g = torch.Generator().manual_seed(5)
    W = torch.randn(H, 16, generator=g, dtype=torch.float64)
    v = torch.randn(H, generator=g, dtype=torch.float64); v = v / v.norm()
    n = W.norm(dim=1).unsqueeze(1)
    before = v @ (W / n)
    after = v @ (pre_step(W, v, 2.0) / n)
    assert torch.allclose(after, -before, atol=1e-12)
    assert float(after.norm()) == pytest.approx(float(before.norm()), rel=1e-12)


# --- the same properties, through the real file-level patch --------------------------

def test_patch_writes_only_the_enumerated_targets(src, dirs, tmp_path):
    dst = _copy(src, str(tmp_path / "dst"))
    tot, mx = patch(src, dst, dirs, K=1.0, verbose=False)
    assert tot == len(TARGETS)
    assert mx > 0.0
    changed = {n for n in {**TARGETS, **UNTOUCHED}
               if not torch.equal(read(src, n), read(dst, n))}
    assert changed == set(TARGETS)


def test_target_without_a_direction_is_left_alone(src, dirs):
    """layers.1 is enumerated as a target but the blob has no 1.attn_out key."""
    _, targets, _, _ = enumerate_targets(src)
    assert "model.layers.1.self_attn.o_proj.weight" in {n for n, _ in targets}
    plan = build_plan(src, dirs[0]["d"])
    assert {n for _, n, _ in plan} == set(TARGETS)


def test_file_level_delta_is_rank_one_per_tensor(src, dirs, tmp_path):
    dst = _copy(src, str(tmp_path / "dst"))
    patch(src, dst, dirs, K=1.0, verbose=False)
    for name in TARGETS:
        dW = (read(dst, name).float() - read(src, name).float())
        mats = dW if dW.dim() == 3 else dW.unsqueeze(0)     # 3-D: one per fused expert
        for e, m in enumerate(mats):
            S = torch.linalg.svdvals(m.double())
            energy = float(S[0] ** 2 / (S ** 2).sum())
            # bf16 storage quantises the written weights, so the measured rank-1 energy
            # is 1 minus a small rounding floor rather than exactly 1 (measured: > 0.99995).
            assert energy > 0.999, (name, e, energy)


def test_each_fused_expert_gets_its_own_edit(src, dirs, tmp_path):
    dst = _copy(src, str(tmp_path / "dst"))
    patch(src, dst, dirs, K=1.0, verbose=False)
    name = "model.layers.0.mlp.experts.down_proj"
    dW = read(dst, name).float() - read(src, name).float()
    assert dW.shape[0] == 2
    for e in range(2):
        assert float(dW[e].abs().max()) > 0.0


def test_max_dw_is_linear_in_K(src, dirs, tmp_path):
    out = {}
    for K in (0.5, 1.0, 1.5):
        dst = _copy(src, str(tmp_path / ("dst%s" % K)))
        out[K] = patch(src, dst, dirs, K=K, verbose=False)[1]
    assert out[1.0] / out[0.5] == pytest.approx(2.0, rel=1e-5)
    assert out[1.5] / out[0.5] == pytest.approx(3.0, rel=1e-5)


def test_repatching_the_same_source_does_not_accumulate(src, dirs, tmp_path):
    """DST is written from SRC every time, so a K change replaces the edit, never stacks.

    Re-running at the same K is a no-op, and arriving at K=1.5 via K=0.5 gives the same
    bytes as going straight there.
    """
    a = _copy(src, str(tmp_path / "a"))
    patch(src, a, dirs, K=1.5, verbose=False)
    want = shard_bytes(a)

    patch(src, a, dirs, K=1.5, verbose=False)            # idempotent
    assert shard_bytes(a) == want

    b = _copy(src, str(tmp_path / "b"))
    patch(src, b, dirs, K=0.5, verbose=False)
    assert shard_bytes(b) != want                        # K really does change the output
    patch(src, b, dirs, K=1.5, verbose=False)
    assert shard_bytes(b) == want                        # ... and 0.5 left no residue


def test_K_zero_is_a_bit_exact_no_op(src, dirs, tmp_path):
    dst = _copy(src, str(tmp_path / "dst"))
    patch(src, dst, dirs, K=0.0, verbose=False)
    assert shard_bytes(dst) == shard_bytes(src)


def test_dst_must_already_be_a_copy(src, dirs, tmp_path):
    empty = str(tmp_path / "empty")
    os.makedirs(empty)
    with pytest.raises(FileNotFoundError, match="DST must already be a copy"):
        patch(src, empty, dirs, K=1.0, verbose=False)


def test_no_direction_blob_raises(src):
    with pytest.raises(ValueError, match="no direction blob"):
        patch(src, src, [], K=1.0, verbose=False)


# --- key mapping --------------------------------------------------------------------

@pytest.mark.parametrize("name,want", [
    ("model.layers.3.self_attn.o_proj.weight", "3.attn_out"),
    ("model.layers.3.self_attn.out_proj.weight", "3.attn_out"),
    ("model.layers.3.attention.wo.weight", "3.attn_out"),
    ("model.layers.3.mlp.down_proj.weight", "3.mlp_out"),
    ("model.layers.3.mlp.dense_4h_to_h.weight", "3.mlp_out"),
])
def test_key_of(name, want):
    assert key_of(name, layer_of(name)) == want


def test_layer_of_returns_none_without_a_layer_segment():
    assert layer_of("model.embed_tokens.weight") is None
    assert layer_of("model.layers.12.mlp.down_proj.weight") == 12


# --- the two modules' shared key convention -----------------------------------------

def test_direction_blob_drives_the_patch_end_to_end(tmp_path, write_ckpt):
    """direction.py emits '<layer>.attn_out' / '<layer>.mlp_out'; patch.py looks those up.

    Nothing else in the suite checks that the producer and the consumer agree on the key
    names, so this runs the real compute_directions output into the real patcher. Still no
    model: the activations are synthetic and D == hidden_size == 8.
    """
    from tempx.direction import compute_directions

    g = torch.Generator().manual_seed(23)
    acts = {"nlast": 1, "snapshot": "synthetic", "sys": "", "items": [
        {"key": "h1", "group": "harmful",
         "acts": torch.randn(2, 1, H, generator=g)},
        *[{"key": "g%d" % i, "group": "harmless",
           "acts": torch.randn(2, 1, H, generator=g)} for i in range(4)],
    ]}
    blob = compute_directions(acts, verbose=False)
    assert set(blob["d"]) == {"0.attn_out", "0.mlp_out", "1.attn_out", "1.mlp_out"}

    names = {"model.layers.%d.%s" % (l, t): s for l in (0, 1) for t, s in (
        ("self_attn.o_proj.weight", (H, H)), ("mlp.down_proj.weight", (H, 16)))}
    gw = torch.Generator().manual_seed(29)
    src = write_ckpt(tmp_path / "s", hidden=H,
                     tensors={n: _rand(s, gw) for n, s in names.items()},
                     shards={n: SHARD for n in names})
    dst = _copy(src, str(tmp_path / "d"))
    tot, mx = patch(src, dst, [blob], K=1.0, verbose=False)
    assert tot == len(names) and mx > 0.0
    for n in names:
        with safe_open(os.path.join(dst, SHARD), framework="pt") as f:
            after = f.get_tensor(n)
        with safe_open(os.path.join(src, SHARD), framework="pt") as f:
            before = f.get_tensor(n)
        assert not torch.equal(before, after), n
