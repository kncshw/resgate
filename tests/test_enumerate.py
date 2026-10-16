"""Target derivation: hidden_size in the OUTPUT position AND an output-projection name.

Everything here runs off safetensors headers and a config.json written into a tmpdir --
no real checkpoint, no network, no torch compute.
"""
from __future__ import annotations

import pytest
import torch

from resgate.enumerate_targets import (EXCLUDE_PARTS, OUT_NAMES, enumerate_targets,
                                     format_report)
from conftest import SHARD, SHARD2

H = 8

# --- tensors that MUST be selected -------------------------------------------------
KEEP = {
    "model.layers.0.self_attn.o_proj.weight": (H, H),
    "model.layers.0.mlp.down_proj.weight": (H, 16),
    "model.layers.1.self_attn.out_proj.weight": (H, H),
    "model.layers.1.mlp.experts.down_proj": (4, H, 16),      # fused MoE: H at shape[1]
    "model.layers.2.mlp.dense_4h_to_h.weight": (H, 32),
    "model.layers.3.attention.wo.weight": (H, H),
    "transformer.h.4.attn.c_proj.weight": (H, H),
}

# --- rejected because hidden_size is not in the output position --------------------
REJECT_SHAPE = {
    "model.layers.0.mlp.up_proj.weight": (16, H),
    "model.layers.0.mlp.gate_proj.weight": (16, H),
    "model.layers.2.self_attn.o_proj.weight": (16, H),       # transposed: H is the INPUT
    "model.layers.3.mlp.experts.down_proj": (H, 4, 16),      # 3-D with H at shape[0]
    "model.embed_tokens.weight": (16, H),
    "model.layers.0.input_layernorm.weight": (H,),           # 1-D
}

# --- rejected because the name carries no output-projection token ------------------
REJECT_ROLE = {
    "model.layers.0.self_attn.q_proj.weight": (H, H),        # reads the residual stream
    "model.layers.0.self_attn.k_proj.weight": (H, H),
    "model.layers.1.mlp.gate.weight": (H, H),                # MoE router
}

# --- excluded by name part, though they pass both filters --------------------------
EXCLUDED = {
    "model.mtp.layers.0.self_attn.o_proj.weight": (H, H),
    "model.vision_tower.encoder.layers.0.self_attn.out_proj.weight": (H, H),
    "visual.blocks.0.attn.c_proj.weight": (H, H),
}

ALL = {**KEEP, **REJECT_SHAPE, **REJECT_ROLE, **EXCLUDED}


@pytest.fixture
def ckpt(tmp_path, write_ckpt):
    tensors = {n: torch.zeros(s, dtype=torch.bfloat16) for n, s in ALL.items()}
    # two shards, to exercise the header cache
    shards = {n: (SHARD if i % 2 == 0 else SHARD2) for i, n in enumerate(sorted(tensors))}
    return write_ckpt(tmp_path / "ckpt", hidden=H, tensors=tensors, shards=shards)


def test_selects_exactly_the_output_projections(ckpt):
    h, keep, rs, rr = enumerate_targets(ckpt)
    assert h == H
    assert {n: s for n, s in keep} == KEEP
    assert rs == len(REJECT_SHAPE)
    assert rr == len(REJECT_ROLE)


def test_hidden_must_be_in_the_output_position(ckpt):
    """A transposed o_proj [*, H] and a 3-D stack with H at shape[0] are both rejected."""
    _, keep, _, _ = enumerate_targets(ckpt)
    names = {n for n, _ in keep}
    assert "model.layers.2.self_attn.o_proj.weight" not in names
    assert "model.layers.3.mlp.experts.down_proj" not in names
    # ... while their correctly-oriented twins are kept
    assert "model.layers.0.self_attn.o_proj.weight" in names
    assert "model.layers.1.mlp.experts.down_proj" in names


def test_shape_alone_is_not_enough(ckpt):
    """q_proj is [H, H] -- it passes the shape test and must fail the role test."""
    _, keep, _, rr = enumerate_targets(ckpt)
    names = {n for n, _ in keep}
    assert not any("q_proj" in n or "k_proj" in n for n in names)
    assert rr == len(REJECT_ROLE)


def test_excluded_parts_are_dropped_and_counted_nowhere(ckpt):
    """mtp / vision_tower / visual pass both filters, so only the name check stops them."""
    _, keep, rs, rr = enumerate_targets(ckpt)
    names = {n for n, _ in keep}
    for n in EXCLUDED:
        assert n not in names
    # they are skipped, not booked as shape or role rejections
    assert rs + rr + len(keep) + len(EXCLUDED) == len(ALL)


def test_negative_case_nothing_qualifies(tmp_path, write_ckpt):
    """A checkpoint with no output projections yields no targets and no crash."""
    d = write_ckpt(tmp_path / "none", hidden=H, tensors={
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(H, H, dtype=torch.bfloat16),
        "model.layers.0.mlp.up_proj.weight": torch.zeros(16, H, dtype=torch.bfloat16),
        "model.norm.weight": torch.zeros(H, dtype=torch.bfloat16),
    })
    h, keep, rs, rr = enumerate_targets(d)
    assert (h, keep, rs, rr) == (H, [], 2, 1)


def test_hidden_size_falls_back_to_text_config(tmp_path, write_ckpt):
    d = write_ckpt(tmp_path / "mm", tensors={
        "model.language_model.layers.0.self_attn.o_proj.weight":
            torch.zeros(H, H, dtype=torch.bfloat16)},
        config={"model_type": "wrapper", "text_config": {"hidden_size": H}})
    h, keep, _, _ = enumerate_targets(d)
    assert h == H and len(keep) == 1


def test_weight_map_entry_missing_from_the_shard_is_skipped(tmp_path, write_ckpt):
    tensors = {"model.layers.0.mlp.down_proj.weight": torch.zeros(H, 16, dtype=torch.bfloat16)}
    shards = {**{n: SHARD for n in tensors}, "model.layers.9.mlp.down_proj.weight": SHARD}
    d = write_ckpt(tmp_path / "stale", hidden=H, tensors=tensors, shards=shards)
    _, keep, rs, rr = enumerate_targets(d)
    assert [n for n, _ in keep] == ["model.layers.0.mlp.down_proj.weight"]
    assert (rs, rr) == (0, 0)


def test_token_lists_match_the_documented_spec():
    """These two tuples ARE the published rule; a silent change invalidates the counts."""
    assert OUT_NAMES == ("o_proj", "out_proj", "down_proj", "wo", "dense_4h_to_h", "c_proj")
    assert EXCLUDE_PARTS == ("mtp", "vision_tower", "visual")


def test_format_report_totals(ckpt):
    h, keep, rs, rr = enumerate_targets(ckpt)
    txt = format_report(ckpt, h, keep, rs, rr)
    assert "hidden=%d" % H in txt
    assert "TOTAL %d targets" % len(KEEP) in txt
    assert "rejected: %d by shape, %d by role" % (rs, rr) in txt
