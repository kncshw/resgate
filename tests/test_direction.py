"""Direction and sep(l), on synthetic activations with hand-computed answers.

The fixture is built so every quantity is known on paper:

  layer 0  harmless  [1,1,0,0] [1,-1,0,0] [1,0,1,0] [1,0,-1,0]   centroid [1,0,0,0], spread 1
           harmful   [4,0,0,3]            gap ||[3,0,0,3]|| = 3*sqrt(2)   sep(0) = 3*sqrt(2)
  layer 1  harmless  10x[[1,2,0,0],[1,-2,0,0],[1,0,2,0],[1,0,-2,0]]  centroid [10,0,0,0], spread 20
           harmful   [10,0,0,40]          gap 40                     sep(1) = 2

Layer 0's raw difference has a component along the harmless centroid (|cos| = 1/sqrt(2)), so
the orthogonalisation has real work to do there; layer 1's does not.
"""
from __future__ import annotations

import math
import statistics as st

import pytest
import torch

from tempx.direction import compute_directions, orthogonalize_direction, unit

D = 4
SEP0 = 3.0 * math.sqrt(2.0)
SEP1 = 2.0

HARMLESS = [
    [[1.0, 1.0, 0.0, 0.0], [10.0, 20.0, 0.0, 0.0]],
    [[1.0, -1.0, 0.0, 0.0], [10.0, -20.0, 0.0, 0.0]],
    [[1.0, 0.0, 1.0, 0.0], [10.0, 0.0, 20.0, 0.0]],
    [[1.0, 0.0, -1.0, 0.0], [10.0, 0.0, -20.0, 0.0]],
]
HARMFUL = [[4.0, 0.0, 0.0, 3.0], [10.0, 0.0, 0.0, 40.0]]
CENTROIDS = [torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.tensor([10.0, 0.0, 0.0, 0.0])]


def _acts(per_layer):
    """[layer][D] -> the (L, 1, D) tensor the capture step saves."""
    return torch.tensor(per_layer, dtype=torch.float32).unsqueeze(1)


def blob(harmful=None, harmless=None):
    harmful = {"h1": HARMFUL} if harmful is None else harmful
    harmless = ({"g%d" % (i + 1): v for i, v in enumerate(HARMLESS)}
                if harmless is None else harmless)
    items = [{"key": k, "group": "harmful", "acts": _acts(v)} for k, v in harmful.items()]
    items += [{"key": k, "group": "harmless", "acts": _acts(v)} for k, v in harmless.items()]
    return {"nlast": 1, "snapshot": "synthetic", "sys": "", "items": items}


@pytest.fixture
def dirs():
    return compute_directions(blob(), verbose=False)


# --- sep(l) -------------------------------------------------------------------------

def test_sep_matches_hand_computed_values(dirs):
    assert dirs["meta"]["sep"] == pytest.approx([SEP0, SEP1], rel=1e-12)


def test_sep_is_scale_free_across_depth(dirs):
    """Layer 1 is layer 0 x10 in norm with 2x the spread, and sep reports the ratio only."""
    sep = dirs["meta"]["sep"]
    assert sep[1] == pytest.approx(2.0, rel=1e-12)       # 40 / 20, not 40 / 1


def test_sep_raises_at_one_harmless_prompt():
    """The denominator is then exactly 0 -- undefined, not small. It must refuse."""
    one = {"g1": HARMLESS[0]}
    with pytest.raises(ValueError, match=r"sep\(l\) is undefined with 1 harmless"):
        compute_directions(blob(harmless=one), verbose=False)


def test_missing_group_raises():
    with pytest.raises(ValueError, match="no 'harmful' group"):
        compute_directions(blob(harmful={}), verbose=False)


# --- the direction ------------------------------------------------------------------

def test_difference_of_means_is_correct():
    """Harmless centroid [1,0,0,0], harmful [1,0,3,4]: d = unit([0,0,3,4]) = [0,0,.6,.8].

    The difference is already orthogonal to the centroid here, so this isolates the
    difference-of-means step from the orthogonalisation.
    """
    hv = [[1.0, 0.0, 3.0, 4.0], [10.0, 0.0, 30.0, 40.0]]
    out = compute_directions(blob(harmful={"h1": hv}), verbose=False)
    for key in ("0.attn_out", "1.attn_out"):
        assert out["d"][key].tolist() == pytest.approx([0.0, 0.0, 0.6, 0.8], abs=1e-6)


def test_only_the_first_harmful_prompt_by_sorted_key_contributes():
    """Documented behaviour: the gap comes from sorted(harmful)[0] alone."""
    a = [[1.0, 0.0, 0.0, 3.0], [10.0, 0.0, 0.0, 30.0]]
    z = [[1.0, 0.0, 99.0, 0.0], [10.0, 0.0, 99.0, 0.0]]
    both = compute_directions(blob(harmful={"a": a, "z": z}), verbose=False)
    only_a = compute_directions(blob(harmful={"a": a}), verbose=False)
    assert both["meta"]["sep"] == pytest.approx(only_a["meta"]["sep"], rel=1e-12)
    assert torch.equal(both["d"]["0.attn_out"], only_a["d"]["0.attn_out"])


def test_orthogonalisation_leaves_the_result_orthogonal_and_unit_norm(dirs):
    for l, centroid in enumerate(CENTROIDS):
        d = dirs["d"]["%d.attn_out" % l].double()
        gd = unit(centroid.double())
        assert float(d.norm()) == pytest.approx(1.0, abs=1e-6)
        assert abs(float(torch.dot(d, gd))) < 1e-6


def test_orthogonalisation_actually_removes_something_at_layer_0(dirs):
    """Before: |cos(raw, centroid)| = 1/sqrt(2). After: 0, and d is the residual axis."""
    raw = unit((torch.tensor(HARMFUL[0]) - CENTROIDS[0]).double())
    gd = unit(CENTROIDS[0].double())
    assert abs(float(torch.dot(raw, gd))) == pytest.approx(1 / math.sqrt(2), rel=1e-9)
    assert dirs["d"]["0.attn_out"].tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0], abs=1e-6)
    assert orthogonalize_direction(raw, gd).tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 1.0], abs=1e-12)


def test_both_channels_of_a_layer_share_its_direction(dirs):
    for l in (0, 1):
        assert torch.equal(dirs["d"]["%d.attn_out" % l], dirs["d"]["%d.mlp_out" % l])
    assert set(dirs["d"]) == {"0.attn_out", "0.mlp_out", "1.attn_out", "1.mlp_out"}


# --- the per-layer weight -----------------------------------------------------------

def test_weights_are_sep_normalised_to_mean_exactly_one(dirs):
    w = dirs["w"]
    assert st.mean(w.values()) == pytest.approx(1.0, abs=1e-12)
    m = (SEP0 + SEP1) / 2
    assert w["0.attn_out"] == pytest.approx(SEP0 / m, rel=1e-12)
    assert w["1.mlp_out"] == pytest.approx(SEP1 / m, rel=1e-12)
    assert w["0.attn_out"] > w["1.attn_out"]          # the better-separated layer is dosed more


def test_uniform_mode_gives_every_layer_weight_one():
    out = compute_directions(blob(), mode="uniform", verbose=False)
    assert set(out["w"].values()) == {1.0}
    assert out["meta"]["sep"] == pytest.approx([SEP0, SEP1], rel=1e-12)   # still reported


def test_keep_drops_weak_layers_and_renormalises_over_what_is_left():
    """KEEP=0.9 drops layer 1 (2.0 < 0.9 * 3.1213); the survivor is renormalised to 1."""
    out = compute_directions(blob(), keep=0.9, verbose=False)
    assert out["meta"]["layers"] == [0]
    assert set(out["d"]) == {"0.attn_out", "0.mlp_out"}
    assert st.mean(out["w"].values()) == pytest.approx(1.0, abs=1e-12)
    assert out["w"]["0.attn_out"] == pytest.approx(1.0, abs=1e-12)


# --- round trip ---------------------------------------------------------------------

def test_saves_and_reloads(tmp_path, dirs):
    p = tmp_path / "dir.pt"
    compute_directions(blob(), p, verbose=False)
    back = torch.load(p, map_location="cpu")
    assert back["w"] == dirs["w"]
    assert back["meta"]["mode"] == "sep" and back["meta"]["keep_thresh"] == 0.0
    assert torch.equal(back["d"]["0.attn_out"], dirs["d"]["0.attn_out"])
