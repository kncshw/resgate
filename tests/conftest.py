"""Shared fixtures. CPU-only, no network, no model downloads.

The tests import `resgate` from `src/` when the package is not installed, so a bare
`pytest` in a checkout works as well as `pip install -e . && pytest`.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(ROOT, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

SHARD = "model-00001-of-00002.safetensors"
SHARD2 = "model-00002-of-00002.safetensors"


@pytest.fixture
def write_ckpt():
    """Write a synthetic safetensors checkpoint and return its directory.

    `tensors` maps tensor name -> torch tensor. `shards` maps name -> shard filename
    (defaults to one shard) and becomes the index's `weight_map`; a name present in
    `shards` but absent from `tensors` is therefore in the weight map without being in
    any shard, which is a case the enumerator has to survive.
    """
    def _write(d, hidden=8, tensors=None, shards=None, config=None):
        from safetensors.torch import save_file

        tensors = {} if tensors is None else tensors
        d = str(d)
        os.makedirs(d, exist_ok=True)
        shards = dict(shards) if shards else {n: SHARD for n in tensors}
        cfg = {"hidden_size": hidden} if config is None else config
        with open(os.path.join(d, "config.json"), "w") as fh:
            json.dump(cfg, fh)
        byfile = {}
        for n, t in tensors.items():
            byfile.setdefault(shards[n], {})[n] = t.contiguous()
        for fn, ts in byfile.items():
            save_file(ts, os.path.join(d, fn), metadata={"format": "pt"})
        with open(os.path.join(d, "model.safetensors.index.json"), "w") as fh:
            json.dump({"metadata": {"total_size": 0}, "weight_map": shards}, fh)
        return d

    return _write
