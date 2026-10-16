"""resgate CLI: `resgate targets|capture|direction|patch`.

Every option also reads the environment variable the original scripts used, so existing
invocations (SRC DST DIRS K MODE KEEP ACTS OUT PROMPTS MODEL DEV DEVMAP MAXMEM BS SYS THINK)
keep working. Explicit flags win over the environment.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence


def _env(name: str, default=None):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="resgate", description="sep(l) abliteration: four steps.")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("targets", help="derive the target tensor set from a checkpoint")
    t.add_argument("dirs", nargs="+", help="checkpoint directories")

    c = sub.add_parser("capture", help="capture last-prompt-token residuals, all layers")
    c.add_argument("--model", default=_env("MODEL"), required=_env("MODEL") is None)
    c.add_argument("--prompts", default=_env("PROMPTS"), required=_env("PROMPTS") is None,
                   help="JSON [{key, group, text}] -- group in {harmful, harmless}")
    c.add_argument("--out", default=_env("OUT"), required=_env("OUT") is None)
    c.add_argument("--sys", dest="sys_prompt", default=_env("SYS", "You are a helpful assistant."))
    c.add_argument("--bs", type=int, default=int(_env("BS", 8)))
    c.add_argument("--dev", default=_env("DEV", "cuda"))
    c.add_argument("--devmap", default=_env("DEVMAP"), help="e.g. auto (needs accelerate)")
    c.add_argument("--maxmem", default=_env("MAXMEM"), help="JSON dict for device_map")
    c.add_argument("--think", default=_env("THINK"),
                   help="'0' -> enable_thinking=False; unset -> chat-template default")

    d = sub.add_parser("direction", help="per-layer directions + sep(l) weights")
    d.add_argument("--acts", default=_env("ACTS"), required=_env("ACTS") is None)
    d.add_argument("--out", default=_env("OUT"), required=_env("OUT") is None)
    d.add_argument("--mode", default=_env("MODE", "sep"), choices=["sep", "uniform"])
    d.add_argument("--keep", type=float, default=float(_env("KEEP", 0.0)),
                   help="drop layers with sep below this fraction of the mean (canonical 0)")

    w = sub.add_parser("patch", help="rank-1 PRE weight edit; DST must be a copy of SRC")
    w.add_argument("--src", default=_env("SRC"), required=_env("SRC") is None)
    w.add_argument("--dst", default=_env("DST"), required=_env("DST") is None)
    w.add_argument("--dirs", default=_env("DIRS"), required=_env("DIRS") is None,
                   help="direction blob; comma-separated for sequential application")
    w.add_argument("-K", "--K", dest="K", type=float, default=float(_env("K", 1.0)))
    return p


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "targets":
        from .enumerate_targets import main as targets_main
        return targets_main(a.dirs)
    if a.cmd == "capture":
        from .capture import capture
        capture(a.model, a.prompts, a.out, sys_prompt=a.sys_prompt, bs=a.bs, dev=a.dev,
                devmap=a.devmap, maxmem=a.maxmem, think=a.think)
        return 0
    if a.cmd == "direction":
        from .direction import compute_directions
        compute_directions(a.acts, a.out, mode=a.mode, keep=a.keep)
        return 0
    if a.cmd == "patch":
        from .patch import patch
        patch(a.src, a.dst, a.dirs, K=a.K)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
