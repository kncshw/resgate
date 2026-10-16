"""Backward-compatible alias for `resgate.ablation.enumerate_targets`."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module(".ablation.enumerate_targets", __package__)
