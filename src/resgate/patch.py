"""Backward-compatible alias for `resgate.ablation.patch`."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module(".ablation.patch", __package__)
