"""Backward-compatible alias for `resgate.ablation.direction`."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module(".ablation.direction", __package__)
