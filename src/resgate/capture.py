"""Backward-compatible alias for `resgate.ablation.capture`."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module(".ablation.capture", __package__)
