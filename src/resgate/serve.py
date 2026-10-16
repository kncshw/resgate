"""Backward-compatible alias for `resgate.gate.serve`."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module(".gate.serve", __package__)
