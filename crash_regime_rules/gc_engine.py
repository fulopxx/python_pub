"""Stable import boundary for the current GC checklist runner.

The implementation remains pinned to gc_regime_1_0_19. These are direct
re-exports, not wrappers or a second engine implementation. Historical modules
and offline audits keep their explicit version imports; in particular, this
module does not upgrade the audit from engine 1.0.18.
"""
from gc_regime_1_0_19 import build_dxy_macro_context, run_gold_checklist

__all__ = ["build_dxy_macro_context", "run_gold_checklist"]
