"""Compatibility import for the packaged exhaustion implementation.

Both import paths resolve to the same module object, including private helpers,
configuration singletons and monkeypatch targets. Keep this file while older
engines, runners, tests or saved Python objects use the historical module name.
"""
import sys as _sys

from market_regime_toolkit import exhaustion as _implementation

_sys.modules[__name__] = _implementation
