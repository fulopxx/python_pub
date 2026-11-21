"""Offline compatibility and spawn checks for the first packaged module."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import importlib
import multiprocessing
from pathlib import Path
import pickle
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import market_exhaustion as legacy
from market_regime_toolkit import exhaustion as packaged


ROOT = Path(__file__).resolve().parent


def _describe_config_in_worker(config):
    from market_regime_toolkit import exhaustion
    return (
        config.symbol,
        type(config).__module__,
        type(config) is exhaustion.ExhaustionConfig,
    )


class PackageLayoutTests(unittest.TestCase):
    def test_import_paths_are_one_module(self):
        self.assertIs(legacy, packaged)
        self.assertIs(sys.modules["market_exhaustion"], packaged)
        self.assertEqual(packaged.__name__, "market_regime_toolkit.exhaustion")

    def test_public_and_private_names_are_preserved(self):
        for name in (
            "ExhaustionConfig", "GC_EXHAUSTION", "ES_EXHAUSTION", "NQ_EXHAUSTION",
            "PROFILES", "evaluate_exhaustion", "format_exhaustion",
            "exhaustion_long_block", "advance_state", "build_index_cross_context",
            "_slot_baseline", "_htf_features", "_session_labels", "_atr",
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(legacy, name), getattr(packaged, name))

    def test_function_globals_share_patch_targets(self):
        self.assertIs(legacy.evaluate_exhaustion.__globals__, packaged.__dict__)
        original = packaged._features
        with patch.object(legacy, "_features") as replacement:
            self.assertIs(packaged._features, replacement)
            self.assertIs(legacy.evaluate_exhaustion.__globals__["_features"], replacement)
        self.assertIs(packaged._features, original)

    def test_current_config_pickle_roundtrip(self):
        for config in (legacy.GC_EXHAUSTION, legacy.ES_EXHAUSTION, legacy.NQ_EXHAUSTION):
            with self.subTest(symbol=config.symbol):
                restored = pickle.loads(pickle.dumps(config))
                self.assertEqual(restored, config)
                self.assertIs(type(restored), packaged.ExhaustionConfig)

    def test_historical_pickle_class_name_still_resolves(self):
        # A fixed, trusted pickle GLOBAL opcode, not an external pickle input.
        historical_class = pickle.loads(b"cmarket_exhaustion\nExhaustionConfig\n.")
        self.assertIs(historical_class, packaged.ExhaustionConfig)

    def test_fresh_import_orders_from_another_working_directory(self):
        for first, second in (
            ("market_exhaustion", "market_regime_toolkit.exhaustion"),
            ("market_regime_toolkit.exhaustion", "market_exhaustion"),
        ):
            code = (
                "import importlib, sys; "
                f"sys.path.insert(0, {str(ROOT)!r}); "
                f"a=importlib.import_module({first!r}); "
                f"b=importlib.import_module({second!r}); "
                "assert a is b; "
                "assert a.ExhaustionConfig is b.ExhaustionConfig"
            )
            with self.subTest(first=first), TemporaryDirectory() as directory:
                result = subprocess.run(
                    [sys.executable, "-c", code], cwd=directory,
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_package_alone_does_not_import_implementations(self):
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(ROOT)!r}); "
            "import market_regime_toolkit; "
            "assert 'market_regime_toolkit.exhaustion' not in sys.modules; "
            "assert 'ib_insync' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_config_crosses_spawn_process_boundary(self):
        with ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            result = executor.submit(
                _describe_config_in_worker, legacy.GC_EXHAUSTION,
            ).result(timeout=30)
        self.assertEqual(result, ("GC", "market_regime_toolkit.exhaustion", True))

    def test_engine_classes_and_audit_pin_remain_consistent(self):
        previous = importlib.import_module("gc_regime_1_0_18")
        current = importlib.import_module("gc_regime_1_0_19")
        audit = importlib.import_module("gc_audit_backtest_v1_3_12")
        self.assertIs(previous.ExhaustionConfig, packaged.ExhaustionConfig)
        self.assertIs(current.ExhaustionConfig, packaged.ExhaustionConfig)
        self.assertIs(audit.GoldRegime, previous.GoldRegime)
        self.assertIs(audit.GC_CONFIG, previous.CONFIG)
        self.assertIs(audit.GC_CONFIG.exhaustion, packaged.GC_EXHAUSTION)


if __name__ == "__main__":
    unittest.main()
