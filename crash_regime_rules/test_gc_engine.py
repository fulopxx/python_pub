"""Offline regression checks for the stable current-GC import boundary."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import gc_engine
import gc_regime_1_0_18 as audit_engine
import gc_regime_1_0_19 as current_engine


ROOT = Path(__file__).resolve().parent


class StableGCEngineTests(unittest.TestCase):
    def test_macro_builder_is_original_function(self):
        self.assertIs(
            gc_engine.build_dxy_macro_context,
            current_engine.build_dxy_macro_context,
        )

    def test_checklist_is_original_function(self):
        self.assertIs(gc_engine.run_gold_checklist, current_engine.run_gold_checklist)

    def test_explicit_public_surface(self):
        self.assertEqual(
            gc_engine.__all__, ["build_dxy_macro_context", "run_gold_checklist"],
        )

    def test_runner_imports_stable_surface(self):
        tree = ast.parse((ROOT / "run_gc_checklist_v1_7.py").read_text(encoding="utf-8"))
        imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        stable = [node for node in imports if node.module == "gc_engine"]
        self.assertEqual(len(stable), 1)
        self.assertEqual(
            [(name.name, name.asname) for name in stable[0].names],
            [("build_dxy_macro_context", None), ("run_gold_checklist", None)],
        )
        self.assertFalse(
            any((node.module or "").startswith("gc_regime") for node in imports),
        )

    def test_runner_import_is_offline_and_preserves_function_identity(self):
        # A fake broker module makes this check independent of installed TWS
        # libraries and raises immediately if module import constructs a client.
        broker = types.ModuleType("ib_insync")
        broker.IB = Mock(side_effect=AssertionError("Unexpected broker client"))
        broker.ContFuture = Mock(side_effect=AssertionError("Unexpected contract"))
        broker.util = types.SimpleNamespace()
        daily_context = types.ModuleType("gc_daily_context")
        daily_context.fetch_gc_daily_context = Mock(
            side_effect=AssertionError("Unexpected daily-data request"),
        )
        name = "_gc_stable_runner_under_test"
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "run_gc_checklist_v1_7.py",
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {
            "ib_insync": broker, "gc_daily_context": daily_context, name: module,
        }):
            spec.loader.exec_module(module)
        self.assertIs(module.build_dxy_macro_context, current_engine.build_dxy_macro_context)
        self.assertIs(module.run_gold_checklist, current_engine.run_gold_checklist)
        broker.IB.assert_not_called()
        broker.ContFuture.assert_not_called()
        daily_context.fetch_gc_daily_context.assert_not_called()

    def test_current_audit_remains_pinned_to_engine_18(self):
        import gc_audit_backtest_v1_3_12 as audit

        for exported, original in (
            ("GC_CONFIG", "CONFIG"),
            ("GoldRegime", "GoldRegime"),
            ("build_gold_snapshot", "build_gold_snapshot"),
            ("calculate_gold_risk_plan", "calculate_gold_risk_plan"),
            ("gold_regime_active", "gold_regime_active"),
        ):
            with self.subTest(exported=exported):
                self.assertIs(getattr(audit, exported), getattr(audit_engine, original))


if __name__ == "__main__":
    unittest.main()
