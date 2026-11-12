"""NDOG/NWOG targets do not bypass entry gates or change stop/sizing."""
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import gc_regime_1_0_18 as previous
import gc_regime_1_0_19 as engine


def gap(low, ce, high, opened="2026-10-19 22:00:00+00:00"):
    return dict(gap_low=low, gap_ce=ce, gap_high=high, open_timestamp=opened)


def snapshot():
    return dict(
        timestamp=pd.Timestamp("2026-10-20 10:00:00+00:00"),
        close=4000., atr_5m=10., rsi_2=50., rsi_5=50., vwap_distance_atr=0.,
        macro_analysis=dict(bull_score=0, bear_score=0, available=False, message="TEST"),
        gap_context=dict(
            latest_ndog=None, latest_nwog=None,
            ndog_reaction=dict(reaction="NO_RECENT_INTERACTION"),
            nwog_reaction=dict(reaction="NO_RECENT_INTERACTION"),
        ),
    )


def plan(s=None, short=False, config=engine.CONFIG):
    return engine.calculate_gold_risk_plan(
        snapshot() if s is None else s,
        engine.GoldRegime.BEAR_TREND if short else engine.GoldRegime.BULL_TREND,
        10_000., config,
    )


class TargetTests(unittest.TestCase):
    def test_no_gap_falls_back_to_original_targets(self):
        p = plan()
        self.assertEqual((p["target_1"], p["target_2"]), (4010., 4020.))
        self.assertEqual(p["target_1_source"], "ATR")

    def test_near_ndog_high_replaces_long_first_target(self):
        s = snapshot()
        s["gap_context"]["latest_ndog"] = gap(4007, 4008, 4009)
        p = plan(s)
        self.assertEqual(p["target_1"], 4009.)
        self.assertEqual(p["target_1_source"], "NDOG GAP_HIGH")
        self.assertEqual(p["target_1_atr_base"], 4010.)
        self.assertEqual(p["target_1_reward_r"], 0.9)

    def test_near_nwog_low_replaces_second_target(self):
        s = snapshot()
        s["gap_context"]["latest_nwog"] = gap(4021, 4022, 4023)
        p = plan(s)
        self.assertEqual(p["target_2"], 4021.)
        self.assertEqual(p["target_2_source"], "NWOG GAP_LOW")
        self.assertEqual(p["target_2_reward_r"], 2.1)

    def test_both_targets_aligned(self):
        s = snapshot()
        s["gap_context"]["latest_ndog"] = gap(4007, 4008, 4009)
        s["gap_context"]["latest_nwog"] = gap(4021, 4022, 4023)
        p = plan(s)
        self.assertEqual((p["target_1"], p["target_2"]), (4009., 4021.))
        self.assertLess(p["entry"], p["target_1"])
        self.assertLess(p["target_1"], p["target_2"])

    def test_ce_is_a_candidate(self):
        s = snapshot()
        s["gap_context"]["latest_ndog"] = gap(4005, 4010, 4015)
        p = plan(s)
        self.assertEqual(p["target_1"], 4010.)
        self.assertEqual(p["target_1_source"], "NDOG GAP_CE")

    def test_short_mirrors_long(self):
        s = snapshot()
        s["gap_context"]["latest_ndog"] = gap(3991, 3992, 3993)
        p = plan(s, short=True)
        self.assertEqual(p["target_1"], 3991.)
        self.assertEqual(p["target_1_source"], "NDOG GAP_LOW")
        self.assertEqual(p["target_1_reward_r"], 0.9)
        self.assertGreater(p["entry"], p["target_1"])
        self.assertGreater(p["target_1"], p["target_2"])

    def test_levels_behind_entry_not_used(self):
        for short, candidate in (
            (False, gap(3980, 3985, 3990)), (True, gap(4005, 4010, 4015))
        ):
            with self.subTest(short=short):
                s = snapshot()
                s["gap_context"]["latest_ndog"] = candidate
                p = plan(s, short=short)
                self.assertEqual(p["target_1_source"], "ATR")
                self.assertEqual(p["target_2_source"], "ATR")

    def test_far_level_not_used(self):
        s = snapshot()
        s["gap_context"]["latest_ndog"] = gap(4030, 4031, 4032)
        p = plan(s)
        self.assertEqual((p["target_1"], p["target_2"]), (4010., 4020.))

    def test_tolerance_boundary_inclusive(self):
        s = snapshot()
        s["gap_context"]["latest_ndog"] = gap(4015, 4016, 4017)
        p = plan(s)
        self.assertEqual(p["target_1"], 4015.)
        self.assertNotEqual(p["target_1_source"], "ATR")

    def test_just_outside_tolerance_not_used_for_first_target(self):
        s = snapshot()
        s["gap_context"]["latest_ndog"] = gap(4015.01, 4016, 4017)
        self.assertEqual(plan(s)["target_1_source"], "ATR")

    def test_custom_proximity(self):
        s = snapshot()
        s["gap_context"]["latest_ndog"] = gap(4007, 4008, 4009)
        p = plan(s, config=replace(engine.CONFIG, gap_target_proximity_atr=0.05))
        self.assertEqual(p["target_1_source"], "ATR")

    def test_zero_proximity_matches_only_exact_level(self):
        s = snapshot()
        s["gap_context"]["latest_ndog"] = gap(4009, 4010, 4011)
        p = plan(s, config=replace(engine.CONFIG, gap_target_proximity_atr=0.0))
        self.assertEqual(p["target_1_source"], "NDOG GAP_CE")
        self.assertEqual(p["target_2_source"], "ATR")

    def test_same_display_price_cannot_be_both_targets(self):
        s = snapshot()
        s["gap_context"]["latest_ndog"] = gap(4015., 4015.001, 4015.002)
        p = plan(s)
        self.assertLess(p["target_1"], p["target_2"])
        self.assertEqual(sum(p[f"target_{n}_source"] != "ATR" for n in (1, 2)), 1)

    def test_deterministic_ties(self):
        s = snapshot()
        s["gap_context"]["latest_ndog"] = gap(4009, 4010, 4011)
        s["gap_context"]["latest_nwog"] = gap(4009, 4010, 4011)
        first = plan(s)
        self.assertEqual(first, plan(s))
        self.assertEqual(first["target_1_source"], "NDOG GAP_CE")

    def test_disabled_matches_all_legacy_plan_fields(self):
        s = snapshot()
        s["gap_context"]["latest_ndog"] = gap(4007, 4008, 4009)
        for short in (False, True):
            old = previous.calculate_gold_risk_plan(
                s, previous.GoldRegime.BEAR_TREND if short else previous.GoldRegime.BULL_TREND,
                10000.,
            )
            new = plan(s, short, replace(engine.CONFIG, gap_targets_enabled=False))
            self.assertEqual({k: new[k] for k in old}, old)

    def test_stop_sizing_and_entry_unchanged_when_targets_move(self):
        s = snapshot()
        baseline = plan(s)
        s["gap_context"]["latest_ndog"] = gap(4007, 4008, 4009)
        aligned = plan(s)
        for key in ("trade_allowed", "side", "entry", "stop_price", "atr_5m",
                    "stop_distance_usd_per_oz", "risk_budget_usd",
                    "gc_risk_per_contract_usd", "mgc_risk_per_contract_usd",
                    "max_gc_contracts", "max_mgc_contracts"):
            self.assertEqual(baseline[key], aligned[key], key)

    def test_input_not_mutated_and_metadata_serializable(self):
        s = snapshot()
        s["gap_context"]["latest_ndog"] = gap(4007, 4008, 4009)
        before = deepcopy(s)
        p = plan(s)
        self.assertEqual(before, s)
        json.dumps(p, allow_nan=False)


class QualityAndGuardTests(unittest.TestCase):
    def test_future_gap_ignored(self):
        s = snapshot()
        s["gap_context"]["latest_ndog"] = gap(4007, 4008, 4009, "2026-10-20T11:00:00Z")
        self.assertEqual(plan(s)["target_1_source"], "ATR")

    def test_missing_or_naive_gap_time_ignored(self):
        for opened in (None, "2026-10-19 22:00:00", "bad"):
            with self.subTest(opened=opened):
                s = snapshot()
                s["gap_context"]["latest_ndog"] = gap(4007, 4008, 4009, opened)
                self.assertEqual(plan(s)["target_1_source"], "ATR")

    def test_missing_or_naive_snapshot_time_ignored(self):
        for asof in (None, "bad", "2026-10-20 10:00:00"):
            with self.subTest(asof=asof):
                s = snapshot()
                s["timestamp"] = asof
                s["gap_context"]["latest_ndog"] = gap(4007, 4008, 4009)
                self.assertEqual(plan(s)["target_1_source"], "ATR")

    def test_invalid_gap_prices_ignored(self):
        for prices in ((4009, 4008, 4007), (4007, 5000, 4009), (4008, 4008, 4008),
                       (np.nan, 4008, 4009), (4007, np.inf, 4009), (-1, 4008, 4009)):
            with self.subTest(prices=prices):
                s = snapshot()
                s["gap_context"]["latest_ndog"] = gap(*prices)
                self.assertEqual(plan(s)["target_1_source"], "ATR")

    def test_config_rejects_invalid_tolerance(self):
        for value in (-0.1, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(engine.CONFIG, gap_target_proximity_atr=value)
        with self.assertRaises(ValueError):
            replace(engine.CONFIG, gap_targets_enabled="False")

    def test_all_entry_guards_remain_in_force(self):
        cases = [
            (False, {"rsi_2": 99}),
            (True, {"rsi_2": 1}),
            (False, {"vwap_distance_atr": 10}),
            (True, {"vwap_distance_atr": -10}),
            (False, {"exhaustion": {"block_new_longs": True, "state": "EXHAUSTION_RISK",
                                    "exhaustion_checks": []}}),
            (False, {"transition_risk": {"block_new_longs": True,
                                         "long_block_reasons": ["test"]}}),
            (True, {"transition_risk": {"block_new_shorts": True,
                                        "short_block_reasons": ["test"]}}),
            (False, {"macro_analysis": {"bull_score": 0, "bear_score": 2}}),
            (True, {"macro_analysis": {"bull_score": 2, "bear_score": 0}}),
        ]
        for short, changes in cases:
            with self.subTest(short=short, changes=changes):
                s = snapshot()
                s.update(changes)
                s["gap_context"]["latest_ndog"] = gap(4007, 4008, 4009)
                self.assertFalse(plan(s, short)["trade_allowed"])
        for short, reaction in ((False, "BEARISH_REJECTION"), (True, "BULLISH_RECLAIM")):
            s = snapshot()
            s["gap_context"]["latest_ndog"] = gap(4007, 4008, 4009)
            s["gap_context"]["ndog_reaction"]["reaction"] = reaction
            self.assertFalse(plan(s, short)["trade_allowed"])

    def test_neutral_and_blowoff_still_blocked(self):
        for regime in (engine.GoldRegime.NEUTRAL, engine.GoldRegime.BLOWOFF_RISK):
            self.assertFalse(engine.calculate_gold_risk_plan(snapshot(), regime, 10000)["trade_allowed"])

    def test_snapshot_and_regime_functions_unchanged(self):
        root = Path(__file__).parent
        old = ast.parse((root/"gc_regime_1_0_18.py").read_text(encoding="utf-8"))
        new = ast.parse((root/"gc_regime_1_0_19.py").read_text(encoding="utf-8"))
        for name in ("build_gold_snapshot", "gold_regime_active", "add_session_key",
                     "build_gap_context", "select_latest_gap"):
            a = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == name)
            b = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(a), ast.dump(b), name)

    def test_report_prints_source_base_targets_and_actual_r(self):
        index = pd.date_range("2026-10-19", periods=2400, freq="min", tz="UTC")
        frame = pd.DataFrame(
            dict(open=4000., high=4002., low=3998., close=4001., volume=100.),
            index=index,
        )
        full = engine.build_gold_snapshot(frame, data_context={"last_bar_is_complete": True})
        minimal = snapshot()
        full.update({k: v for k, v in minimal.items() if k != "gap_context"})
        candidate = engine.make_gap(
            "NDOG", 4007., 4009., pd.Timestamp("2026-10-19T21:00:00Z"),
            pd.Timestamp("2026-10-19T22:00:00Z"),
        )
        full["gap_context"]["latest_ndog"] = candidate
        full["gap_context"]["ndog_reaction"]["reaction"] = "NO_RECENT_INTERACTION"
        full["gap_context"]["nwog_reaction"]["reaction"] = "NO_RECENT_INTERACTION"
        full["exhaustion"]["block_new_longs"] = False
        full["transition_risk"]["block_new_longs"] = False
        result = engine.gold_regime_active(full)
        result["regime"] = engine.GoldRegime.BULL_TREND
        with patch.object(engine, "build_gold_snapshot", return_value=full), patch.object(
            engine, "gold_regime_active", return_value=result
        ):
            text = engine.run_gold_checklist(frame)
        self.assertIn("VERSION 1.0.19", text)
        self.assertIn("NDOG GAP_HIGH", text)
        self.assertIn("Eredeti ATR-célárak", text)
        self.assertIn("0.90R", text)


if __name__ == "__main__":
    unittest.main()
