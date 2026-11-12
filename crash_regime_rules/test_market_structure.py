"""Offline causal ICT/GC breaker tests. No TWS, orders or market quotes."""
from __future__ import annotations

import ast
import contextlib
import copy
import io
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import market_structure as m
import market_transition as transition
import gc_breaker_monitor as gc
import gc_regime_1_0_19 as engine
import run_gc_breaker_monitor as runner

CFG = m.DEFAULT_STRUCTURE


def mirror(bars):
    x = bars.copy()
    x["open"], x["close"] = 300-bars.open, 300-bars.close
    x["high"], x["low"] = 300-bars.low, 300-bars.high
    return x


def appended(bars, values):
    candidates = m._expected_ends(bars.index[-1]+pd.Timedelta(minutes=1),
                                  bars.index[-1]+pd.Timedelta(days=3), 240, CFG)
    extra = pd.DataFrame([values], index=candidates[:1],
                         columns=["open", "high", "low", "close", "volume"])
    extra["start"] = extra.index-pd.Timedelta(hours=4)
    extra["scheduled_minutes"] = 240
    return pd.concat([bars, extra])


def breaker(bars=None, config=CFG):
    detected = m.detect_zones(gc.demo_htf_bars() if bars is None else bars, 240, config)
    return next(z for z in detected["zones"] if z["kind"] == "BREAKER")


def five_after(zone, rows):
    idx = pd.date_range(zone["known_at"]+pd.Timedelta(minutes=5), periods=len(rows), freq="5min")
    data = pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"])
    data["volume"], data["prior_atr"] = 100, 1.0
    return data


class DetectionTests(unittest.TestCase):
    def test_automatic_bullish_breaker_bounds_parent_and_evidence(self):
        z = breaker()
        self.assertEqual((z["direction"], z["low"], z["high"]), (1, 100, 102))
        self.assertEqual(z["midpoint"], 101)
        self.assertEqual(z["lifecycle"], "ACTIVE")
        self.assertTrue(z["evidence"]["sweep_confirmed"])
        self.assertTrue(z["evidence"]["displaced_structure_break"])
        self.assertTrue(z["parent_id"])
        self.assertLess(z["origin_at"], z["known_at"])
        self.assertFalse(z["execution_allowed"])

    def test_bearish_mirror_symmetry(self):
        z = breaker(mirror(gc.demo_htf_bars()))
        self.assertEqual((z["direction"], z["low"], z["high"]), (-1, 198, 200))

    def test_fvg_three_candle_geometry_and_delayed_known_time(self):
        b = gc.demo_htf_bars()
        r = m.detect_zones(b.iloc[:21], 240, CFG)
        fvg = next(z for z in r["zones"] if z["kind"] == "FVG")
        self.assertEqual((fvg["low"], fvg["high"]), (97, 100))
        self.assertEqual(fvg["known_at"], b.index[20])
        self.assertEqual(fvg["origin_at"], b.index[19])

    def test_order_block_waits_for_fvg_third_bar(self):
        b = gc.demo_htf_bars()
        earlier = m.detect_zones(b.iloc[:20], 240, CFG)
        self.assertFalse(any(z["kind"] == "ORDER_BLOCK" for z in earlier["zones"]))
        later = m.detect_zones(b.iloc[:21], 240, CFG)
        z = next(z for z in later["zones"] if z["kind"] == "ORDER_BLOCK")
        self.assertEqual(z["known_at"], b.index[20])
        self.assertTrue(z["evidence"]["fvg_confirmed"])

    def test_inversion_fvg_not_breaker_and_preserves_bounds(self):
        r = m.detect_zones(gc.demo_htf_bars(), 240, CFG)
        inv = next(z for z in r["zones"] if z["kind"] == "INVERSION_FVG")
        source = next(z for z in r["zones"] if z["id"] == inv["parent_id"])
        self.assertEqual(inv["direction"], -source["direction"])
        self.assertEqual((inv["low"], inv["high"]), (source["low"], source["high"]))
        self.assertEqual(source["lifecycle"], "INVALIDATED")

    def test_no_sweep_no_strict_breaker(self):
        b = gc.demo_htf_bars()
        b.iloc[24, b.columns.get_loc("low")] = 93.5
        r = m.detect_zones(b, 240, CFG)
        self.assertFalse(any(z["kind"] == "BREAKER" for z in r["zones"]))
        self.assertTrue(any(z["kind"] == "ORDER_BLOCK" and z["lifecycle"] == "INVALIDATED"
                            for z in r["zones"]))

    def test_no_known_swing_break_no_strict_breaker(self):
        b = gc.demo_htf_bars()
        b.iloc[19, b.columns.get_loc("high")] = 102  # tied high prevents confirmed pivot
        r = m.detect_zones(b, 240, CFG)
        self.assertFalse(any(z["kind"] == "BREAKER" for z in r["zones"]))

    def test_relaxed_diagnostic_profile_still_has_no_permission(self):
        b = gc.demo_htf_bars()
        b.iloc[19, b.columns.get_loc("high")] = 102
        z = breaker(b, replace(CFG, require_breaker_mss=False))
        self.assertFalse(z["evidence"]["displaced_structure_break"])
        self.assertFalse(z["execution_allowed"])

    def test_wick_through_breaker_is_not_htf_invalidation(self):
        b = appended(gc.demo_htf_bars(), [103.5, 104, 95, 103, 100])
        z = breaker(b)
        self.assertEqual(z["lifecycle"], "ACTIVE")
        self.assertIsNone(z["invalidated_at"])

    def test_htf_close_through_far_edge_invalidates(self):
        b = appended(gc.demo_htf_bars(), [103.5, 104, 97, 98, 100])
        z = breaker(b)
        self.assertEqual(z["lifecycle"], "INVALIDATED")
        self.assertEqual(z["invalidated_at"], b.index[-1])

    def test_missing_htf_bar_prevents_false_breaker_and_marks_old_zone(self):
        b = gc.demo_htf_bars().drop(gc.demo_htf_bars().index[22])
        r = m.detect_zones(b, 240, CFG)
        self.assertGreater(r["missing_scheduled_bars"], 0)
        self.assertFalse(any(z["kind"] == "BREAKER" for z in r["zones"]))
        self.assertTrue(any(z["lifecycle"] == "UNVERIFIED_AFTER_HTF_GAP" for z in r["zones"]))

    def test_pivot_and_fvg_events_never_known_before_reference(self):
        r = m.detect_zones(gc.demo_htf_bars(), 240, CFG)
        for event in r["events"]:
            if "reference_known_at" in event:
                self.assertLessEqual(event["reference_known_at"], event["at"])
        for z in r["zones"]:
            self.assertLess(z["origin_at"], z["known_at"])

    def test_zone_boundaries_and_id_not_repainted_by_later_prices(self):
        a = breaker()
        b = breaker(appended(gc.demo_htf_bars(), [104.5, 120, 103, 115, 100]))
        for field in ("id", "low", "high", "midpoint", "known_at", "origin_at"):
            self.assertEqual(a[field], b[field])

    def test_body_boundary_profile_uses_body_not_wicks(self):
        z = breaker(config=replace(CFG, zone_bounds="BODY"))
        self.assertEqual((z["low"], z["high"]), (100.2, 101.7))

    def test_displacement_filter_can_reject_all_zones(self):
        r = m.detect_zones(gc.demo_htf_bars(), 240, replace(CFG, displacement_body_atr=100))
        self.assertFalse(r["zones"])

    def test_order_block_origin_sweep_evidence_is_explicit(self):
        r = m.detect_zones(gc.demo_htf_bars(), 240, CFG)
        z = next(z for z in r["zones"] if z["kind"] == "ORDER_BLOCK")
        self.assertTrue(z["evidence"]["origin_candle_sweep"])

    def test_bounded_zone_storage(self):
        r = m.detect_zones(gc.demo_htf_bars(), 240, replace(CFG, max_zones_per_timeframe=2))
        self.assertLessEqual(len(r["zones"]), 2)


class ReactionTests(unittest.TestCase):
    def setUp(self):
        self.z = breaker()

    def test_closed_bullish_retest_and_current_touch(self):
        data = five_after(self.z, [[101.5, 103, 101, 102.8]])
        z = m.react_to_zone(self.z, data, data.index[-1], CFG)
        self.assertEqual(z["last_reaction"], "LTF_RETEST_HELD")
        self.assertTrue(z["reaction_fresh"])
        self.assertTrue(z["current_interaction"])
        self.assertEqual(z["current_position"], "ABOVE_ZONE")

    def test_touch_inside_is_not_confirmed(self):
        data = five_after(self.z, [[102.5, 102.6, 100.5, 101]])
        z = m.react_to_zone(self.z, data, data.index[-1], CFG)
        self.assertEqual(z["last_reaction"], "TOUCH_UNCONFIRMED")
        self.assertIsNone(z["confirmed_retest_at"])
        self.assertEqual(z["current_position"], "INSIDE_ZONE")
        self.assertTrue(z["ce_touched"])

    def test_large_range_touch_not_good_retest(self):
        data = five_after(self.z, [[101.5, 105, 99, 103]])
        z = m.react_to_zone(self.z, data, data.index[-1], CFG)
        self.assertEqual(z["last_reaction"], "TOUCH_UNCONFIRMED")
        self.assertIsNone(z["confirmed_retest_at"])

    def test_5m_failure_not_4h_invalidation(self):
        data = five_after(self.z, [[102.5, 103, 98.5, 99]])
        z = m.react_to_zone(self.z, data, data.index[-1], CFG)
        self.assertEqual(z["last_reaction"], "LTF_FAILED_RETEST")
        self.assertEqual(z["lifecycle"], "ACTIVE")
        self.assertIsNone(z["invalidated_at"])

    def test_later_reclaim_preserves_failed_history(self):
        data = five_after(self.z, [[102.5, 103, 98.5, 99], [101, 103, 100.8, 102.8]])
        z = m.react_to_zone(self.z, data, data.index[-1], CFG)
        self.assertEqual(z["last_reaction"], "LTF_RECLAIM_FROM_WRONG_SIDE")
        self.assertEqual(z["failed_ltf_at"], data.index[0])
        self.assertEqual(z["reclaimed_at"], data.index[-1])
        self.assertIsNone(z["confirmed_retest_at"])

    def test_no_same_confirmation_candle_retest(self):
        data = five_after(self.z, [[101.5, 103, 101, 102.8]])
        data.index -= pd.Timedelta(minutes=5)
        z = m.react_to_zone(self.z, data, self.z["known_at"], CFG)
        self.assertIsNone(z["first_touch_at"])
        self.assertIsNone(z["confirmed_retest_at"])

    def test_first_missing_five_minute_stops_later_false_confirmation(self):
        data = five_after(self.z, [[103, 104, 102.5, 103.5], [101.5, 103, 101, 102.8]])
        data = data.iloc[1:]
        z = m.react_to_zone(self.z, data, data.index[-1], CFG)
        self.assertEqual(z["reaction_quality"], "OBSERVATION_GAP")
        self.assertEqual(z["last_reaction"], "UNVERIFIED_OBSERVATIONS")
        self.assertIsNone(z["confirmed_retest_at"])
        self.assertFalse(z["current_interaction"])

    def test_gap_after_known_retest_is_historical_not_current(self):
        data = five_after(self.z, [[101.5, 103, 101, 102.8], [103, 104, 102.5, 103.5],
                                  [101.5, 103, 101, 102.8]])
        data = data.drop(data.index[1])
        z = m.react_to_zone(self.z, data, data.index[-1], CFG)
        self.assertEqual(z["confirmed_retest_at"], data.index[0])
        self.assertEqual(z["reaction_evaluated_until"], data.index[0])
        self.assertFalse(z["reaction_fresh"])

    def test_missing_atr_does_not_confirm(self):
        data = five_after(self.z, [[101.5, 103, 101, 102.8]])
        data["prior_atr"] = np.nan
        z = m.react_to_zone(self.z, data, data.index[-1], CFG)
        self.assertIsNone(z["confirmed_retest_at"])
        self.assertEqual(z["reaction_quality"], "PARTIAL_ATR_HISTORY")

    def test_repeated_return_is_not_presented_as_fresh_unmitigated_zone(self):
        data = five_after(self.z, [[101.5, 103, 101, 102.8], [103, 104, 102.5, 103.5],
                                  [101.5, 103, 101, 102.8]])
        z = m.react_to_zone(self.z, data, data.index[-1], CFG)
        self.assertEqual(z["visit_count"], 2)
        self.assertTrue(z["repeated_retest_warning"])
        self.assertEqual(z["touch_count"], 2)

    def test_no_later_observation_with_elapsed_schedule_is_unknown(self):
        data = five_after(self.z, [[101.5, 103, 101, 102.8]]).iloc[:0]
        cutoff = self.z["known_at"]+pd.Timedelta(minutes=10)
        z = m.react_to_zone(self.z, data, cutoff, CFG)
        self.assertEqual(z["reaction_quality"], "OBSERVATION_GAP")
        self.assertEqual(z["last_reaction"], "UNVERIFIED_OBSERVATIONS")

    def test_expiry_does_not_leave_active_label(self):
        data = five_after(self.z, [[101.5, 103, 101, 102.8]])
        z = m.react_to_zone(self.z, data, self.z["expires_at"], CFG)
        self.assertEqual(z["lifecycle"], "EXPIRED")
        self.assertFalse(z["current_interaction"])

    def test_bearish_retest_mirror(self):
        z = breaker(mirror(gc.demo_htf_bars()))
        data = five_after(z, [[198.5, 199, 197, 197.2]])
        result = m.react_to_zone(z, data, data.index[-1], CFG)
        self.assertEqual(result["last_reaction"], "LTF_RETEST_HELD")
        self.assertEqual(result["current_position"], "BELOW_ZONE")


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.x = gc.demo_history()
        self.context = {"mode": "SYNTHETIC", "last_bar_is_complete": True}

    def test_full_minute_pipeline_automatic_gc_breaker_and_retest(self):
        r = gc.evaluate_gc_breaker_monitor(self.x, data_context=self.context)
        self.assertEqual(r["active_bullish"], 1)
        self.assertEqual(r["zones"][0]["last_reaction"], "LTF_RETEST_HELD")
        self.assertFalse(r["execution_allowed"])

    def test_latest_forming_minute_mutation_does_not_change_result(self):
        a = gc.evaluate_gc_breaker_monitor(self.x)
        self.x.iloc[-1] = [105, 150, 50, 105, 100000]
        self.assertEqual(a, gc.evaluate_gc_breaker_monitor(self.x))

    def test_forming_four_hour_cannot_create_confirmed_breaker(self):
        end = gc.demo_htf_bars().index[-1]
        # Drop the last minute of the HTF confirmation bar, so it cannot close.
        x = self.x.loc[self.x.index < end-pd.Timedelta(minutes=1)]
        r = gc.evaluate_gc_breaker_monitor(x, data_context=self.context)
        self.assertFalse(r["zones"])

    def test_partial_three_hour_session_bar_is_not_four_hour(self):
        from market_transition import _bars, _prepare, _active_minutes, GC_TRANSITION
        end = self.x.index[-1]+pd.Timedelta(minutes=1)
        prepared = _active_minutes(_prepare(self.x, GC_TRANSITION), GC_TRANSITION)
        raw = _bars(prepared, 240, end, GC_TRANSITION)
        self.assertTrue((raw.scheduled_minutes < 240).any())
        r = m.evaluate_market_structure(self.x, config=replace(CFG, timeframes_minutes=(240,)),
                                        data_context=self.context)
        self.assertEqual(r["timeframes"]["4h"]["bars"], int((raw.scheduled_minutes == 240).sum()))

    def test_sufficient_history_not_required_for_basic_vwap_whipsaw(self):
        from test_market_transition import history
        r = transition.evaluate_transition_risk(history())
        self.assertEqual(r["ict_structure"]["state"], "INSUFFICIENT_HISTORY")
        self.assertIn(r["whipsaw"]["state"], ("LOW", "ELEVATED", "HIGH"))

    def test_observation_layer_never_changes_legacy_gates(self):
        a = transition.evaluate_transition_risk(self.x, data_context=self.context)
        cfg = replace(transition.GC_TRANSITION, structure=replace(CFG, enabled=False))
        b = transition.evaluate_transition_risk(self.x, config=cfg, data_context=self.context)
        for key in ("block_new_longs", "block_new_shorts", "long_block_reasons",
                    "short_block_reasons", "confirmation_conflict", "timeframes", "whipsaw"):
            self.assertEqual(a[key], b[key])

    def test_shared_transition_includes_all_three_ict_timeframes(self):
        r = transition.evaluate_transition_risk(self.x, data_context=self.context)
        self.assertEqual(set(r["ict_structure"]["timeframes"]), {"1h", "2h", "4h"})
        text = "\n".join(transition.format_transition_risk(r))
        self.assertIn("ICT ÁRSTRUKTÚRA", text)
        self.assertIn("BREAKER-JELÖLT", text)
        self.assertIn("ÓRÁS VWAP-FORDULÓK", text)

    def test_gc_snapshot_and_report_have_automatic_breaker(self):
        s = engine.build_gold_snapshot(self.x, data_context=self.context)
        self.assertEqual(s["gc_breaker_monitor"]["active_bullish"], 1)
        text = engine.run_gold_checklist(self.x, data_context=self.context)
        self.assertIn("GC / GOLD REGIME ENGINE VERSION 1.0.19", text)
        self.assertIn("BREAKER-JELÖLT", text)

    def test_stale_wall_source_cannot_claim_current_reaction(self):
        r = gc.evaluate_gc_breaker_monitor(self.x, data_context={
            **self.context, "observed_at": self.x.index[-1]+pd.Timedelta(hours=2)})
        self.assertEqual(r["state"], "DATA_UNAVAILABLE")
        self.assertTrue(all(not z["current_interaction"] and not z["reaction_fresh"] for z in r["zones"]))

    def test_future_source_cannot_claim_current_reaction(self):
        r = gc.evaluate_gc_breaker_monitor(self.x, data_context={
            **self.context, "observed_at": self.x.index[-1]-pd.Timedelta(minutes=5)})
        self.assertEqual(r["source_quality"], "FUTURE_SOURCE")
        self.assertEqual(r["state"], "DATA_UNAVAILABLE")

    def test_disabled_and_short_history_not_false_no_zone(self):
        r = gc.evaluate_gc_breaker_monitor(self.x, config=replace(CFG, enabled=False))
        self.assertEqual(r["state"], "DISABLED")
        r = gc.evaluate_gc_breaker_monitor(self.x.iloc[:1000])
        self.assertEqual(r["state"], "INSUFFICIENT_4H_HISTORY")

    def test_duplicate_minute_and_naive_input_rejected(self):
        with self.assertRaises(ValueError):
            m.evaluate_market_structure(pd.concat([self.x, self.x.iloc[-1:]]))
        x = self.x.copy()
        x.index = x.index.tz_localize(None)
        with self.assertRaises(ValueError):
            m.evaluate_market_structure(x)

    def test_invalid_config_rejected(self):
        for kwargs in (dict(zone_bounds="AUTO"), dict(pivot_right=0),
                       dict(min_fvg_atr=np.nan), dict(displacement_body_share=2),
                       dict(timeframes_minutes=(240, 240))):
            with self.assertRaises(ValueError):
                replace(CFG, **kwargs)

    def test_strict_export_round_trip_and_demo_isolation(self):
        with TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(runner.main(["--demo", "--output-dir", folder]), 0)
            file = Path(folder)/"demo"/"gc_breaker_latest.json"
            saved = json.loads(file.read_text(encoding="utf-8"))
            self.assertEqual(saved["active_bullish"], 1)
            self.assertTrue((Path(folder)/"demo"/"gc_breaker_zones.csv").exists())
            self.assertFalse((Path(folder)/"gc_breaker_latest.json").exists())
        self.assertIn("BULLISH BREAKER", out.getvalue())

    def test_formatter_does_not_mutate_zones(self):
        r = m.evaluate_market_structure(self.x, data_context=self.context)
        before = copy.deepcopy(r)
        m.format_market_structure(r)
        self.assertEqual(before, r)

    def test_no_order_account_or_network_calls_in_new_modules(self):
        forbidden = {"placeOrder", "exerciseOptions", "cancelOrder", "positions",
                     "accountSummary", "reqMktData", "reqHistoricalData"}
        for module in (m, gc, runner):
            tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
            calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute)}
            self.assertFalse(calls & forbidden)


if __name__ == "__main__":
    unittest.main()
