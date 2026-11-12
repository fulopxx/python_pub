"""Offline regression tests. Run: python -m unittest test_market_exhaustion -v"""
from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

import market_exhaustion as m
import index_regime as idx
import gc_regime_1_0_13 as gc
import gc_audit_backtest as audit


def history(n: int = 6001, kind: str = "noise") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    stamps = pd.date_range("2026-09-01", periods=n, freq="min", tz="UTC")
    price = 4200 + np.cumsum(rng.normal(0, .15, n))
    if kind == "climax":
        price = np.full(n, 4200.0)
        price[-601:] += np.linspace(0, 30, 601)
        price[-31:] = 4228.0
    out = pd.DataFrame(dict(open=price, high=price + .3, low=price - .3,
                            close=price, volume=np.full(n, 40.0)), index=stamps)
    if kind == "climax":
        # A closed failed-breakout bar with a time-slot volume climax.
        out.iloc[-6:-1, out.columns.get_loc("high")] = 4235.0
        out.iloc[-6:-1, out.columns.get_loc("open")] = 4230.0
        out.iloc[-6:-1, out.columns.get_loc("low")] = 4227.5
        out.iloc[-6:-1, out.columns.get_loc("close")] = 4228.0
        out.iloc[-6:-1, out.columns.get_loc("volume")] = 160.0
    return out


def example_row(**updates):
    row = dict(atr=1.0, close=109.0, open=110.0, high=110.0,
               higher_low=105.0, peak=112.0)
    row.update(updates)
    return row


class ExhaustionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = m.ES_EXHAUSTION
        self.ts = pd.Timestamp("2026-10-12 14:00", tz="UTC")
        self.risk_gates = dict(sufficient=True, parabolic=True, exhaustion=True)
        self.normal_gates = dict(sufficient=True, parabolic=False, exhaustion=False)

    def latched(self):
        return m.advance_state({}, self.ts, example_row(), self.risk_gates, self.cfg)

    def test_risk_does_not_authorize_a_short(self):
        mem = self.latched()
        self.assertEqual(mem["state"], "EXHAUSTION_RISK")
        self.assertIsNone(mem["broken_at"])

    def test_recent_extension_can_precede_exhaustion(self):
        extension = dict(sufficient=True, parabolic=True, exhaustion=False)
        first = m.advance_state({}, self.ts, example_row(), extension, self.cfg)
        later = dict(sufficient=True, parabolic=False, exhaustion=False,
                     exhaustion_evidence=True)
        second = m.advance_state(first, self.ts + pd.Timedelta(minutes=10),
                                 example_row(), later, self.cfg)
        self.assertEqual(second["state"], "EXHAUSTION_RISK")
        expired = m.advance_state(first, self.ts + pd.Timedelta(minutes=65),
                                  example_row(), later, self.cfg)
        self.assertEqual(expired["state"], "NORMAL")

    def test_break_and_retest_must_be_separate_bars(self):
        mem = m.advance_state(self.latched(), self.ts + pd.Timedelta(minutes=5),
                              example_row(close=104, open=106, high=107),
                              self.normal_gates, self.cfg)
        self.assertEqual(mem["state"], "EXHAUSTION_RISK")
        self.assertIsNotNone(mem["broken_at"])
        mem = m.advance_state(mem, self.ts + pd.Timedelta(minutes=10),
                              example_row(close=104, open=105, high=105.1),
                              self.normal_gates, self.cfg)
        self.assertEqual(mem["state"], "BREAKDOWN_CONFIRMED")

    def test_duplicate_does_not_renew_latch(self):
        first = self.latched()
        repeat = m.advance_state(first, self.ts, example_row(),
                                 self.risk_gates, self.cfg)
        self.assertEqual(first, repeat)

    def test_latch_survives_falling_rsi_and_expires(self):
        first = self.latched()
        second = m.advance_state(first, self.ts + pd.Timedelta(minutes=5),
                                 example_row(), self.normal_gates, self.cfg)
        self.assertEqual(second["state"], "EXHAUSTION_RISK")
        third = m.advance_state(second, self.ts + pd.Timedelta(minutes=245),
                                example_row(), self.normal_gates, self.cfg)
        self.assertEqual(third["state"], "NORMAL")

    def test_two_new_high_closes_reset_old_risk(self):
        first = self.latched()
        second = m.advance_state(first, self.ts + pd.Timedelta(minutes=5),
                                 example_row(close=114), self.normal_gates, self.cfg)
        self.assertEqual(second["state"], "EXHAUSTION_RISK")
        third = m.advance_state(second, self.ts + pd.Timedelta(minutes=10),
                                example_row(close=115), self.risk_gates, self.cfg)
        self.assertEqual(third["state"], "PARABOLIC_EXTENSION")
        self.assertNotIn("broken_at", third)

    def test_short_history_is_unknown_and_blocks_long(self):
        d = history(181)
        out = m.evaluate_exhaustion(d, "GC")
        self.assertEqual(out["state"], "UNKNOWN")
        self.assertTrue(out["block_new_longs"])
        self.assertFalse(out["execution_ready"])
        again = m.evaluate_exhaustion(d, "GC", previous_state=out["memory"])
        self.assertEqual(again["state"], "UNKNOWN")

    def test_future_and_partial_minutes_cannot_change_completed_features(self):
        d = history()
        full = m.evaluate_exhaustion(d, "GC")
        changed = d.copy()
        changed.iloc[-1, changed.columns.get_loc("high")] += 1000
        spike = m.evaluate_exhaustion(changed, "GC")
        for key, value in full["metrics"].items():
            if pd.isna(value):
                self.assertTrue(pd.isna(spike["metrics"][key]), key)
            else:
                self.assertEqual(value, spike["metrics"][key], key)
        # Compare full-history features at an earlier completion to a prefix.
        clean = m._clean(d)
        cutoff = clean.index[-1]
        bars = m._closed_bars(clean[clean.index < cutoff], 5, cutoff)
        hourly = m._closed_bars(clean[clean.index < cutoff], 60, cutoff)
        daily = m._daily_from_minutes(clean[clean.index < cutoff], cutoff)
        all_features = m._features(clean[clean.index < cutoff], bars, hourly, daily,
                                   m.GC_EXHAUSTION)
        end = bars.index[-31]
        prefix = clean.loc[clean.index < end]
        p5 = m._closed_bars(prefix, 5, end)
        p60 = m._closed_bars(prefix, 60, end)
        pdaily = m._daily_from_minutes(prefix, end)
        pf = m._features(prefix, p5, p60, pdaily, m.GC_EXHAUSTION)
        for key in ("atr", "rsi", "h_extension", "h_acceleration", "higher_low",
                    "volume_ratio_slot", "divergence"):
            a, b = all_features.loc[end, key], pf.loc[end, key]
            if pd.isna(a):
                self.assertTrue(pd.isna(b), key)
            else:
                self.assertAlmostEqual(float(a), float(b), places=10, msg=key)

    def test_na_context_and_delayed_never_execution_ready(self):
        out = m.evaluate_exhaustion(history(), "NQ", data_context={"mode": "DELAYED"})
        self.assertTrue(all(c["passed"] is None for c in out["context_checks"]))
        self.assertFalse(out["execution_ready"])

    def test_stale_realtime_blocks_new_long(self):
        d = history()
        out = m.evaluate_exhaustion(d, "ES", data_context={
            "mode": "REALTIME", "as_of": d.index[-1] + pd.Timedelta(minutes=30)
        })
        self.assertFalse(out["execution_ready"])
        self.assertTrue(out["block_new_longs"])

    def test_profiles_and_context_time_validation(self):
        self.assertNotEqual(m.NQ_EXHAUSTION.hourly_extension_atr,
                            m.ES_EXHAUSTION.hourly_extension_atr)
        for age, expected in ((5, True), (20, None), (-1, None)):
            ctx = {"vix_stress": {"value": True,
                                 "timestamp": self.ts - pd.Timedelta(minutes=age)}}
            rows = m._context_checks("ES", ctx, self.ts)
            self.assertIs(next(c for c in rows if c["label"].startswith("VIX"))["passed"],
                          expected)

    def test_invalid_prices_and_duplicates_rejected(self):
        d = history()
        with self.assertRaises(ValueError):
            m.evaluate_exhaustion(pd.concat([d, d.tail(1)]), "GC")
        d.iloc[-10, d.columns.get_loc("low")] = 10000
        with self.assertRaises(ValueError):
            m.evaluate_exhaustion(d, "GC")

    def test_rsi_monotonic_is_100_not_50(self):
        rsi = m._rsi(pd.Series(np.arange(50.0)))
        self.assertEqual(rsi.iloc[-1], 100)

    def test_dst_fall_back_is_unambiguous(self):
        d = history(4321)
        d.index = pd.date_range("2026-10-31 00:00", periods=len(d), freq="min", tz="UTC")
        clean = m._clean(d)
        result = m.evaluate_exhaustion(d, "ES")
        self.assertIn("state", result)
        daily = m._daily_from_minutes(clean, clean.index[-1])
        self.assertTrue(all(t.hour == 17 for t in daily.index))

    def test_real_ohlcv_climax_activates_the_overlay(self):
        result = m.evaluate_exhaustion(history(14161, "climax"), "ES")
        self.assertEqual(result["state"], "EXHAUSTION_RISK")
        self.assertTrue(result["block_new_longs"])
        self.assertFalse(result["short_setup_candidate"])

    def test_disabled_layer_preserves_legacy_mode(self):
        out = m.evaluate_exhaustion(history(), "ES",
                                   config=replace(self.cfg, enabled=False))
        self.assertEqual(out["state"], "DISABLED")
        self.assertFalse(out["block_new_longs"])

    def test_optional_previous_state_matches_replay(self):
        d = history()
        early = m.evaluate_exhaustion(d.iloc[:-15], "ES")
        full = m.evaluate_exhaustion(d, "ES")
        incremental = m.evaluate_exhaustion(d, "ES", previous_state=early["memory"])
        self.assertEqual(full["state"], incremental["state"])
        self.assertEqual(full["block_new_longs"], incremental["block_new_longs"])
        with self.assertRaises(ValueError):
            m.evaluate_exhaustion(d, "NQ", previous_state=early["memory"])

    def test_gc_es_nq_reports_and_long_gates(self):
        d = history()
        symbols = [
            ("GC", gc.build_gold_snapshot(d), gc.GoldRegime.BULL_TREND,
             gc.calculate_gold_risk_plan),
            ("ES", idx.build_index_snapshot(d, idx.ES_CONFIG),
             idx.IndexRegime.BULL_TREND, idx.calculate_index_risk_plan),
            ("NQ", idx.build_index_snapshot(d, idx.NQ_CONFIG),
             idx.IndexRegime.BULL_TREND, idx.calculate_index_risk_plan),
        ]
        for symbol, snapshot, regime, plan in symbols:
            overlay = snapshot["exhaustion"]
            overlay.update(state="EXHAUSTION_RISK", block_new_longs=True)
            risk = plan(snapshot, regime, 10000)
            self.assertFalse(risk["trade_allowed"], symbol)
            self.assertIn("EXHAUSTION_RISK", risk["reason"])
            self.assertIn(symbol, "\n".join(m.format_exhaustion(overlay)))
        report = idx.run_index_checklist(d, idx.NQ_CONFIG)
        self.assertIn("NQ BLOW-OFF / PARABOLIKUS", report)
        self.assertIn("CRASH / VOLATILITY", report)
        report = gc.run_gold_checklist(d)
        self.assertIn("GC BLOW-OFF / PARABOLIKUS", report)
        self.assertNotIn("LONG RECLAIM SETUP", report)

    def test_qqq_advisory_remains_callable(self):
        d = history()
        es, nq = (idx.build_index_snapshot(d, p) for p in (idx.ES_CONFIG, idx.NQ_CONFIG))
        result = idx.build_qqq_put_advisory(
            es, idx.index_regime_active(es), nq, idx.index_regime_active(nq)
        )
        self.assertIn("entry_allowed", result)

    def test_cross_adapter_and_summary(self):
        d = history()
        contexts = idx.build_index_cross_context(d, d * 2)
        self.assertIn("peer_non_confirmation", contexts["ES"])
        self.assertIn("relative_strength_falling", contexts["NQ"])
        self.assertNotIn("vix_stress", contexts["ES"])
        summary = idx.build_cross_market_risk_summary(
            idx.IndexRegime.BULL_TREND, idx.IndexRegime.BULL_TREND,
            es_exhaustion=dict(state="EXHAUSTION_RISK", block_new_longs=True),
        )
        self.assertIn("KÖZÖS INDEX LONG VÉDELEM", summary)

    def test_future_external_daily_bars_are_ignored(self):
        d = history()
        end = d.index[-1].normalize()
        stamps = pd.date_range(end - pd.Timedelta(days=40), periods=50,
                               freq="D", tz="UTC")
        values = np.linspace(4150, 4220, len(stamps))
        daily = pd.DataFrame(dict(open=values, high=values + 2, low=values - 2,
                                  close=values, volume=100), index=stamps)
        a = m.evaluate_exhaustion(d, "GC", daily_history=daily)
        future = daily.index > end
        daily.loc[future, "high"] += 9999
        b = m.evaluate_exhaustion(d, "GC", daily_history=daily)
        self.assertEqual(a["metrics"]["daily_extension_atr"],
                         b["metrics"]["daily_extension_atr"])

    def test_audit_observation_and_summary_export_new_fields(self):
        d = history()
        s = gc.build_gold_snapshot(d)
        r = gc.gold_regime_active(s)
        plan = gc.calculate_gold_risk_plan(s, r["regime"], 10000)
        row = audit.build_observation_record(s, r, plan, 4000)
        self.assertIn("exhaustion_state", row)
        for h in audit.FORWARD_HORIZONS_MINUTES:
            for kind in ("directional", "long", "short"):
                row[f"forward_{kind}_{h}m_atr"] = 0.0
        summary = audit.build_summary_dataframe(pd.DataFrame([row]),
                                                pd.DataFrame(columns=["event_type"]))
        self.assertTrue(summary.label.str.startswith("EXHAUSTION_").any())


if __name__ == "__main__":
    unittest.main()
