"""Offline DXY MTF tests: python -m unittest test_gc_dxy_mtf -v."""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

import gc_regime_1_0_14 as gc
import gc_audit_backtest as audit


def dx(n=1800):
    index = pd.date_range("2026-10-11 18:00", periods=n, freq="min",
                          tz="America/New_York")
    close = 103. - np.arange(n) * .0005
    return pd.DataFrame(dict(open=close, high=close+.005, low=close-.005,
                             close=close, volume=100.), index=index)


def context(**updates):
    out = dict(dxy_price=102., dxy_vwap=102.1,
               dxy_change_15m_pct=-.06, dxy_change_1h_pct=-.1,
               dxy_change_4h_pct=-.2, dxy_change_1d_pct=-.4)
    out.update(updates)
    return out


class DXYMTFTests(unittest.TestCase):
    def test_four_exact_elapsed_returns(self):
        x = dx()
        r = gc.build_dxy_macro_context(x)
        for name, minutes in (("15m", 15), ("1h", 60), ("4h", 240), ("1d", 1440)):
            expected = 100*(x.close.iloc[-2]/x.close.iloc[-2-minutes]-1)
            self.assertAlmostEqual(r[f"dxy_change_{name}_pct"], expected)
            self.assertEqual(r["dxy_horizons"][name]["status"], "OK")

    def test_not_row_based_when_minute_missing(self):
        x = dx()
        target = x.index[-2]-pd.Timedelta(hours=1)
        x = x.drop(x.index[(x.index > target) & (x.index < target+pd.Timedelta(minutes=10))])
        r = gc.build_dxy_macro_context(x)
        expected = 100*(x.close.iloc[-2]/x.loc[target, "close"]-1)
        self.assertAlmostEqual(r["dxy_change_1h_pct"], expected)

    def test_reference_gap_is_na(self):
        x = dx()
        target = x.index[-2]-pd.Timedelta(hours=1)
        x = x.drop(x.index[(x.index >= target-pd.Timedelta(minutes=8)) &
                          (x.index <= target)])
        r = gc.build_dxy_macro_context(x)
        self.assertTrue(np.isnan(r["dxy_change_1h_pct"]))
        self.assertEqual(r["dxy_horizons"]["1h"]["status"], "REFERENCE_GAP")

    def test_short_history_daily_na(self):
        r = gc.build_dxy_macro_context(dx(600))
        self.assertTrue(np.isfinite(r["dxy_change_4h_pct"]))
        self.assertTrue(np.isnan(r["dxy_change_1d_pct"]))

    def test_weekend_never_substituted_with_friday(self):
        x = dx(600)
        friday = x.copy()
        friday.index -= pd.Timedelta(days=3)
        r = gc.build_dxy_macro_context(pd.concat([friday, x]))
        self.assertTrue(np.isnan(r["dxy_change_1d_pct"]))
        self.assertEqual(r["dxy_horizons"]["1d"]["status"], "REFERENCE_GAP")

    def test_final_minute_excluded(self):
        x = dx()
        before = gc.build_dxy_macro_context(x)
        x.iloc[-1] = [105, 106, 100, 105, 100000]
        after = gc.build_dxy_macro_context(x)
        self.assertEqual(before["dxy_price"], after["dxy_price"])
        self.assertEqual(before["dxy_timestamp"], after["dxy_timestamp"])

    def test_future_bars_excluded_at_cutoff(self):
        x = dx()
        cutoff = x.index[-101]
        a = gc.build_dxy_macro_context(x, as_of=cutoff)
        b = gc.build_dxy_macro_context(x.loc[x.index < cutoff], as_of=cutoff)
        self.assertEqual(a["dxy_price"], b["dxy_price"])
        self.assertLessEqual(a["dxy_timestamp"], cutoff)

    def test_stale_source_not_carried_forward(self):
        x = dx()
        r = gc.build_dxy_macro_context(x, as_of=x.index[-1]+pd.Timedelta(minutes=30))
        self.assertTrue(np.isnan(r["dxy_price"]))
        self.assertEqual(r["dxy_horizons"]["4h"]["status"], "STALE_CURRENT_PRICE")

    def test_partial_session_vwap_na(self):
        r = gc.build_dxy_macro_context(dx(600).iloc[5:])
        self.assertTrue(np.isnan(r["dxy_vwap"]))
        self.assertEqual(r["dxy_vwap_status"], "PARTIAL_SESSION")

    def test_sunday_session_vwap(self):
        x = dx(600)
        r = gc.build_dxy_macro_context(x)
        self.assertAlmostEqual(r["dxy_vwap"], x.close.iloc[:-1].mean())

    def test_dst_elapsed_change(self):
        x = dx(600)
        x.index = pd.date_range("2026-10-31 23:00", periods=600, freq="min",
                                tz="America/New_York")
        r = gc.build_dxy_macro_context(x)
        a = r["dxy_timestamp"]
        b = r["dxy_horizons"]["4h"]["reference_timestamp"]
        self.assertEqual(a-b, pd.Timedelta(hours=4))

    def test_naive_utc_matches_aware(self):
        x = dx()
        a = gc.build_dxy_macro_context(x)
        x.index = x.index.tz_convert("UTC").tz_localize(None)
        b = gc.build_dxy_macro_context(x)
        self.assertEqual(a["dxy_price"], b["dxy_price"])

    def test_duplicate_timestamps_rejected(self):
        x = dx()
        with self.assertRaises(ValueError):
            gc.build_dxy_macro_context(pd.concat([x, x.iloc[-1:]]))

    def test_invalid_rows_dropped(self):
        x = dx()
        x.iloc[20, x.columns.get_loc("close")] = -1
        self.assertTrue(np.isfinite(gc.build_dxy_macro_context(x)["dxy_change_4h_pct"]))

    def test_empty_data_na(self):
        r = gc.build_dxy_macro_context(pd.DataFrame())
        self.assertTrue(np.isnan(r["dxy_change_1d_pct"]))

    def test_longer_weakness_survives_15m_bounce(self):
        r = gc.analyze_gold_macro_context(context(dxy_change_15m_pct=.1))
        self.assertTrue(r["strong_bullish"])
        self.assertFalse(r["strong_bearish"])
        self.assertEqual(r["dxy_short_term_state"], "SHORT_TERM_BOUNCE_IN_DXY_DOWNTREND")

    def test_1h_bounce_does_not_override_4h_daily(self):
        r = gc.analyze_gold_macro_context(context(dxy_change_1h_pct=.2))
        self.assertTrue(r["strong_bullish"])

    def test_symmetric_strength(self):
        r = gc.analyze_gold_macro_context(context(
            dxy_change_1h_pct=.1, dxy_change_4h_pct=.2, dxy_change_1d_pct=.4))
        self.assertTrue(r["strong_bearish"])
        self.assertEqual(r["dxy_short_term_state"], "SHORT_TERM_DIP_IN_DXY_UPTREND")

    def test_mixed_4h_daily_no_strong_confirmation(self):
        r = gc.analyze_gold_macro_context(context(dxy_change_1d_pct=.4))
        self.assertFalse(r["strong_bullish"])
        self.assertFalse(r["strong_bearish"])

    def test_unavailable_daily_explicit_fallback(self):
        r = gc.analyze_gold_macro_context(context(dxy_change_1d_pct=np.nan))
        self.assertTrue(r["strong_bullish"])
        self.assertEqual(r["confirmation_basis"], "1H_4H_1D_UNAVAILABLE")

    def test_known_flat_daily_not_missing(self):
        r = gc.analyze_gold_macro_context(context(dxy_change_1d_pct=0))
        self.assertFalse(r["strong_bullish"])

    def test_opposing_yield_vetoes_strong_macro(self):
        r = gc.analyze_gold_macro_context(context(real_yield_change_15m_bps=2.))
        self.assertFalse(r["strong_bullish"])

    def test_legacy_15m_context_still_works(self):
        r = gc.analyze_gold_macro_context(
            dict(dxy_price=102., dxy_vwap=102.1, dxy_change_15m_pct=-.1))
        self.assertTrue(r["strong_bullish"])
        self.assertEqual(r["confirmation_basis"], "LEGACY_15M_VWAP_YIELD")

    def test_no_data_no_confirmation(self):
        r = gc.analyze_gold_macro_context(None)
        self.assertFalse(r["strong_bullish"])
        self.assertIn("Nincs", "\n".join(gc.format_macro(r)))

    def test_future_and_stale_external_macro_invalidated(self):
        cutoff = pd.Timestamp("2026-10-12 10:00", tz="UTC")
        for delta in (pd.Timedelta(minutes=1), -pd.Timedelta(minutes=20)):
            r = gc.analyze_gold_macro_context(
                context(dxy_timestamp=cutoff+delta), as_of=cutoff)
            self.assertFalse(r["strong_bullish"])
            self.assertTrue(np.isnan(r["dxy_change_4h_pct"]))

    def test_none_string_external_values_safe(self):
        r = gc.analyze_gold_macro_context(
            context(dxy_change_4h_pct=None, dxy_change_1d_pct="N/A"))
        self.assertFalse(r["strong_bullish"])

    def test_report_separates_na_and_no(self):
        r = gc.analyze_gold_macro_context(context(dxy_change_1d_pct=np.nan))
        text = "\n".join(gc.format_macro(r))
        self.assertIn("1D (24h)", text)
        self.assertIn("[N/A] DXY 1D (24h) gyengül", text)
        self.assertIn("/6", text)

    def test_audit_exports_mtf_without_fabricating_missing_data(self):
        r = audit.macro_audit_fields({"macro_analysis": gc.analyze_gold_macro_context(context())})
        self.assertEqual(r["dxy_change_4h_pct"], -.2)
        absent = audit.macro_audit_fields({})
        self.assertTrue(np.isnan(absent["dxy_change_4h_pct"]))

    def test_risk_plan_uses_longer_confirmation(self):
        snapshot = {
            "close": 4200., "atr_5m": 3., "rsi_2": 50., "rsi_5": 50.,
            "vwap_distance_atr": 0.,
            "gap_context": {"ndog_reaction": {"reaction": "NO_RECENT_INTERACTION"},
                            "nwog_reaction": {"reaction": "NO_RECENT_INTERACTION"}},
            "macro_analysis": gc.analyze_gold_macro_context(context(dxy_change_15m_pct=.1)),
        }
        plan = gc.calculate_gold_risk_plan(snapshot, gc.GoldRegime.BEAR_TREND, 10000.)
        self.assertFalse(plan["trade_allowed"])
        self.assertIn("macro context", plan["details"])


if __name__ == "__main__":
    unittest.main()
