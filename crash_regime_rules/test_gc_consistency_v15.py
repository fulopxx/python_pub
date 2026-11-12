"""Offline tests for the user's current filenames, closed bars and daily data."""
from __future__ import annotations

import importlib
import sys
import types
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

import gc_regime_1_0_17 as gc
import gc_audit_backtest_v1_3_9 as audit
import gc_daily_context as daily
import market_exhaustion as exhaustion
import market_transition as transition


def minute_history(n=1001):
    idx = pd.date_range("2026-10-11 18:00", periods=n, freq="min",
                        tz="America/New_York")
    p = 4200 + np.arange(n)*.01
    return pd.DataFrame(dict(open=p-.01, high=p+.1, low=p-.1,
                             close=p, volume=100.), index=idx)


def daily_raw():
    days = pd.bdate_range(end="2026-10-09", periods=40)
    p = 4190 + np.arange(40)*.2
    return pd.DataFrame(dict(date=days.date, open=p-.1, high=p+3, low=p-3,
                             close=p, volume=10000.))


def ib_stub():
    stub = types.ModuleType("ib_insync")
    stub.IB = type("IB", (), {})
    class Contract:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    stub.Contract = stub.ContFuture = Contract
    stub.util = types.SimpleNamespace(df=lambda bars: pd.DataFrame(bars))
    return stub


class ConsistencyTests(unittest.TestCase):
    def test_default_excludes_forming_minute(self):
        x = minute_history()
        s = gc.build_gold_snapshot(x)
        self.assertEqual(s["close"], x.close.iloc[-2])
        self.assertEqual(s["timestamp"], x.index[-1])

    def test_incomplete_five_minute_omitted(self):
        s = gc.build_gold_snapshot(minute_history(1003))
        self.assertEqual(s["last_closed_5m_at"].minute % 5, 0)
        self.assertLess(s["last_closed_5m_at"], s["timestamp"])

    def test_whole_bar_metadata_includes_last_minute(self):
        x = minute_history()
        s = gc.build_gold_snapshot(x, data_context={"last_bar_is_complete": True})
        self.assertEqual(s["close"], x.close.iloc[-1])
        self.assertEqual(s["timestamp"], x.index[-1]+pd.Timedelta(minutes=1))

    def test_closed_boundaries_and_extension_match(self):
        s = gc.build_gold_snapshot(minute_history())
        self.assertEqual(s["last_closed_5m_at"], s["exhaustion"]["asof"])
        self.assertEqual(s["timestamp"], s["transition_risk"]["asof"])
        self.assertAlmostEqual(s["vwap_distance_atr"],
                               s["exhaustion"]["metrics"]["vwap_extension_atr"])

    def test_last_minute_mutation_cannot_change_snapshot(self):
        x = minute_history()
        a = gc.build_gold_snapshot(x)
        x.iloc[-1] = [4300, 4400, 4000, 4300, 100000]
        b = gc.build_gold_snapshot(x)
        for key in ("close", "vwap", "atr_1m", "atr_5m", "adx_5m", "rsi_2"):
            self.assertEqual(a[key], b[key])

    def test_naive_utc_input_supported(self):
        x = minute_history()
        a = gc.build_gold_snapshot(x)
        x.index = x.index.tz_convert("UTC").tz_localize(None)
        b = gc.build_gold_snapshot(x)
        self.assertEqual(a["close"], b["close"])

    def test_partial_five_minute_no_false_complete(self):
        x = minute_history(101)
        cutoff = x.index[-1]
        bars = gc.resample_ohlcv(x.loc[x.index < cutoff], as_of=cutoff)
        self.assertTrue(((bars.index+pd.Timedelta(minutes=5)) <= cutoff).all())

    def test_daily_trading_date_uses_availability_not_midnight(self):
        df = daily.daily_bars_to_dataframe(daily_raw(), 123, pd.Timestamp("2026-10-12", tz="UTC"))
        self.assertEqual(df.index[-1].tz_convert("America/New_York").hour, 5)
        self.assertEqual(df.index[-1].tz_convert("America/New_York").date(),
                         pd.Timestamp("2026-10-10").date())

    def test_current_daily_bar_never_leaks(self):
        raw = daily_raw()
        raw.loc[len(raw)] = [pd.Timestamp("2026-10-12").date(), 4200, 4203, 4197, 4200, 10000]
        cutoff = pd.Timestamp("2026-10-12 08:20", tz="America/New_York")
        df = daily.daily_bars_to_dataframe(raw, 123, cutoff)
        self.assertTrue((df.index <= cutoff).all())
        self.assertEqual(len(df), 40)

    def test_compact_ib_daily_dates_supported(self):
        raw = daily_raw()
        raw["date"] = [pd.Timestamp(x).strftime("%Y%m%d") for x in raw.date]
        self.assertEqual(len(daily.daily_bars_to_dataframe(raw, 123,
                            pd.Timestamp("2026-10-12", tz="UTC"))), 40)

    def test_daily_extension_available_with_separate_daily_data(self):
        x = minute_history()
        df = daily.daily_bars_to_dataframe(daily_raw(), 123, x.index[-1])
        s = gc.build_gold_snapshot(x, daily_history=df, data_context={"contract_id": 123})
        self.assertTrue(s["exhaustion"]["daily_extension_available"])
        self.assertTrue(np.isfinite(s["exhaustion"]["metrics"]["daily_extension_atr"]))

    def test_wrong_contract_daily_rejected(self):
        x = minute_history()
        df = daily.daily_bars_to_dataframe(daily_raw(), 124, x.index[-1])
        with self.assertRaises(ValueError):
            gc.build_gold_snapshot(x, daily_history=df, data_context={"contract_id": 123})

    def test_stale_daily_stays_na(self):
        x = minute_history()
        df = daily.daily_bars_to_dataframe(daily_raw(), 123, x.index[-1])
        df.index -= pd.Timedelta(days=14)
        s = gc.build_gold_snapshot(x, daily_history=df)
        self.assertFalse(s["exhaustion"]["daily_extension_available"])

    def test_failed_daily_feed_returns_none_not_zero(self):
        class IB:
            def reqHistoricalData(self, *args, **kwargs):
                raise RuntimeError("feed unavailable")
        with TemporaryDirectory() as folder, patch.dict(sys.modules, {"ib_insync": ib_stub()}):
            df, status = daily.fetch_gc_daily_context(
                IB(), types.SimpleNamespace(conId=123, exchange="COMEX"),
                minute_history().index[-1], cache_dir=folder)
        self.assertIsNone(df)
        self.assertEqual(status, "UNAVAILABLE_OR_INSUFFICIENT")

    def test_daily_cache_skips_second_request(self):
        class IB:
            calls = 0
            def reqHistoricalData(self, contract, **kwargs):
                self.calls += 1
                self.contract = contract
                self.options = kwargs
                return daily_raw().to_dict("records")
        ib = IB()
        with TemporaryDirectory() as folder, patch.dict(sys.modules, {"ib_insync": ib_stub()}):
            contract = types.SimpleNamespace(conId=123, exchange="COMEX")
            a, status_a = daily.fetch_gc_daily_context(ib, contract,
                            minute_history().index[-1], cache_dir=folder)
            b, status_b = daily.fetch_gc_daily_context(ib, contract,
                            minute_history().index[-1], cache_dir=folder)
        self.assertEqual(ib.calls, 1)
        self.assertEqual(status_b, "CACHE")
        self.assertEqual(ib.contract.conId, 123)
        self.assertEqual(ib.options["barSizeSetting"], "1 day")
        self.assertEqual(len(a), len(b))

    def test_slope_independent_of_timestamp_resolution(self):
        if not hasattr(pd.date_range("2026-10-11", periods=2, freq="h"), "as_unit"):
            self.skipTest("A pandas ezen verziója nem támogat több index-felbontást.")
        for unit in ("ns", "us"):
            idx = pd.date_range("2026-10-11", periods=12, freq="h", tz="UTC").as_unit(unit)
            slope, fit = exhaustion._slope(pd.Series(np.arange(12)*2., index=idx), 2.)
            self.assertAlmostEqual(slope, 1.)
            self.assertAlmostEqual(fit, 1.)

    def test_risk_gate_no_new_veto_does_not_override_neutral(self):
        s = gc.build_gold_snapshot(minute_history())
        p = gc.calculate_gold_risk_plan(s, gc.GoldRegime.NEUTRAL, 10000.)
        self.assertFalse(p["trade_allowed"])

    def test_current_version_banner(self):
        text = gc.run_gold_checklist(minute_history())
        self.assertIn("ENGINE VERSION 1.0.15", text)
        self.assertIn("Utolsó lezárt 1m", text)

    def test_renamed_audit_import(self):
        self.assertIs(audit.build_gold_snapshot, gc.build_gold_snapshot)

    def test_macro_missing_checks_count(self):
        m = gc.analyze_gold_macro_context(dict(
            dxy_price=102., dxy_vwap=102.1, dxy_change_15m_pct=-.0049,
            dxy_change_1h_pct=-.0049, dxy_change_4h_pct=-.0245, dxy_change_1d_pct=-.093))
        self.assertEqual(m["available_check_count"], 5)
        self.assertEqual(m["bull_score"], 1)
        self.assertEqual(m["state"], "NO STRONG MACRO CONFIRMATION")

    def test_expired_cross_is_explicitly_historical(self):
        from test_market_transition import timeframe_fixture
        args = list(timeframe_fixture())
        args[-1] += pd.Timedelta(hours=2)
        later = args[0].iloc[-1:].copy()
        later.index += pd.Timedelta(hours=2)
        args[0] = pd.concat([args[0], later])
        idx = pd.date_range(args[4].index[0], args[-1]-pd.Timedelta(minutes=1), freq="min")
        args[4] = args[4].reindex(idx).ffill()
        args[2] = pd.Series(100., index=idx)
        r = transition._timeframe(*args, 60, transition.GC_TRANSITION)
        self.assertEqual(r["state"], "TURN_EXPIRED")
        self.assertFalse(r["active_turn"])
        self.assertLess(r["evaluated_until"], args[-1])
        self.assertFalse(r["observation_window_complete"])
        self.assertTrue(any("értékelt" in c["label"].lower() for c in r["checks"]))

    def test_renamed_runner_adapter_import(self):
        with patch.dict(sys.modules, {"ib_insync": ib_stub()}):
            runner = importlib.import_module("run_gc_checklist_v1_5")
            import gc_regime_1_0_16 as current_gc
        self.assertIs(runner.run_gold_checklist, current_gc.run_gold_checklist)
        self.assertTrue(runner.FETCH_GC_DAILY_CONTEXT)


if __name__ == "__main__":
    unittest.main()
