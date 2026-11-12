"""Offline tests with a fake IB client; no connection to TWS or market-data API."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import logging
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

def history(n: int = 6001) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    stamps = pd.date_range("2026-09-01", periods=n, freq="min", tz="UTC")
    price = 4200 + np.cumsum(rng.normal(0, .15, n))
    return pd.DataFrame(
        dict(open=price, high=price + .3, low=price - .3,
             close=price, volume=np.full(n, 40.0)), index=stamps,
    )


class Contract:
    def __init__(self, symbol, exchange, currency):
        self.symbol, self.exchange, self.currency = symbol, exchange, currency
        self.conId = 0
        self.localSymbol = symbol


class FakeIB:
    latest_instance = None

    def __init__(self):
        type(self).latest_instance = self
        self.connected = False
        self.disconnected = False
        self.calls = []
        self.es = history()
        self.nq = self.es.copy()
        self.nq[["open", "high", "low", "close"]] *= 5
        asof = self.es.index[-1]
        self.vix_bars = [
            dict(date=asof - pd.Timedelta(minutes=i), close=20.0)
            for i in range(5, 0, -1)
        ]
        self.vix3m_bars = [
            dict(date=asof - pd.Timedelta(minutes=i), close=25.0)
            for i in range(5, 0, -1)
        ]

    def connect(self, *args, **kwargs):
        self.calls.append(("connect", args, kwargs))
        self.connected = True

    def reqMarketDataType(self, value):
        self.calls.append(("mode", value))

    def qualifyContracts(self, *contracts):
        for i, c in enumerate(contracts, start=1):
            c.conId = i
        return list(contracts)

    def reqTickers(self, *contracts):
        # Intentionally stale close with a fresh RECEIPT time.
        return [
            types.SimpleNamespace(last=np.nan, close=88.0, time=pd.Timestamp.now(tz="UTC")),
            types.SimpleNamespace(last=np.nan, close=99.0, time=pd.Timestamp.now(tz="UTC")),
        ]

    def sleep(self, seconds):
        pass

    def reqHistoricalData(self, contract, **kwargs):
        self.calls.append(("history", contract.symbol, kwargs))
        if contract.symbol == "VIX":
            return self.vix_bars
        if contract.symbol == "VIX3M":
            return self.vix3m_bars
        df = self.es if contract.symbol == "ES" else self.nq
        return df.rename_axis("date").reset_index().to_dict("records")

    def isConnected(self):
        return self.connected

    def disconnect(self):
        self.connected = False
        self.disconnected = True


stub = types.ModuleType("ib_insync")
stub.IB, stub.ContFuture, stub.Index = FakeIB, Contract, Contract
stub.util = types.SimpleNamespace(df=lambda bars: pd.DataFrame(bars))
spec = importlib.util.spec_from_file_location(
    "_es_nq_runner_under_test", Path(__file__).with_name("run_es_nq_checklist.py")
)
runner = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"ib_insync": stub}):
    spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.old_log_level = runner.log.level
        runner.log.setLevel(logging.CRITICAL)
        self.ib = FakeIB()
        self.vix = Contract("VIX", "CBOE", "USD")
        self.vix3m = Contract("VIX3M", "CBOE", "USD")
        self.ib.qualifyContracts(self.vix, self.vix3m)
        self.asof = self.ib.es.index[-1]

    def tearDown(self):
        runner.log.setLevel(self.old_log_level)

    def fetch_context(self):
        return runner.fetch_vix_context(
            self.ib, self.vix, self.vix3m, as_of=self.asof
        )

    def test_timestamped_history_not_stale_snapshot_is_used(self):
        ctx = self.fetch_context()
        self.assertEqual(ctx["vix"], 20)
        self.assertEqual(ctx["vix3m"], 25)
        self.assertEqual(ctx["timestamp"], self.asof)
        self.assertNotEqual(ctx["timestamp"], ctx["received_at"])

    def test_fresh_receipt_time_is_not_source_time(self):
        self.ib.vix_bars = []
        with patch.object(runner, "DISPLAY_UNTIMESTAMPED_VIX_SNAPSHOT", True):
            with patch.object(self.ib, "reqTickers", wraps=self.ib.reqTickers) as snapshots:
                self.assertIsNone(self.fetch_context())
                snapshots.assert_called_once()

    def test_stale_history_is_not_fresh_context(self):
        self.ib.vix_bars = [dict(date=self.asof - pd.Timedelta(minutes=20), close=20)]
        self.assertIsNone(self.fetch_context())

    def test_future_unclosed_value_is_not_used(self):
        self.ib.vix_bars.append(dict(date=self.asof, close=999))
        self.assertEqual(self.fetch_context()["vix"], 20)

    def test_snapshot_failure_can_fall_back_to_history(self):
        with patch.object(runner, "DISPLAY_UNTIMESTAMPED_VIX_SNAPSHOT", True):
            with patch.object(self.ib, "reqTickers", side_effect=RuntimeError("snapshot unavailable")):
                self.assertEqual(self.fetch_context()["vix"], 20)

    def test_missing_history_permissions_fail_soft(self):
        with patch.object(self.ib, "reqHistoricalData", side_effect=RuntimeError("permission")):
            self.assertIsNone(self.fetch_context())

    def test_unqualified_vix_is_optional(self):
        self.vix.conId = 0
        self.assertIsNone(self.fetch_context())

    def test_utc_intraday_request_and_deduplication(self):
        es = Contract("ES", "CME", "USD")
        self.ib.qualifyContracts(es)
        d = runner.fetch_historical_1m_data(self.ib, es, "ES")
        self.assertEqual(str(d.index.tz), "UTC")
        self.assertEqual(d.attrs["contract_id"], es.conId)
        self.assertEqual(self.ib.calls[-1][2]["formatDate"], 2)
        records = d.rename_axis("date").reset_index().to_dict("records")
        records += records[-1:]
        self.assertEqual(len(runner.bars_to_dataframe(records)), len(d))

    def test_same_context_reaches_both_reports_and_summary(self):
        ctx = self.fetch_context()
        bundle = runner.build_analysis_bundle(
            self.ib.es, self.ib.nq, ctx, clock=self.asof + pd.Timedelta(minutes=15)
        )
        for symbol in ("ES", "NQ"):
            snapshot = bundle[symbol]["snapshot"]
            self.assertTrue(snapshot["vix_analysis"]["available"])
            self.assertEqual(snapshot["exhaustion"]["data_mode"], "DELAYED_FROZEN")
            self.assertFalse(snapshot["exhaustion"]["execution_ready"])
            self.assertIn(f"{symbol} BLOW-OFF / PARABOLIKUS", bundle[symbol]["report"])
            checks = snapshot["exhaustion"]["context_checks"]
            vix = next(c for c in checks if c["label"].startswith("VIX"))
            self.assertIs(vix["passed"], False)  # valid, non-stress 20/25 ratio
            relative = next(c for c in checks if "NQ/ES" in c["label"])
            self.assertIsNotNone(relative["passed"])
        self.assertIn("entry_allowed", bundle["qqq_advisory"])

    def test_async_indices_disable_peer_context(self):
        older_nq = self.ib.nq.iloc[:-12]
        bundle = runner.build_analysis_bundle(self.ib.es, older_nq, None, clock=self.asof)
        self.assertTrue(bundle["warnings"])
        for symbol in ("ES", "NQ"):
            self.assertNotIn("peer_non_confirmation", bundle["cross_context"][symbol])
            self.assertNotIn("relative_strength_falling", bundle["cross_context"][symbol])

    def test_daily_file_is_optional_and_not_downloaded(self):
        with patch.dict(runner.DAILY_HISTORY_CSV, {"ES": None}):
            self.assertIsNone(runner.load_daily_history("ES"))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "daily.csv"
            d = self.ib.es.iloc[::1440].rename_axis("date").reset_index()
            d.to_csv(path, index=False)
            with patch.dict(runner.DAILY_HISTORY_CSV, {"ES": str(path)}):
                frame = runner.load_daily_history("ES")
                self.assertEqual(str(frame.index.tz), "UTC")

    def test_main_fake_tws_end_to_end_and_disconnect(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            runner.main()
        client = FakeIB.latest_instance
        self.assertTrue(client.disconnected)
        text = output.getvalue()
        self.assertIn("ES BLOW-OFF / PARABOLIKUS", text)
        self.assertIn("NQ BLOW-OFF / PARABOLIKUS", text)
        self.assertIn("QQQ LONG PUT", text)
        self.assertIn("nem valós idejű végrehajtási engedély", text)
        self.assertEqual(client.calls[0][2]["clientId"], 30)

    def test_real_failure_raises_and_disconnects(self):
        with patch.object(FakeIB, "qualifyContracts", return_value=[]):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    runner.main()
        self.assertTrue(FakeIB.latest_instance.disconnected)


if __name__ == "__main__":
    unittest.main()
