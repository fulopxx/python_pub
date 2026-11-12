"""Offline causal regression tests: python -m unittest test_market_transition -v."""
from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd

import market_transition as m
import gc_regime_1_0_19 as gc
import gc_audit_backtest_v1_3_12 as audit


def history(n=1000, whipsaw=False):
    stamps = pd.date_range("2026-10-11 18:00", periods=n, freq="min",
                           tz="America/New_York")
    close = 4200 + np.arange(n) * .01
    out = pd.DataFrame(dict(open=close-.01, high=close+.1, low=close-.1,
                            close=close, volume=100.), index=stamps)
    if whipsaw:
        # Discarded last minute remains untouched. Thirty known closed bars
        # alternate direction, have two long wicks and abruptly expanded range.
        k = 30
        mid = close[-k-2]
        values = mid + np.where(np.arange(k) % 2, .3, -.3)
        out.iloc[-k-1:-1] = np.column_stack((
            np.full(k, mid), np.full(k, mid+2), np.full(k, mid-2),
            values, np.full(k, 100.),
        ))
    return out


def timeframe_fixture(minutes=60, direction=1, retest=True, failed=False):
    start = pd.Timestamp("2026-10-11 18:00", tz=m.GC_TRANSITION.timezone)
    cross = start + pd.Timedelta(minutes=2*minutes)
    ends = pd.DatetimeIndex([start + pd.Timedelta(minutes=minutes), cross])
    close = [100-direction, 100+direction]
    bars = pd.DataFrame(dict(open=close, close=close,
                             low=np.array(close)-.2, high=np.array(close)+.2,
                             volume=100, session=start.date()), index=ends)
    five_ends = pd.date_range(start+pd.Timedelta(minutes=5),
                              cross+pd.Timedelta(minutes=5 if retest else 0),
                              freq="5min")
    five = pd.DataFrame(dict(open=101., high=101.2, low=100.8,
                              close=101., volume=100), index=five_ends)
    if retest:
        five.iloc[-1] = [100., 100.7, 99.3, 100+.5*direction, 100]
    if failed:
        five.iloc[-1] = [100., 101.2, 98.8, 100-direction, 100]
    cutoff = five.index[-1]
    idx = pd.date_range(start, cutoff-pd.Timedelta(minutes=1), freq="min")
    current = pd.DataFrame(dict(open=101., high=101.2, low=100.8,
                                 close=101., volume=100), index=idx)
    vwap = pd.Series(100., index=idx)
    atr = pd.Series(1., index=five.index)
    return bars, five, vwap, atr, current, cutoff


def tf(minutes=60, direction=1, retest=True, failed=False):
    return m._timeframe(*timeframe_fixture(minutes, direction, retest, failed),
                        minutes, m.GC_TRANSITION)


class TransitionTests(unittest.TestCase):
    def test_bullish_retest_each_timeframe(self):
        for minutes in (60, 120, 240):
            with self.subTest(minutes=minutes):
                result = tf(minutes)
                self.assertEqual(result["state"], "BULLISH_RETEST_CONFIRMED")
                self.assertEqual(result["confirmed_direction"], 1)
                self.assertGreater(result["retest_at"], result["cross_at"])

    def test_bearish_retest(self):
        self.assertEqual(tf(direction=-1)["state"], "BEARISH_RETEST_CONFIRMED")

    def test_cross_bar_cannot_retest_itself(self):
        result = tf(retest=False)
        self.assertEqual(result["state"], "BULLISH_CROSS_PENDING")
        self.assertIsNone(result["retest_at"])
        self.assertEqual(result["confirmed_direction"], 0)

    def test_failed_reclaim(self):
        result = tf(failed=True)
        self.assertEqual(result["state"], "FAILED_BULLISH_RECLAIM")
        self.assertEqual(result["confirmed_direction"], 0)

    def test_turn_expires(self):
        args = list(timeframe_fixture())
        args[-1] += pd.Timedelta(hours=2)
        later = args[0].iloc[-1:].copy()
        later.index += pd.Timedelta(hours=2)
        args[0] = pd.concat([args[0], later])
        idx = pd.date_range(args[4].index[0], args[-1]-pd.Timedelta(minutes=1),
                            freq="min")
        args[4] = args[4].reindex(idx).ffill()
        args[2] = pd.Series(100., index=idx)
        result = m._timeframe(*args, 60, m.GC_TRANSITION)
        self.assertEqual(result["state"], "TURN_EXPIRED")

    def test_forming_reclaim_not_confirmation(self):
        args = list(timeframe_fixture(retest=False))
        args[0]["close"] = 99.
        args[0]["open"] = 99.
        args[0]["high"] = 99.2
        args[0]["low"] = 98.8
        extra = args[4].iloc[-1:].copy()
        extra.index += pd.Timedelta(minutes=1)
        args[4] = pd.concat([args[4], extra])
        args[2] = args[2].reindex(args[4].index, fill_value=100.)
        args[5] += pd.Timedelta(minutes=1)
        result = m._timeframe(*args, 60, m.GC_TRANSITION)
        self.assertEqual(result["provisional_direction"], 1)
        self.assertEqual(result["confirmed_direction"], 0)

    def test_session_reset_is_not_cross(self):
        args = list(timeframe_fixture())
        args[0].iloc[0, args[0].columns.get_loc("session")] -= pd.Timedelta(days=1)
        result = m._timeframe(*args, 60, m.GC_TRANSITION)
        self.assertIsNone(result["cross_at"])
        self.assertEqual(result["state"], "UNKNOWN")

    def test_high_whipsaw_blocks_both(self):
        result = m.evaluate_transition_risk(history(whipsaw=True))
        self.assertEqual(result["whipsaw"]["state"], "HIGH")
        self.assertTrue(result["block_new_longs"])
        self.assertTrue(result["block_new_shorts"])

    def test_smooth_trend_low_whipsaw(self):
        result = m.evaluate_transition_risk(history())
        self.assertEqual(result["whipsaw"]["state"], "LOW")

    def test_latest_minute_is_excluded(self):
        x = history()
        before = m.evaluate_transition_risk(x)
        x.iloc[-1] = [4400, 4500, 4100, 4200, 10000]
        after = m.evaluate_transition_risk(x)
        self.assertEqual(before["whipsaw"]["metrics"], after["whipsaw"]["metrics"])
        self.assertEqual(before["asof"], after["asof"])

    def test_complete_metadata_advances_cutoff(self):
        x = history()
        a = m.evaluate_transition_risk(x)
        b = m.evaluate_transition_risk(x, data_context={"last_bar_is_complete": True})
        self.assertEqual(b["asof"]-a["asof"], pd.Timedelta(minutes=1))

    def test_missing_recent_minutes_unknown(self):
        x = history().drop(history().index[-12:-7])
        result = m.evaluate_transition_risk(x)
        self.assertEqual(result["whipsaw"]["state"], "UNKNOWN")
        self.assertTrue(result["block_new_shorts"])

    def test_truncated_session_vwap_unknown(self):
        result = m.evaluate_transition_risk(history().iloc[30:])
        self.assertEqual(result["whipsaw"]["state"], "UNKNOWN")

    def test_zero_volume_unknown(self):
        x = history()
        x.volume = 0
        self.assertEqual(m.evaluate_transition_risk(x)["whipsaw"]["state"], "UNKNOWN")

    def test_duplicates_rejected(self):
        x = history()
        with self.assertRaises(ValueError):
            m.evaluate_transition_risk(pd.concat([x, x.iloc[-1:]]))

    def test_partial_htf_is_not_closed(self):
        x = history(121)
        r = m.evaluate_transition_risk(x)
        self.assertEqual(r["timeframes"]["4h"]["state"], "UNKNOWN")
        self.assertIsNone(r["timeframes"]["4h"]["last_closed_at"])

    def test_missing_htf_minute_rejects_bucket(self):
        x = history(241)
        x = x.drop(x.index[100])
        b = m._bars(x, 240, x.index[-1], m.GC_TRANSITION)
        self.assertTrue(b.empty)

    def test_shortened_last_session_bucket(self):
        x = history(1380)
        cutoff = x.index[-1]+pd.Timedelta(minutes=1)
        b = m._bars(x, 240, cutoff, m.GC_TRANSITION)
        self.assertEqual(b.scheduled_minutes.iloc[-1], 180)
        self.assertEqual(b.index[-1].hour, 17)

    def test_dst_calendar_bounds(self):
        for day, hours in (("2026-03-07", 22), ("2026-10-31", 24)):
            a, b = m._session_bounds(pd.Timestamp(day).date(), m.GC_TRANSITION)
            self.assertEqual((b-a).total_seconds()/3600, hours)
            self.assertEqual(a.hour, 18)
            self.assertEqual(b.hour, 17)

    def test_pending_turn_gates_both(self):
        with patch.object(m, "_timeframe", return_value=tf(retest=False)):
            result = m.evaluate_transition_risk(history())
        self.assertTrue(result["block_new_longs"])
        self.assertTrue(result["block_new_shorts"])

    def test_confirmed_bull_blocks_short_not_long(self):
        with patch.object(m, "_timeframe", return_value=tf()):
            result = m.evaluate_transition_risk(history())
        self.assertFalse(result["block_new_longs"])
        self.assertTrue(result["block_new_shorts"])

    def test_confirmed_conflict_blocks_both(self):
        with patch.object(m, "_timeframe", side_effect=[tf(), tf(direction=-1), tf()]):
            result = m.evaluate_transition_risk(history())
        self.assertTrue(result["confirmation_conflict"])
        self.assertTrue(result["block_new_longs"])
        self.assertTrue(result["block_new_shorts"])

    def test_disabled_overlay(self):
        r = m.evaluate_transition_risk(pd.DataFrame(),
                                       config=replace(m.GC_TRANSITION, enabled=False))
        self.assertFalse(r["block_new_shorts"])

    def test_gc_integration_and_risk_gate(self):
        snapshot = gc.build_gold_snapshot(history(whipsaw=True))
        for direction in (gc.GoldRegime.BULL_TREND, gc.GoldRegime.BEAR_TREND):
            risk = gc.calculate_gold_risk_plan(snapshot, direction, 10000.)
            self.assertFalse(risk["trade_allowed"])
        report = gc.run_gold_checklist(history())
        for label in ("1H FORDULÓ CHECKLIST", "2H FORDULÓ CHECKLIST",
                      "4H FORDULÓ CHECKLIST", "1m WHIPSAW-KOCKÁZAT"):
            self.assertIn(label, report)
        fields = audit.transition_audit_fields(snapshot)
        self.assertEqual(fields["whipsaw_state"], "HIGH")
        self.assertIn("htf_4h_state", fields)

    def test_audit_summary_has_whipsaw_and_transition_groups(self):
        x = history(whipsaw=True)
        snapshot = gc.build_gold_snapshot(x)
        rows = []
        for regime in (gc.GoldRegime.BULL_TREND, gc.GoldRegime.BEAR_TREND):
            plan = gc.calculate_gold_risk_plan(snapshot, regime, 10000.)
            record = audit.build_observation_record(snapshot, {"regime": regime},
                                                     plan, len(x)-100)
            audit.record_forward_metrics(record, x, len(x)-100, snapshot["atr_5m"])
            rows.append(record)
        observations = pd.DataFrame(rows)
        events = observations.assign(event_type="BLOCKED_TRADE")
        summary = audit.build_summary_dataframe(observations, events)
        self.assertIn("WHIPSAW_HIGH_30M_DIRECTIONAL", summary.to_string())
        self.assertIn("TRANSITION_BLOCKED_BEAR_30M", summary.to_string())


if __name__ == "__main__":
    unittest.main()
