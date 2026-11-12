"""Causal candidate-layer regression; synthetic data, no user CSV dependency."""
from __future__ import annotations

import json
import unittest
from dataclasses import replace

import pandas as pd

import market_structure as m
from gc_breaker_monitor import summarize_gc_breakers
from market_transition import _session_bounds


CFG = m.DEFAULT_STRUCTURE


def fixture():
    idx = m._candidate_ends(pd.Timestamp("2026-10-05 22:00", tz=CFG.timezone),
                            pd.Timestamp("2026-10-13 14:00", tz=CFG.timezone), 240, CFG)
    b = pd.DataFrame([[103, 104, 102, 103, 100]]*len(idx), index=idx,
                     columns=["open", "high", "low", "close", "volume"], dtype=float)
    starts, durations = [], []
    for stamp in idx:
        day = (stamp.tz_localize(None)-pd.Timedelta(hours=18)).date()
        opened, closed = _session_bounds(day, CFG)
        full = stamp != closed
        starts.append(stamp-pd.Timedelta(hours=4) if full else opened+pd.Timedelta(hours=20))
        durations.append(240 if full else int((closed-starts[-1])/pd.Timedelta(minutes=1)))
    b["start"], b["scheduled_minutes"] = starts, durations
    target = idx.get_loc(pd.Timestamp("2026-10-12 17:00", tz=CFG.timezone))
    rows = [
        [102, 104, 98, 100, 100],
        [100, 102, 99, 101, 100],
        [100.5, 102, 100, 101.5, 100],  # short session-close origin
        [101.5, 103, 95, 100.2, 100],  # local low takeout
        [100.2, 108, 99, 107, 100],    # separate displaced reclaim
        [107, 108, 104, 105, 100],
        [105, 106, 102, 103, 100],
        [103, 103.5, 95, 96, 100],    # still later displaced failure
    ]
    b.loc[idx[target-2:target+6], ["open", "high", "low", "close", "volume"]] = rows
    return b


def target_zone(b=None, cfg=CFG):
    zones = m.detect_breaker_candidates(fixture() if b is None else b, 240, cfg)
    return next(z for z in zones if z["low"] == 100 and z["high"] == 102)


class CandidateTests(unittest.TestCase):
    def test_short_origin_explicit_not_strict_and_no_permission(self):
        z = target_zone()
        self.assertEqual(z["direction"], -1)
        self.assertEqual(z["kind"], "BREAKER_CANDIDATE")
        self.assertEqual(z["evidence"]["source_scheduled_minutes"], 180)
        self.assertFalse(z["evidence"]["source_is_full_timeframe"])
        self.assertFalse(z["evidence"]["qualified_ob"])
        self.assertFalse(z["execution_allowed"])

    def test_no_future_knowledge_before_failure(self):
        b = fixture()
        z = target_zone(b)
        earlier = m.detect_breaker_candidates(b.loc[b.index < z["known_at"]], 240, CFG)
        self.assertFalse(any(x["low"] == 100 and x["high"] == 102 for x in earlier))

    def test_three_distinct_closed_stage_times(self):
        z = target_zone()
        ev = z["evidence"]
        self.assertLess(z["origin_at"], pd.Timestamp(ev["sweep_at"]))
        self.assertLess(pd.Timestamp(ev["sweep_at"]), pd.Timestamp(ev["reclaimed_at"]))
        self.assertLess(pd.Timestamp(ev["reclaimed_at"]), z["known_at"])

    def test_mirrored_bullish_pattern(self):
        b = fixture()
        original = b.copy()
        b["open"], b["close"] = 300-original.open, 300-original.close
        b["high"], b["low"] = 300-original.low, 300-original.high
        zones = m.detect_breaker_candidates(b, 240, CFG)
        z = next(z for z in zones if z["low"] == 198 and z["high"] == 200)
        self.assertEqual(z["direction"], 1)

    def test_disabled_layer_empty(self):
        self.assertEqual(m.detect_breaker_candidates(fixture(), 240,
                         replace(CFG, enable_breaker_candidates=False)), [])

    def test_missing_scheduled_reclaim_bar_prevents_candidate(self):
        b = fixture()
        z = target_zone(b)
        b = b.drop(pd.Timestamp(z["evidence"]["reclaimed_at"]))
        self.assertFalse(any(x["low"] == 100 and x["high"] == 102
                             for x in m.detect_breaker_candidates(b, 240, CFG)))

    def test_partial_confirmation_not_called_four_hour(self):
        b = fixture()
        b.loc[b.index[-1], "scheduled_minutes"] = 180
        self.assertFalse(any(x["low"] == 100 and x["high"] == 102
                             for x in m.detect_breaker_candidates(b, 240, CFG)))

    def test_insufficient_history_empty(self):
        self.assertEqual(m.detect_breaker_candidates(fixture().iloc[-8:], 240, CFG), [])

    def test_strict_output_unchanged_by_candidate_switch(self):
        b = fixture().loc[lambda x: x.scheduled_minutes == 240]
        a = m.detect_zones(b, 240, CFG)
        c = m.detect_zones(b, 240, replace(CFG, enable_breaker_candidates=False))
        self.assertEqual(a, c)

    def test_gc_summary_keeps_candidates_separate(self):
        zone = m._serialize_zone(target_zone())
        result = summarize_gc_breakers(dict(state="ACTIVE_ZONES", asof=zone["known_at"],
                     timeframes={"4h": dict(quality="OK", zones=[], breaker_candidates=[zone])}))
        self.assertEqual(result["state"], "WEAKER_BREAKER_CANDIDATES_ONLY")
        self.assertEqual(result["active_bearish"], 0)
        self.assertEqual(result["active_candidates"], 1)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
