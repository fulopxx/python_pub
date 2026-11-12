"""Gyorsítási módosítások egyenértékűségi tesztjei.

A 2026-os optimalizálás nem változtathatja az elemzési kimeneteket:
- market_exhaustion 0.1.3: vektorizált session-címkék, _slot_baseline,
  tömbalapú _htf_features;
- market_structure 0.1.2: _span_checker a sorról sorra maszkolt
  _span_complete helyett;
- gc_regime_1_0_16: vektorizált add_session_key.

Ezek a tesztek a régi implementációk másolataival (referencia) hasonlítanak
össze, így a gyorsítás nem csak historikus ablakokon, hanem szintetikus
határeseteken (DST-őszire visszateszőlt helyi idősáv, duplikált slot,
ütemezési rés) is ellenőrzött.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import market_exhaustion
from market_exhaustion import _slot_baseline, _htf_features, _session_labels
from market_structure import _span_checker, _span_complete, StructureConfig
import gc_regime_1_0_16 as engine


CFG = StructureConfig()


def _reference_session_labels(idx: pd.DatetimeIndex) -> pd.Index:
    # A 0.1.2-es market_exhaustion eredeti implementációja.
    local = idx.tz_localize(None)
    return pd.Index((local - pd.Timedelta(hours=18)).date)


def _reference_baseline(volume: pd.Series, min_sessions: int) -> pd.Series:
    # A 0.1.2-es _features-ben használt per-slot groupby transform.
    def baseline(s: pd.Series) -> pd.Series:
        sessions = _reference_session_labels(s.index - pd.Timedelta(minutes=1))
        first_per_session = pd.Series(
            s.to_numpy(float), index=sessions
        ).groupby(level=0, sort=False).first()
        prior = first_per_session.shift(1).expanding(
            min_periods=min_sessions
        ).median()
        return pd.Series(prior.reindex(sessions).to_numpy(), index=s.index)

    slots = volume.index.hour * 60 + volume.index.minute
    return volume.groupby(slots).transform(baseline)


def _reference_htf_features(df: pd.DataFrame, minimum: int) -> pd.DataFrame:
    # A 0.1.2-es _htf_features eredeti, pandas-szeletelő változata.
    from market_exhaustion import _atr

    def _slope(series: pd.Series, atr: float) -> tuple[float, float]:
        y = series.to_numpy(float)
        x = np.asarray((series.index - series.index[0]) / pd.Timedelta(hours=1))
        if len(y) < 2 or atr <= 0 or np.ptp(x) == 0:
            return np.nan, np.nan
        slope, intercept = np.polyfit(x, y, 1)
        residual = np.sum((y - (slope * x + intercept)) ** 2)
        variance = np.sum((y - y.mean()) ** 2)
        return slope / atr, 1 - residual / variance if variance > 0 else 0.0

    f = pd.DataFrame(index=df.index)
    if df.empty:
        return f
    atr = _atr(df)
    ema = df.close.ewm(span=20, adjust=False, min_periods=20).mean()
    f["extension"] = ((df.close - ema) / atr.replace(0, np.nan)).where(
        np.arange(len(df)) >= minimum - 1
    )
    f["acceleration"] = np.nan
    f["slope"] = np.nan
    f["r2"] = np.nan
    acceleration = np.full(len(df), np.nan)
    slopes = np.full(len(df), np.nan)
    fits = np.full(len(df), np.nan)
    for i in range(max(minimum - 1, 11), len(df)):
        a = float(atr.iloc[i])
        old, _ = _slope(df.close.iloc[i - 11:i - 5], a)
        new, r2 = _slope(df.close.iloc[i - 5:i + 1], a)
        acceleration[i] = new / old if old > 0 and new > 0 else 0.0
        slopes[i] = new
        fits[i] = r2
    f["acceleration"], f["slope"], f["r2"] = acceleration, slopes, fits
    return f


class SessionLabelTests(unittest.TestCase):
    def test_labels_match_reference_across_dst_fallback(self):
        # 2026-11-01: 2:00 EDT -> 1:00 EST visszatesőlés New Yorkban.
        idx = pd.date_range("2026-10-31 20:00", "2026-11-01 04:00",
                            freq="1min", tz="America/New_York")
        self.assertEqual(
            [stamp.date() for stamp in _session_labels(idx)],
            list(_reference_session_labels(idx)),
        )

    def test_session_boundary_18h_belongs_to_next_session(self):
        idx = pd.DatetimeIndex(
            [pd.Timestamp("2026-11-02 17:55", tz="America/New_York"),
             pd.Timestamp("2026-11-02 18:00", tz="America/New_York"),
             pd.Timestamp("2026-11-02 18:05", tz="America/New_York")]
        )
        labels = list(_session_labels(idx))
        # 17:55 helyi idő még a november 1-jei session, 18:00 már a új.
        self.assertEqual(labels[0], pd.Timestamp("2026-11-01"))
        self.assertEqual(labels[1], pd.Timestamp("2026-11-02"))
        self.assertEqual(labels[2], pd.Timestamp("2026-11-02"))
        self.assertEqual(
            [stamp.date() for stamp in labels],
            list(_reference_session_labels(idx)),
        )


class SlotBaselineTests(unittest.TestCase):
    def _synthetic_volume(self, duplicated_slot: bool) -> pd.Series:
        days = ["2026-10-28", "2026-10-29", "2026-10-30", "2026-10-31",
                "2026-11-02"]
        stamps = []
        for day in days:
            for minute in (0, 15, 30, 45):
                stamps.append(pd.Timestamp(f"{day} 08:{minute:02d}",
                                           tz="America/New_York"))
        if duplicated_slot:
            # DST-őszi visszatesőléskor ugyanaz a helyi 5 perces slot
            # kétszer szerepel ugyanabban a sessionben; az ismételt példány
            # csak UTC-ből konstruálható egyértelműen.
            stamps += [pd.Timestamp("2026-11-01 05:15", tz="UTC").tz_convert(
                           "America/New_York"),
                       pd.Timestamp("2026-11-01 06:15", tz="UTC").tz_convert(
                           "America/New_York")]
        index = pd.DatetimeIndex(stamps).sort_values()
        values = np.arange(1, len(index) + 1, dtype=float) * 10.0
        return pd.Series(values, index=index)

    def test_baseline_matches_reference_transform(self):
        for duplicated in (False, True):
            with self.subTest(duplicated_slot=duplicated):
                volume = self._synthetic_volume(duplicated)
                slots = volume.index.hour * 60 + volume.index.minute
                labels = np.asarray(
                    _session_labels(volume.index - pd.Timedelta(minutes=1))
                )
                reference = _reference_baseline(volume, min_sessions=2)
                fast = _slot_baseline(
                    volume.to_numpy(float), np.asarray(slots), labels, 2
                )
                reference_values = reference.to_numpy(float)
                for row in range(len(volume)):
                    ref, new = reference_values[row], fast[row]
                    if np.isnan(ref):
                        self.assertTrue(np.isnan(new), f"row={row}")
                    else:
                        self.assertEqual(ref, new, f"row={row}")

    def test_baseline_requires_min_sessions(self):
        volume = self._synthetic_volume(False)
        slots = volume.index.hour * 60 + volume.index.minute
        labels = np.asarray(
            _session_labels(volume.index - pd.Timedelta(minutes=1))
        )
        fast = _slot_baseline(volume.to_numpy(float), np.asarray(slots),
                              labels, 2)
        first_two_sessions = 8  # napi 4 slot; a 3. session (k=2) az első érvényes
        self.assertTrue(np.isnan(fast[:first_two_sessions]).all())
        self.assertTrue(np.isfinite(fast[first_two_sessions:]).all())


class HtfFeaturesTests(unittest.TestCase):
    def test_htf_features_exactly_matches_reference(self):
        rng = np.random.default_rng(20261019)
        stamps = pd.date_range("2026-09-01", periods=70, freq="1h",
                               tz="America/New_York")
        close = 4000.0 + np.cumsum(rng.normal(0, 3.0, len(stamps)))
        bars = pd.DataFrame(
            {"open": close - rng.random(len(stamps)),
             "high": close + rng.random(len(stamps)),
             "low": close - rng.random(len(stamps)),
             "close": close,
             "volume": rng.integers(100, 900, len(stamps)).astype(float)},
            index=stamps,
        )
        for minimum in (25, 30, 40):
            with self.subTest(minimum=minimum):
                reference = _reference_htf_features(bars, minimum)
                fast = _htf_features(bars, minimum)
                self.assertEqual(list(reference.columns), list(fast.columns))
                for column in reference.columns:
                    left = reference[column].to_numpy(float)
                    right = fast[column].to_numpy(float)
                    self.assertTrue(
                        np.array_equal(np.isnan(left), np.isnan(right)),
                        f"column={column}, minimum={minimum} (NaN-minta)",
                    )
                    comparable = ~np.isnan(left)
                    self.assertTrue(
                        np.allclose(left[comparable], right[comparable],
                                    rtol=0.0, atol=0.0),
                        f"column={column}, minimum={minimum}",
                    )


class SpanCheckerTests(unittest.TestCase):
    def test_span_checker_matches_complete_on_gaps(self):
        rng = np.random.default_rng(2026101902)
        expected = pd.date_range("2026-09-01", "2026-09-20 23:00",
                                 freq="60min", tz="UTC")
        # véletlenszerű hiányzó ütemezett zárássávok
        keep = rng.random(len(expected)) > 0.25
        index = expected[keep]
        complete = _span_checker(index, expected)
        queries = [
            (expected[3], expected[9]),
            (expected[3], expected[3]),
            (expected[10], expected[40]),
            (expected[50], expected[120]),
            (expected[100], expected[100]),
            (expected[400], expected[460]),
        ]
        for start, end in queries:
            self.assertEqual(
                complete(start, end),
                _span_complete(index, expected, start, end),
                f"start={start}, end={end}",
            )


class EngineSessionKeyTests(unittest.TestCase):
    def test_vectorized_add_session_key_matches_per_row(self):
        idx = pd.date_range("2026-10-05 17:00", periods=5000, freq="1min",
                            tz="America/New_York")
        frame = pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
             "volume": 1.0},
            index=idx,
        )
        result = engine.add_session_key(frame)
        per_row = [engine.session_key(stamp) for stamp in idx]
        self.assertEqual(list(result["session_key"]), per_row)


if __name__ == "__main__":
    unittest.main()
