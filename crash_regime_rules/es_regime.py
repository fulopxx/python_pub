#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRASH REGIME ALGORITHMIC CHECKLIST – Runnable Prototype
========================================================
- Generates synthetic market states (Normal / Blow-off / Crash / Chop)
- Evaluates regime filters, position sizing, entry/exit rules
- Prints a human-readable CHECKLIST REPORT for each scenario
- No external deps beyond stdlib + numpy/pandas (standard data stack)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Literal, Dict, Any
from enum import Enum

# ──────────────────────────────────────────────────────────────
# 1. CONFIG / THRESHOLDS  ( módosítsd a saját risk paramétereidre )
# ──────────────────────────────────────────────────────────────
class Regime(str, Enum):
    NORMAL      = "NORMAL_TREND"
    BLOWOFF     = "EXPONENTIAL_BLOWOFF"
    CRASH       = "CRASH_LIQUIDATION"
    CHOP        = "CHOPPY_LOW_VOL"

@dataclass(frozen=True)
class Thresholds:
    # Regime detection
    vix_backwardation_ratio: float = 1.15      # VIX / VIX3M > this
    vix_absolute_min: float      = 30.0
    spy_vwap_zscore_max: float   = -2.5        # price below VWAP (sigma)
    breadth_ad_max: float        = 0.30        # advancing / declining < 30%
    gamma_flip_threshold: float  = 0.0         # dealer gamma < 0 (short gamma)
    adx_min: float               = 35.0
    rsi_2_max: float             = 95.0        # extreme overbought (blowoff)
    cvd_divergence_lookback: int = 20

    # Position sizing
    base_risk_pct: float         = 0.005       # 0.5% equity per trade
    crash_size_multiplier: float = 0.4         # 40% of base size in crash
    blowoff_size_multiplier: float = 0.6       # 60% in blowoff
    max_hold_min_crash: int      = 90
    max_hold_min_blowoff: int    = 45

    # Entry / Exit
    entry_rsi_5m_min: float      = 55.0        # short the rip
    entry_cvd_rising: bool       = True
    entry_vol_quiet_pct: float   = 0.7         # volume < 70% avg
    trail_psar_af0: float        = 0.015
    trail_psar_af_max: float     = 0.15
    fib_target_ext: float        = 1.618

TH = Thresholds()

# ──────────────────────────────────────────────────────────────
# 2. SYNTHETIC DATA GENERATOR (helyettesíti a live feed-et)
# ──────────────────────────────────────────────────────────────
def synth_market_state(regime: Regime, n_bars: int = 500) -> pd.DataFrame:
    """
    Létrehoz egy 1-perces OHLCV + VIX + Breadth + CVD + Dealer Gamma DataFrame-et
    a megadott régió típussal. CSAK TESZTRE / DEMO-RA!
    """
    rng = np.random.default_rng(42)
    t = np.arange(n_bars)
    base = 4500.0  # ES level

    if regime == Regime.NORMAL:
        drift = 0.02 / (252*390)  # ~5% év átlag
        vol = 1.2
        vix = 14 + rng.normal(0, 1.5, n_bars)
        breadth = 0.55 + rng.normal(0, 0.1, n_bars)
        gamma = 500_000 + rng.normal(0, 50_000, n_bars)  # dealers long gamma
        cvd_drift = 100

    elif regime == Regime.BLOWOFF:
        drift = 0.15 / (252*390)  # parabolic melt-up
        vol = 1.8
        vix = 11 + rng.normal(0, 1, n_bars) * np.exp(-t/200)  # VIX crush
        breadth = 0.75 + rng.normal(0, 0.05, n_bars)          # narrow leadership
        gamma = -200_000 + rng.normal(0, 30_000, n_bars)      # dealers SHORT gamma
        cvd_drift = 500  # aggressive buying

    elif regime == Regime.CRASH:
        drift = -0.40 / (252*390)  # -40% annualized intraday
        vol = 5.5
        vix = 38 + rng.normal(0, 5, n_bars) * (1 + t/500)     # VIX spike
        breadth = 0.12 + rng.normal(0, 0.08, n_bars)          # panic selling
        gamma = -1_200_000 + rng.normal(0, 100_000, n_bars)   # deep short gamma
        cvd_drift = -800  # aggressive selling

    else:  # CHOP
        drift = 0.0
        vol = 0.6
        vix = 16 + rng.normal(0, 1, n_bars)
        breadth = 0.50 + rng.normal(0, 0.07, n_bars)
        gamma = 100_000 + rng.normal(0, 40_000, n_bars)
        cvd_drift = 0

    # Price path
    returns = rng.normal(drift, vol/10000, n_bars)  # vol in index points per min
    # add fat tails for crash
    if regime == Regime.CRASH:
        jumps = rng.choice([0, -1, -2, -3], n_bars, p=[0.97, 0.02, 0.008, 0.002])
        returns += jumps * vol * 1.5

    close = base * np.exp(np.cumsum(returns))
    high  = close * (1 + np.abs(rng.normal(0, vol*0.3/10000, n_bars)))
    low   = close * (1 - np.abs(rng.normal(0, vol*0.3/10000, n_bars)))
    open_ = np.roll(close, 1); open_[0] = base
    volume = rng.lognormal(12, 0.4, n_bars).astype(int)
    if regime == Regime.CRASH: volume *= 3
    if regime == Regime.BLOWOFF: volume *= 2

    # VWAP anchor (session start)
    vwap = np.cumsum(close * volume) / np.cumsum(volume)
    spy_vwap_z = (close - vwap) / (close * 0.005)  # approx 0.5% intraday std

    # CVD (Cumulative Volume Delta) – simplified
    cvd = np.cumsum(cvd_drift + rng.normal(0, 2000, n_bars))

    # Indicators
    df = pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume,
        'vix': np.clip(vix, 8, 80),
        'vix_3m': np.clip(vix * 1.15, 10, 90),  # term structure proxy
        'breadth_ad': np.clip(breadth, 0.01, 0.99),
        'dealer_gamma': gamma,
        'cvd': cvd,
        'vwap': vwap,
        'spy_vwap_z': spy_vwap_z,
    }, index=pd.date_range('2026-10-12 09:30', periods=n_bars, freq='1min', tz='America/New_York'))

    # Technicals
    df['rsi_2']  = _rsi(df['close'], 2)
    df['rsi_5m'] = _rsi(df['close'], 5)
    df['adx_14'] = _adx(df['high'], df['low'], df['close'], 14)
    df['atr_14'] = _atr(df['high'], df['low'], df['close'], 14)
    df['vol_avg_20'] = df['volume'].rolling(20).mean()
    return df

def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _atr(high, low, close, period):
    tr = np.maximum(high - low, np.maximum((high - close.shift()).abs(), (low - close.shift()).abs()))
    return tr.ewm(alpha=1/period, adjust=False).mean()

def _adx(high, low, close, period):
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr = _atr(high, low, close, 1)
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr.ewm(alpha=1/period).mean())
    minus_di = 100 * (-minus_dm.ewm(alpha=1/period).mean() / tr.ewm(alpha=1/period).mean())
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1/period).mean()

# ──────────────────────────────────────────────────────────────
# 3. CORE LOGIC – a checklist függvények
# ──────────────────────────────────────────────────────────────
@dataclass
class MarketSnapshot:
    vix: float
    vix_3m: float
    spy_vwap_z: float
    breadth_ad: float
    dealer_gamma: float
    adx_14: float
    rsi_2: float
    rsi_5m: float
    cvd: float
    cvd_5m_ago: float
    volume: float
    vol_avg_20: float
    close: float
    vwap: float
    atr_14: float

def snapshot_from_df(df: pd.DataFrame, idx: int = -1) -> MarketSnapshot:
    row = df.iloc[idx]
    cvd_now = row['cvd']
    cvd_5m = df['cvd'].iloc[max(0, idx-5)]
    return MarketSnapshot(
        vix=row['vix'], vix_3m=row['vix_3m'], spy_vwap_z=row['spy_vwap_z'],
        breadth_ad=row['breadth_ad'], dealer_gamma=row['dealer_gamma'],
        adx_14=row['adx_14'], rsi_2=row['rsi_2'], rsi_5m=row['rsi_5m'],
        cvd=cvd_now, cvd_5m_ago=cvd_5m,
        volume=row['volume'], vol_avg_20=row['vol_avg_20'],
        close=row['close'], vwap=row['vwap'], atr_14=row['atr_14']
    )

def crash_regime_active(snap: MarketSnapshot, th: Thresholds) -> Dict[str, Any]:
    checks = {
        'vix_backwardation': snap.vix > snap.vix_3m * th.vix_backwardation_ratio,
        'vix_elevated': snap.vix > th.vix_absolute_min,
        'price_below_vwap': snap.spy_vwap_z < th.spy_vwap_zscore_max,
        'breadth_collapse': snap.breadth_ad < th.breadth_ad_max,
        'dealers_short_gamma': snap.dealer_gamma < th.gamma_flip_threshold,
        'strong_trend_adx': snap.adx_14 > th.adx_min,
    }
    all_true = all(checks.values())
    return {'regime': Regime.CRASH if all_true else Regime.NORMAL, 'checks': checks}

def blowoff_regime_active(snap: MarketSnapshot, th: Thresholds) -> Dict[str, Any]:
    checks = {
        'adx_strong': snap.adx_14 > th.adx_min,
        'rsi_2_extreme': snap.rsi_2 > th.rsi_2_max,
        'vix_complacent': snap.vix < 15,  # VIX crush
        'dealers_short_gamma': snap.dealer_gamma < 0,
        'price_above_vwap': snap.spy_vwap_z > 3.0,
    }
    all_true = all(checks.values())
    return {'regime': Regime.BLOWOFF if all_true else Regime.NORMAL, 'checks': checks}

def position_sizing(equity: float, regime: Regime, atr: float, th: Thresholds) -> Dict[str, float]:
    base_risk = equity * th.base_risk_pct
    mult = {Regime.CRASH: th.crash_size_multiplier,
            Regime.BLOWOFF: th.blowoff_size_multiplier}.get(regime, 1.0)
    risk_dollars = base_risk * mult
    # Futures: 1 pont = $50 (ES), $20 (NQ), $5 (MES), $2 (MNQ) – itt ES $50 feltételezve
    pt_value = 50.0
    stop_dist_pts = max(atr * 0.8, 8.0)  # min 8 pts stop
    contracts = max(1, int(risk_dollars / (stop_dist_pts * pt_value)))
    return {'risk_dollars': risk_dollars, 'stop_dist_pts': stop_dist_pts, 'contracts': contracts}

def entry_signal_crash(snap: MarketSnapshot, th: Thresholds) -> Dict[str, Any]:
    # SHORT THE RIP: failed rally into VWAP/EMA, low volume, CVD rising (divergence)
    checks = {
        'rsi_5m_overbought': snap.rsi_5m > th.entry_rsi_5m_min,
        'cvd_rising_divergence': snap.cvd > snap.cvd_5m_ago,  # price down, cvd up
        'volume_dry': snap.volume < snap.vol_avg_20 * th.entry_vol_quiet_pct,
        'price_below_vwap': snap.close < snap.vwap,
    }
    return {'signal': 'SHORT' if all(checks.values()) else 'NONE', 'checks': checks}

def exit_management_crash(snap: MarketSnapshot, entry_price: float, side: Literal['SHORT','LONG'], th: Thresholds) -> Dict[str, Any]:
    # PSAR trailing approx: use ATR multiple for demo
    trail_dist = snap.atr_14 * 1.5
    if side == 'SHORT':
        stop_price = entry_price + trail_dist
        target_1 = entry_price - snap.atr_14 * 1.0
        target_2 = entry_price - snap.atr_14 * 2.5
        target_3 = entry_price - (entry_price - snap.low if hasattr(snap,'low') else entry_price*0.02) * th.fib_target_ext
    else:
        stop_price = entry_price - trail_dist
        target_1 = entry_price + snap.atr_14 * 1.0
        target_2 = entry_price + snap.atr_14 * 2.5
        target_3 = entry_price + (snap.high - entry_price if hasattr(snap,'high') else entry_price*0.02) * th.fib_target_ext
    return {
        'trail_stop_pts': trail_dist,
        'stop_price': round(stop_price, 2),
        'targets': [round(t,2) for t in [target_1, target_2, target_3]],
        'max_hold_min': th.max_hold_min_crash,
        'time_exit_rule': f'Force close after {th.max_hold_min_crash} min if no target hit'
    }

# ──────────────────────────────────────────────────────────────
# 4. REPORT GENERATOR – szép szöveges kimenet
# ──────────────────────────────────────────────────────────────
def format_checklist(title: str, checks: Dict[str, bool]) -> str:
    lines = [f"\n{'='*60}", f" {title}", f"{'='*60}"]
    for k, v in checks.items():
        status = "✅ PASS" if v else "❌ FAIL"
        lines.append(f"  [{status}] {k}")
    lines.append(f"{'='*60}")
    return "\n".join(lines)

def run_full_checklist(equity: float = 100_000) -> str:
    report = ["\n" + "="*70,
              " CRASH REGIME ALGORITHMIC CHECKLIST – LIVE SIMULATION",
              "="*70]

    for regime in [Regime.NORMAL, Regime.BLOWOFF, Regime.CRASH, Regime.CHOP]:
        df = synth_market_state(regime, n_bars=600)
        snap = snapshot_from_df(df, -1)  # last bar = "now"

        # 1. Regime Detection
        crash_res = crash_regime_active(snap, TH)
        blowoff_res = blowoff_regime_active(snap, TH)
        detected = crash_res['regime'] if crash_res['regime'] != Regime.NORMAL else blowoff_res['regime']

        report.append(format_checklist(f"REGIME DETECTION → {detected.value}", crash_res['checks']))
        if detected == Regime.BLOWOFF:
            report.append(format_checklist("BLOWOFF CONFIRMATION", blowoff_res['checks']))

        # 2. Position Sizing
        pos = position_sizing(equity, detected, snap.atr_14, TH)
        report.append(f"\n📏 POSITION SIZING (Equity ${equity:,.0f})")
        report.append(f"   Risk/Trade: ${pos['risk_dollars']:,.0f} | Stop Dist: {pos['stop_dist_pts']:.1f} pts | Contracts: {pos['contracts']}")

        # 3. Entry Signal (only if crash or blowoff)
        if detected in (Regime.CRASH, Regime.BLOWOFF):
            entry = entry_signal_crash(snap, TH)
            report.append(format_checklist(f"ENTRY SIGNAL ({entry['signal']})", entry['checks']))

            if entry['signal'] == 'SHORT':
                exit_plan = exit_management_crash(snap, snap.close, 'SHORT', TH)
                report.append(f"\n🎯 EXIT PLAN (SHORT @ {snap.close:.2f})")
                report.append(f"   Trail Stop: {exit_plan['trail_stop_pts']:.1f} pts → Stop Price: {exit_plan['stop_price']}")
                report.append(f"   Targets: T1={exit_plan['targets'][0]} | T2={exit_plan['targets'][1]} | T3={exit_plan['targets'][2]}")
                report.append(f"   ⏱ Max Hold: {exit_plan['max_hold_min']} min ({exit_plan['time_exit_rule']})")

        report.append("\n" + "-"*70)

    return "\n".join(report)

# ──────────────────────────────────────────────────────────────
# 5. MAIN – ha scriptként futtatod
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        print(run_full_checklist(equity=100_000))
    except Exception as e:
        import traceback, sys
        print(f"\n❌ HIBA: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)