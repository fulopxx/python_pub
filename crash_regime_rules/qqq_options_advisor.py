"""QQQ put / put debit spread scanner, research/paper ONLY.

Reads a timestamped index advisory and queries market data, never orders.
Bid/ask and Greeks have no exchange-source timestamp in this interface:
all selections are explicitly unverified paper review candidates.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
import textwrap
from typing import Any

import pandas as pd

VERSION = "0.1.5"
BASE_DIR = Path(__file__).resolve().parent
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 50
    market_data_type: int = 3
    target_dte: int = 7
    min_dte: int = 4
    max_dte: int = 14
    target_abs_delta: float = .45
    long_delta_min: float = .35
    long_delta_max: float = .55
    short_delta_min: float = .15
    short_delta_max: float = .30
    min_open_interest: int = 100
    max_spread_fraction: float = .15
    max_spread_usd: float = .30
    premium_budget_usd: float = 500.0
    assumed_entry_fee_usd_per_leg: float = 1.0
    max_greek_underlier_difference_fraction: float = .01
    max_expiries: int = 3
    max_strikes_per_expiry: int = 12
    max_contracts: int = 36
    batch_size: int = 12
    quote_wait_seconds: float = 8.0
    max_quote_wait_seconds: float = 20.0
    max_source_age_minutes: float = 45.0
    max_advisory_run_age_minutes: float = 15.0
    max_context_skew_minutes: float = 15.0
    require_quote_sizes: bool = True
    max_leg_collection_skew_seconds: float = 30.0

    def __post_init__(self) -> None:
        numeric = (self.premium_budget_usd, self.assumed_entry_fee_usd_per_leg,
                   self.target_abs_delta, self.long_delta_min, self.long_delta_max,
                   self.short_delta_min, self.short_delta_max, self.max_spread_fraction,
                   self.max_spread_usd, self.quote_wait_seconds,
                   self.max_greek_underlier_difference_fraction, self.max_source_age_minutes,
                   self.max_advisory_run_age_minutes, self.max_context_skew_minutes,
                   self.max_leg_collection_skew_seconds, self.max_quote_wait_seconds)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("All numeric configuration values must be finite.")
        if not 1 <= self.min_dte <= self.target_dte <= self.max_dte <= 365:
            raise ValueError("DTE: 1 <= min <= target <= max <= 365.")
        if not 0 < self.long_delta_min <= self.target_abs_delta <= self.long_delta_max < 1:
            raise ValueError("Invalid target/long delta range.")
        if not 0 < self.short_delta_min <= self.short_delta_max < 1:
            raise ValueError("Invalid short delta range.")
        if self.premium_budget_usd <= 0 or self.min_open_interest < 0:
            raise ValueError("Positive premium budget and nonnegative OI required.")
        if (self.batch_size < 1 or self.max_contracts < 1 or self.max_expiries < 1
                or self.max_strikes_per_expiry < 1 or self.quote_wait_seconds < 0
                or self.max_quote_wait_seconds < self.quote_wait_seconds
                or self.assumed_entry_fee_usd_per_leg < 0
                or self.max_spread_fraction <= 0 or self.max_spread_usd <= 0):
            raise ValueError("Invalid request/spread/fee configuration.")


def number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) and abs(result) < 1e100 else None
    except (TypeError, ValueError, OverflowError):
        return None


def utc(value: Any) -> pd.Timestamp:
    if value is None:
        raise ValueError("Missing timestamp.")
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Timezone-aware timestamp required.")
    return stamp.tz_convert("UTC")


def load_advisory(path: str | Path) -> dict[str, Any] | None:
    try:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("Advisory JSON must be an object.")
        return result
    except (OSError, ValueError) as error:
        log.warning("Index advisory input N/A: %s", error)
        return None


def advisory_gate(packet: dict[str, Any] | None, now: pd.Timestamp,
                  cfg: Config) -> dict[str, Any]:
    result = dict(quality="MISSING", paper_allowed=False, state="N/A",
                  structure=None, asof=None, reasons=["Futtasd előbb az ES/NQ runnert."])
    if not packet:
        return result
    try:
        if packet.get("schema_version") != 1 or packet.get("source") != "ES_NQ_REGIME_RUNNER":
            raise ValueError("Ismeretlen index advisory séma/forrás.")
        observed, asof = utc(packet.get("observed_at")), utc(packet.get("asof"))
        clock = utc(now)
        if not asof <= observed <= clock:
            raise ValueError("Jövőbeli vagy hibás advisory időpont.")
        if ((clock-asof).total_seconds()/60 > cfg.max_source_age_minutes
                or (clock-observed).total_seconds()/60 > cfg.max_advisory_run_age_minutes):
            result.update(quality="STALE", reasons=["Az index advisory túl régi."])
            return result
        advisory = packet["advisory"]
        indices = packet["indices"]
        result.update(quality="OK", state=advisory["state"], asof=asof.isoformat(),
                      structure=advisory.get("recommended_structure"),
                      reasons=list(advisory.get("reasons", [])))
        if packet.get("execution_allowed") is not False or advisory.get("execution_allowed") is not False:
            raise ValueError("Az input nem a várt paper-only advisory.")
        if advisory.get("entry_scope") != "PAPER_ONLY":
            raise ValueError("Hiányzó paper scope.")
        if any(indices[s]["data_quality"] != "OK" or utc(indices[s]["asof"]) != asof
               for s in ("ES", "NQ")):
            result["reasons"].append("ES/NQ core adatminőség vagy közös időpont hibás.")
            return result
        if indices["NQ"].get("short_overlay_blocked") is not False:
            result["reasons"].append("Az NQ short overlay kapu zárt vagy hiányzik.")
            return result
        allowed = (advisory.get("entry_allowed") is True
                   and advisory.get("trade_candidate") is True
                   and advisory["state"] == "QQQ_PUT_CANDIDATE"
                   and result["structure"] in (
                       "QQQ_LONG_PUT_OR_DEBIT_SPREAD", "QQQ_PUT_DEBIT_SPREAD_PREFERRED"))
        result["paper_allowed"] = allowed
        if not allowed:
            result["reasons"].append("Az index advisory nem enged friss paper put setupot.")
    except (ValueError, TypeError, KeyError) as error:
        result.update(quality="INVALID", paper_allowed=False, reasons=[str(error)])
    return result


def underlying_context(raw: pd.DataFrame, now: pd.Timestamp, cfg: Config) -> dict[str, Any]:
    result = dict(quality="MISSING", price=None, asof=None, age_minutes=None,
                  rth_vwap=None, below_rth_vwap=None, paper_price_gate=False)
    if raw is None or raw.empty or not {"date", "high", "low", "close", "volume"}.issubset(raw):
        return result
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")+pd.Timedelta(minutes=1)
    frame = frame.set_index("date").sort_index()
    for key in ("high", "low", "close", "volume"):
        frame[key] = pd.to_numeric(frame[key], errors="coerce")
    frame = frame.loc[(frame.index <= now) & frame.index.notna()]
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    valid = frame[["high", "low", "close", "volume"]].apply(lambda col: col.map(number)).notna().all(axis=1)
    valid &= ((frame[["high", "low", "close"]] > 0).all(axis=1)
              & (frame.volume >= 0) & (frame.high >= frame.close) & (frame.low <= frame.close))
    frame = frame.loc[valid]
    if frame.empty:
        return result
    stamp = frame.index[-1]
    price = float(frame.close.iloc[-1])
    age = (now-stamp).total_seconds()/60
    result.update(quality="OK", price=price, asof=stamp.isoformat(), age_minutes=age)
    if age > cfg.max_source_age_minutes:
        result["quality"] = "STALE"
        return result
    local_now = now.tz_convert("America/New_York")
    local = frame.index.tz_convert("America/New_York")
    session = frame.loc[(local.date == local_now.date())
                        & (local.hour*60+local.minute > 570)
                        & (local.hour*60+local.minute <= 960)]
    if not (local_now.weekday() < 5 and 570 <= local_now.hour*60+local_now.minute < 960):
        result["quality"] = "OUTSIDE_RTH"
        return result
    if session.empty or session.index[0].tz_convert("America/New_York").strftime("%H:%M") != "09:31":
        result["quality"] = "PARTIAL_RTH_SESSION"
        return result
    if len(session) < 20 or session.volume.sum() <= 0:
        result["quality"] = "INSUFFICIENT_RTH_HISTORY"
        return result
    expected_minutes = int((session.index[-1]-session.index[0])/pd.Timedelta(minutes=1))+1
    max_gap = session.index.to_series().diff().dt.total_seconds().div(60).max()
    if len(session)/expected_minutes < .90 or max_gap > 2:
        result["quality"] = "INCOMPLETE_RTH_SESSION"
        return result
    tp = (session.high+session.low+session.close)/3
    vwap = float((tp*session.volume).sum()/session.volume.sum())
    result.update(rth_vwap=vwap, below_rth_vwap=price < vwap,
                  paper_price_gate=price < vwap)
    return result


def select_chain(chains: list[Any], underlying_conid: int) -> Any | None:
    valid = [c for c in chains if getattr(c, "tradingClass", "") == "QQQ"
             and number(getattr(c, "multiplier", None)) == 100
             and positive_id(getattr(c, "underlyingConId", None)) == positive_id(underlying_conid)
             and positive_id(underlying_conid) is not None]
    return sorted(valid, key=lambda c: (getattr(c, "exchange", "") != "SMART",
                                        getattr(c, "exchange", "")))[0] if valid else None


def select_contract_specs(chain: Any, spot: float, now: pd.Timestamp,
                          cfg: Config) -> list[dict[str, Any]]:
    today = now.tz_convert("America/New_York").date()
    expiries = []
    for text in chain.expirations:
        try:
            date = datetime.strptime(str(text), "%Y%m%d").date()
            dte = (date-today).days
            if cfg.min_dte <= dte <= cfg.max_dte:
                expiries.append((abs(dte-cfg.target_dte), date, str(text), dte))
        except (TypeError, ValueError):
            continue
    strikes = sorted({float(s) for s in chain.strikes
                      if number(s) is not None and spot*.85 <= float(s) <= spot*1.03})
    if not strikes:
        return []
    # Denser near-ATM ladder. This is NOT a fabricated delta calculation.
    chosen = []
    for ratio in (1.00, .9975, .995, 1.005, .9925, .99, .985, .98, .975, .97, .96, .95):
        strike = min(strikes, key=lambda s: abs(s-spot*ratio))
        if strike not in chosen:
            chosen.append(strike)
    for strike in sorted(strikes, key=lambda s: abs(s-spot)):
        if strike not in chosen:
            chosen.append(strike)
    chosen = chosen[:cfg.max_strikes_per_expiry]
    result = []
    for _, _, expiry, dte in sorted(expiries)[:cfg.max_expiries]:
        for strike in chosen:
            result.append(dict(expiry=expiry, dte=dte, strike=strike))
    return result[:cfg.max_contracts]


def positive_id(value: Any) -> int | None:
    value = number(value)
    return int(value) if value is not None and value > 0 and value.is_integer() else None


def expiry_date(value: Any) -> str | None:
    """Exact day or explicit day + trading time; never accept month-only dates."""
    text = str(value or "").strip()
    if not re.fullmatch(r"\d{8}(?: \d{2}:\d{2}:\d{2}(?: [A-Za-z_+/\-0-9]+)?)?", text):
        return None
    try:
        datetime.strptime(text[:8], "%Y%m%d")
        if len(text) > 8:
            datetime.strptime(text[9:17], "%H:%M:%S")
        return text[:8]
    except ValueError:
        return None


def definition_rejections(detail: Any, expiry: str, underlying_conid: Any,
                          spot: float) -> list[str]:
    """All rejection reasons, not one silent boolean. Identity stays mandatory."""
    c = getattr(detail, "contract", None)
    if c is None:
        return ["CONTRACT_MISSING"]
    reasons = []
    for attr, expected, reason in (
        ("symbol", "QQQ", "SYMBOL_MISMATCH"),
        ("right", "P", "NOT_PUT"),
        ("currency", "USD", "CURRENCY_MISMATCH"),
        ("tradingClass", "QQQ", "TRADING_CLASS_NOT_QQQ"),
    ):
        if getattr(c, attr, "") != expected:
            reasons.append(reason)
    under = positive_id(getattr(detail, "underConId", None))
    target = positive_id(underlying_conid)
    if under is None:
        reasons.append("UNDERLYING_CONID_MISSING_OR_INVALID")
    elif target is None or under != target:
        reasons.append("UNDERLYING_CONID_MISMATCH")
    if getattr(c, "secType", "OPT") != "OPT":
        reasons.append("NOT_OPTION")
    if number(getattr(c, "multiplier", None)) != 100:
        reasons.append("MULTIPLIER_NOT_100")
    date = expiry_date(getattr(c, "lastTradeDateOrContractMonth", None))
    if date is None:
        reasons.append("CONTRACT_EXPIRY_INVALID")
    elif date != expiry:
        reasons.append("CONTRACT_EXPIRY_MISMATCH")
    real_date = getattr(detail, "realExpirationDate", "")
    if real_date and expiry_date(real_date) != expiry:
        reasons.append("REAL_EXPIRATION_DATE_MISMATCH")
    if positive_id(getattr(c, "conId", None)) is None:
        reasons.append("CONID_MISSING_OR_INVALID")
    strike = number(getattr(c, "strike", None))
    if strike is None or strike <= 0:
        reasons.append("STRIKE_INVALID")
    elif not spot*.85 <= strike <= spot*1.03:
        reasons.append("STRIKE_OUTSIDE_PRICE_BAND")
    return reasons


def definition_sample(detail: Any, reasons: list[str]) -> dict[str, Any]:
    """Bounded public-contract evidence; no account/contact information."""
    c = getattr(detail, "contract", None)
    fields = ("conId", "secType", "symbol", "localSymbol", "lastTradeDateOrContractMonth",
              "strike", "right", "exchange", "currency", "tradingClass", "multiplier")
    return dict(contract={k: str(getattr(c, k, ""))[:160] for k in fields},
                under_conid=str(getattr(detail, "underConId", ""))[:160],
                real_expiration_date=str(getattr(detail, "realExpirationDate", ""))[:160],
                reasons=reasons)


def discover_contracts(ib: Any, api: Any, chain: Any, spot: float,
                       now: pd.Timestamp, cfg: Config) -> tuple[list[Any], list[str], dict]:
    """Resolve actual puts for selected expiries, NOT expiry x global strikes.

    At most max_expiries narrowly scoped definition requests. No fallback to
    guessed contracts if definition lookup fails. Quotes remain bounded.
    """
    seeds = select_contract_specs(chain, spot, now, cfg)
    expiries = list(dict.fromkeys(s["expiry"] for s in seeds))
    options, notes, counts = [], [], {}
    seen = set()
    for expiry in expiries:
        query = api.Option("QQQ", expiry, 0, "P", "SMART",
                           multiplier="100", currency="USD", tradingClass="QQQ")
        report = dict(status="LOOKUP_PENDING", available=0, selected=0,
                      raw_returned=0, rejected_definitions=0, rejection_counts={},
                      rejected_examples=[], normalized_expiry_count=0,
                      expected_underlying_conid=positive_id(chain.underlyingConId),
                      strike_band=[spot*.85, spot*1.03],
                      query=dict(symbol="QQQ", expiry=expiry, right="P", strike=0,
                                 exchange="SMART", trading_class="QQQ", multiplier="100"))
        counts[expiry] = report
        try:
            details = list(ib.reqContractDetails(query) or [])
        except Exception as error:
            notes.append(f"{expiry}: lejáratspecifikus kontraktuslista N/A: {error}")
            report.update(status="LOOKUP_FAILED", error=f"{type(error).__name__}: {error}")
            continue
        report["raw_returned"] = len(details)
        actual = {}
        rejected = Counter()
        examples_seen = set()
        for detail in details:
            reasons = definition_rejections(detail, expiry, chain.underlyingConId, spot)
            if reasons:
                report["rejected_definitions"] += 1
                rejected.update(reasons)
                signature = tuple(reasons)
                if signature not in examples_seen and len(report["rejected_examples"]) < 3:
                    report["rejected_examples"].append(definition_sample(detail, reasons))
                    examples_seen.add(signature)
                continue
            c = detail.contract
            strike = number(c.strike)
            if c.lastTradeDateOrContractMonth != expiry:
                report["normalized_expiry_count"] += 1
            actual.setdefault(strike, c)
        report.update(available=len(actual), rejection_counts=dict(rejected))
        if actual:
            report["status"] = "OK"
        elif not details:
            report["status"] = "EMPTY_CONTRACT_DETAILS"
        elif set(rejected) == {"STRIKE_OUTSIDE_PRICE_BAND"}:
            report["status"] = "NO_CONTRACTS_IN_PRICE_BAND"
        else:
            report["status"] = "ALL_CONTRACT_DETAILS_REJECTED"
        if not actual:
            notes.append(f"{expiry}: kontraktusfeloldás {report['status']}; "
                         f"nyers API-válasz={len(details)}, kizárási okok={dict(rejected)}. "
                         "Ez nem a teljes QQQ opciós lánc hiányának bizonyítéka.")
            continue
        # Use only strikes proven to exist on THIS expiry.
        from types import SimpleNamespace
        local_chain = SimpleNamespace(expirations={expiry}, strikes=set(actual))
        for spec in select_contract_specs(local_chain, spot, now, cfg):
            c = actual[spec["strike"]]
            conid = positive_id(c.conId)
            if conid in seen or len(options) >= cfg.max_contracts:
                continue
            option = api.Option("QQQ", expiry, spec["strike"], "P", "SMART",
                                multiplier="100", currency="USD", tradingClass="QQQ",
                                conId=conid,
                                localSymbol=str(getattr(c, "localSymbol", "") or ""))
            seen.add(conid)
            options.append(option)
            counts[expiry]["selected"] += 1
    return options, notes, counts


def wait_for_fields(ib: Any, tickers: list[Any], cfg: Config) -> float:
    """Wait at least the base interval, then bounded polling for missing fields.

    A timeout is not permission to impute OI/Greeks or relax any filter.
    """
    if not tickers:
        return 0.0
    waited = cfg.quote_wait_seconds
    ib.sleep(waited)
    while waited < cfg.max_quote_wait_seconds:
        stamp = pd.Timestamp.now(tz="UTC")
        rows = [quote_row(t, cfg, stamp) for t in tickers]
        complete = all(
            r["mid"] is not None and r["delta"] is not None
            and r["open_interest"] is not None
            and (not cfg.require_quote_sizes or all(
                r[k] is not None and r[k] >= 1 for k in ("bid_size", "ask_size")))
            for r in rows)
        if complete:
            break
        step = min(1.0, cfg.max_quote_wait_seconds-waited)
        ib.sleep(step)
        waited += step
    return waited


def quote_row(ticker: Any, cfg: Config, now: pd.Timestamp) -> dict[str, Any]:
    c = ticker.contract
    row = dict(
        conid=int(getattr(c, "conId", 0)), symbol=getattr(c, "symbol", ""),
        local_symbol=getattr(c, "localSymbol", ""), expiry=str(getattr(c, "lastTradeDateOrContractMonth", "") or "")[:8],
        strike=number(getattr(c, "strike", None)), right=getattr(c, "right", ""),
        multiplier=number(getattr(c, "multiplier", None)), currency=getattr(c, "currency", ""),
        trading_class=getattr(c, "tradingClass", ""),
        bid=number(getattr(ticker, "bid", None)), ask=number(getattr(ticker, "ask", None)),
        bid_size=number(getattr(ticker, "bidSize", None)), ask_size=number(getattr(ticker, "askSize", None)),
        open_interest=number(getattr(ticker, "putOpenInterest", None)),
        delta=None, gamma=None, vega=None, theta=None, iv=None, greek_underlier=None,
        greeks_source=None, collected_at=now.isoformat(),
        market_data_type_reported=getattr(ticker, "marketDataType", None),
        quote_source_timestamp=None, quote_time_quality="UNVERIFIED_SOURCE_TIME",
        oi_time_quality="ASOF_DATE_NOT_SUPPLIED", execution_allowed=False)
    for field in ("modelGreeks", "bidGreeks", "askGreeks"):
        greek = getattr(ticker, field, None)
        if greek is None:
            continue
        delta, iv, und = (number(getattr(greek, x, None)) for x in ("delta", "impliedVol", "undPrice"))
        if delta is not None and -1 < delta < 0 and iv is not None and iv > 0 and und is not None and und > 0:
            row.update(delta=delta, iv=iv, greek_underlier=und, greeks_source=field)
            for key in ("gamma", "vega", "theta"):
                row[key] = number(getattr(greek, key, None))
            break
    try:
        row["dte"] = (datetime.strptime(row["expiry"], "%Y%m%d").date()
                      - now.tz_convert("America/New_York").date()).days
    except ValueError:
        row["dte"] = None
    if row["open_interest"] is not None and row["open_interest"] < 0:
        row["open_interest"] = None
    bid, ask = row["bid"], row["ask"]
    valid_book = bid is not None and ask is not None and 0 < bid <= ask
    row["mid"] = (bid+ask)/2 if valid_book else None
    row["spread_usd"] = ask-bid if valid_book else None
    row["spread_fraction"] = (ask-bid)/row["mid"] if valid_book else None
    return row


def liquidity_rejections(row: dict[str, Any], spot: float, cfg: Config) -> list[str]:
    issues = []
    if (row["symbol"] != "QQQ" or row["right"] != "P" or row["currency"] != "USD"
            or row["multiplier"] != 100 or row["trading_class"] != "QQQ"
            or row["conid"] <= 0 or row["strike"] is None or row["strike"] <= 0):
        issues.append("NONSTANDARD_OR_INVALID_CONTRACT")
    if row["dte"] is None or not cfg.min_dte <= row["dte"] <= cfg.max_dte:
        issues.append("DTE_OUTSIDE_RANGE")
    if row["mid"] is None:
        issues.append("BID_ASK_MISSING_OR_INVALID")
    else:
        if row["spread_usd"] > cfg.max_spread_usd or row["spread_fraction"] > cfg.max_spread_fraction:
            issues.append("SPREAD_TOO_WIDE")
        if cfg.require_quote_sizes and any(row[k] is None or row[k] < 1 for k in ("bid_size", "ask_size")):
            issues.append("QUOTE_SIZE_MISSING")
    if row["open_interest"] is None:
        issues.append("OPEN_INTEREST_NA")
    elif row["open_interest"] != int(row["open_interest"]):
        issues.append("OPEN_INTEREST_INVALID")
    elif row["open_interest"] < cfg.min_open_interest:
        issues.append("OPEN_INTEREST_TOO_LOW")
    if row["delta"] is None or row["iv"] is None or row["greek_underlier"] is None:
        issues.append("GREEKS_OR_IV_NA")
    elif abs(row["greek_underlier"]/spot-1) > cfg.max_greek_underlier_difference_fraction:
        issues.append("GREEK_UNDERLIER_MISMATCH")
    if row["market_data_type_reported"] in (2, 4):
        issues.append("FROZEN_QUOTE")
    return issues


def screen(rows: list[dict[str, Any]], spot: float, cfg: Config) -> tuple[list, list]:
    longs, shorts = [], []
    for row in rows:
        issues = liquidity_rejections(row, spot, cfg)
        row["rejections"] = issues
        row["liquidity_greeks_pass"] = not issues
        row["long_delta_pass"] = row["delta"] is not None and cfg.long_delta_min <= abs(row["delta"]) <= cfg.long_delta_max
        row["short_delta_pass"] = row["delta"] is not None and cfg.short_delta_min <= abs(row["delta"]) <= cfg.short_delta_max
        row["long_cost_usd"] = (row["ask"]*100+cfg.assumed_entry_fee_usd_per_leg
                                if row["ask"] is not None and row["ask"] > 0 else None)
        row["long_budget_pass"] = row["long_cost_usd"] is not None and row["long_cost_usd"] <= cfg.premium_budget_usd
        row["long_rejections"] = (
            ([] if row["long_delta_pass"] else ["LONG_DELTA_OUTSIDE_TARGET_BAND"])
            + ([] if row["long_budget_pass"] else ["LONG_PREMIUM_OVER_BUDGET_OR_NA"]))
        row["long_expiry_breakeven"] = (
            row["strike"]-row["long_cost_usd"]/100
            if row["strike"] is not None and row["long_cost_usd"] is not None else None)
        if issues:
            continue
        if row["long_delta_pass"]:
            # Keep over-budget long legs for possible cheaper spreads.
            longs.append(row)
        if row["short_delta_pass"]:
            shorts.append(row)
    return longs, shorts


def build_spreads(longs: list[dict], shorts: list[dict], cfg: Config,
                  limit: int | None = 5) -> list[dict]:
    results = []
    for long in longs:
        for short in shorts:
            if (long["expiry"] != short["expiry"] or long["strike"] <= short["strike"]
                    or long["conid"] == short["conid"]):
                continue
            width = long["strike"]-short["strike"]
            debit = long["ask"]-short["bid"]
            skew = abs((utc(long["collected_at"])-utc(short["collected_at"])).total_seconds())
            if skew > cfg.max_leg_collection_skew_seconds:
                continue
            if not 0 < debit < width:
                continue
            fee = 2*cfg.assumed_entry_fee_usd_per_leg
            cost = debit*100+fee
            if cost > cfg.premium_budget_usd or cost >= width*100:
                continue
            results.append(dict(
                expiry=long["expiry"], dte=long["dte"],
                long_conid=long["conid"], short_conid=short["conid"],
                long_strike=long["strike"], short_strike=short["strike"],
                long_delta=long["delta"], short_delta=short["delta"],
                indicative_debit=debit, width=width,
                opening_cost_usd=cost, theoretical_max_loss_usd=cost,
                theoretical_max_profit_usd=width*100-cost,
                expiry_breakeven=long["strike"]-debit-fee/100,
                net_delta=long["delta"]-short["delta"],
                score=abs(abs(long["delta"])-cfg.target_abs_delta)*10+
                      abs(long["dte"]-cfg.target_dte)*.05+
                      long["spread_fraction"]+short["spread_fraction"],
                quote_quality="UNVERIFIED_SOURCE_TIME_AND_COMBO_NOT_QUOTED",
                collection_time_skew_seconds=skew,
                opening_fee_usd=fee,
                long_ask_cost_usd=long["ask"]*100,
                short_bid_credit_usd=short["bid"]*100,
                net_gamma=(long["gamma"]-short["gamma"]
                           if long.get("gamma") is not None and short.get("gamma") is not None else None),
                net_theta=(long["theta"]-short["theta"]
                           if long.get("theta") is not None and short.get("theta") is not None else None),
                net_vega=(long["vega"]-short["vega"]
                          if long.get("vega") is not None and short.get("vega") is not None else None),
                long_leg=dict(long), short_leg=dict(short),
                execution_allowed=False))
    ordered = sorted(results, key=lambda x: (x["score"], x["opening_cost_usd"]))
    return ordered if limit is None else ordered[:limit]


def evaluate(rows: list[dict], underlying: dict, packet: dict | None,
             now: pd.Timestamp, cfg: Config) -> dict[str, Any]:
    underlying = dict(underlying)
    try:
        final_age = (utc(now)-utc(underlying.get("asof"))).total_seconds()/60
        underlying["age_minutes"] = final_age
        if final_age < 0 or final_age > cfg.max_source_age_minutes:
            underlying.update(quality="FUTURE" if final_age < 0 else "STALE",
                              paper_price_gate=False)
    except (TypeError, ValueError):
        underlying.update(quality="MISSING", paper_price_gate=False)
    gate = advisory_gate(packet, now, cfg)
    spot = number(underlying.get("price"))
    result = dict(version=VERSION, observed_at=now.isoformat(), config=asdict(cfg),
                  state="DATA_UNAVAILABLE", underlying=underlying, regime_gate=gate,
                  options=rows, paper_long_put_candidates=[], paper_spread_candidates=[],
                  diagnostic_long_puts=[], diagnostic_spreads=[],
                  rejection_counts={}, execution_allowed=False, notes=[],
                  blockers=[], field_coverage={})
    if spot is None or spot <= 0:
        result["notes"].append("Nincs használható QQQ történeti ár; opciószűrés nem értékelhető.")
        result["blockers"].append("Nincs használható QQQ történeti ár.")
        if gate["quality"] != "OK":
            result["blockers"].append(f"Index advisory adatminőség: {gate['quality']}.")
        return result
    longs, shorts = screen(rows, spot, cfg)
    long_reviews = [r for r in longs if r["long_budget_pass"]]
    long_reviews.sort(key=lambda r: (
        abs(abs(r["delta"])-cfg.target_abs_delta)*10+abs(r["dte"]-cfg.target_dte)*.05+r["spread_fraction"],
        r["long_cost_usd"]))
    spreads = build_spreads(longs, shorts, cfg, limit=None)
    result.update(diagnostic_long_puts=long_reviews, diagnostic_spreads=spreads,
                  rejection_counts=dict(Counter(
                      reason for row in rows for reason in row["rejections"]+row["long_rejections"])))
    aligned = gate["asof"] is not None and underlying.get("asof") is not None and abs(
        (utc(gate["asof"])-utc(underlying["asof"])).total_seconds()/60) <= cfg.max_context_skew_minutes
    ready = (gate["paper_allowed"] and underlying.get("quality") == "OK"
             and underlying.get("paper_price_gate", False) and aligned)
    if ready:
        if gate["structure"] == "QQQ_LONG_PUT_OR_DEBIT_SPREAD":
            result["paper_long_put_candidates"] = long_reviews[:5]
        result["paper_spread_candidates"] = spreads[:5]
        result["state"] = ("PAPER_CANDIDATES_UNVERIFIED_QUOTES"
                           if result["paper_long_put_candidates"] or spreads else "NO_OPTION_MATCH")
    elif gate["quality"] != "OK":
        result["state"] = "REGIME_INPUT_NA_OR_STALE"
    elif not gate["paper_allowed"]:
        result["state"] = "REGIME_BLOCKS_NEW_PUT"
    else:
        result["state"] = "QQQ_PRICE_OR_TIME_ALIGNMENT_BLOCKS_NEW_PUT"
    result["notes"] += [
        "Minden jelölt kizárólag ellenőrizendő PAPER jelölt; nem végrehajtási ajánlat.",
        "Ticker.time/lekérési idő nem a bid/ask/Greeks tőzsdei forrásideje.",
        "Az OI dátumát ez az API-válasz nem igazolja; az OI nem valós idejű likviditásmérő.",
        "106-os 30 napos underlying IV-t nem használunk strike-specifikus IV helyett.",
        "Spread: long ask - short bid, nem végrehajtható combo jegyzés és nem garantált teljesülés.",
        "A költség egy kontraktusra számolt, becsült nyitási díjjal; zárási díj/slippage nincs benne.",
        "QQQ opciók részvényleszállításosak; exercise/assignment és pin-risk külön ellenőrzendő.",
        "Nem használ számlaegyenleget/pozíciót a szűréshez és nem kezeli a meglévő novemberi opcióidat.",
    ]
    if not aligned:
        result["notes"].append("A QQQ és az index forrásidőpontja nem megfelelően illesztett vagy hiányzik.")
    blockers = []
    if gate["quality"] != "OK":
        blockers.append(f"Index advisory hiányzik/hibás/elavult: {gate['quality']}.")
    elif not gate["paper_allowed"]:
        blockers.append(f"Index paper kapu zárt: {gate['state']}.")
    if underlying.get("quality") != "OK":
        blockers.append(f"QQQ core adatminőség: {underlying.get('quality', 'N/A')}.")
    if underlying.get("below_rth_vwap") is False:
        blockers.append("QQQ nincs a saját RTH VWAP alatt.")
    if not underlying.get("paper_price_gate", False) and underlying.get("below_rth_vwap") is not False:
        blockers.append("QQQ RTH/session árkapu nem igazolt.")
    if not aligned:
        blockers.append("QQQ/index forrásidő-illesztés nem igazolt.")
    if not rows:
        requested = underlying.get("chain", {}).get("requested_contract_count", 0)
        if requested:
            blockers.append("Ellenőrzött kontraktus van, de egyetlen opciós jegyzéssor sem gyűlt össze.")
        else:
            blockers.append("Nincs jegyzésig eljutott opciós kontraktus: a kontraktusfeloldást kell vizsgálni. "
                            "A delta/OI/spread/prémiumszűrők még nem értékelhetők.")
    elif not long_reviews:
        blockers.append("A vizsgált kontraktusokból nincs minden long-put szűrőn átjutó találat.")
    if rows and not spreads:
        blockers.append("A vizsgált kontraktusokból nincs minden spread-szűrőn átjutó pár.")
    result["blockers"] = blockers
    result["field_coverage"] = dict(
        quoted_contracts=len(rows),
        valid_bid_ask=sum(r["mid"] is not None for r in rows),
        greeks_iv=sum(r["delta"] is not None and r["iv"] is not None for r in rows),
        oi_received=sum(r["open_interest"] is not None for r in rows),
        oi_threshold_pass=sum(r["open_interest"] is not None
                              and r["open_interest"] >= cfg.min_open_interest for r in rows),
        long_leg_matches=len(longs), short_leg_matches=len(shorts),
        budget_long_matches=len(long_reviews), spread_matches=len(spreads))
    return result


def collect(ib: Any, api: Any, cfg: Config) -> tuple[dict, list[dict], list[str]]:
    """Bounded read-only requests. No paid regulatory snapshots."""
    active = []
    notes = []
    api_errors = []
    event = getattr(ib, "errorEvent", None)
    def on_error(req_id, code, message, contract=None):
        if code in (200, 354, 10089, 10090, 10091, 10167, 10168):
            api_errors.append(dict(code=int(code), request_id=req_id,
                                   conid=getattr(contract, "conId", None),
                                   message=str(message)))
    if event is not None:
        event += on_error
    try:
        stock = api.Stock("QQQ", "SMART", "USD", primaryExchange="NASDAQ")
        qualified = ib.qualifyContracts(stock)
        if not qualified or stock.conId <= 0:
            raise RuntimeError("QQQ Stock nem validálható.")
        active.append(stock)
        ib.reqMktData(stock, "", False, False)
        bars = ib.reqHistoricalData(
            stock, endDateTime="", durationStr="2 D", barSizeSetting="1 min",
            whatToShow="TRADES", useRTH=True, formatDate=2, timeout=30)
        now = pd.Timestamp.now(tz="UTC")
        underlying = underlying_context(api.util.df(bars), now, cfg)
        if underlying["price"] is None:
            return underlying, [], ["QQQ történeti ár N/A; chain-scan kihagyva."]
        chains = ib.reqSecDefOptParams("QQQ", "", "STK", stock.conId)
        chain = select_chain(chains, stock.conId)
        if chain is None:
            return underlying, [], ["Nincs standard QQQ/100 opciós chain."]
        options, discovery_notes, counts = discover_contracts(
            ib, api, chain, underlying["price"], now, cfg)
        notes += discovery_notes
        underlying["chain"] = dict(exchange=chain.exchange, trading_class=chain.tradingClass,
                                  expiry_count=len(chain.expirations), strike_count=len(chain.strikes),
                                  selected_expiries=list(counts),
                                  requested_contract_count=len(options),
                                  discovery_by_expiry=counts,
                                  discovery="EXPIRY_SCOPED_CONTRACT_DETAILS")
        rows = []
        waits = []
        for start in range(0, len(options), cfg.batch_size):
            batch = options[start:start+cfg.batch_size]
            tickers = []
            subscribed = []
            try:
                for option in batch:
                    try:
                        ticker = ib.reqMktData(option, "101", False, False)
                        active.append(option)
                        subscribed.append(option)
                        tickers.append(ticker)
                    except Exception as error:
                        notes.append(f"Opciós adat N/A, conId={option.conId}: {error}")
                waits.append(wait_for_fields(ib, tickers, cfg))
                stamp = pd.Timestamp.now(tz="UTC")
                for t in tickers:
                    row = quote_row(t, cfg, stamp)
                    row["api_error_codes"] = sorted({
                        e["code"] for e in api_errors if e["conid"] == row["conid"]})
                    rows.append(row)
            finally:
                for option in subscribed:
                    try:
                        ib.cancelMktData(option)
                    except Exception:
                        pass
                    active.remove(option)
        underlying["qualified_quote_count"] = len(rows)
        underlying["collection"] = dict(
            api_error_counts=dict(Counter(e["code"] for e in api_errors)),
            batch_wait_seconds=waits,
            subscription_warning=any(e["code"] in (354, 10089, 10090, 10091) for e in api_errors))
        if underlying["collection"]["subscription_warning"]:
            notes.append("API előfizetés/jogosultság figyelmeztetés érkezett. "
                         "A visszakapott mezőket külön értékeljük; ez nem teljes feed-kiesés.")
        notes.append("Korlátozott kontraktusminta: a nincs találat nem jelenti a teljes chain vizsgálatát.")
        return underlying, rows, list(dict.fromkeys(notes))
    finally:
        if event is not None:
            event -= on_error
        for contract in active:
            try:
                ib.cancelMktData(contract)
            except Exception:
                pass


def fmt(value: Any, digits: int = 2) -> str:
    value = number(value)
    return f"{value:.{digits}f}" if value is not None else "N/A"


PARAMETER_NAMES = {
    "conid": "Kontraktus conId", "local_symbol": "Tőzsdei opcióazonosító",
    "symbol": "Alaptermék", "expiry": "Lejárat", "dte": "DTE (naptári nap)",
    "strike": "Strike (USD)", "right": "P/C", "currency": "Deviza",
    "multiplier": "Kontraktusszorzó", "trading_class": "Trading class",
    "bid": "Bid (USD/részvény)", "ask": "Ask (USD/részvény)", "mid": "Mid (USD/részvény)",
    "bid_size": "Bid méret (API)", "ask_size": "Ask méret (API)",
    "spread_usd": "Láb bid/ask különbség (USD)", "spread_fraction": "Láb bid/ask különbség (%)",
    "delta": "Delta", "gamma": "Gamma", "theta": "Theta", "vega": "Vega",
    "iv": "IV (%)", "open_interest": "Open interest (OI)",
    "greeks_source": "Greeks forrás", "greek_underlier": "Greeks mögötti underlying ár",
    "market_data_type_reported": "Jelzett adatmód (1/2/3/4)",
    "collected_at": "Begyűjtés ideje (nem forrásidő)",
    "quote_source_timestamp": "Jegyzés igazolt forrásideje",
    "quote_time_quality": "Jegyzés időminősége", "oi_time_quality": "OI időminősége",
    "api_error_codes": "API hibakódok",
    "rejections": "Likviditási/Greeks kizárási okok",
    "liquidity_greeks_pass": "Likviditás/Greeks szűrők",
    "long_delta_pass": "Önálló LONG delta-szűrő",
    "short_delta_pass": "Spread SHORT delta-szűrő",
    "long_budget_pass": "Önálló LONG keretszűrő",
    "long_rejections": "Önálló LONG kizárási okok",
    "long_cost_usd": "Önálló LONG nyitási költség (USD)",
    "long_expiry_breakeven": "Önálló LONG lejárati breakeven",
    "long_conid": "LONG conId", "short_conid": "SHORT conId",
    "long_strike": "LONG strike", "short_strike": "SHORT strike",
    "long_delta": "LONG delta", "short_delta": "SHORT delta",
    "indicative_debit": "Indikatív debit (USD/részvény)",
    "width": "Strike-szélesség (USD/részvény)",
    "opening_cost_usd": "Nyitási nettó költség (USD)",
    "theoretical_max_loss_usd": "Elméleti max. veszteség (USD)",
    "theoretical_max_profit_usd": "Elméleti max. profit (USD)",
    "expiry_breakeven": "Lejárati breakeven (USD)",
    "opening_fee_usd": "Feltételezett nyitási díj (USD)",
    "long_ask_cost_usd": "LONG ask x 100 (díj nélkül)",
    "short_bid_credit_usd": "SHORT bid x 100 (díj nélkül)",
    "net_delta": "Nettó delta (LONG - SHORT)",
    "net_gamma": "Nettó gamma (LONG - SHORT)",
    "net_theta": "Nettó theta (LONG - SHORT)",
    "net_vega": "Nettó vega (LONG - SHORT)",
    "score": "Rangsorolási score (nem esély)",
    "quote_quality": "Jegyzés minősége",
    "collection_time_skew_seconds": "Lábak begyűjtési eltérése (mp)",
    "execution_allowed": "Valós végrehajtás engedélyezett",
}


def diagnostic_status(result: dict, kind: str) -> str:
    gate = result["regime_gate"]
    if gate.get("quality") != "OK":
        return "INDEXADAT MIATT NEM BELÉPHETŐ"
    if not gate.get("paper_allowed"):
        return "REZSIM MIATT NEM BELÉPHETŐ"
    if result.get("state") == "QQQ_PRICE_OR_TIME_ALIGNMENT_BLOCKS_NEW_PUT":
        return "QQQ/IDŐKAPU MIATT NEM BELÉPHETŐ"
    if kind == "LONG" and gate.get("structure") != "QQQ_LONG_PUT_OR_DEBIT_SPREAD":
        return "STRUKTÚRA MIATT NEM BELÉPHETŐ"
    return "CSAK DIAGNOSZTIKA; NEM VÉGREHAJTHATÓ"


def parameter_value(key: str, value: Any) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return "N/A"
    if isinstance(value, bool):
        return "IGEN" if value else "NEM"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if key in ("iv", "spread_fraction") and number(value) is not None:
        return f"{float(value)*100:.6g}% (nyers: {float(value):.12g})"
    if isinstance(value, float):
        return f"{value:.10g}" if math.isfinite(value) else "N/A"
    return str(value)


def console_table(headers: list[str], rows: list[list[Any]], widths: list[int]) -> list[str]:
    """ASCII borders, aligned and wrapped cells; no external console dependency."""
    border = "+"+"+".join("-"*(w+2) for w in widths)+"+"
    lines = [border]
    for i, row in enumerate([headers, *rows]):
        cells = [textwrap.wrap(str(v).replace("\n", " "), w, break_long_words=True,
                               break_on_hyphens=False) or [""]
                 for v,w in zip(row,widths)]
        for j in range(max(map(len,cells))):
            lines.append("| "+" | ".join((cell[j] if j<len(cell) else "").ljust(w)
                                          for cell,w in zip(cells,widths))+" |")
        if i == 0:
            lines.append(border)
    lines.append(border)
    return lines


def render_diagnostics(result: dict[str, Any]) -> list[str]:
    """Every retained match, with every collected parameter, never entry permission."""
    longs = result.get("diagnostic_long_puts", [])
    spreads = result.get("diagnostic_spreads", [])
    gate, u, cfg = result["regime_gate"], result["underlying"], result["config"]
    lines = ["", "="*116, "KÜLÖN DIAGNOSZTIKAI TALÁLATOK | NEM BELÉPÉSI UTASÍTÁS",
             f"{len(longs)} long / {len(spreads)} spread; minden szűrőtalálat a lekért mintából, nem a teljes chain.",
             f"Index: {gate.get('state')} | minőség: {gate.get('quality')} | "
             f"paper kapu: {'NYITOTT' if gate.get('paper_allowed') else 'ZÁRT'}",
             f"QQQ: {fmt(u.get('price'))} | QQQ áridő: {u.get('asof') or 'N/A'} | "
             f"VWAP: {fmt(u.get('rth_vwap'))} | index áridő: {gate.get('asof') or 'N/A'}",
             f"Keret: {fmt(cfg['premium_budget_usd'])} USD | nyitási díj/láb: "
             f"{fmt(cfg['assumed_entry_fee_usd_per_leg'])} USD | 1 kontraktus / 1 spread.",
             f"LONG delta abs.: {cfg['long_delta_min']}..{cfg['long_delta_max']} | "
             f"SHORT delta abs.: {cfg['short_delta_min']}..{cfg['short_delta_max']} | "
             f"OI minimum: {cfg['min_open_interest']} | max. bid/ask: "
             f"{cfg['max_spread_usd']} USD és {cfg['max_spread_fraction']*100:g}%.",
             "Árak/Greeks/OI forrásideje nem igazolt; adatmód 3 = delayed, nem élő végrehajtási ár.",
             "Spread: LONG ask - SHORT bid; nincs combo jegyzés és nincs garantált teljesülés.",
             "Greeks nettó = LONG - SHORT, az API egységében; nem USD-re konvertált kontraktusérték.",
             "Konzol: olvashatóságért kerekített értékek; a JSON őrzi a numerikus mezőket.",
             "Max. veszteség/profit: elméleti lejárati payoff, feltételezett nyitási díjjal; "
             "zárási díj, slippage és exercise/assignment-kockázat külön kezelendő.",
             "SHORT láb önálló LONG-szűrője lehet NEM: ez nem a SHORT láb kizárása."]
    lines += [f"Tiltás/akadály: {text}" for text in result.get("blockers", [])]
    lines += [f"Rezsim indok: {text}" for text in gate.get("reasons", [])]
    lines += ["", "DIAGNOSZTIKAI LONG PUT TALÁLATOK"]
    long_status = diagnostic_status(result, "LONG")
    if not longs:
        lines.append("Nincs minden önálló long-put opciószűrőn átjutó diagnosztikai találat.")
    else:
        lines += console_table(
            ["#", "Lejárat", "DTE", "Strike", "Bid / ask", "Delta", "Költség USD", "Státusz"],
            [[i,r.get("expiry"),r.get("dte"),fmt(r.get("strike")),
              f"{fmt(r.get('bid'))}/{fmt(r.get('ask'))}",fmt(r.get("delta"),4),
              fmt(r.get("long_cost_usd")),long_status] for i,r in enumerate(longs,1)],
            [3,8,3,8,13,8,11,34])
        for i, row in enumerate(longs,1):
            lines += ["", f"LONG #{i} | {long_status}",
                      *console_table(["Paraméter", "Érték"],
                                     [[PARAMETER_NAMES.get(k,k),parameter_value(k,v)]
                                      for k,v in row.items()], [39,72])]
    lines += ["", "DIAGNOSZTIKAI PUT DEBIT SPREAD TALÁLATOK"]
    spread_status = diagnostic_status(result, "SPREAD")
    if not spreads:
        lines.append("Nincs minden spread-szűrőn átjutó diagnosztikai pár.")
    else:
        lines += console_table(
            ["#", "Lejárat", "DTE", "LONG / SHORT", "Debit", "Max.loss USD", "Max.profit USD", "Státusz"],
            [[i,s.get("expiry"),s.get("dte"),
              f"{fmt(s.get('long_strike'))}/{fmt(s.get('short_strike'))}",
              fmt(s.get("indicative_debit")),fmt(s.get("theoretical_max_loss_usd")),
              fmt(s.get("theoretical_max_profit_usd")),spread_status] for i,s in enumerate(spreads,1)],
            [3,8,3,15,7,12,14,34])
        # Saved old results may contain only leg conIds; resolve without guessing.
        quotes = {r.get("conid"):r for r in result.get("options", [])}
        for i,s in enumerate(spreads,1):
            lines += ["", f"SPREAD #{i} | {spread_status}",
                      *console_table(["Spread paraméter", "Érték"],
                                     [[PARAMETER_NAMES.get(k,k),parameter_value(k,v)]
                                      for k,v in s.items() if k not in ("long_leg","short_leg")],
                                     [39,72])]
            long = s.get("long_leg") or quotes.get(s.get("long_conid"), {})
            short = s.get("short_leg") or quotes.get(s.get("short_conid"), {})
            keys = list(dict.fromkeys([*long,*short]))
            if keys:
                lines += [f"SPREAD #{i} LÁBAK | LONG = VÉTEL | SHORT = ELADÁS",
                          *console_table(["Láb paraméter", "LONG (vétel)", "SHORT (eladás)"],
                                         [[PARAMETER_NAMES.get(k,k),parameter_value(k,long.get(k)),
                                           parameter_value(k,short.get(k))] for k in keys], [39,35,35])]
            else:
                lines.append("ADATHIÁNY: a spread lábainak részletes jegyzése nem áll rendelkezésre.")
    lines += ["", "DIAGNOSZTIKA VÉGE | VÉGREHAJTÁSI ENGEDÉLY: NEM", "="*116]
    return lines


def show(result: dict[str, Any]) -> None:
    u, gate = result["underlying"], result["regime_gate"]
    print("\n"+"="*100)
    print(f"QQQ OPTIONS ADVISOR {VERSION} | PAPER / READ-ONLY")
    print(f"ÖSSZESÍTETT STÁTUSZ: {result['state']}")
    print("Ez a fő státusz; az alábbi akadályok egyszerre is fennállhatnak.")
    for blocker in result.get("blockers", []):
        print(f"  AKADÁLY: {blocker}")
    if result.get("advisory_path"):
        print(f"Index JSON elvárt helye: {result['advisory_path']}")
    if gate["quality"] == "MISSING":
        print("Teendő: run_es_nq_checklist.py 1.2.1 vagy újabb, sikeres futás és JSON-export.")
        print("Ha a JSON más könyvtárban van: --advisory-json \"teljes_elérési_út\".")
    print(f"QQQ: {fmt(u['price'])} | áridő: {u.get('asof') or 'N/A'} | adatminőség: {u['quality']}")
    below = u.get("below_rth_vwap")
    print(f"QQQ RTH VWAP: {fmt(u.get('rth_vwap'))} | Alatta: {'N/A' if below is None else 'IGEN' if below else 'NEM'}")
    print(f"Index advisory: {gate['state']} | minőség: {gate['quality']} | forrásidő: {gate['asof'] or 'N/A'}")
    print(f"Paper rezsimkapu: {'NYITOTT' if gate['paper_allowed'] else 'ZÁRT'} | struktúra: {gate['structure']}")
    print(f"Keretszűrő: {result['config']['premium_budget_usd']:.2f} USD / 1 kontraktus vagy 1 spread")
    print(f"Kizárási okok darabszámai: {result['rejection_counts']}")
    coverage = result.get("field_coverage", {})
    if coverage:
        print(f"Adatlefedettség: {coverage['quoted_contracts']} kontraktus | "
              f"bid/ask: {coverage['valid_bid_ask']} | Greeks/IV: {coverage['greeks_iv']} | "
              f"OI érkezett: {coverage['oi_received']} | OI >= minimum: {coverage['oi_threshold_pass']}")
        print(f"Lábszűrés: long delta + likviditás: {coverage['long_leg_matches']} | "
              f"ebből long kereten belül: {coverage['budget_long_matches']} | "
              f"short delta + likviditás: {coverage['short_leg_matches']} | "
              f"kereten belüli spreadpár: {coverage['spread_matches']}")
    collection = u.get("collection", {})
    if collection:
        print(f"API hibakódok (darab): {collection['api_error_counts']} | "
              f"batch várakozás (mp): {collection['batch_wait_seconds']}")
    for expiry, discovery in u.get("chain", {}).get("discovery_by_expiry", {}).items():
        print(f"Kontraktusfeloldás {expiry}: {discovery['status']} | "
              f"nyers API-válasz: {discovery.get('raw_returned', 'N/A')} | "
              f"ársávban ellenőrzött: {discovery['available']} | kiválasztott: {discovery['selected']}")
        print(f"  Elvárt underlying conId: {discovery.get('expected_underlying_conid', 'N/A')} | "
              f"strike-sáv: {discovery.get('strike_band', 'N/A')} | "
              f"kizárt definíció: {discovery.get('rejected_definitions', 'N/A')}")
        if discovery.get("error"):
            print(f"  Feloldási hiba: {discovery['error']}")
        if discovery.get("rejection_counts"):
            print(f"  Definíciós kizárási okok: {discovery['rejection_counts']}")
        for sample in discovery.get("rejected_examples", []):
            c = sample["contract"]
            print(f"  Kizárt példa: conId={c['conId']} | "
                  f"underConId={sample['under_conid']} | "
                  f"lejárat={c['lastTradeDateOrContractMonth']!r} | "
                  f"realExpirationDate={sample['real_expiration_date']!r} | "
                  f"strike={c['strike']} | class={c['tradingClass']!r} | "
                  f"szorzó={c['multiplier']!r} | ok={sample['reasons']}")
    if not result["options"]:
        print("Nincs opciós jegyzéssor: az üres kizáráslista NEM azt jelenti, hogy a szűrők teljesültek.")
        print("A nulla bid/ask/Greeks/OI itt még nem bizonyít előfizetési hiányt.")
    print("A visszakapott mező nem igazolja a forrásidőt vagy a valós idejű jogosultságot.")
    print("Lejárat: DTE = naptári nap. Delta: negatív putdelta, cél abszolút delta.")
    print("-"*100)
    print("LEKÉRT PUTOK: KIJELZÉS, NEM BELÉPÉSI UTASÍTÁS")
    print("LEJÁRAT | DTE | STRIKE | BID / ASK | DELTA | IV % | OI | SPREAD % | SZŰRÉS")
    for row in result["options"]:
        statuses = row.get("rejections", []) or (["LIQUIDITY/GREEKS OK"]+row.get("long_rejections", []))
        print(f"{row['expiry']} | {row['dte']} | {fmt(row['strike'])} | "
              f"{fmt(row['bid'])}/{fmt(row['ask'])} | {fmt(row['delta'], 3)} | "
              f"{fmt(row['iv']*100 if row['iv'] is not None else None)} | "
              f"{fmt(row['open_interest'], 0)} | "
              f"{fmt(row['spread_fraction']*100 if row['spread_fraction'] is not None else None)} | "
              f"{','.join(statuses)}")
        print(f"  Greeks forrás: {row['greeks_source'] or 'N/A'} | gamma={fmt(row['gamma'], 4)} | "
              f"theta={fmt(row['theta'], 4)} | vega={fmt(row['vega'], 4)} | "
              f"Greeks underlying={fmt(row['greek_underlier'])}")
        short_pass = row.get("liquidity_greeks_pass", False) and row["delta"] is not None and (
            result["config"]["short_delta_min"] <= abs(row["delta"]) <= result["config"]["short_delta_max"])
        print(f"  LONG láb delta: {'OK' if row.get('long_delta_pass') else 'NEM'} | "
              f"LONG költség: {fmt(row.get('long_cost_usd'))} USD | "
              f"keret: {'OK' if row.get('long_budget_pass') else 'NEM'} | "
              f"spread SHORT láb összes szűrő: {'OK' if short_pass else 'NEM'} | "
              f"jelzett adatmód: {row.get('market_data_type_reported', 'N/A')} | "
              f"API kódok: {row.get('api_error_codes', [])}")
    for title, candidates in (
        ("PAPER LONG PUT JELÖLTEK (forrásidő nem igazolt)", result["paper_long_put_candidates"]),
        ("PAPER DEBIT SPREAD JELÖLTEK (combo nincs jegyezve)", result["paper_spread_candidates"]),
    ):
        print("\n"+title)
        if not candidates:
            print("Nincs megfelelő jelölt.")
        for candidate in candidates:
            if "long_strike" in candidate:
                print(f"{candidate['expiry']} | {candidate['long_strike']:g}/{candidate['short_strike']:g} PUT | "
                      f"indikatív debit={candidate['indicative_debit']:.2f} | "
                      f"nyitási költség={candidate['opening_cost_usd']:.2f} USD | "
                      f"elméleti maxprofit={candidate['theoretical_max_profit_usd']:.2f} USD | "
                      f"nettó delta={candidate['net_delta']:.3f}")
            else:
                print(f"{candidate['local_symbol']} | delta={candidate['delta']:.3f} | "
                      f"ask-alapú költség/max prémiumvesztés={candidate['long_cost_usd']:.2f} USD | "
                      f"lejárati breakeven={candidate['long_expiry_breakeven']:.2f}")
    if not result["paper_long_put_candidates"] and not result["paper_spread_candidates"]:
        print(f"\nCsak diagnosztikai szűrőtalálat: {len(result['diagnostic_long_puts'])} long / "
              f"{len(result['diagnostic_spreads'])} spread. Ezek NEM entry-jelöltek.")
    print("\n".join(render_diagnostics(result)))
    for reason in gate["reasons"]:
        print(f"- Rezsim: {reason}")
    for note in result["notes"]:
        print(f"- {note}")
    print("Valós végrehajtási engedély: NEM. Orderküldés / exercise / pozíciókezelés: NINCS.")
    print("="*100, flush=True)


def save_result(result: dict[str, Any], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=folder, delete=False) as file:
            name = file.name
            json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(name, folder/"qqq_options_latest.json")
    finally:
        if name and os.path.exists(name):
            os.unlink(name)
    export = []
    for row in result["options"]:
        record = dict(row)
        record["rejections"] = ";".join(record.get("rejections", []))
        record["long_rejections"] = ";".join(record.get("long_rejections", []))
        export.append(record)
    if export:
        pd.DataFrame(export).to_csv(folder/"qqq_options_quotes.csv", index=False)
    else:
        pd.DataFrame(columns=["conid", "expiry", "strike", "bid", "ask", "delta", "iv",
                              "open_interest", "rejections"]).to_csv(
                                  folder/"qqq_options_quotes.csv", index=False)
    # Export the same diagnostic lists, including the block label and full legs.
    # Empty runs overwrite old diagnostic CSVs instead of retaining stale matches.
    for key, kind in (("diagnostic_long_puts","LONG"), ("diagnostic_spreads","SPREAD")):
        records = []
        for rank, candidate in enumerate(result.get(key, []), 1):
            record = dict(diagnostic_rank=rank, diagnostic_status=diagnostic_status(result,kind),
                          regime_state=result["regime_gate"].get("state"))
            for k,v in candidate.items():
                if k in ("long_leg","short_leg"):
                    for leg_key,leg_value in v.items():
                        record[f"{k}_{leg_key}"] = parameter_value(leg_key,leg_value) if isinstance(
                            leg_value,(dict,list)) else leg_value
                else:
                    record[k] = parameter_value(k,v) if isinstance(v,(dict,list)) else v
            records.append(record)
        frame = pd.DataFrame(records) if records else pd.DataFrame(
            columns=["diagnostic_rank","diagnostic_status","regime_state","expiry","execution_allowed"])
        frame.to_csv(folder/f"qqq_options_{key}.csv",index=False)
    (folder/"qqq_options_diagnostics.txt").write_text(
        "\n".join(render_diagnostics(result))+"\n",encoding="utf-8")


def demo_result(cfg: Config) -> dict[str, Any]:
    """Synthetic examples, never replace a real latest report."""
    from types import SimpleNamespace
    now = pd.Timestamp("2026-10-13T15:30:00Z")
    source = now-pd.Timedelta(minutes=15)
    packet = dict(schema_version=1, source="ES_NQ_REGIME_RUNNER", observed_at=now.isoformat(),
                  asof=source.isoformat(), execution_allowed=False,
                  indices={s: dict(asof=source.isoformat(), data_quality="OK",
                                   short_overlay_blocked=False) for s in ("ES", "NQ")},
                  advisory=dict(state="QQQ_PUT_CANDIDATE", entry_scope="PAPER_ONLY",
                                execution_allowed=False, entry_allowed=True,
                                trade_candidate=True, recommended_structure="QQQ_PUT_DEBIT_SPREAD_PREFERRED",
                                reasons=["SZINTETIKUS DEMO, nem piaci rezsim."]))
    underlying = dict(price=600, asof=source.isoformat(), quality="OK",
                      rth_vwap=603, below_rth_vwap=True, paper_price_gate=True)
    expiry = (now+pd.Timedelta(days=cfg.target_dte)).strftime("%Y%m%d")
    rows = []
    for conid, strike, bid, ask, delta in [(1, 600, 4.1, 4.2, -.45), (2, 590, 1.5, 1.6, -.20)]:
        c = SimpleNamespace(conId=conid, symbol="QQQ", localSymbol=f"DEMO-{conid}",
                            lastTradeDateOrContractMonth=expiry, strike=strike, right="P",
                            multiplier="100", currency="USD", tradingClass="QQQ")
        g = SimpleNamespace(delta=delta, impliedVol=.30, undPrice=600, gamma=.01, vega=.2, theta=-.1)
        t = SimpleNamespace(contract=c, bid=bid, ask=ask, bidSize=10, askSize=10,
                            putOpenInterest=500, modelGreeks=g, marketDataType=3)
        rows.append(quote_row(t, cfg, now))
    result = evaluate(rows, underlying, packet, now, cfg)
    result["state"] = "DEMO_"+result["state"]
    result["data_mode"] = "SYNTHETIC_DEMO_NOT_MARKET_DATA"
    result["notes"].insert(0, "DEMO: minden ár, Greeks, OI és rezsim szintetikus, NEM jelenlegi piaci adat.")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QQQ put/debit spread paper scanner; no orders.")
    parser.add_argument("--budget", type=float, default=None,
                        help="Override Config.premium_budget_usd only when explicitly supplied.")
    parser.add_argument("--target-dte", type=int, default=None)
    parser.add_argument("--target-delta", type=float, default=None)
    parser.add_argument("--min-oi", type=int, default=None)
    parser.add_argument("--quote-wait", type=float, default=None,
                        help="Minimum streaming wait per batch, seconds.")
    parser.add_argument("--max-quote-wait", type=float, default=None,
                        help="Maximum bounded wait per batch for missing OI/Greeks.")
    parser.add_argument("--demo", action="store_true", help="Synthetic offline demo, no TWS.")
    parser.add_argument("--advisory-json", type=Path,
                        default=BASE_DIR/"index_regime_output"/"qqq_regime_advisory.json")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR/"qqq_options_output")
    args = parser.parse_args(argv)
    try:
        overrides = {
            field: value for field, value in (
                ("premium_budget_usd", args.budget),
                ("target_dte", args.target_dte),
                ("target_abs_delta", args.target_delta),
                ("min_open_interest", args.min_oi),
                ("quote_wait_seconds", args.quote_wait),
                ("max_quote_wait_seconds", args.max_quote_wait),
            ) if value is not None
        }
        cfg = replace(Config(), **overrides)
    except ValueError as error:
        parser.error(str(error))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logging.getLogger("ib_insync.wrapper").setLevel(logging.WARNING)
    print(f"QQQ OPTIONS ADVISOR {VERSION} | EGYSZERI PAPER FUTÁS", flush=True)
    print(f"TWS {cfg.host}:{cfg.port} | clientId={cfg.client_id} | kért adatmód={cfg.market_data_type}", flush=True)
    print(f"Tényleges prémiumkeret: {cfg.premium_budget_usd:.2f} USD | "
          f"forrás: {'--budget argumentum' if args.budget is not None else 'Config.premium_budget_usd'}", flush=True)
    if args.demo:
        result = demo_result(cfg)
        show(result)
        save_result(result, args.output_dir/"demo")
        print(f"DEMO export (külön könyvtár): {args.output_dir/'demo'}")
        return 0
    packet = load_advisory(args.advisory_json)
    initial_gate = advisory_gate(packet, pd.Timestamp.now(tz="UTC"), cfg)
    print(f"Index JSON: {args.advisory_json.resolve()} | minőség: {initial_gate['quality']}", flush=True)
    if initial_gate["quality"] != "OK":
        print("Csak adatdiagnosztika: hiányzó/hibás/elavult JSON nem enged paper belépést. "
              "Futtasd a friss ES/NQ runnert; az inputot nem pótoljuk becsléssel.", flush=True)
    ib = None
    try:
        import ib_insync as api
        ib = api.IB()
        ib.RequestTimeout = 30
        ib.connect(cfg.host, cfg.port, clientId=cfg.client_id, timeout=20, readonly=True)
        ib.reqMarketDataType(cfg.market_data_type)
        underlying, rows, notes = collect(ib, api, cfg)
        now = pd.Timestamp.now(tz="UTC")
        result = evaluate(rows, underlying, packet, now, cfg)
        result["advisory_path"] = str(args.advisory_json.resolve())
        result["notes"] += notes
        show(result)
        save_result(result, args.output_dir)
        print(f"Exportok: {args.output_dir}")
        return 0
    except Exception as error:
        print(f"\nÖSSZESÍTETT STÁTUSZ: LEKÉRÉSI / PROGRAMHIBA | {type(error).__name__}: {error}")
        print("Nincs végrehajtási engedély. Hiányzó ib_insync esetén: pip install ib_insync")
        try:
            result = evaluate([], dict(price=None, quality="MISSING", asof=None),
                              packet, pd.Timestamp.now(tz="UTC"), cfg)
            result["state"] = "COLLECTION_ERROR"
            result["notes"].append(f"{type(error).__name__}: {error}")
            save_result(result, args.output_dir)
        except Exception as export_error:
            print(f"Hibastátusz exportja sem sikerült: {export_error}")
        return 1
    finally:
        if ib is not None and ib.isConnected():
            ib.disconnect()
            print("TWS kapcsolat bontva.")


if __name__ == "__main__":
    raise SystemExit(main())
