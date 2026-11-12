from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from ib_insync import IB, ContFuture, Index, util

from nq_regime import (
    get_nq_regime,
    get_nq_risk_plan,
    nq_regime_summary,
    run_nq_checklist,
)


# ===================================================================
# TWS / IBKR BEÁLLÍTÁSOK
# ===================================================================

IB_HOST = "127.0.0.1"

# TWS Paper Trading:
IB_PORT = 7497

# TWS Live Trading esetén:
# IB_PORT = 7496

# Ne legyen azonos más futó scripted clientId-jával.
CLIENT_ID = 40

# Kizárólag az elméleti risk sizinghez.
EQUITY_USD = 10_000.0

# NDOG/NWOG és prior-session szintekhez.
HISTORICAL_DURATION = "10 D"

BAR_SIZE = "1 min"
USE_RTH = False

# 4 = Delayed market data.
MARKET_DATA_TYPE = 4


# ===================================================================
# LOGOLÁS
# ===================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

log = logging.getLogger(__name__)


# ===================================================================
# SEGÉDFÜGGVÉNYEK
# ===================================================================

def is_valid_price(value: Any) -> bool:
    """
    TWS delayed tickernél a -1, 0 vagy NaN nem használható ár.
    """
    if value is None:
        return False

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False

    return (
        math.isfinite(numeric_value)
        and numeric_value > 0
    )


def get_ticker_price(ticker: Any) -> float:
    """
    Használható árat keres a tickerből.

    Sorrend:
    last -> close -> marketPrice -> bid -> ask
    """
    values = [
        getattr(ticker, "last", None),
        getattr(ticker, "close", None),
    ]

    for value in values:
        if is_valid_price(value):
            return float(value)

    try:
        market_price = ticker.marketPrice()

        if is_valid_price(market_price):
            return float(market_price)

    except Exception:
        pass

    values = [
        getattr(ticker, "bid", None),
        getattr(ticker, "ask", None),
    ]

    for value in values:
        if is_valid_price(value):
            return float(value)

    return np.nan


def bars_to_dataframe(bars: list[Any]) -> pd.DataFrame:
    """
    IBKR HistoricalData listából OHLCV DataFrame.
    """
    if not bars:
        raise RuntimeError(
            "Az IBKR nem adott vissza történeti NQ adatot."
        )

    df = util.df(bars)

    if df.empty:
        raise RuntimeError(
            "Az NQ történeti DataFrame üres."
        )

    if "date" not in df.columns:
        raise RuntimeError(
            f"Nincs 'date' oszlop. Elérhető oszlopok: {list(df.columns)}"
        )

    df = df.set_index("date")

    df.columns = [
        str(column).lower()
        for column in df.columns
    ]

    required_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise RuntimeError(
            f"Hiányzó OHLCV oszlopok: {missing_columns}. "
            f"Elérhető oszlopok: {list(df.columns)}"
        )

    df = df[required_columns].dropna()
    df = df.sort_index()

    if len(df) < 100:
        raise RuntimeError(
            f"Túl kevés NQ adat: {len(df)} db bar."
        )

    return df


def fetch_vix_context(
    ib: IB,
    vix_contract: Any,
    vix3m_contract: Any,
) -> dict[str, float] | None:
    """
    Opcionális VIX / VIX3M context.

    NQ-nál ez nem specifikus NQ volatility input,
    de széles piaci stressz-contextként hasznos.
    """
    log.info("📈 VIX / VIX3M snapshot lekérése...")

    tickers = ib.reqTickers(
        vix_contract,
        vix3m_contract,
    )

    ib.sleep(2)

    if len(tickers) < 2:
        return None

    vix_value = get_ticker_price(tickers[0])
    vix3m_value = get_ticker_price(tickers[1])

    if (
        not is_valid_price(vix_value)
        or not is_valid_price(vix3m_value)
    ):
        log.warning(
            "⚠️ VIX/VIX3M nem elérhető. "
            "NQ price-action engine ettől még működik."
        )
        return None

    log.info(
        f"   ✅ VIX={vix_value:.2f} | "
        f"VIX3M={vix3m_value:.2f} | "
        f"Ratio={vix_value / vix3m_value:.3f}"
    )

    return {
        "vix": float(vix_value),
        "vix3m": float(vix3m_value),
        "vix_change_pct": np.nan,
    }


# ===================================================================
# FŐ FUTTATÓ
# ===================================================================

def main() -> None:
    ib = IB()

    try:
        # -----------------------------------------------------------
        # 1. TWS kapcsolat
        # -----------------------------------------------------------
        print("Kapcsolódás a TWS-hez...")

        ib.connect(
            IB_HOST,
            IB_PORT,
            clientId=CLIENT_ID,
            timeout=20,
        )

        print("Kapcsolódás sikeres.")

        ib.reqMarketDataType(MARKET_DATA_TYPE)

        print(
            f"Market data mode: "
            f"{'DELAYED' if MARKET_DATA_TYPE == 4 else 'LIVE / DEFAULT'}"
        )

        # -----------------------------------------------------------
        # 2. Kontraktusok
        # -----------------------------------------------------------
        nq_contract = ContFuture(
            "NQ",
            "CME",
            "USD",
        )

        vix_contract = Index(
            "VIX",
            "CBOE",
            "USD",
        )

        vix3m_contract = Index(
            "VIX3M",
            "CBOE",
            "USD",
        )

        print("NQ / VIX kontraktusok validálása...")

        qualified = ib.qualifyContracts(
            nq_contract,
            vix_contract,
            vix3m_contract,
        )

        if not qualified:
            raise RuntimeError(
                "Nincs validált NQ/VIX kontraktus."
            )

        print("Kontraktusok:")

        for contract in qualified:
            print(
                f"  ✅ {contract.localSymbol} | "
                f"Symbol={contract.symbol} | "
                f"Exchange={contract.exchange} | "
                f"conId={contract.conId}"
            )

        # -----------------------------------------------------------
        # 3. VIX context
        # -----------------------------------------------------------
        vix_context = fetch_vix_context(
            ib=ib,
            vix_contract=vix_contract,
            vix3m_contract=vix3m_contract,
        )

        # -----------------------------------------------------------
        # 4. NQ historical 1m adat
        # -----------------------------------------------------------
        print(
            f"NQ {HISTORICAL_DURATION} / {BAR_SIZE} "
            f"adat letöltése..."
        )

        bars_nq = ib.reqHistoricalData(
            nq_contract,
            endDateTime="",
            durationStr=HISTORICAL_DURATION,
            barSizeSetting=BAR_SIZE,
            whatToShow="TRADES",
            useRTH=USE_RTH,
            formatDate=1,
        )

        df_nq = bars_to_dataframe(bars_nq)

        print(
            f"Letöltött NQ barok száma: {len(df_nq):,}"
        )

        # -----------------------------------------------------------
        # 5. Rövid összegzés
        # -----------------------------------------------------------
        print("\n" + "=" * 72)
        print("NQ RÖVID REGIME ÖSSZEFOGLALÓ")
        print("=" * 72)

        print(
            nq_regime_summary(
                df_nq_1m=df_nq,
                vix_context=vix_context,
            )
        )

        # -----------------------------------------------------------
        # 6. Teljes NQ report
        # -----------------------------------------------------------
        report = run_nq_checklist(
            df_nq_1m=df_nq,
            equity_usd=EQUITY_USD,
            vix_context=vix_context,
        )

        print(report)

        # -----------------------------------------------------------
        # 7. Risk plan objektum
        # -----------------------------------------------------------
        nq_risk_plan = get_nq_risk_plan(
            df_nq_1m=df_nq,
            equity_usd=EQUITY_USD,
            vix_context=vix_context,
        )

        print("\n" + "=" * 72)
        print("NQ / MNQ GYORS RISK-PLAN ÖSSZEFOGLALÓ")
        print("=" * 72)

        if nq_risk_plan["trade_allowed"]:
            print(
                f"Irány:                  {nq_risk_plan['side']}"
            )

            print(
                f"Elméleti entry:         "
                f"{nq_risk_plan['entry']:.2f}"
            )

            print(
                f"ATR stop:               "
                f"{nq_risk_plan['stop_price']:.2f}"
            )

            print(
                f"Max NQ kontraktus:      "
                f"{nq_risk_plan['max_standard_contracts']}"
            )

            print(
                f"Max MNQ kontraktus:     "
                f"{nq_risk_plan['max_micro_contracts']}"
            )

        else:
            print("Automatikus trade:      TILTVA")
            print(
                f"Ok:                     "
                f"{nq_risk_plan['reason']}"
            )

            if "details" in nq_risk_plan:
                print(
                    f"Konkrét tiltás:         "
                    f"{nq_risk_plan['details']}"
                )

            if "next_setup" in nq_risk_plan:
                print(
                    f"Következő setup:        "
                    f"{nq_risk_plan['next_setup']}"
                )

        print("=" * 72)

    except Exception as error:
        print("\n" + "=" * 72)
        print("KRITIKUS HIBA")
        print("=" * 72)
        print(f"Típus: {type(error).__name__}")
        print(f"Üzenet: {error}")
        print("=" * 72)

        log.exception("💥 NQ runner hiba:")

    finally:
        if ib.isConnected():
            ib.disconnect()
            print("\nTWS kapcsolat bontva.")


if __name__ == "__main__":
    main()