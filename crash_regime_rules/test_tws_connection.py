# test_tws_connection_ROBUST.py
from ib_insync import IB, Index, Stock, Future, util
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger(__name__)

IB_HOST = '127.0.0.1'
IB_PORT = 7497
CLIENT_ID = 1  # Ha fut más script, növeld (2, 3...)


# --- Segédfüggvények ---
def calc_rsi(s: pd.Series, p: int) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / p, adjust=False).mean()
    dn = -d.clip(upper=0).ewm(alpha=1 / p, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def calc_atr(h, l, c, p=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / p, adjust=False).mean()


def get_liquid_es_contract(ib: IB) -> Future:
    """
    Visszaadja az éppen legjobban likvid ES futures kontraktot.
    Lekéri a láncot, leteszteli az első 3 lejártat (front, 2nd, 3rd),
    és kiválasztja a legmagasabb (ATR * Volume) pontszámúat.
    """
    log.info("🔍 ES Futures lánc lekérdezése (Contract Details)...")
    details = ib.reqContractDetails(Future('ES', '', 'CME'))

    today_str = datetime.now().strftime('%Y%m')
    futures = []
    for d in details:
        c = d.contract
        # Csak sima ES futures (nem Micro MES, nem opció, USD, multiplier=50)
        if c.secType == 'FUT' and c.exchange == 'CME' and c.currency == 'USD' and c.multiplier == '50':
            expiry = c.lastTradeDateOrContractMonth[:6]
            if expiry >= today_str:  # Csak jövőbeli lejártak
                futures.append(c)

    futures.sort(key=lambda x: x.lastTradeDateOrContractMonth)

    if not futures:
        raise RuntimeError("Nem található érvényes ES futures kontrakt.")

    log.info(f"   Talált lejártak (front 4): {[f.localSymbol for f in futures[:4]]}")

    # Teszteljük az első 3-at
    candidates = futures[:3]
    best_contract = None
    best_score = -1.0

    for f in candidates:
        try:
            log.info(f"   📊 Likviditás teszt: {f.localSymbol}...")
            bars = ib.reqHistoricalData(f, '', '1 D', '1 min', 'TRADES', False, 1)
            if not bars:
                log.warning(f"      -> Nincs adat (lejárt/üres).")
                continue

            df = util.df(bars).set_index('date')
            if len(df) < 20:
                log.warning(f"      -> Túl kevés bárka ({len(df)}).")
                continue

            c, h, l, v = df['close'], df['high'], df['low'], df['volume']
            atr = calc_atr(h, l, c, 14).iloc[-1]
            avg_vol = v.rolling(20).mean().iloc[-1]

            if np.isnan(atr) or np.isnan(avg_vol) or atr <= 0 or avg_vol <= 0:
                continue

            # Pontszám: Volatilitás * Likviditás (minél több, annál jobb)
            score = atr * avg_vol
            log.info(f"      -> ATR={atr:.2f}, AvgVol={avg_vol:,.0f}, SCORE={score:,.0f}")

            if score > best_score:
                best_score = score
                best_contract = f

        except Exception as e:
            log.warning(f"      -> Hiba tesztelés közben: {e}")
            continue

    if not best_contract:
        log.warning("Nem találtunk likvid kontraktot a top 3-ban, fallback az elsőre.")
        best_contract = futures[0]

    log.info(
        f"🏆 Kiválasztott LIKVID ES Kontrakt: {best_contract.localSymbol} (conId: {best_contract.conId}, Score: {best_score:,.0f})")
    return best_contract


# --- MAIN ---
def main():
    ib = IB()
    try:
        log.info(f"🔌 Csatlakozás: {IB_HOST}:{IB_PORT} (clientId={CLIENT_ID})...")
        ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID, timeout=15)
        ib.reqMarketDataType(4)  # 4 = Delayed (ingyenes)
        log.info("✅ Csatlakozás OK + Delayed Mode")

        # 1. Statikus Kontraktok
        vix_con = Index('VIX', 'CBOE', 'USD')
        vix3m_con = Index('VIX3M', 'CBOE', 'USD')
        spy_con = Stock('SPY', 'SMART', 'USD', primaryExchange='ARCA')

        # 2. DINAMIKUS ES KONTRAKT (Robosztus választás)
        es_con = get_liquid_es_contract(ib)

        # 3. Validálás
        qualified = ib.qualifyContracts(vix_con, vix3m_con, spy_con, es_con)
        for c in qualified:
            log.info(f"   ✅ {c.localSymbol} (conId: {c.conId})")

        # 4. Snapshot
        log.info("📸 Snapshot lekérése (várakozás 5 mp delayed feed-re)...")
        tickers = ib.reqTickers(*qualified)
        ib.sleep(5)
        for t in tickers:
            sym = t.contract.localSymbol
            last = t.last if t.last and not np.isnan(t.last) else (
                t.close if t.close and not np.isnan(t.close) else 'NAN')
            log.info(f"   {sym}: Last={last} Bid={t.bid} Ask={t.ask}")

        # 5. Történeti adatok
        log.info("📊 1 perces bárkák letöltése...")
        bars_es = ib.reqHistoricalData(es_con, '', '1 D', '1 min', 'TRADES', False, 1)  # RTH=False (Globex)
        bars_spy = ib.reqHistoricalData(spy_con, '', '1 D', '1 min', 'TRADES', True, 1)  # RTH=True (Cash)

        if not bars_es or not bars_spy:
            log.error("❌ NEM KAPTUNK BÁR KÁT! -> Market Data Subscriptions (Delayed) + TWS Restart.")
            return

        df_es = util.df(bars_es).set_index('date')
        df_spy = util.df(bars_spy).set_index('date')
        log.info(f"   ES bárkák: {len(df_es)} db | SPY bárkák: {len(df_spy)} db")

        # 6. Számítások
        c, h, l, v = df_es['close'], df_es['high'], df_es['low'], df_es['volume']
        typical = (h + l + c) / 3
        vwap_es = (typical * v).cumsum() / v.cumsum()

        atr14 = calc_atr(h, l, c, 14).iloc[-1]
        rsi2 = calc_rsi(c, 2).iloc[-1]
        rsi5 = calc_rsi(c, 5).iloc[-1]
        vol_avg20 = v.rolling(20).mean().iloc[-1]

        # SPY VWAP normalizálva ES skálára (SPY * 10 ≈ ES)
        spy_typ = (df_spy['high'] + df_spy['low'] + df_spy['close']) / 3
        spy_vwap = (spy_typ * df_spy['volume']).cumsum() / df_spy['volume'].cumsum()
        spy_vwap_z = (c.iloc[-1] - (spy_vwap.iloc[-1] * 10)) / (c.iloc[-1] * 0.005)

        # VIX
        vix_t = next(t for t in tickers if t.contract.symbol == 'VIX')
        vix3m_t = next(t for t in tickers if t.contract.symbol == 'VIX3M')
        vix_val = vix_t.last if vix_t.last and not np.isnan(vix_t.last) else vix_t.close
        vix3m_val = vix3m_t.last if vix3m_t.last and not np.isnan(vix3m_t.last) else vix3m_t.close

        # 7. KIMENET
        print("\n" + "=" * 60)
        print("📋 CHECKLIST INPUT ADATOK (ROBUSTUS - LIKVID KONTRAKT)")
        print("=" * 60)
        print(f"ES Kontrakt:     {es_con.localSymbol}  <-- AUTO SELECTED LIQUID")
        print(f"ES Last:         {c.iloc[-1]:.2f}")
        print(f"ES VWAP:         {vwap_es.iloc[-1]:.2f}")
        print(f"SPY VWAP*10:     {spy_vwap.iloc[-1] * 10:.2f}")
        print(f"SPY/ES Z-Score:  {spy_vwap_z:.2f}  ( |Z| > 2.5 = Divergencia )")
        print(f"VIX Spot:        {vix_val:.2f}" if vix_val and not np.isnan(vix_val) else "VIX Spot:        N/A")
        print(f"VIX 3M:          {vix3m_val:.2f}" if vix3m_val and not np.isnan(vix3m_val) else "VIX 3M:          N/A")
        if vix_val and vix3m_val and not np.isnan(vix_val): print(f"VIX/VIX3M Ratio: {vix_val / vix3m_val:.3f}")
        print(f"ATR(14):         {atr14:.2f}  <-- HA < 5 PONT: PROBLÉMA VAN!")
        print(f"RSI(2):          {rsi2:.1f}")
        print(f"RSI(5):          {rsi5:.1f}")
        print(f"Volume:          {v.iloc[-1]:,.0f}  (Avg20: {vol_avg20:,.0f}) Ratio: {v.iloc[-1] / vol_avg20:.2f}x")
        print("-" * 60)
        print("⚠️ HIÁNYZÓ KÜLÖN FEED: Breadth (Adv/Dec), Dealer GEX, CVD (Delta)")
        print("=" * 60)

    except Exception as e:
        log.exception("💥 KRITIKUS HIBA:")
    finally:
        if ib.isConnected():
            ib.disconnect()
            log.info("🔌 Lebontva.")


if __name__ == '__main__':
    main()