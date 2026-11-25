"""
Unusual Options Activity alert (free data).

Scans option chains from Yahoo Finance (via yfinance) and flags contracts where
today's volume is unusually large compared with open interest, i.e. new, big,
often short-dated bets such as "50,000 x $160 calls expiring tomorrow".

Alerts are sent as phone push notifications via ntfy.sh (free, no account):
  1. Install the "ntfy" app (iOS/Android).
  2. Subscribe to a hard-to-guess topic name, e.g. "uoa-alerts-x7k29q".
  3. Set NTFY_TOPIC to that same name (env var or GitHub secret).

Run:  pip install yfinance pandas requests
      NTFY_TOPIC=uoa-alerts-x7k29q python unusual_options_alert.py

Notes / limits:
  - Yahoo data is delayed ~15 min and open interest updates once per day,
    so this catches activity during the day, not instantly.
  - A big trade can be a hedge or spread, not a directional bet.
  - Not financial advice.
"""

import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# ---------------- Settings (override with env vars) ----------------
TICKERS = os.getenv("TICKERS", "NVDA,AAPL,TSLA,AMD,MSFT,META,AMZN,GOOGL,SPY,QQQ").split(",")
MAX_DAYS_TO_EXPIRY = int(os.getenv("MAX_DAYS_TO_EXPIRY", "14"))   # short-dated bets only
MIN_VOLUME = int(os.getenv("MIN_VOLUME", "2000"))                 # contracts traded today
MIN_VOL_OI_RATIO = float(os.getenv("MIN_VOL_OI_RATIO", "3"))      # volume vs. open interest
MIN_PREMIUM_USD = float(os.getenv("MIN_PREMIUM_USD", "500000"))   # volume * price * 100
MIN_OTM_PCT = float(os.getenv("MIN_OTM_PCT", "3"))                # % out of the money
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "")
STATE_FILE = Path(os.getenv("STATE_FILE", "uoa_seen.json"))       # avoid duplicate alerts


def spot_price(tk: yf.Ticker) -> float:
    hist = tk.history(period="1d", interval="1m")
    if hist.empty:
        hist = tk.history(period="5d")
    return float(hist["Close"].iloc[-1])


def scan_ticker(symbol: str) -> list[dict]:
    tk = yf.Ticker(symbol)
    try:
        spot = spot_price(tk)
        expiries = tk.options
    except Exception as e:
        print(f"[{symbol}] skipped: {e}", file=sys.stderr)
        return []

    hits = []
    today = date.today()
    for exp in expiries:
        dte = (date.fromisoformat(exp) - today).days
        if dte < 0 or dte > MAX_DAYS_TO_EXPIRY:
            continue
        try:
            chain = tk.option_chain(exp)
        except Exception as e:
            print(f"[{symbol} {exp}] chain error: {e}", file=sys.stderr)
            continue

        for kind, df in (("CALL", chain.calls), ("PUT", chain.puts)):
            if df.empty:
                continue
            df = df.copy()
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
            df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0)
            price = df["lastPrice"].where(df["lastPrice"] > 0, (df["bid"] + df["ask"]) / 2)
            df["premium"] = df["volume"] * price * 100
            df["vol_oi"] = df["volume"] / df["openInterest"].clip(lower=1)
            if kind == "CALL":
                df["otm_pct"] = (df["strike"] / spot - 1) * 100
            else:
                df["otm_pct"] = (1 - df["strike"] / spot) * 100

            flagged = df[
                (df["volume"] >= MIN_VOLUME)
                & (df["vol_oi"] >= MIN_VOL_OI_RATIO)
                & (df["premium"] >= MIN_PREMIUM_USD)
                & (df["otm_pct"] >= MIN_OTM_PCT)
            ]
            for _, r in flagged.iterrows():
                hits.append({
                    "id": r["contractSymbol"],
                    "symbol": symbol,
                    "type": kind,
                    "strike": float(r["strike"]),
                    "expiry": exp,
                    "dte": dte,
                    "spot": round(spot, 2),
                    "otm_pct": round(float(r["otm_pct"]), 1),
                    "volume": int(r["volume"]),
                    "oi": int(r["openInterest"]),
                    "vol_oi": round(float(r["vol_oi"]), 1),
                    "premium": float(r["premium"]),
                    "iv": round(float(r.get("impliedVolatility", 0) or 0) * 100, 1),
                })
    return hits


def fmt(h: dict) -> str:
    return (f"{h['symbol']} ${h['strike']:g} {h['type']} exp {h['expiry']} ({h['dte']}d) | "
            f"vol {h['volume']:,} vs OI {h['oi']:,} ({h['vol_oi']}x) | "
            f"~${h['premium']/1e6:.2f}M premium | spot {h['spot']} ({h['otm_pct']}% OTM) | IV {h['iv']}%")


def load_seen() -> set:
    try:
        data = json.loads(STATE_FILE.read_text())
        return set(data.get(str(date.today()), []))
    except Exception:
        return set()


def save_seen(seen: set) -> None:
    STATE_FILE.write_text(json.dumps({str(date.today()): sorted(seen)}))


def notify(lines: list[str]) -> None:
    if not NTFY_TOPIC:
        print("(NTFY_TOPIC not set - printing only)")
        return
    body = "\n\n".join(lines)[:3900]
    requests.post(
        f"https://ntfy.sh/{NTFY_TOPIC}",
        data=body.encode("utf-8"),
        headers={"Title": f"Unusual options: {len(lines)} new", "Priority": "high", "Tags": "rotating_light"},
        timeout=15,
    )


def main() -> None:
    print(f"Scan at {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC for {', '.join(TICKERS)}")
    hits = []
    for t in TICKERS:
        hits.extend(scan_ticker(t.strip().upper()))
    hits.sort(key=lambda h: h["premium"], reverse=True)

    seen = load_seen()
    new = [h for h in hits if h["id"] not in seen]
    for h in hits:
        print(("NEW  " if h in new else "seen ") + fmt(h))
    if not hits:
        print("No unusual activity matched the filters.")

    if new:
        notify([fmt(h) for h in new[:15]])
        seen.update(h["id"] for h in new)
    save_seen(seen)


if __name__ == "__main__":
    main()
