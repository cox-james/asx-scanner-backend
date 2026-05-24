import os
import json
import asyncio
from datetime import datetime
import anthropic
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="ASX Proxy Selloff Scanner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Stock universe ─────────────────────────────────────────────────────────
# Filtered to financials + insurance based on backtest evidence (Jan 2023 - Dec 2024).
# These names consistently mean-revert after US-driven selloffs due to high
# ETF weighting and retail flow exposure. Other low-US-revenue names (WOW, APA,
# TCL, BHP) showed no tradeable edge in backtest.
ASX_STOCKS = [
    {"ticker": "CBA",  "yf": "CBA.AX",  "name": "Commonwealth Bank",       "sector": "Financials", "us_revenue": 2, "description": "Australian retail bank, almost entirely domestic operations"},
    {"ticker": "NAB",  "yf": "NAB.AX",  "name": "National Australia Bank", "sector": "Financials", "us_revenue": 3, "description": "Domestic bank with small offshore presence"},
    {"ticker": "ANZ",  "yf": "ANZ.AX",  "name": "ANZ Banking Group",       "sector": "Financials", "us_revenue": 2, "description": "Asia-Pacific focused bank, minimal US exposure"},
    {"ticker": "WBC",  "yf": "WBC.AX",  "name": "Westpac",                 "sector": "Financials", "us_revenue": 2, "description": "Australian retail and business bank"},
    {"ticker": "MPL",  "yf": "MPL.AX",  "name": "Medibank",                "sector": "Insurance",  "us_revenue": 0, "description": "Australian private health insurer"},
    {"ticker": "SUN",  "yf": "SUN.AX",  "name": "Suncorp",                 "sector": "Financials", "us_revenue": 0, "description": "Australian bank and insurer"},
    {"ticker": "IAG",  "yf": "IAG.AX",  "name": "Insurance Australia",     "sector": "Insurance",  "us_revenue": 0, "description": "Australian general insurer, no US operations"},
]

# ── Live price fetching ────────────────────────────────────────────────────
def fetch_all_live_data() -> dict:
    asx_prices = {}
    us_data = {}

    asx_tickers = [s["yf"] for s in ASX_STOCKS]
    try:
        data = yf.download(
            tickers=" ".join(asx_tickers),
            period="2d", interval="1d",
            auto_adjust=True, progress=False, threads=True,
        )
        if not data.empty and "Close" in data.columns:
            close = data["Close"]
            if hasattr(close, "columns"):
                for yf_ticker in asx_tickers:
                    try:
                        col = close[yf_ticker].dropna()
                        if len(col) >= 2:
                            prev, curr = float(col.iloc[-2]), float(col.iloc[-1])
                            asx_prices[yf_ticker.replace(".AX","")] = round((curr-prev)/prev*100, 2)
                    except Exception:
                        pass
    except Exception as e:
        print(f"ASX fetch error: {e}")

    us_tickers = {
        "spy": "^GSPC", "qqq": "^IXIC", "vix": "^VIX",
        "dxy": "DX-Y.NYB", "us10y": "^TNX",
    }
    for key, ticker in us_tickers.items():
        try:
            hist = yf.Ticker(ticker).history(period="2d", interval="1d")
            if len(hist) >= 2:
                prev, curr = float(hist["Close"].iloc[-2]), float(hist["Close"].iloc[-1])
                if key in ("vix", "us10y", "dxy"):
                    us_data[key] = round(curr, 2)
                    us_data[key+"_chg"] = round(curr - prev, 2)
                else:
                    us_data[key] = round((curr-prev)/prev*100, 2)
        except Exception as e:
            print(f"US ticker {ticker} error: {e}")

    return {"asx": asx_prices, "us": us_data}


async def interpret_market(client: anthropic.AsyncAnthropic, us_data: dict) -> dict:
    prompt = f"""Today's overnight US market data:

S&P 500: {us_data.get('spy', 'N/A')}%
Nasdaq: {us_data.get('qqq', 'N/A')}%
VIX: {us_data.get('vix', 'N/A')} (change: {us_data.get('vix_chg', 'N/A')})
US 10Y yield: {us_data.get('us10y', 'N/A')}% (change: {us_data.get('us10y_chg', 'N/A')})
USD index (DXY): {us_data.get('dxy', 'N/A')} (change: {us_data.get('dxy_chg', 'N/A')})

Identify the most likely trigger driving markets today.

Reply ONLY with this JSON:
{{
  "trigger": <short label>,
  "description": <one sentence explaining what drove markets>,
  "market_is_actionable": <true if S&P is down more than 1.5 percent, else false>,
  "most_exposed_sectors": <list of US sectors genuinely affected>,
  "least_exposed_sectors": <list of ASX sectors with no real connection to trigger>
}}"""
    try:
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=512,
            system="You are a senior market strategist. Reply with ONLY a raw JSON object.",
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.replace("```json","").replace("```","").strip()
        s, e = text.index("{"), text.rindex("}")
        return json.loads(text[s:e+1])
    except Exception as ex:
        print(f"Market interpretation error: {ex}")
        spy = us_data.get("spy", 0) or 0
        return {"trigger": f"S&P {spy:+.1f}%", "description": "Auto-derived",
                "market_is_actionable": spy < -1.5,
                "most_exposed_sectors": [], "least_exposed_sectors": []}


async def score_stock(client, market_interp, us_data, stock, actual_move):
    market_is_down = market_interp.get("market_is_actionable", False)
    move_ctx = (f"This stock has moved {'+' if actual_move >= 0 else ''}{actual_move}% today on the ASX."
                if actual_move is not None
                else "No live ASX price data available.")

    prompt = f"""LIVE MARKET CONDITIONS:
Trigger: {market_interp['trigger']}
Context: {market_interp['description']}
S&P 500: {us_data.get('spy','N/A')}%  |  Nasdaq: {us_data.get('qqq','N/A')}%  |  VIX: {us_data.get('vix','N/A')}
US 10Y: {us_data.get('us10y','N/A')}%  |  DXY: {us_data.get('dxy','N/A')}

ASX STOCK: {stock['ticker']} — {stock['name']}
Sector: {stock['sector']}  |  US Revenue: {stock['us_revenue']}%
Business: {stock['description']}
Live move: {move_ctx}

This stock is in a curated list of names (banks + insurers) that historically mean-revert after US-driven selloffs.
How irrational is today's selloff in this stock given the actual US trigger?
{'' if market_is_down else 'NOTE: Market is not materially down — score conservatively.'}

Reply ONLY with this JSON:
{{
  "irrationalityScore": <integer 1-10>,
  "thesis": <one sentence — why this is a buying opportunity given today's trigger>,
  "riskFlag": <one sentence — one genuine reason the selloff could be rational>,
  "action": <"STRONG BUY" | "BUY" | "WATCH" | "PASS">
}}

Be conservative — only score 8+ if the selloff is clearly irrational given the trigger."""

    try:
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=400,
            system="You are a senior ASX equity analyst. Reply with ONLY a raw JSON object.",
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.replace("```json","").replace("```","").strip()
        s, e = text.index("{"), text.rindex("}")
        scored = json.loads(text[s:e+1])

        if scored.get("irrationalityScore", 0) >= 8 and market_is_down:
            return {
                "ticker": stock["ticker"], "name": stock["name"], "sector": stock["sector"],
                "us_revenue": stock["us_revenue"],
                "irrationality_score": scored["irrationalityScore"],
                "actual_move": actual_move,
                "thesis": scored.get("thesis", ""),
                "risk_flag": scored.get("riskFlag", ""),
                "action": scored.get("action", "WATCH"),
            }
    except Exception as ex:
        print(f"Score error {stock['ticker']}: {ex}")
    return None


class ScanResponse(BaseModel):
    signals: list[dict]
    trigger: str
    description: str
    spy_move: float | None
    qqq: float | None
    vix: float | None
    us10y: float | None
    dxy: float | None
    prices_live: bool
    candidates_scanned: int
    fetched_at: str
    entry_rule: str
    holding_rule: str


@app.post("/scan", response_model=ScanResponse)
async def run_scan():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    candidates = ASX_STOCKS

    loop = asyncio.get_event_loop()
    price_data = await loop.run_in_executor(None, fetch_all_live_data)
    asx_prices = price_data.get("asx", {})
    us_data = price_data.get("us", {})
    prices_live = bool(asx_prices)
    print(f"Fetched {len(asx_prices)} ASX prices")

    market_interp = await interpret_market(client, us_data)
    print(f"Market: {market_interp.get('trigger')} — actionable: {market_interp.get('market_is_actionable')}")

    semaphore = asyncio.Semaphore(3)
    async def score_with_limit(stock):
        async with semaphore:
            return await score_stock(client, market_interp, us_data, stock, asx_prices.get(stock["ticker"]))

    results = await asyncio.gather(*[score_with_limit(s) for s in candidates])
    signals = [r for r in results if r is not None]
    signals.sort(key=lambda s: s["irrationality_score"], reverse=True)

    return ScanResponse(
        signals=signals,
        trigger=market_interp.get("trigger", "Unknown"),
        description=market_interp.get("description", ""),
        spy_move=us_data.get("spy"),
        qqq=us_data.get("qqq"),
        vix=us_data.get("vix"),
        us10y=us_data.get("us10y"),
        dxy=us_data.get("dxy"),
        prices_live=prices_live,
        candidates_scanned=len(candidates),
        fetched_at=datetime.utcnow().isoformat() + "Z",
        entry_rule="Enter at Day 1 close — do NOT buy on signal day (backtest: -0.14% Day 1, +1.88% Day 10)",
        holding_rule="Hold minimum 5 days, target 10 days (5D: +0.97% / 60% win, 10D: +1.88% / 58% win)",
    )


@app.get("/prices")
async def get_prices():
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fetch_all_live_data)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
