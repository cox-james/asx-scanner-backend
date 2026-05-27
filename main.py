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
# Expanded universe across 5 sectors. Financials/insurance validated in 5yr backtest.
# Resources, healthcare, REITs and utilities added as diversification — same ETF
# flow inefficiency thesis, different sectors.
ASX_STOCKS = [
    # Financials & Insurance
    {"ticker": "CBA",  "yf": "CBA.AX",  "name": "Commonwealth Bank",       "sector": "Financials",   "us_revenue": 2,  "description": "Australian retail bank, almost entirely domestic operations"},
    {"ticker": "NAB",  "yf": "NAB.AX",  "name": "National Australia Bank", "sector": "Financials",   "us_revenue": 3,  "description": "Domestic bank with small offshore presence"},
    {"ticker": "ANZ",  "yf": "ANZ.AX",  "name": "ANZ Banking Group",       "sector": "Financials",   "us_revenue": 2,  "description": "Asia-Pacific focused bank, minimal US exposure"},
    {"ticker": "WBC",  "yf": "WBC.AX",  "name": "Westpac",                 "sector": "Financials",   "us_revenue": 2,  "description": "Australian retail and business bank"},
    {"ticker": "MPL",  "yf": "MPL.AX",  "name": "Medibank",                "sector": "Insurance",    "us_revenue": 0,  "description": "Australian private health insurer"},
    {"ticker": "SUN",  "yf": "SUN.AX",  "name": "Suncorp",                 "sector": "Financials",   "us_revenue": 0,  "description": "Australian bank and insurer"},
    {"ticker": "IAG",  "yf": "IAG.AX",  "name": "Insurance Australia",     "sector": "Insurance",    "us_revenue": 0,  "description": "Australian general insurer, no US operations"},
    # Resources — China revenue, zero US exposure, dragged by US ETF flows
    {"ticker": "FMG",  "yf": "FMG.AX",  "name": "Fortescue",               "sector": "Materials",    "us_revenue": 0,  "description": "Iron ore miner, sells entirely to China — zero US revenue"},
    {"ticker": "RIO",  "yf": "RIO.AX",  "name": "Rio Tinto",               "sector": "Materials",    "us_revenue": 4,  "description": "Global miner, primarily Asian commodity revenues"},
    {"ticker": "MIN",  "yf": "MIN.AX",  "name": "Mineral Resources",       "sector": "Materials",    "us_revenue": 0,  "description": "Australian mining services and lithium, zero US revenue"},
    # Healthcare — domestic hospitals and pathology, no US earnings exposure
    {"ticker": "RHC",  "yf": "RHC.AX",  "name": "Ramsay Health Care",      "sector": "Healthcare",   "us_revenue": 0,  "description": "Australian private hospital operator, purely domestic earnings"},
    {"ticker": "HLS",  "yf": "HLS.AX",  "name": "Healius",                 "sector": "Healthcare",   "us_revenue": 0,  "description": "Australian pathology and imaging, zero US exposure"},
    {"ticker": "SHL",  "yf": "SHL.AX",  "name": "Sonic Healthcare",        "sector": "Healthcare",   "us_revenue": 28, "description": "Global pathology, significant US laboratory network"},
    # REITs — domestic cash flows, sold off irrationally on US rate moves
    {"ticker": "GMG",  "yf": "GMG.AX",  "name": "Goodman Group",           "sector": "REITs",        "us_revenue": 8,  "description": "Industrial REIT, global logistics properties, locked-in funding costs"},
    {"ticker": "SCG",  "yf": "SCG.AX",  "name": "Scentre Group",           "sector": "REITs",        "us_revenue": 0,  "description": "Australian Westfield shopping centres, zero US exposure"},
    {"ticker": "CHC",  "yf": "CHC.AX",  "name": "Charter Hall",            "sector": "REITs",        "us_revenue": 0,  "description": "Australian commercial property fund manager, domestic focus"},
    # Utilities & Infrastructure — regulated domestic cash flows
    {"ticker": "APA",  "yf": "APA.AX",  "name": "APA Group",               "sector": "Utilities",    "us_revenue": 0,  "description": "Australian gas pipelines, government-regulated, purely domestic"},
    {"ticker": "TCL",  "yf": "TCL.AX",  "name": "Transurban",              "sector": "Infrastructure","us_revenue": 12, "description": "Toll roads, mostly Australian with some Virginia assets"},
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
  "market_is_actionable": <true if S&P is down more than 1.5 percent AND VIX is below 35, else false>,
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
                "market_is_actionable": spy < -1.5 and (us_data.get("vix") or 0) <= 35,
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

        vix_ok = (us_data.get("vix") or 0) <= 35
        if scored.get("irrationalityScore", 0) >= 7 and market_is_down and vix_ok and (actual_move is None or actual_move < -0.3):
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
        entry_rule="Enter at Day 1 close — only fires when VIX ≤ 35 and stock down 0.3%+ on signal day",
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
