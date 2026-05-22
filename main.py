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
ASX_STOCKS = [
    {"ticker": "CBA",  "yf": "CBA.AX",  "name": "Commonwealth Bank",       "sector": "Financials",       "us_revenue": 2,  "description": "Australian retail bank, almost entirely domestic operations"},
    {"ticker": "BHP",  "yf": "BHP.AX",  "name": "BHP Group",               "sector": "Materials",        "us_revenue": 5,  "description": "Global miner, revenues primarily from Asia commodity sales"},
    {"ticker": "NAB",  "yf": "NAB.AX",  "name": "National Australia Bank", "sector": "Financials",       "us_revenue": 3,  "description": "Domestic bank with small offshore presence"},
    {"ticker": "WES",  "yf": "WES.AX",  "name": "Wesfarmers",              "sector": "Consumer Disc.",   "us_revenue": 1,  "description": "Retail conglomerate, almost exclusively Australian"},
    {"ticker": "ANZ",  "yf": "ANZ.AX",  "name": "ANZ Banking Group",       "sector": "Financials",       "us_revenue": 2,  "description": "Asia-Pacific focused bank, minimal US exposure"},
    {"ticker": "WBC",  "yf": "WBC.AX",  "name": "Westpac",                 "sector": "Financials",       "us_revenue": 2,  "description": "Australian retail and business bank"},
    {"ticker": "WOW",  "yf": "WOW.AX",  "name": "Woolworths",              "sector": "Consumer Staples", "us_revenue": 0,  "description": "Australian supermarket chain, zero US exposure"},
    {"ticker": "FMG",  "yf": "FMG.AX",  "name": "Fortescue",               "sector": "Materials",        "us_revenue": 1,  "description": "Iron ore miner, sells entirely to China"},
    {"ticker": "RIO",  "yf": "RIO.AX",  "name": "Rio Tinto",               "sector": "Materials",        "us_revenue": 4,  "description": "Global miner, primarily Asian commodity revenues"},
    {"ticker": "TLS",  "yf": "TLS.AX",  "name": "Telstra",                 "sector": "Telecom",          "us_revenue": 0,  "description": "Australian telco, zero US exposure"},
    {"ticker": "COL",  "yf": "COL.AX",  "name": "Coles Group",             "sector": "Consumer Staples", "us_revenue": 0,  "description": "Australian supermarket, zero offshore revenue"},
    {"ticker": "APA",  "yf": "APA.AX",  "name": "APA Group",               "sector": "Utilities",        "us_revenue": 0,  "description": "Australian gas pipelines, purely domestic"},
    {"ticker": "MPL",  "yf": "MPL.AX",  "name": "Medibank",                "sector": "Insurance",        "us_revenue": 0,  "description": "Australian private health insurer"},
    {"ticker": "IAG",  "yf": "IAG.AX",  "name": "Insurance Australia",     "sector": "Insurance",        "us_revenue": 0,  "description": "Australian general insurer, no US operations"},
    {"ticker": "SUN",  "yf": "SUN.AX",  "name": "Suncorp",                 "sector": "Financials",       "us_revenue": 0,  "description": "Australian bank and insurer"},
    {"ticker": "ORG",  "yf": "ORG.AX",  "name": "Origin Energy",           "sector": "Energy",           "us_revenue": 2,  "description": "Australian energy retailer and LNG exporter to Asia"},
    {"ticker": "TCL",  "yf": "TCL.AX",  "name": "Transurban",              "sector": "Infrastructure",   "us_revenue": 12, "description": "Toll roads, mostly Australian with some Virginia assets"},
]

# ── Live price fetching ────────────────────────────────────────────────────
def fetch_all_live_data() -> dict:
    """Fetch today's % move for ASX stocks + US overnight data."""
    asx_prices = {}
    us_data = {}

    # ASX stocks
    asx_tickers = [s["yf"] for s in ASX_STOCKS if s["us_revenue"] < 15]
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
            else:
                col = close.dropna()
                if len(col) >= 2:
                    prev, curr = float(col.iloc[-2]), float(col.iloc[-1])
                    asx_prices[asx_tickers[0].replace(".AX","")] = round((curr-prev)/prev*100, 2)
    except Exception as e:
        print(f"ASX fetch error: {e}")

    # US overnight: S&P, Nasdaq, VIX, 10Y yield, DXY
    us_tickers = {
        "spy": "^GSPC",
        "qqq": "^IXIC",
        "vix": "^VIX",
        "dxy": "DX-Y.NYB",
        "us10y": "^TNX",
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
                    us_data[key+"_price"] = round(curr, 2)
        except Exception as e:
            print(f"US ticker {ticker} error: {e}")

    return {"asx": asx_prices, "us": us_data}


# ── Claude: interpret market conditions ───────────────────────────────────
async def interpret_market(client: anthropic.AsyncAnthropic, us_data: dict) -> dict:
    """Ask Claude to interpret the overnight data and identify the trigger."""
    prompt = f"""Here is today's overnight US market data:

S&P 500: {us_data.get('spy', 'N/A')}%
Nasdaq: {us_data.get('qqq', 'N/A')}%
VIX: {us_data.get('vix', 'N/A')} (change: {us_data.get('vix_chg', 'N/A')})
US 10Y yield: {us_data.get('us10y', 'N/A')}% (change: {us_data.get('us10y_chg', 'N/A')})
USD index (DXY): {us_data.get('dxy', 'N/A')} (change: {us_data.get('dxy_chg', 'N/A')})

Based on these numbers, identify:
1. The most likely trigger/narrative driving markets today
2. Whether conditions warrant scanning for irrational ASX selloffs
3. Which ASX sectors are most likely to be irrationally sold off given this trigger

Reply ONLY with this JSON:
{{
  "trigger": <short label e.g. "US tech selloff" or "Risk-off: VIX spike">,
  "description": <one sentence explaining what drove markets>,
  "market_is_actionable": <true if S&P is down more than 0.5%, else false>,
  "most_exposed_sectors": <list of US sectors genuinely affected>,
  "least_exposed_sectors": <list of ASX sectors with no real connection to trigger>
}}"""

    try:
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system="You are a senior market strategist. Reply with ONLY a raw JSON object — no markdown.",
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text
        clean = text.replace("```json","").replace("```","").strip()
        s, e = clean.index("{"), clean.rindex("}")
        return json.loads(clean[s:e+1])
    except Exception as ex:
        print(f"Market interpretation error: {ex}")
        spy = us_data.get("spy", 0) or 0
        return {
            "trigger": f"Market move (S&P {spy:+.1f}%)",
            "description": "Automated market data — no narrative available.",
            "market_is_actionable": spy < -0.5,
            "most_exposed_sectors": [],
            "least_exposed_sectors": [],
        }


# ── Score a single stock ───────────────────────────────────────────────────
async def score_stock(
    client: anthropic.AsyncAnthropic,
    market_interp: dict,
    us_data: dict,
    stock: dict,
    actual_move: float | None,
) -> dict | None:

    market_is_down = market_interp.get("market_is_actionable", False)

    move_ctx = (
        f"This stock has actually moved {'+' if actual_move >= 0 else ''}{actual_move}% today on the ASX."
        if actual_move is not None
        else "No live ASX price data — estimate likely move."
    )

    prompt = f"""LIVE MARKET CONDITIONS:
Trigger: {market_interp['trigger']}
Context: {market_interp['description']}
S&P 500: {us_data.get('spy','N/A')}%  |  Nasdaq: {us_data.get('qqq','N/A')}%  |  VIX: {us_data.get('vix','N/A')}
US 10Y yield: {us_data.get('us10y','N/A')}%  |  USD index: {us_data.get('dxy','N/A')}
Sectors genuinely affected: {', '.join(market_interp.get('most_exposed_sectors', []))}

ASX STOCK: {stock['ticker']} — {stock['name']}
Sector: {stock['sector']}  |  US Revenue: {stock['us_revenue']}%
Business: {stock['description']}
Actual price move today: {move_ctx}

Given the real market trigger above, how irrational is any selloff in this stock?
A stock with near-zero US revenue being sold off purely due to ETF flows is irrational.
If the stock has actually moved, assess whether that real move is justified by its fundamentals.
{'' if market_is_down else 'NOTE: Market is not materially down — score conservatively.'}

Reply ONLY with this JSON:
{{
  "irrationalityScore": <integer 1-10>,
  "thesis": <one sentence: why this is a buying opportunity given today's specific trigger>,
  "riskFlag": <one sentence: one genuine reason the selloff could be rational>,
  "action": <"STRONG BUY" | "BUY" | "WATCH" | "PASS">
}}"""

    try:
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system="You are a senior ASX equity analyst. Reply with ONLY a raw JSON object — no markdown.",
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text
        clean = text.replace("```json","").replace("```","").strip()
        s, e = clean.index("{"), clean.rindex("}")
        scored = json.loads(clean[s:e+1])

        if scored.get("irrationalityScore", 0) >= 6 and market_is_down:
            return {
                "ticker": stock["ticker"],
                "name": stock["name"],
                "sector": stock["sector"],
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


# ── Models ─────────────────────────────────────────────────────────────────
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


# ── Main scan endpoint ─────────────────────────────────────────────────────
@app.post("/scan", response_model=ScanResponse)
async def run_scan():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    candidates = [s for s in ASX_STOCKS if s["us_revenue"] < 15]

    # Step 1: fetch all live data
    loop = asyncio.get_event_loop()
    price_data = await loop.run_in_executor(None, fetch_all_live_data)
    asx_prices = price_data.get("asx", {})
    us_data = price_data.get("us", {})
    prices_live = bool(asx_prices)
    print(f"Fetched {len(asx_prices)} ASX prices, US data keys: {list(us_data.keys())}")

    # Step 2: Claude interprets the market conditions
    market_interp = await interpret_market(client, us_data)
    print(f"Market: {market_interp.get('trigger')} — actionable: {market_interp.get('market_is_actionable')}")

    # Step 3: score each stock against real conditions
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
