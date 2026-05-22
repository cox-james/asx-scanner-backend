import os
import json
import asyncio
from datetime import datetime, time
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

# US market tickers for overnight data
US_TICKERS = {
    "spy": "^GSPC",   # S&P 500
    "qqq": "^IXIC",   # Nasdaq
    "vix": "^VIX",    # VIX
}

# ── Live price fetching ────────────────────────────────────────────────────
def fetch_live_prices() -> dict:
    """Fetch today's % change for all ASX candidates + US overnight data."""
    prices = {}

    # ASX stocks — today's % move
    asx_tickers = [s["yf"] for s in ASX_STOCKS if s["us_revenue"] < 15]
    try:
        data = yf.download(
            tickers=" ".join(asx_tickers),
            period="2d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if not data.empty and "Close" in data.columns:
            close = data["Close"]
            # Handle both single and multi-ticker responses
            if hasattr(close, "columns"):
                for yf_ticker in asx_tickers:
                    try:
                        col = close[yf_ticker]
                        if len(col.dropna()) >= 2:
                            prev, curr = float(col.dropna().iloc[-2]), float(col.dropna().iloc[-1])
                            pct = round((curr - prev) / prev * 100, 2)
                            # Map back to ASX ticker
                            asx_t = yf_ticker.replace(".AX", "")
                            prices[asx_t] = pct
                    except Exception:
                        pass
            else:
                # Single ticker
                col = close
                if len(col.dropna()) >= 2:
                    prev, curr = float(col.dropna().iloc[-2]), float(col.dropna().iloc[-1])
                    pct = round((curr - prev) / prev * 100, 2)
                    prices[asx_tickers[0].replace(".AX", "")] = pct
    except Exception as e:
        print(f"ASX price fetch error: {e}")

    # US overnight data
    us_data = {}
    try:
        for key, ticker in US_TICKERS.items():
            t = yf.Ticker(ticker)
            hist = t.history(period="2d", interval="1d")
            if len(hist) >= 2:
                prev = float(hist["Close"].iloc[-2])
                curr = float(hist["Close"].iloc[-1])
                if key == "vix":
                    us_data[key] = round(curr, 1)
                else:
                    us_data[key] = round((curr - prev) / prev * 100, 2)
    except Exception as e:
        print(f"US data fetch error: {e}")

    return {"asx": prices, "us": us_data}


# ── Models ─────────────────────────────────────────────────────────────────
class ScanRequest(BaseModel):
    label: str
    spy_move: float
    qqq: float
    vix: float
    description: str
    use_live_prices: bool = True   # fetch real ASX prices when True

class StockSignal(BaseModel):
    ticker: str
    name: str
    sector: str
    us_revenue: int
    irrationality_score: int
    actual_move: float | None       # real observed move today
    estimated_move: float | None    # Claude's estimate if no live data
    thesis: str
    risk_flag: str
    action: str

class MarketSnapshot(BaseModel):
    spy_move: float | None
    qqq: float | None
    vix: float | None
    source: str   # "live" or "user_provided"

class ScanResponse(BaseModel):
    signals: list[StockSignal]
    scan_label: str
    market: MarketSnapshot
    candidates_scanned: int
    prices_live: bool
    fetched_at: str

# ── Score a single stock ───────────────────────────────────────────────────
async def score_stock(
    client: anthropic.AsyncAnthropic,
    market: ScanRequest,
    stock: dict,
    actual_move: float | None,
) -> StockSignal | None:

    market_is_down = market.spy_move < -0.5

    if actual_move is not None:
        move_context = f"The stock has actually moved {'+' if actual_move >= 0 else ''}{actual_move}% today on the ASX."
    else:
        move_context = "No live price data available — estimate likely move based on typical ASX beta."

    prompt = f"""OVERNIGHT US MARKET:
Trigger: {market.label}
S&P 500: {market.spy_move}%  |  Nasdaq: {market.qqq}%  |  VIX: {market.vix}
Context: {market.description}

ASX STOCK: {stock['ticker']} — {stock['name']}
Sector: {stock['sector']}  |  US Revenue: {stock['us_revenue']}%
Business: {stock['description']}
Live price move today: {move_context}

How irrational is any selloff in this stock given the US trigger?
Stocks with near-zero US revenue have no fundamental reason to sell on US-specific news — only ETF flow contagion applies.
If the stock has actually moved, assess whether that real move is irrational given its fundamentals.
{'' if market_is_down else 'NOTE: US market is not materially down — score conservatively.'}

Reply ONLY with this JSON (no markdown, no extra text):
{{
  "irrationalityScore": <integer 1-10; 10 = completely irrational selloff>,
  "estimatedMove": <estimated % move if no live data, else null>,
  "thesis": <one sentence: why this is a buying opportunity>,
  "riskFlag": <one sentence: key reason the selloff might be partially rational>,
  "action": <"STRONG BUY" | "BUY" | "WATCH" | "PASS">
}}"""

    try:
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system="You are a senior ASX equity analyst specialising in ETF-driven mispricings. Reply with ONLY a raw JSON object — no markdown, no text outside the JSON.",
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text
        clean = text.replace("```json", "").replace("```", "").strip()
        start, end = clean.index("{"), clean.rindex("}")
        scored = json.loads(clean[start:end+1])

        if scored.get("irrationalityScore", 0) >= 6 and market_is_down:
            return StockSignal(
                ticker=stock["ticker"],
                name=stock["name"],
                sector=stock["sector"],
                us_revenue=stock["us_revenue"],
                irrationality_score=scored["irrationalityScore"],
                actual_move=actual_move,
                estimated_move=scored.get("estimatedMove") if actual_move is None else None,
                thesis=scored.get("thesis", ""),
                risk_flag=scored.get("riskFlag", ""),
                action=scored.get("action", "WATCH"),
            )
    except Exception as e:
        print(f"Error scoring {stock['ticker']}: {e}")
    return None


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.post("/scan", response_model=ScanResponse)
async def run_scan(req: ScanRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    candidates = [s for s in ASX_STOCKS if s["us_revenue"] < 15]

    # Fetch live prices in a thread (yfinance is sync)
    live_prices = {}
    live_us = {}
    prices_live = False

    if req.use_live_prices:
        try:
            loop = asyncio.get_event_loop()
            price_data = await loop.run_in_executor(None, fetch_live_prices)
            live_prices = price_data.get("asx", {})
            live_us = price_data.get("us", {})
            prices_live = bool(live_prices)
            print(f"Live prices fetched: {len(live_prices)} stocks")
        except Exception as e:
            print(f"Live price fetch failed: {e}")

    # Use live US data if available and user didn't provide explicit scenario numbers
    effective_spy = live_us.get("spy", req.spy_move) if live_us else req.spy_move
    effective_qqq = live_us.get("qqq", req.qqq) if live_us else req.qqq
    effective_vix = live_us.get("vix", req.vix) if live_us else req.vix

    # Override request with live US data
    req.spy_move = effective_spy
    req.qqq = effective_qqq
    req.vix = effective_vix

    # Score all stocks concurrently
    semaphore = asyncio.Semaphore(3)

    async def score_with_limit(stock):
        async with semaphore:
            actual_move = live_prices.get(stock["ticker"])
            return await score_stock(client, req, stock, actual_move)

    results = await asyncio.gather(*[score_with_limit(s) for s in candidates])
    signals = [r for r in results if r is not None]
    signals.sort(key=lambda s: s.irrationality_score, reverse=True)

    return ScanResponse(
        signals=signals,
        scan_label=req.label,
        market=MarketSnapshot(
            spy_move=effective_spy,
            qqq=effective_qqq,
            vix=effective_vix,
            source="live" if live_us else "user_provided",
        ),
        candidates_scanned=len(candidates),
        prices_live=prices_live,
        fetched_at=datetime.utcnow().isoformat() + "Z",
    )


@app.get("/prices")
async def get_prices():
    """Standalone endpoint to check live price data."""
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, fetch_live_prices)
    return data


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
