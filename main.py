import os
import json
import asyncio
import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="ASX Proxy Selloff Scanner")

# Allow requests from anywhere (Claude artifact, your phone, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Stock universe ────────────────────────────────────────────────────────
ASX_STOCKS = [
    {"ticker": "CBA",  "name": "Commonwealth Bank",        "sector": "Financials",       "us_revenue": 2,  "description": "Australian retail bank, almost entirely domestic operations"},
    {"ticker": "BHP",  "name": "BHP Group",                "sector": "Materials",        "us_revenue": 5,  "description": "Global miner, revenues primarily from Asia commodity sales"},
    {"ticker": "NAB",  "name": "National Australia Bank",  "sector": "Financials",       "us_revenue": 3,  "description": "Domestic bank with small offshore presence"},
    {"ticker": "WES",  "name": "Wesfarmers",               "sector": "Consumer Disc.",   "us_revenue": 1,  "description": "Retail conglomerate, almost exclusively Australian"},
    {"ticker": "ANZ",  "name": "ANZ Banking Group",        "sector": "Financials",       "us_revenue": 2,  "description": "Asia-Pacific focused bank, minimal US exposure"},
    {"ticker": "WBC",  "name": "Westpac",                  "sector": "Financials",       "us_revenue": 2,  "description": "Australian retail and business bank"},
    {"ticker": "WOW",  "name": "Woolworths",               "sector": "Consumer Staples", "us_revenue": 0,  "description": "Australian supermarket chain, zero US exposure"},
    {"ticker": "FMG",  "name": "Fortescue",                "sector": "Materials",        "us_revenue": 1,  "description": "Iron ore miner, sells entirely to China"},
    {"ticker": "RIO",  "name": "Rio Tinto",                "sector": "Materials",        "us_revenue": 4,  "description": "Global miner, primarily Asian commodity revenues"},
    {"ticker": "TLS",  "name": "Telstra",                  "sector": "Telecom",          "us_revenue": 0,  "description": "Australian telco, zero US exposure"},
    {"ticker": "COL",  "name": "Coles Group",              "sector": "Consumer Staples", "us_revenue": 0,  "description": "Australian supermarket, zero offshore revenue"},
    {"ticker": "APA",  "name": "APA Group",                "sector": "Utilities",        "us_revenue": 0,  "description": "Australian gas pipelines, purely domestic"},
    {"ticker": "MPL",  "name": "Medibank",                 "sector": "Insurance",        "us_revenue": 0,  "description": "Australian private health insurer"},
    {"ticker": "IAG",  "name": "Insurance Australia",      "sector": "Insurance",        "us_revenue": 0,  "description": "Australian general insurer, no US operations"},
    {"ticker": "SUN",  "name": "Suncorp",                  "sector": "Financials",       "us_revenue": 0,  "description": "Australian bank and insurer"},
    {"ticker": "ORG",  "name": "Origin Energy",            "sector": "Energy",           "us_revenue": 2,  "description": "Australian energy retailer and LNG exporter to Asia"},
    {"ticker": "TCL",  "name": "Transurban",               "sector": "Infrastructure",   "us_revenue": 12, "description": "Toll roads, mostly Australian with some Virginia assets"},
]

# ── Request / response models ─────────────────────────────────────────────
class ScanRequest(BaseModel):
    label: str           # e.g. "US Tech Selloff"
    spy_move: float      # e.g. -2.1
    qqq: float           # e.g. -2.8
    vix: float           # e.g. 24.0
    description: str     # one sentence context

class StockSignal(BaseModel):
    ticker: str
    name: str
    sector: str
    us_revenue: int
    irrationality_score: int
    estimated_move: float | None
    thesis: str
    risk_flag: str
    action: str

class ScanResponse(BaseModel):
    signals: list[StockSignal]
    scan_label: str
    spy_move: float
    qqq: float
    vix: float
    candidates_scanned: int

# ── Score a single stock ──────────────────────────────────────────────────
async def score_stock(client: anthropic.AsyncAnthropic, market: ScanRequest, stock: dict) -> StockSignal | None:
    market_is_down = market.spy_move < -0.5
    prompt = f"""OVERNIGHT US MARKET:
Trigger: {market.label}
S&P 500: {market.spy_move}%  |  Nasdaq: {market.qqq}%  |  VIX: {market.vix}
Context: {market.description}

ASX STOCK: {stock['ticker']} — {stock['name']}
Sector: {stock['sector']}  |  US Revenue: {stock['us_revenue']}%
Business: {stock['description']}

How irrational is a selloff in this stock given the US trigger?
Stocks with near-zero US revenue have no fundamental reason to sell on US-specific news — only ETF flow contagion applies.
{'' if market_is_down else 'NOTE: US market is not materially down — score conservatively, likely PASS.'}

Reply ONLY with this JSON (no markdown, no extra text):
{{
  "irrationalityScore": <integer 1-10; 10 = completely irrational selloff>,
  "estimatedMove": <estimated ASX open % move as float e.g. -1.4>,
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
        # Strip markdown fences if present
        clean = text.replace("```json", "").replace("```", "").strip()
        start, end = clean.index("{"), clean.rindex("}")
        scored = json.loads(clean[start:end+1])

        # Only return if signal is strong enough
        if scored.get("irrationalityScore", 0) >= 6 and market_is_down:
            return StockSignal(
                ticker=stock["ticker"],
                name=stock["name"],
                sector=stock["sector"],
                us_revenue=stock["us_revenue"],
                irrationality_score=scored["irrationalityScore"],
                estimated_move=scored.get("estimatedMove"),
                thesis=scored.get("thesis", ""),
                risk_flag=scored.get("riskFlag", ""),
                action=scored.get("action", "WATCH"),
            )
    except Exception as e:
        print(f"Error scoring {stock['ticker']}: {e}")
    return None

# ── Main scan endpoint ────────────────────────────────────────────────────
@app.post("/scan", response_model=ScanResponse)
async def run_scan(req: ScanRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    candidates = [s for s in ASX_STOCKS if s["us_revenue"] < 15]

    # Score all stocks concurrently (faster, but rate-limit friendly with semaphore)
    semaphore = asyncio.Semaphore(3)  # max 3 concurrent API calls

    async def score_with_limit(stock):
        async with semaphore:
            return await score_stock(client, req, stock)

    results = await asyncio.gather(*[score_with_limit(s) for s in candidates])
    signals = [r for r in results if r is not None]
    signals.sort(key=lambda s: s.irrationality_score, reverse=True)

    return ScanResponse(
        signals=signals,
        scan_label=req.label,
        spy_move=req.spy_move,
        qqq=req.qqq,
        vix=req.vix,
        candidates_scanned=len(candidates),
    )

@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
