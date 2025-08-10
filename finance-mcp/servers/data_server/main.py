"""
MCP data server which fetches market data from yfinance API and makes it available to agent to call
"""

from fastmcp import FastMCP
from pydantic import BaseModel, Field
import pandas as pd
import yfinance as yf

app = FastMCP("data-server")

# Pydantic Models for request and response
class BarsRequest(BaseModel):
    # design BarsRequest based on what your tool actually needs to call yfinance.download()
    ticker: str
    start: str            # "YYYY-MM-DD"
    end: str              # "YYYY-MM-DD"
    interval: str = Field(default="1d", description="1d, 1wk, 1mo")

class Bar(BaseModel):
    ts: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class BarsResponse(BaseModel):
    ticker: str
    bars: list[Bar]

@app.tool()
def get_bars(req: BarsRequest) -> BarsResponse:
    df = yf.download(req.ticker, start=req.start, end=req.end, interval=req.interval, progress=False)
    if df.empty:
        return BarsResponse(ticker=req.ticker, bars=[])
    df = df.rename(columns=str.lower).reset_index()
    bars = [
        Bar(
            ts=str(row["date"]) if "date" in row else str(row["index"]),
            open=float(row["open"]), high=float(row["high"]),
            low=float(row["low"]), close=float(row["close"]),
            volume=float(row["volume"]),
        )
        for _, row in df.iterrows()
    ]
    return BarsResponse(ticker=req.ticker, bars=bars)

if __name__ == "__main__":
    app.run()