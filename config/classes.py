from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalyzedStock:
    ticker: str
    base_symbol: str
    yfinance_symbol: str
    name: str
    score: float
    atr20: float
    current_price: float
    sma100: float
    max_gap_percent: float | None
    slope: float
    r_squared: float


@dataclass(frozen=True)
class OrderRequest:
    ticker: str
    quantity: float
