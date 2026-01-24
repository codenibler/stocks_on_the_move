from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

from core.classes import AnalyzedStock, OrderRequest

logger = logging.getLogger(__name__)


def extract_position_map(
    positions: Iterable[dict],
    *,
    allowed_tickers: Iterable[str],
) -> Dict[str, float]:
    allowed = set(allowed_tickers)
    position_map: Dict[str, float] = {}
    for position in positions:
        instrument = position.get("instrument", {})
        ticker = instrument.get("ticker")
        if not ticker or ticker not in allowed:
            continue
        qty = float(position.get("quantityAvailableForTrading") or 0.0)
        if qty <= 0:
            continue
        position_map[ticker] = qty
    return position_map


def build_sell_orders(
    positions: Iterable[dict],
    *,
    keep_tickers: Iterable[str],
) -> List[OrderRequest]:
    keep = set(keep_tickers)
    orders: List[OrderRequest] = []
    for position in positions:
        instrument = position.get("instrument", {})
        ticker = instrument.get("ticker")
        if not ticker:
            continue
        qty = float(position.get("quantityAvailableForTrading") or 0.0)
        if qty <= 0:
            continue
        if ticker in keep:
            continue
        orders.append(OrderRequest(ticker=ticker, quantity=-qty))
        logger.info("Scheduled sell for %s quantity=%s", ticker, qty)
    logger.info("Total sell orders scheduled: %s", len(orders))
    return orders


def calculate_holdings_value(
    positions: Iterable[dict],
    *,
    keep_tickers: Iterable[str],
) -> float:
    keep = set(keep_tickers)
    total_value = 0.0
    for position in positions:
        instrument = position.get("instrument", {})
        ticker = instrument.get("ticker")
        if not ticker or ticker not in keep:
            continue
        qty = float(position.get("quantityAvailableForTrading") or 0.0)
        price = position.get("currentPrice")
        if qty <= 0 or price is None:
            continue
        total_value += float(price) * qty
    return total_value


def build_buy_orders(
    ranked: List[AnalyzedStock],
    *,
    positions_by_ticker: Dict[str, float],
    cash: float,
    total_equity: float,
    risk_fraction: float,
    top_n: int,
) -> Tuple[List[OrderRequest], float]:
    orders: List[OrderRequest] = []
    remaining_cash = cash

    for stock in ranked[:top_n]:
        target_qty = (risk_fraction * total_equity) / stock.atr20
        current_qty = positions_by_ticker.get(stock.ticker, 0.0)
        delta_qty = target_qty - current_qty
        if delta_qty <= 0:
            logger.debug("No buy needed for %s (target=%s current=%s)", stock.ticker, target_qty, current_qty)
            continue
        required_cash = delta_qty * stock.current_price
        if required_cash <= 0:
            continue
        if required_cash > remaining_cash:
            delta_qty = remaining_cash / stock.current_price
            required_cash = delta_qty * stock.current_price
        if delta_qty <= 0:
            break
        orders.append(OrderRequest(ticker=stock.ticker, quantity=round(delta_qty, 6)))
        remaining_cash -= required_cash
        logger.info(
            "Scheduled buy for %s qty=%s price=%.4f cash_left=%.2f",
            stock.ticker,
            delta_qty,
            stock.current_price,
            remaining_cash,
        )
        if remaining_cash <= 0:
            break

    logger.info("Total buy orders scheduled: %s", len(orders))
    return orders, remaining_cash
