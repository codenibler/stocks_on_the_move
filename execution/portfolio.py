from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

from core.classes import AnalyzedStock, OrderRequest

logger = logging.getLogger(__name__)


# Set current holding to 0 if not in top 100 momentum ranking /
# price < SMA100 / gap >= 15% in last 90 days
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
    gap_lookback_days: int,
) -> Tuple[List[OrderRequest], float]:
    orders: List[OrderRequest] = []
    remaining_cash = cash
    seen: set[str] = set()
    max_position_value = total_equity * 0.10

    for stock in ranked[:top_n]:
        if stock.ticker in seen:
            logger.debug("Skipping duplicate ticker in buy list: %s", stock.ticker)
            continue
        seen.add(stock.ticker)
        raw_target_qty = (risk_fraction * total_equity) / stock.atr20
        max_qty_by_value = None
        if stock.current_price > 0:
            max_qty_by_value = max_position_value / stock.current_price
        target_qty = raw_target_qty
        if max_qty_by_value is not None and max_qty_by_value > 0:
            target_qty = min(raw_target_qty, max_qty_by_value)
        logger.info(
            "Position size for %s | ((%.3f) * %.3f) / %.3f = %.3f | cap10%%_qty=%s | final_target_qty=%.3f",
            stock.ticker,
            total_equity,
            risk_fraction,
            stock.atr20,
            raw_target_qty,
            f"{max_qty_by_value:.3f}" if max_qty_by_value is not None else "n/a",
            target_qty,
        )
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
        orders.append(OrderRequest(ticker=stock.ticker, quantity=round(delta_qty, 3)))
        remaining_cash -= required_cash
        gap_text = "n/a"
        if stock.max_gap_percent is not None:
            gap_text = f"{stock.max_gap_percent:.3f}%"
        logger.info(
            "Scheduled buy for %s | qty=%.3f | price=%.3f | score=%.3f | atr20=%.3f | sma100=%.3f | max_gap_%sd=%s "
            "| target_qty=%.3f | current_qty=%.3f | required_cash=%.3f | cash_left=%.3f | name=%s | r2=%.3f | slope=%.3f",
            stock.ticker,
            delta_qty,
            stock.current_price,
            stock.score,
            stock.atr20,
            stock.sma100,
            gap_lookback_days,
            gap_text,
            target_qty,
            current_qty,
            required_cash,
            remaining_cash,
            stock.name,
            stock.r_squared,
            stock.slope,
        )
        if remaining_cash <= 0:
            break

    logger.info("Total buy orders scheduled: %s", len(orders))
    return orders, remaining_cash
