from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

from config.classes import AnalyzedStock, OrderRequest

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
    drop_reasons: dict[str, str] | None = None,
) -> List[OrderRequest]:
    keep = set(keep_tickers)
    if drop_reasons is None:
        drop_reasons = {}
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
        reason = drop_reasons.get(ticker, "ranked_position > top_n")
        logger.debug("Sell reason lookup for %s -> %s", ticker, reason)
        logger.info("Scheduled sell for %s quantity=%s reason=%s", ticker, qty, reason)
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


def build_rebalance_orders(
    ranked: List[AnalyzedStock],
    *,
    positions_by_ticker: Dict[str, float],
    cash: float,
    total_equity: float,
    risk_fraction: float,
    top_n: int,
    gap_lookback_days: int,
    rebalance_threshold: float,
    max_position_fraction: float,
) -> Tuple[List[OrderRequest], List[OrderRequest], List[OrderRequest], float]:
    new_buy_orders: List[OrderRequest] = []
    rebalance_buy_orders: List[OrderRequest] = []
    sell_orders: List[OrderRequest] = []
    remaining_cash = cash
    seen: set[str] = set()
    max_position_value = total_equity * max_position_fraction
    cap_label = f"cap{max_position_fraction * 100:.0f}%_qty"
    plans: List[dict] = []

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
            "Position size for %s | ((%.3f) * %.3f) / %.3f = %.3f | %s=%s | final_target_qty=%.3f",
            stock.ticker,
            total_equity,
            risk_fraction,
            stock.atr20,
            raw_target_qty,
            cap_label,
            f"{max_qty_by_value:.3f}" if max_qty_by_value is not None else "n/a",
            target_qty,
        )
        current_qty = positions_by_ticker.get(stock.ticker, 0.0)
        is_existing = current_qty > 0
        delta_qty = target_qty - current_qty
        change_pct = None
        if current_qty > 0:
            change_pct = abs(delta_qty) / current_qty
        should_adjust = False
        if current_qty > 0:
            should_adjust = change_pct is not None and change_pct > rebalance_threshold
        elif delta_qty > 0:
            should_adjust = True
        plans.append({
            "stock": stock,
            "target_qty": target_qty,
            "current_qty": current_qty,
            "delta_qty": delta_qty,
            "change_pct": change_pct,
            "should_adjust": should_adjust,
            "is_existing": is_existing,
        })

    expected_sell = sum(
        abs(plan["delta_qty"]) * plan["stock"].current_price
        for plan in plans
        if plan["should_adjust"] and plan["delta_qty"] < 0 and plan["is_existing"]
    )
    if expected_sell > 0:
        remaining_cash += expected_sell

    for plan in plans:
        if not plan["should_adjust"]:
            continue
        if plan["delta_qty"] >= 0:
            continue
        if not plan["is_existing"]:
            continue
        stock = plan["stock"]
        sell_qty = round(abs(plan["delta_qty"]), 3)
        if sell_qty <= 0:
            continue
        change_pct = plan["change_pct"]
        logger.info(
            "Rebalance sell for %s | current_qty=%.3f | target_qty=%.3f | delta=%.3f | change_pct=%s | threshold=%.2f%%",
            stock.ticker,
            plan["current_qty"],
            plan["target_qty"],
            plan["delta_qty"],
            f"{change_pct * 100:.2f}%" if change_pct is not None else "n/a",
            rebalance_threshold * 100.0,
        )
        sell_orders.append(OrderRequest(ticker=stock.ticker, quantity=-sell_qty))

    for plan in plans:
        if not plan["should_adjust"]:
            continue
        if plan["delta_qty"] <= 0:
            continue
        if not plan["is_existing"]:
            continue
        stock = plan["stock"]
        current_qty = plan["current_qty"]
        target_qty = plan["target_qty"]
        delta_qty = plan["delta_qty"]
        if current_qty > 0:
            logger.info(
                "Rebalance buy for %s | current_qty=%.3f | target_qty=%.3f | delta=%.3f | change_pct=%.2f%% | threshold=%.2f%%",
                stock.ticker,
                current_qty,
                target_qty,
                delta_qty,
                (plan["change_pct"] or 0.0) * 100.0,
                rebalance_threshold * 100.0,
            )
        required_cash = delta_qty * stock.current_price
        if required_cash <= 0:
            continue
        if required_cash > remaining_cash:
            delta_qty = remaining_cash / stock.current_price
            required_cash = delta_qty * stock.current_price
        if delta_qty <= 0:
            break
        delta_qty = round(delta_qty, 3)
        rebalance_buy_orders.append(OrderRequest(ticker=stock.ticker, quantity=delta_qty))
        remaining_cash -= required_cash
        gap_text = "n/a"
        if stock.max_gap_percent is not None:
            gap_text = f"{stock.max_gap_percent:.3f}%"
        logger.info(
            "Scheduled rebalance buy for %s | qty=%.3f | price=%.3f | score=%.3f | atr20=%.3f | sma100=%.3f | max_gap_%sd=%s "
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

    for plan in plans:
        if not plan["should_adjust"]:
            continue
        if plan["delta_qty"] <= 0:
            continue
        if plan["is_existing"]:
            continue
        stock = plan["stock"]
        current_qty = plan["current_qty"]
        target_qty = plan["target_qty"]
        delta_qty = plan["delta_qty"]
        required_cash = delta_qty * stock.current_price
        if required_cash <= 0:
            continue
        if required_cash > remaining_cash:
            delta_qty = remaining_cash / stock.current_price
            required_cash = delta_qty * stock.current_price
        if delta_qty <= 0:
            break
        delta_qty = round(delta_qty, 3)
        new_buy_orders.append(OrderRequest(ticker=stock.ticker, quantity=delta_qty))
        remaining_cash -= required_cash
        gap_text = "n/a"
        if stock.max_gap_percent is not None:
            gap_text = f"{stock.max_gap_percent:.3f}%"
        logger.info(
            "Scheduled new buy for %s | qty=%.3f | price=%.3f | score=%.3f | atr20=%.3f | sma100=%.3f | max_gap_%sd=%s "
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

    logger.info("Total rebalance sell orders scheduled: %s", len(sell_orders))
    logger.info("Total rebalance buy orders scheduled: %s", len(rebalance_buy_orders))
    logger.info("Total new buy orders scheduled: %s", len(new_buy_orders))
    return new_buy_orders, rebalance_buy_orders, sell_orders, remaining_cash
