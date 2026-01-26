from __future__ import annotations

import logging
import os
import time

from indicators import analytics
from indicators import charts
from indicators import strategy
from config.config import get_runtime_config, get_strategy_config, get_trading212_config
from config.logging_utils import setup_logging
from data_fetching import market_data
from execution import portfolio
from execution.trading212_client import Trading212Client, Trading212Error
from stock_universe import constituents
from tqdm import tqdm

logger = logging.getLogger(__name__)


def check_sp500_trend(strategy_config) -> bool:
    logger.info("Checking S&P500 trend using %s", strategy_config.sp500_ticker)
    df = market_data.fetch_history_with_retries(
        strategy_config.sp500_ticker,
        period=strategy_config.sp500_lookback,
        interval=strategy_config.price_interval,
        retries=strategy_config.retries,
        retry_sleep_seconds=strategy_config.retry_sleep_seconds,
    )
    if df is None or df.empty:
        logger.warning("Unable to fetch S&P500 data, defaulting to risk-off")
        return False
    df = market_data.normalize_price_frame(df)
    price_series = market_data.select_price_series(df)
    if price_series is None or price_series.dropna().empty:
        logger.warning("S&P500 data missing Close values, defaulting to risk-off")
        return False
    sma200 = analytics.calculate_sma(price_series, strategy_config.sma_long)
    if sma200 is None:
        logger.warning("S&P500 SMA%s unavailable, defaulting to risk-off", strategy_config.sma_long)
        return False
    current_price = float(price_series.dropna().iloc[-1])
    logger.info("S&P500 last close %.2f vs SMA%s %.2f", current_price, strategy_config.sma_long, sma200)
    risk_on = current_price >= sma200
    trend_label = "ABOVE" if risk_on else "BELOW"
    logger.info(
        "S&P500 trend: %s SMA%s (price %s SMA)",
        trend_label,
        strategy_config.sma_long,
        ">= " if risk_on else "< ",
    )
    return risk_on


def execute_orders(client: Trading212Client, orders, *, extended_hours: bool) -> None:
    for order in orders:
        try:
            response = client.place_market_order(
                ticker=order.ticker,
                quantity=order.quantity,
                extended_hours=extended_hours,
            )
            order_id = None
            status = None
            if isinstance(response, dict):
                order_id = response.get("id")
                status = response.get("status")
            logger.info("Order placed for %s qty=%s id=%s status=%s", order.ticker, order.quantity, order_id, status)
        except Trading212Error as exc:
            logger.error("Order failed for %s qty=%s: %s", order.ticker, order.quantity, exc)


def _sleep_with_progress(seconds: float, *, label: str) -> None:
    total_seconds = max(0.0, seconds)
    whole_seconds = int(total_seconds)
    remainder = total_seconds - whole_seconds
    if whole_seconds > 0:
        for _ in tqdm(range(whole_seconds), desc=label, unit="s"):
            time.sleep(1)
    if remainder > 0:
        time.sleep(remainder)


def main() -> None:
    runtime_config = get_runtime_config()
    log_dir = setup_logging(
        runtime_config.log_root,
        runtime_config.run_log_root,
        runtime_config.log_level,
    )

    strategy_config = get_strategy_config()
    trading_config = get_trading212_config()

    logger.info("Run configuration: base_url=%s top_n=%s risk_fraction=%.2f", trading_config.base_url, strategy_config.top_n, strategy_config.risk_fraction)
    logger.info("Lookbacks: history=%s momentum=%s sp500=%s", strategy_config.history_lookback, strategy_config.momentum_lookback, strategy_config.sp500_lookback)

    client = Trading212Client(
        api_key=trading_config.api_key,
        api_secret=trading_config.api_secret,
        base_url=trading_config.base_url,
        timeout_seconds=trading_config.timeout_seconds,
        retries=trading_config.retries,
        retry_sleep_seconds=trading_config.retry_sleep_seconds,
    )

    links = constituents.load_constituent_links()
    logger.info("Scraping constituents from %s wikipedia pages", len(links))
    wiki_symbols, index_membership = constituents.scrape_wikipedia_constituents_with_sources(links)

    instruments = constituents.fetch_trading212_instruments(client)
    tradable = constituents.filter_tradable_instruments(instruments)
    matched = constituents.cross_reference_constituents(wiki_symbols, tradable)
    symbols_dir = os.path.join(log_dir, "symbols")
    constituents.save_universe_snapshot(
        wiki_symbols,
        tradable,
        matched,
        output_root=symbols_dir,
        use_date_subdir=False,
        index_membership=index_membership,
    )
    constituents.save_unmatched_symbols(
        wiki_symbols,
        matched,
        output_dir=symbols_dir,
    )

    ranked, duplicate_tickers = strategy.analyze_universe(
        matched,
        config=strategy_config,
        log_dir=log_dir,
    )
    top_ranked = ranked[: strategy_config.top_n]
    top_tickers = [stock.ticker for stock in top_ranked]
    logger.info("Top %s tickers selected", len(top_tickers))

    risk_on = check_sp500_trend(strategy_config)

    positions = client.get_positions()
    if not isinstance(positions, list):
        raise Trading212Error("Positions response was not a list.")
    logger.info("Retrieved %s open positions", len(positions))

    sell_orders = portfolio.build_sell_orders(positions, keep_tickers=top_tickers)
    if sell_orders:
        execute_orders(client, sell_orders, extended_hours=trading_config.extended_hours)
    else:
        logger.info("No sell orders required")

    summary = client.get_account_summary()
    cash = None
    if isinstance(summary, dict):
        cash = summary.get("cash", {}).get("availableToTrade")
    if cash is None:
        raise RuntimeError("Account summary missing cash.availableToTrade")

    holdings_value = portfolio.calculate_holdings_value(positions, keep_tickers=top_tickers)
    total_equity = float(cash) + holdings_value
    logger.info("Cash available=%.2f holdings value=%.2f total equity=%.2f", cash, holdings_value, total_equity)

    positions_map = portfolio.extract_position_map(positions, allowed_tickers=top_tickers)
    new_buy_orders, rebalance_buy_orders, rebalance_sell_orders, remaining_cash = portfolio.build_rebalance_orders(
        ranked,
        positions_by_ticker=positions_map,
        cash=float(cash),
        total_equity=total_equity,
        risk_fraction=strategy_config.risk_fraction,
        top_n=strategy_config.top_n,
        gap_lookback_days=strategy_config.gap_lookback_days,
        rebalance_threshold=strategy_config.rebalance_threshold,
        max_position_fraction=strategy_config.max_position_fraction,
    )

    if rebalance_sell_orders:
        execute_orders(client, rebalance_sell_orders, extended_hours=trading_config.extended_hours)
    else:
        logger.info("No rebalance sell orders required")

    if not risk_on:
        logger.info("S&P500 below SMA%s. Exiting after sells/rebalance.", strategy_config.sma_long)
        logger.info(
            "Waiting %.1f seconds before fetching positions for holdings pie chart",
            runtime_config.holdings_pie_delay_seconds,
        )
        _sleep_with_progress(runtime_config.holdings_pie_delay_seconds, label="Waiting for fills")
        updated_positions = client.get_positions()
        if isinstance(updated_positions, list):
            charts.plot_holdings_pie(updated_positions, output_dir=log_dir)
        else:
            logger.warning("Unable to refresh positions for holdings pie chart.")
        return

    if rebalance_buy_orders:
        execute_orders(client, rebalance_buy_orders, extended_hours=trading_config.extended_hours)
    else:
        logger.info("No rebalance buy orders required")

    if new_buy_orders:
        execute_orders(client, new_buy_orders, extended_hours=trading_config.extended_hours)
    else:
        logger.info("No new buy orders required")

    logger.info(
        "Waiting %.1f seconds before fetching positions for holdings pie chart",
        runtime_config.holdings_pie_delay_seconds,
    )
    _sleep_with_progress(runtime_config.holdings_pie_delay_seconds, label="Waiting for fills")
    updated_positions = client.get_positions()
    if isinstance(updated_positions, list):
        charts.plot_holdings_pie(updated_positions, output_dir=log_dir)
    else:
        logger.warning("Unable to refresh positions for holdings pie chart.")

    logger.info("Run complete. Remaining cash estimate: %.2f", remaining_cash)


if __name__ == "__main__":
    main()
