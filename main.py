from __future__ import annotations

import html
import logging
import os
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
import yfinance as yf

from indicators import analytics
from indicators import charts
from indicators import strategy
from config.config import (
    get_runtime_config,
    get_strategy_config,
    get_telegram_config,
    get_trading212_config,
)
from config.logging_utils import setup_logging
from data_fetching import market_data
from execution import portfolio
from execution.telegram_client import TelegramClient, send_rebalance_report, TelegramError
from execution.trading212_client import Trading212Client, Trading212Error
from reports import generate_rebalance_report
from stock_universe import constituents
from tqdm import tqdm

logger = logging.getLogger(__name__)

def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s value %r; using default %.2f", name, raw, default)
        return default


if _env_bool("YFINANCE_DEBUG_MODE_ON", default=False) or _env_bool("DEBUG_MODE_ON", default=False):
    yf.enable_debug_mode()

RANKING_CHART_FILENAMES = (
    "top1to25momentum.png",
    "top26to50momentum.png",
    "top51to75momentum.png",
    "top76to100momentum.png",
)


def check_sp500_trend(strategy_config) -> tuple[bool, float | None, float | None]:
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
        return False, None, None
    df = market_data.normalize_price_frame(df)
    price_series = market_data.select_price_series(df)
    if price_series is None or price_series.dropna().empty:
        logger.warning("S&P500 data missing Close values, defaulting to risk-off")
        return False, None, None
    sma200 = analytics.calculate_sma(price_series, strategy_config.sma_long)
    if sma200 is None:
        logger.warning("S&P500 SMA%s unavailable, defaulting to risk-off", strategy_config.sma_long)
        return False, None, None
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
    return risk_on, current_price, sma200


def execute_orders(client: Trading212Client, orders, *, extended_hours: bool, stage: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for order in orders:
        side = "BUY" if order.quantity > 0 else "SELL"
        entry = {
            "stage": stage,
            "side": side,
            "ticker": order.ticker,
            "quantity": order.quantity,
            "order_id": None,
            "status": None,
            "error": None,
        }
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
            entry["order_id"] = order_id
            entry["status"] = status
            logger.info("Order placed for %s qty=%s id=%s status=%s", order.ticker, order.quantity, order_id, status)
        except Trading212Error as exc:
            entry["error"] = str(exc)
            entry["status"] = "ERROR"
            logger.error("Order failed for %s qty=%s: %s", order.ticker, order.quantity, exc)
        results.append(entry)
    return results


def _sleep_with_progress(seconds: float, *, label: str) -> None:
    total_seconds = max(0.0, seconds)
    whole_seconds = int(total_seconds)
    remainder = total_seconds - whole_seconds
    if whole_seconds > 0:
        for _ in tqdm(range(whole_seconds), desc=label, unit="s"):
            time.sleep(1)
    if remainder > 0:
        time.sleep(remainder)


def _sleep_for_summary_rate_limit() -> None:
    delay = max(0.0, _env_float("SUMMARY_RATE_LIMIT_SECONDS", 5.0))
    time.sleep(delay)


def _format_telegram_message(
    *,
    universe_summary: Dict[str, Any],
    analysis_summary: Dict[str, Any],
    regime_summary: Dict[str, Any],
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scraped = universe_summary.get("scraped_count", 0)
    matched = universe_summary.get("matched_count", 0)
    unmatched = universe_summary.get("unmatched_count", 0)
    index_counts = universe_summary.get("index_counts", {}) or {}
    match_stats = universe_summary.get("match_stats", {}) or {}
    drop_counts = analysis_summary.get("drop_counts", {}) or {}
    ranked_count = analysis_summary.get("ranked_count", 0)
    duplicate_count = analysis_summary.get("duplicate_count", 0)
    momentum_stats = analysis_summary.get("momentum_stats", {}) or {}
    score_stats = momentum_stats.get("score") or {}
    slope_stats = momentum_stats.get("slope") or {}
    r2_stats = momentum_stats.get("r_squared") or {}

    regime_price = regime_summary.get("price")
    regime_sma = regime_summary.get("sma")
    regime_window = regime_summary.get("sma_window")
    regime_ticker = regime_summary.get("ticker", "^GSPC")
    regime_risk_on = bool(regime_summary.get("risk_on"))
    comparator = ">=" if regime_risk_on else "<"
    emoji = "🟢" if regime_risk_on else "🔴"
    if isinstance(regime_price, (int, float)) and isinstance(regime_sma, (int, float)):
        regime_text = (
            f"{emoji} Regime Filter: {regime_ticker} last close {regime_price:.2f} "
            f"{comparator} SMA{regime_window} {regime_sma:.2f}"
        )
    else:
        regime_text = f"{emoji} Regime Filter: {regime_ticker} data unavailable (SMA{regime_window})"

    lines = [
        "Hey boss, here's your portfolio!",
        "",
        f"These are the results from your {now} rebalance:",
        "",
        f"<b>Universe Scraped</b>: {scraped} symbols",
        f"    S&P 500: {index_counts.get('SP500', 0)}",
        f"    S&P 400: {index_counts.get('SP400', 0)}",
        f"    S&P 600: {index_counts.get('SP600', 0)}",
        f"<b>Matched Instruments</b>: {matched}",
        f"    Normalized base ticker matches: {match_stats.get('base_matched', 0)}",
        f"    Dot/slash variant matches: {match_stats.get('variant_matched', 0)}",
        f"    ShortName metadata matches: {match_stats.get('short_matched', 0)}",
        f"<b>Unmatched Symbols</b>: {unmatched}",
        regime_text,
        "",
        "Here's the state of your risk gate:",
        "index_price_charts.png",
        "",
        f"Of the {matched} instruments, {ranked_count} remain. These were the reasons for dropouts:",
    ]

    for key in sorted(drop_counts.keys()):
        lines.append(f"{key}: {drop_counts.get(key, 0)}")
    lines.append("dropCountsBarChart.png")

    error_keys = [
        "no_data",
        "no_momentum_data",
        "nan_close",
        "momentum_failed",
        "atr_missing",
        "missing_close",
        "sma_missing",
        "missing_ticker",
        "missing_base_symbol",
    ]

    lines.extend(
        [
            "",
            "Summary",
            f"Errors during momentum prep: {sum(drop_counts.get(k, 0) for k in error_keys)}",
            f"Below SMA filter: {drop_counts.get('below_sma', 0)}",
            f"Gap >= 15% filter: {drop_counts.get('gap', 0)}",
            f"Ranked stocks: {ranked_count}",
            f"Duplicate tickers removed: {duplicate_count}",
            "<b>Momentum Stats</b>:",
        ]
    )

    if score_stats:
        lines.append(f"Score min/max: {score_stats.get('min', 0):.4f} / {score_stats.get('max', 0):.4f}")
        lines.append(f"Score mean/median: {score_stats.get('mean', 0):.4f} / {score_stats.get('median', 0):.4f}")
    if slope_stats:
        lines.append(f"Slope min/max: {slope_stats.get('min', 0):.6f} / {slope_stats.get('max', 0):.6f}")
        lines.append(f"Slope mean/median: {slope_stats.get('mean', 0):.6f} / {slope_stats.get('median', 0):.6f}")
    if r2_stats:
        lines.append(f"R^2 min/max: {r2_stats.get('min', 0):.4f} / {r2_stats.get('max', 0):.4f}")
        lines.append(f"R^2 mean/median: {r2_stats.get('mean', 0):.4f} / {r2_stats.get('median', 0):.4f}")
    lines.extend(
        [
            "momentum_scores.png",
            "momentum_slopes.png",
            "momentum_r2.png",
        ]
    )

    lines.extend(["", "These were the Top 100 momentum ranking stocks:"])
    lines.extend(RANKING_CHART_FILENAMES)
    lines.extend(
        [
            "",
            "this was your portfolio before today's rebalance:",
            "pre_rebalance_pie_chart.png",
            "",
            "And these were the orders we sent in:",
            "order_summary_code",
            "Order_submission_summary, as png.",
            "",
            "This is your portfolio now, after the rebalance:",
            "holdings_pie.png",
            "",
            "And this is how they're spread out on the indices.",
            "index_exposure_bar.png",
        ]
    )

    return "\n".join(lines)


def _build_telegram_blocks(
    message_text: str,
    *,
    attachments: Sequence[Tuple[str, object, str]],
) -> list[dict]:
    blocks: list[dict] = []
    remaining = message_text
    while remaining:
        next_marker = None
        next_pos = None
        next_payload = None
        next_type = None
        for marker, payload, attach_type in attachments:
            if not marker or payload is None:
                continue
            pos = remaining.find(marker)
            if pos == -1:
                continue
            if next_pos is None or pos < next_pos:
                next_pos = pos
                next_marker = marker
                next_payload = payload
                next_type = attach_type
        if next_marker is None or next_pos is None:
            text = remaining.strip()
            if text:
                blocks.append({"type": "text", "text": text})
            break
        before = remaining[:next_pos].strip()
        if before:
            blocks.append({"type": "text", "text": before})
        if next_type == "text":
            if isinstance(next_payload, list):
                for item in next_payload:
                    blocks.append({"type": "text", "text": str(item)})
            else:
                blocks.append({"type": "text", "text": str(next_payload)})
        else:
            blocks.append({"type": next_type or "photo", "path": str(next_payload)})
        remaining = remaining[next_pos + len(next_marker):]
    return blocks


def _send_telegram_report(
    telegram_config,
    *,
    report_path: str,
    pages_dir: str,
    message_blocks: list[dict],
    pre_pdf_blocks: Optional[list[dict]] = None,
) -> None:
    if not telegram_config.enabled:
        return
    if not telegram_config.api_token or not telegram_config.user_id:
        logger.warning("Telegram enabled but missing TELEGRAM_API_TOKEN or TELEGRAM_USER_ID.")
        return
    try:
        _send_telegram_alert(telegram_config, message="====================")
        send_rebalance_report(
            api_token=telegram_config.api_token,
            chat_id=str(telegram_config.user_id),
            report_path=report_path,
            pages_dir=pages_dir,
            send_delay_seconds=telegram_config.send_delay_seconds,
            message_blocks=message_blocks,
            pre_pdf_blocks=pre_pdf_blocks,
        )
        _send_telegram_alert(telegram_config, message="====================")
    except TelegramError as exc:
        logger.warning("Telegram notification failed: %s", exc)


def _send_telegram_alert(telegram_config, *, message: str) -> None:
    if not telegram_config.enabled:
        return
    if not telegram_config.api_token or not telegram_config.user_id:
        logger.warning("Telegram enabled but missing TELEGRAM_API_TOKEN or TELEGRAM_USER_ID.")
        return
    try:
        client = TelegramClient(api_token=telegram_config.api_token, timeout_seconds=telegram_config.timeout_seconds)
        client.send_message(str(telegram_config.user_id), message)
    except TelegramError as exc:
        logger.warning("Telegram alert failed: %s", exc)


def _existing_path(path: Optional[str]) -> Optional[str]:
    if path and os.path.isfile(path):
        return path
    return None


def _build_ticker_name_map(
    instruments: Sequence[dict],
    ranked: Sequence[object],
    positions: Sequence[dict],
) -> Dict[str, str]:
    name_map: Dict[str, str] = {}
    for inst in instruments:
        ticker = inst.get("ticker")
        if not ticker:
            continue
        name = inst.get("shortName") or inst.get("name")
        if name:
            name_map.setdefault(ticker, str(name))
    for stock in ranked:
        ticker = getattr(stock, "ticker", None)
        if not ticker:
            continue
        name = getattr(stock, "name", None)
        if name:
            name_map.setdefault(str(ticker), str(name))
    for position in positions:
        instrument = position.get("instrument", {})
        ticker = instrument.get("ticker")
        if not ticker:
            continue
        name = instrument.get("shortName") or instrument.get("name")
        if name:
            name_map.setdefault(str(ticker), str(name))
    return name_map


def _format_qty(qty: float) -> str:
    value = abs(qty)
    text = f"{value:.3f}"
    text = text.rstrip("0").rstrip(".")
    return text if text else "0"


def _format_order_table(order_results: Sequence[Dict[str, Any]], name_map: Dict[str, str]) -> str:
    rows: list[tuple[str, str, str, str]] = []
    for order in order_results:
        ticker = order.get("ticker")
        qty_raw = order.get("quantity")
        if not ticker or qty_raw is None:
            continue
        try:
            qty = float(qty_raw)
        except (TypeError, ValueError):
            continue
        orb = "🟢" if qty > 0 else "🔴"
        name = name_map.get(str(ticker), "")
        qty_text = _format_qty(qty)
        rows.append((orb, str(ticker), str(name), qty_text))

    header_ticker = "Ticker"
    header_company = "Company"
    header_qty = "QTY"
    ticker_width = max(len(header_ticker), *(len(row[1]) for row in rows)) if rows else len(header_ticker)
    company_width = max(len(header_company), *(len(row[2]) for row in rows)) if rows else len(header_company)
    qty_width = max(len(header_qty), *(len(row[3]) for row in rows)) if rows else len(header_qty)

    header = f"{header_ticker:<{ticker_width}}  {header_company:<{company_width}}  {header_qty:>{qty_width}}"
    separator = "-" * len(header)
    lines = [header, separator]
    if not rows:
        lines.append("No orders sent.")
        return "\n".join(lines)

    for orb, ticker, company, qty in rows:
        lines.append(
            f"{orb} {ticker:<{ticker_width}}  {company:<{company_width}}  {qty:>{qty_width}}"
        )
    return "\n".join(lines)


def _split_preformatted_blocks(text: str, *, max_chars: int = 3800) -> list[str]:
    escaped = html.escape(text)
    if not escaped:
        return ["<pre></pre>"]
    lines = escaped.splitlines()
    blocks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        line_len = len(line) + 1
        if current and current_len + line_len > max_chars:
            blocks.append("<pre>" + "\n".join(current) + "</pre>")
            current = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += line_len
    if current:
        blocks.append("<pre>" + "\n".join(current) + "</pre>")
    return blocks


def _format_order_code_blocks(
    *,
    order_results: Sequence[Dict[str, Any]],
    name_map: Dict[str, str],
    max_chars: int = 3800,
) -> list[str]:
    table = _format_order_table(order_results, name_map)
    if not table:
        return ["<pre>No orders sent.</pre>"]
    return _split_preformatted_blocks(table, max_chars=max_chars)


def _format_pending_positions_blocks(
    order_results: Sequence[Dict[str, Any]],
    name_map: Dict[str, str],
    *,
    max_chars: int = 3800,
) -> list[str]:
    lines = ["These are the positions still pending:"]
    pending_rows: list[str] = []
    for order in order_results:
        ticker = order.get("ticker")
        qty_raw = order.get("quantity")
        status = order.get("status")
        if not ticker or qty_raw is None:
            continue
        if not _is_pending_status(status):
            continue
        try:
            qty = float(qty_raw)
        except (TypeError, ValueError):
            continue
        name = name_map.get(str(ticker), "")
        pending_rows.append(f"🟡 {ticker}, {name}, {_format_qty(qty)}")
    if not pending_rows:
        lines.append("None.")
    else:
        lines.extend(pending_rows)
    return _split_preformatted_blocks("\n".join(lines), max_chars=max_chars)


def _is_pending_status(status: Optional[str]) -> bool:
    if status is None:
        return True
    if not isinstance(status, str):
        return True
    normalized = status.strip().upper()
    if normalized in {"FILLED", "EXECUTED", "COMPLETED", "SUCCESS", "CLOSED"}:
        return False
    if normalized in {"REJECTED", "CANCELLED", "CANCELED", "ERROR", "FAILED", "DECLINED"}:
        return False
    return True


def main() -> None:
    runtime_config = get_runtime_config()
    log_dir = setup_logging(
        runtime_config.log_root,
        runtime_config.run_log_root,
        runtime_config.log_level,
    )

    strategy_config = get_strategy_config()
    trading_config = get_trading212_config()
    telegram_config = get_telegram_config()

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
    index_counts = {"SP500": 0, "SP400": 0, "SP600": 0, "UNIDENTIFIED": 0}
    for indexes in index_membership.values():
        if not indexes:
            index_counts["UNIDENTIFIED"] += 1
            continue
        for index_label in indexes:
            if index_label in index_counts:
                index_counts[index_label] += 1
            else:
                index_counts["UNIDENTIFIED"] += 1

    instruments = constituents.fetch_trading212_instruments(client)
    tradable = constituents.filter_tradable_instruments(instruments)
    matched_result = constituents.cross_reference_constituents(wiki_symbols, tradable, return_stats=True)
    if isinstance(matched_result, tuple):
        matched, match_stats = matched_result
    else:
        matched = matched_result
        match_stats = {}
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
    unmatched_symbols = constituents.compute_unmatched_symbols(wiki_symbols, matched)

    ranked, duplicate_tickers, analysis_summary = strategy.analyze_universe(
        matched,
        config=strategy_config,
        log_dir=log_dir,
    )
    drop_counts = analysis_summary.get("drop_counts", {}) if isinstance(analysis_summary, dict) else {}
    drop_counts_chart_path = charts.plot_drop_counts_bar(
        drop_counts,
        output_dir=log_dir,
    )
    error_keys = [
        "no_data",
        "no_momentum_data",
        "nan_close",
        "momentum_failed",
        "atr_missing",
        "missing_close",
        "sma_missing",
        "missing_ticker",
        "missing_base_symbol",
    ]
    error_total = sum(drop_counts.get(key, 0) for key in error_keys)
    ranked_count = analysis_summary.get("ranked_count", 0) if isinstance(analysis_summary, dict) else 0
    duplicate_count = analysis_summary.get("duplicate_count", 0) if isinstance(analysis_summary, dict) else 0
    charts.plot_summary_counts(
        {
            "Errors": error_total,
            "Below SMA": drop_counts.get("below_sma", 0),
            "Gap >= 15%": drop_counts.get("gap", 0),
            "Ranked": ranked_count,
            "Duplicates": duplicate_count,
        },
        output_dir=log_dir,
    )
    no_data_threshold = max(0, int(strategy_config.no_data_abort_threshold))
    if no_data_threshold and drop_counts.get("no_data", 0) > no_data_threshold:
        logger.error(
            "Aborting run: no_data drop count exceeded %s (no_data=%s)",
            no_data_threshold,
            drop_counts.get("no_data"),
        )
        _send_telegram_alert(
            telegram_config,
            message="====================\nDawg, check the script out. Shit hit the fan.\n=====================",
        )
        return
    universe_summary = {
        "scraped_count": len(wiki_symbols),
        "matched_count": len(matched),
        "unmatched_count": len(unmatched_symbols),
        "match_stats": match_stats,
        "index_counts": index_counts,
    }
    top_ranked = ranked[: strategy_config.top_n]
    top_tickers = [stock.ticker for stock in top_ranked]
    logger.info("Top %s tickers selected", len(top_tickers))

    risk_on, sp500_price, sp500_sma = check_sp500_trend(strategy_config)

    positions = client.get_positions()
    if not isinstance(positions, list):
        raise Trading212Error("Positions response was not a list.")
    logger.info("Retrieved %s open positions", len(positions))
    ticker_name_map = _build_ticker_name_map(matched, ranked, positions)
    pre_cash = None
    _sleep_for_summary_rate_limit()
    summary_pre = client.get_account_summary()
    if isinstance(summary_pre, dict):
        pre_cash = summary_pre.get("cash", {}).get("availableToTrade")
    charts.plot_holdings_pie(
        positions,
        output_dir=log_dir,
        filename="pre_rebalance_pie_chart.png",
        cash_value=float(pre_cash) if pre_cash is not None else None,
    )

    order_results: List[Dict[str, Any]] = []
    sell_orders = portfolio.build_sell_orders(positions, keep_tickers=top_tickers)
    if sell_orders:
        order_results.extend(
            execute_orders(
                client,
                sell_orders,
                extended_hours=trading_config.extended_hours,
                stage="initial_sell",
            )
        )
    else:
        logger.info("No sell orders required")

    _sleep_for_summary_rate_limit()
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
        order_results.extend(
            execute_orders(
                client,
                rebalance_sell_orders,
                extended_hours=trading_config.extended_hours,
                stage="rebalance_sell",
            )
        )
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
        cash_for_chart = None
        _sleep_for_summary_rate_limit()
        summary_after = client.get_account_summary()
        if isinstance(summary_after, dict):
            cash_for_chart = summary_after.get("cash", {}).get("availableToTrade")
        if isinstance(updated_positions, list):
            charts.plot_holdings_pie(
                updated_positions,
                output_dir=log_dir,
                cash_value=float(cash_for_chart) if cash_for_chart is not None else None,
            )
            index_exposure_path = charts.plot_index_exposure_bar(
                updated_positions,
                output_dir=log_dir,
                cash_value=float(cash_for_chart) if cash_for_chart is not None else None,
            )
        else:
            logger.warning("Unable to refresh positions for holdings pie chart.")
            updated_positions = []
            index_exposure_path = None

        index_price_path = charts.plot_index_price_charts(
            {
                "S&P 500": strategy_config.sp500_ticker,
                "S&P 400": strategy_config.sp400_ticker,
                "S&P 600": strategy_config.sp600_ticker,
            },
            output_dir=log_dir,
            period=f"{365 + strategy_config.sma_long}d",
            interval=strategy_config.price_interval,
            retries=strategy_config.retries,
            retry_sleep_seconds=strategy_config.retry_sleep_seconds,
        )

        order_summary_blocks = _format_order_code_blocks(
            order_results=order_results,
            name_map=ticker_name_map,
        )
        momentum_charts = [
            os.path.join(log_dir, "momentum_charts", "rankings", name)
            for name in RANKING_CHART_FILENAMES
        ]
        regime_summary = {
            "risk_on": False,
            "price": sp500_price,
            "sma": sp500_sma,
            "sma_window": strategy_config.sma_long,
            "ticker": strategy_config.sp500_ticker,
        }
        report_path = generate_rebalance_report(
            output_dir=log_dir,
            report_date=date.today(),
            universe_summary=universe_summary,
            analysis_summary=analysis_summary,
            order_results=order_results,
            regime_summary=regime_summary,
            momentum_chart_paths=momentum_charts,
            pre_pie_path=os.path.join(log_dir, "momentum_charts", "holdings_charts", "pre_rebalance_pie_chart.png"),
            post_pie_path=os.path.join(log_dir, "momentum_charts", "holdings_charts", "holdings_pie.png"),
            index_exposure_path=index_exposure_path,
            index_price_path=index_price_path,
            drop_counts_chart_path=drop_counts_chart_path,
            page_images_dir=os.path.join(log_dir, "report_pages"),
        )
        message_text = _format_telegram_message(
            universe_summary=universe_summary,
            analysis_summary=analysis_summary,
            regime_summary=regime_summary,
        )
        ranking_markers = [
            (name, _existing_path(os.path.join(log_dir, "momentum_charts", "rankings", name)))
            for name in RANKING_CHART_FILENAMES
        ]
        pending_blocks = _format_pending_positions_blocks(order_results, ticker_name_map)
        message_blocks = _build_telegram_blocks(
            message_text,
            attachments=[
                ("index_price_charts.png", _existing_path(index_price_path), "photo"),
                ("dropCountsBarChart.png", _existing_path(drop_counts_chart_path), "photo"),
                ("momentum_scores.png", _existing_path(os.path.join(log_dir, "momentum_charts", "regression_metrics", "momentum_scores.png")), "photo"),
                ("momentum_slopes.png", _existing_path(os.path.join(log_dir, "momentum_charts", "regression_metrics", "momentum_slopes.png")), "photo"),
                ("momentum_r2.png", _existing_path(os.path.join(log_dir, "momentum_charts", "regression_metrics", "momentum_r2.png")), "photo"),
                *[(marker, path, "photo") for marker, path in ranking_markers],
                ("pre_rebalance_pie_chart.png", _existing_path(os.path.join(log_dir, "momentum_charts", "holdings_charts", "pre_rebalance_pie_chart.png")), "photo"),
                ("order_summary_code", order_summary_blocks, "text"),
                ("holdings_pie.png", _existing_path(os.path.join(log_dir, "momentum_charts", "holdings_charts", "holdings_pie.png")), "photo"),
                ("index_exposure_bar.png", _existing_path(index_exposure_path), "photo"),
            ],
        )
        _send_telegram_report(
            telegram_config,
            report_path=report_path,
            pages_dir=os.path.join(log_dir, "report_pages"),
            message_blocks=message_blocks,
            pre_pdf_blocks=[{"type": "text", "text": block} for block in pending_blocks],
        )
        return

    if rebalance_buy_orders:
        order_results.extend(
            execute_orders(
                client,
                rebalance_buy_orders,
                extended_hours=trading_config.extended_hours,
                stage="rebalance_buy",
            )
        )
    else:
        logger.info("No rebalance buy orders required")

    if new_buy_orders:
        order_results.extend(
            execute_orders(
                client,
                new_buy_orders,
                extended_hours=trading_config.extended_hours,
                stage="new_buy",
            )
        )
    else:
        logger.info("No new buy orders required")

    logger.info(
        "Waiting %.1f seconds before fetching positions for holdings pie chart",
        runtime_config.holdings_pie_delay_seconds,
    )
    _sleep_with_progress(runtime_config.holdings_pie_delay_seconds, label="Waiting for fills")
    updated_positions = client.get_positions()
    cash_for_chart = None
    _sleep_for_summary_rate_limit()
    summary_after = client.get_account_summary()
    if isinstance(summary_after, dict):
        cash_for_chart = summary_after.get("cash", {}).get("availableToTrade")
    if isinstance(updated_positions, list):
        charts.plot_holdings_pie(
            updated_positions,
            output_dir=log_dir,
            cash_value=float(cash_for_chart) if cash_for_chart is not None else None,
        )
        index_exposure_path = charts.plot_index_exposure_bar(
            updated_positions,
            output_dir=log_dir,
            cash_value=float(cash_for_chart) if cash_for_chart is not None else None,
        )
    else:
        logger.warning("Unable to refresh positions for holdings pie chart.")
        updated_positions = []
        index_exposure_path = None

    index_price_path = charts.plot_index_price_charts(
        {
            "S&P 500": strategy_config.sp500_ticker,
            "S&P 400": strategy_config.sp400_ticker,
            "S&P 600": strategy_config.sp600_ticker,
        },
        output_dir=log_dir,
        period=f"{365 + strategy_config.sma_long}d",
        interval=strategy_config.price_interval,
        retries=strategy_config.retries,
        retry_sleep_seconds=strategy_config.retry_sleep_seconds,
    )

    order_summary_blocks = _format_order_code_blocks(
        order_results=order_results,
        name_map=ticker_name_map,
    )
    momentum_charts = [
        os.path.join(log_dir, "momentum_charts", "rankings", name)
        for name in RANKING_CHART_FILENAMES
    ]
    regime_summary = {
        "risk_on": risk_on,
        "price": sp500_price,
        "sma": sp500_sma,
        "sma_window": strategy_config.sma_long,
        "ticker": strategy_config.sp500_ticker,
    }
    report_path = generate_rebalance_report(
        output_dir=log_dir,
        report_date=date.today(),
        universe_summary=universe_summary,
        analysis_summary=analysis_summary,
        order_results=order_results,
        regime_summary=regime_summary,
        momentum_chart_paths=momentum_charts,
        pre_pie_path=os.path.join(log_dir, "momentum_charts", "holdings_charts", "pre_rebalance_pie_chart.png"),
        post_pie_path=os.path.join(log_dir, "momentum_charts", "holdings_charts", "holdings_pie.png"),
        index_exposure_path=index_exposure_path,
        index_price_path=index_price_path,
        drop_counts_chart_path=drop_counts_chart_path,
        page_images_dir=os.path.join(log_dir, "report_pages"),
    )
    message_text = _format_telegram_message(
        universe_summary=universe_summary,
        analysis_summary=analysis_summary,
        regime_summary=regime_summary,
    )
    ranking_markers = [
        (name, _existing_path(os.path.join(log_dir, "momentum_charts", "rankings", name)))
        for name in RANKING_CHART_FILENAMES
    ]
    pending_blocks = _format_pending_positions_blocks(order_results, ticker_name_map)
    message_blocks = _build_telegram_blocks(
        message_text,
        attachments=[
            ("index_price_charts.png", _existing_path(index_price_path), "photo"),
            ("dropCountsBarChart.png", _existing_path(drop_counts_chart_path), "photo"),
            ("momentum_scores.png", _existing_path(os.path.join(log_dir, "momentum_charts", "regression_metrics", "momentum_scores.png")), "photo"),
            ("momentum_slopes.png", _existing_path(os.path.join(log_dir, "momentum_charts", "regression_metrics", "momentum_slopes.png")), "photo"),
            ("momentum_r2.png", _existing_path(os.path.join(log_dir, "momentum_charts", "regression_metrics", "momentum_r2.png")), "photo"),
            *[(marker, path, "photo") for marker, path in ranking_markers],
            ("pre_rebalance_pie_chart.png", _existing_path(os.path.join(log_dir, "momentum_charts", "holdings_charts", "pre_rebalance_pie_chart.png")), "photo"),
            ("order_summary_code", order_summary_blocks, "text"),
            ("holdings_pie.png", _existing_path(os.path.join(log_dir, "momentum_charts", "holdings_charts", "holdings_pie.png")), "photo"),
            ("index_exposure_bar.png", _existing_path(index_exposure_path), "photo"),
        ],
    )
    _send_telegram_report(
        telegram_config,
        report_path=report_path,
        pages_dir=os.path.join(log_dir, "report_pages"),
        message_blocks=message_blocks,
        pre_pdf_blocks=[{"type": "text", "text": block} for block in pending_blocks],
    )

    logger.info("Run complete. Remaining cash estimate: %.2f", remaining_cash)


if __name__ == "__main__":
    main()
