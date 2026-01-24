from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from . import analytics
from . import charts
from config.config import StrategyConfig
from config.classes import AnalyzedStock
from data_fetching import market_data
from stock_universe.constituents import extract_trading212_base_symbol

logger = logging.getLogger(__name__)


def _lookback_to_offset(lookback: str) -> Optional[pd.DateOffset]:
    lookback = lookback.strip().lower()
    if lookback.endswith("mo"):
        return pd.DateOffset(months=int(lookback[:-2]))
    if lookback.endswith("y"):
        return pd.DateOffset(years=int(lookback[:-1]))
    if lookback.endswith("d"):
        return pd.DateOffset(days=int(lookback[:-1]))
    return None


def _slice_lookback(df: pd.DataFrame, lookback: str) -> pd.DataFrame:
    offset = _lookback_to_offset(lookback)
    if offset is None:
        return df
    cutoff = datetime.utcnow() - offset
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
    if df.index.tz is not None:
        df = df.tz_convert(None)
    return df[df.index >= cutoff]


def analyze_universe(
    instruments: List[dict],
    *,
    config: StrategyConfig,
    log_dir: str,
) -> Tuple[List[AnalyzedStock], List[str]]:
    logger.info("Analyzing %s instruments for momentum", len(instruments))
    results: List[AnalyzedStock] = []
    drop_counts: Dict[str, int] = {}
    symbol_entries: List[tuple[dict, str]] = []

    for inst in instruments:
        ticker = inst.get("ticker")
        if not ticker:
            drop_counts["missing_ticker"] = drop_counts.get("missing_ticker", 0) + 1
            continue
        base_symbol = extract_trading212_base_symbol(ticker)
        if not base_symbol:
            drop_counts["missing_base_symbol"] = drop_counts.get("missing_base_symbol", 0) + 1
            continue
        symbol_entries.append((inst, base_symbol))

    resolved_histories = market_data.resolve_history_for_symbols(
        {base_symbol for _, base_symbol in symbol_entries},
        period=config.history_lookback,
        interval=config.price_interval,
        retries=config.retries,
        retry_sleep_seconds=config.retry_sleep_seconds,
        batch_size=config.batch_size,
    )

    for inst, base_symbol in symbol_entries:
        ticker = inst.get("ticker", "")
        resolved = resolved_histories.get(base_symbol)
        if not resolved:
            logger.info("Dropping %s: no yfinance data", base_symbol)
            drop_counts["no_data"] = drop_counts.get("no_data", 0) + 1
            continue
        yfinance_symbol, df_full = resolved
        if df_full is None or df_full.empty:
            logger.info("Dropping %s: no yfinance data", base_symbol)
            drop_counts["no_data"] = drop_counts.get("no_data", 0) + 1
            continue
        df_full = market_data.normalize_price_frame(df_full.sort_index())

        momentum_df = _slice_lookback(df_full, config.momentum_lookback)
        if momentum_df is None or momentum_df.empty:
            logger.info("Dropping %s: no momentum data", base_symbol)
            drop_counts["no_momentum_data"] = drop_counts.get("no_momentum_data", 0) + 1
            continue

        price_series = market_data.select_price_series(momentum_df)
        if price_series is None or price_series.dropna().empty:
            logger.info("Dropping %s: NaN price series", base_symbol)
            drop_counts["nan_close"] = drop_counts.get("nan_close", 0) + 1
            continue

        momentum = analytics.calculate_momentum_score(
            price_series,
            annualization_factor=config.annualization_factor,
        )
        if not momentum:
            logger.info("Dropping %s: momentum regression failed", base_symbol)
            drop_counts["momentum_failed"] = drop_counts.get("momentum_failed", 0) + 1
            continue
        score, slope, r_squared = momentum

        atr20 = analytics.calculate_atr(df_full, config.atr_period)
        if atr20 is None or atr20 <= 0:
            logger.info("Dropping %s: ATR unavailable", base_symbol)
            drop_counts["atr_missing"] = drop_counts.get("atr_missing", 0) + 1
            continue

        full_price = market_data.select_price_series(df_full)
        if full_price is None or full_price.dropna().empty:
            logger.info("Dropping %s: missing price data", base_symbol)
            drop_counts["missing_close"] = drop_counts.get("missing_close", 0) + 1
            continue

        current_price = float(full_price.dropna().iloc[-1])
        sma100 = analytics.calculate_sma(full_price, config.sma_short)
        if sma100 is None:
            logger.info("Dropping %s: SMA%s unavailable", base_symbol, config.sma_short)
            drop_counts["sma_missing"] = drop_counts.get("sma_missing", 0) + 1
            continue
        if current_price < sma100:
            logger.info("Dropping %s: price %.2f below SMA%s %.2f", base_symbol, current_price, config.sma_short, sma100)
            drop_counts["below_sma"] = drop_counts.get("below_sma", 0) + 1
            continue

        gap_pct = analytics.find_max_gap_percent(
            df_full,
            lookback_days=config.gap_lookback_days,
        )
        if gap_pct is not None and gap_pct >= (config.gap_threshold * 100.0):
            logger.info(
                "Dropping %s: max gap %.2f%% >= %.2f%% in last %s days",
                base_symbol,
                gap_pct,
                config.gap_threshold * 100.0,
                config.gap_lookback_days,
            )
            drop_counts["gap"] = drop_counts.get("gap", 0) + 1
            continue

        results.append(
            AnalyzedStock(
                ticker=ticker,
                base_symbol=base_symbol,
                yfinance_symbol=yfinance_symbol or base_symbol,
                name=inst.get("name", ""),
                score=score,
                atr20=atr20,
                current_price=current_price,
                sma100=sma100,
                max_gap_percent=gap_pct,
                slope=slope,
                r_squared=r_squared,
            )
        )

    ticker_counts: Dict[str, int] = {}
    for stock in results:
        ticker_counts[stock.ticker] = ticker_counts.get(stock.ticker, 0) + 1

    duplicates = sorted(ticker for ticker, count in ticker_counts.items() if count > 1)
    duplicate_count = sum(count - 1 for count in ticker_counts.values() if count > 1)

    unique_by_ticker: Dict[str, AnalyzedStock] = {}
    for stock in results:
        existing = unique_by_ticker.get(stock.ticker)
        if existing is None or stock.score > existing.score:
            unique_by_ticker[stock.ticker] = stock

    ranked = sorted(unique_by_ticker.values(), key=lambda stock: stock.score, reverse=True)
    logger.info("Momentum ranking complete. Kept %s stocks", len(ranked))
    if duplicate_count:
        logger.info("Removed %s duplicate tickers from rankings", duplicate_count)
        logger.info("Duplicate tickers detected: %s", len(duplicates))
        for ticker in duplicates:
            logger.info("Duplicate ticker: %s", ticker)
    if drop_counts:
        logger.info("Drop summary: %s", drop_counts)

    charts.plot_momentum_buckets(
        ranked,
        output_dir=log_dir,
        bucket_size=config.chart_bucket,
    )
    return ranked, duplicates


def _chunked_list(items: List[str], size: int) -> List[List[str]]:
    if size <= 0:
        return [items]
    return [items[idx : idx + size] for idx in range(0, len(items), size)]
