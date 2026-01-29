from __future__ import annotations

import logging
import os
import random
import time
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from fake_useragent import UserAgent

logger = logging.getLogger(__name__)

_USER_AGENT: Optional[UserAgent] = None


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s value %r; using default %.2f", name, raw, default)
        return default


def _get_random_user_agent() -> Optional[str]:
    global _USER_AGENT
    if _USER_AGENT is None:
        try:
            _USER_AGENT = UserAgent()
        except Exception as exc:
            logger.warning("Failed to initialize fake-useragent: %s", exc)
            return None
    try:
        return _USER_AGENT.random
    except Exception as exc:
        logger.warning("Failed to generate random user agent: %s", exc)
        return None


def _apply_yfinance_user_agent() -> None:
    user_agent = _get_random_user_agent()
    if not user_agent:
        return
    try:
        from yfinance.data import YfData

        session = YfData()._session
        if session is None:
            return
        session.headers.update({"User-Agent": user_agent})
    except Exception as exc:
        logger.warning("Failed to set yfinance User-Agent: %s", exc)


def _batch_sleep_bounds() -> tuple[float, float]:
    lower = _get_env_float("YFINANCE_BATCH_SLEEP_LOWER", 1.0)
    upper = _get_env_float("YFINANCE_BATCH_SLEEP_UPPER", 5.0)
    if lower < 0:
        lower = 0.0
    if upper < lower:
        upper = lower
    return lower, upper


def _sleep_between_batches() -> None:
    lower, upper = _batch_sleep_bounds()
    if upper <= 0:
        return
    delay = random.uniform(lower, upper)
    if delay <= 0:
        return
    logger.info("Sleeping %.3f seconds before next yfinance batch", delay)
    time.sleep(delay)


def build_yfinance_tickers(base_symbol: str) -> List[str]:
    tickers = []
    symbol = base_symbol.strip().upper()
    if not symbol:
        return tickers
    tickers.append(symbol)
    stripped = symbol.strip("._-/0123456789")
    if stripped and stripped != symbol:
        tickers.append(stripped)
    if "_" in symbol:
        tickers.append(symbol.replace("_", "-"))
        tickers.append(symbol.replace("_", "."))
        tickers.append(symbol.replace("_", "/"))
    if "." in symbol:
        tickers.append(symbol.replace(".", "-"))
    if "/" in symbol:
        tickers.append(symbol.replace("/", "-"))
        tickers.append(symbol.replace("/", "."))
    return list(dict.fromkeys(tickers))


def fetch_history_with_retries(
    ticker: str,
    *,
    period: str,
    interval: str,
    retries: int,
    retry_sleep_seconds: float,
) -> Optional[pd.DataFrame]:
    for attempt in range(1, retries + 1):
        try:
            logger.debug(
                "Fetching yfinance data: ticker=%s period=%s interval=%s attempt=%s",
                ticker,
                period,
                interval,
                attempt,
            )
            _apply_yfinance_user_agent()
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            if df is None or df.empty:
                logger.warning("Empty yfinance data for %s; skipping retries", ticker)
                return None
            else:
                df = df.dropna(how="all")
                if df.empty:
                    logger.warning("All-NaN yfinance data for %s; skipping retries", ticker)
                    return None
                else:
                    return df
        except Exception as exc:
            logger.warning("Error fetching %s on attempt %s: %s", ticker, attempt, exc)
        time.sleep(retry_sleep_seconds)
    return None


def fetch_history_with_tickers(
    base_symbol: str,
    *,
    period: str,
    interval: str,
    retries: int,
    retry_sleep_seconds: float,
) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    tickers = build_yfinance_tickers(base_symbol)
    for ticker in tickers:
        df = fetch_history_with_retries(
            ticker,
            period=period,
            interval=interval,
            retries=retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        if df is not None and not df.empty:
            logger.info("Resolved %s to yfinance ticker %s", base_symbol, ticker)
            return ticker, df
        logger.info("No data for yfinance ticker %s (base %s)", ticker, base_symbol)
    return None, None


def resolve_history_for_symbols(
    base_symbols: Iterable[str],
    *,
    period: str,
    interval: str,
    retries: int,
    retry_sleep_seconds: float,
    batch_size: int,
) -> Dict[str, Tuple[str, pd.DataFrame]]:
    symbols = sorted({symbol for symbol in base_symbols if symbol})
    tickers_map = {symbol: build_yfinance_tickers(symbol) for symbol in symbols}
    base_tickers = sorted(
        {variants[0] for variants in tickers_map.values() if variants}
    )

    resolved: Dict[str, Tuple[str, pd.DataFrame]] = {}
    batch_results = fetch_history_batch(
        base_tickers,
        period=period,
        interval=interval,
        retries=retries,
        retry_sleep_seconds=retry_sleep_seconds,
        batch_size=batch_size,
    )

    for base_symbol in symbols:
        variants = tickers_map.get(base_symbol, [])
        if not variants:
            continue
        base_ticker = variants[0]
        resolved_tuple = batch_results.get(base_ticker)
        if not resolved_tuple:
            continue
        resolved[base_symbol] = resolved_tuple
        logger.info("Resolved %s to yfinance ticker %s", base_symbol, resolved_tuple[0])

    unresolved_count = len([symbol for symbol in symbols if symbol not in resolved])
    if unresolved_count:
        logger.info("No yfinance data resolved for %s symbols.", unresolved_count)

    return resolved


def fetch_history_batch(
    tickers: List[str],
    *,
    period: str,
    interval: str,
    retries: int,
    retry_sleep_seconds: float,
    batch_size: int,
) -> Dict[str, Tuple[str, pd.DataFrame]]:
    results: Dict[str, Tuple[str, pd.DataFrame]] = {}
    sorted_tickers = sorted(tickers)
    chunks = list(_chunked(sorted_tickers, batch_size))
    for chunk_index, chunk in enumerate(chunks):
        batch_frames: Dict[str, pd.DataFrame] = {}
        try:
            if chunk:
                start_idx = chunk_index * batch_size + 1
                end_idx = start_idx + len(chunk) - 1
                logger.info(
                    "Fetching batch of tickers %s-%s (%s:%s)",
                    start_idx,
                    end_idx,
                    chunk[0],
                    chunk[-1],
                )
            _apply_yfinance_user_agent()
            batch_df = yf.download(
                chunk,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=True,
                group_by="ticker",
            )
            batch_frames = _split_batch_dataframe(batch_df, chunk)
        except Exception as exc:
            logger.warning("Batch download failed for %s tickers: %s", len(chunk), exc)

        succeeded = sum(1 for ticker in chunk if batch_frames.get(ticker) is not None and not batch_frames[ticker].empty)
        failed = len(chunk) - succeeded
        logger.info(
            "Batch result: %s succeeded, %s failed",
            succeeded,
            failed,
        )

        for ticker in chunk:
            df = batch_frames.get(ticker)
            if df is None or df.empty:
                logger.info("Retrying %s individually after batch miss", ticker)
                df = fetch_history_with_retries(
                    ticker,
                    period=period,
                    interval=interval,
                    retries=retries,
                    retry_sleep_seconds=retry_sleep_seconds,
                )
            used_ticker = ticker
            if df is None or df.empty:
                variants = build_yfinance_tickers(ticker)
                if len(variants) > 1:
                    for variant in variants[1:]:
                        logger.info("Trying variant %s for base %s", variant, ticker)
                        df = fetch_history_with_retries(
                            variant,
                            period=period,
                            interval=interval,
                            retries=retries,
                            retry_sleep_seconds=retry_sleep_seconds,
                        )
                        if df is None or df.empty:
                            continue
                        used_ticker = variant
                        break
            if df is None or df.empty:
                continue
            results[ticker] = (used_ticker, df)
        if chunk_index < len(chunks) - 1:
            _sleep_between_batches()
    return results


def normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    if "Open" not in df.columns.get_level_values(0) and "Open" in df.columns.get_level_values(1):
        df = df.swaplevel(axis=1).sort_index(axis=1)
        logger.info("Swapped yfinance column levels to normalize OHLC layout.")
    normalized = pd.DataFrame(index=df.index)
    for field in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
        if field not in df.columns.get_level_values(0):
            continue
        series = _coerce_series(df.xs(field, level=0, axis=1), label=field)
        if series is not None:
            normalized[field] = series
    if normalized.empty:
        return df
    return normalized


def select_price_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    df = normalize_price_frame(df)
    if isinstance(df.columns, pd.MultiIndex):
        for field in ("Adj Close", "Close"):
            if field in df.columns.get_level_values(0):
                return _coerce_series(df.xs(field, level=0, axis=1), label=field)
        return None
    for field in ("Adj Close", "Close"):
        if field in df.columns:
            return _coerce_series(df[field], label=field)
    return None


def _coerce_series(data: pd.DataFrame | pd.Series, *, label: str) -> Optional[pd.Series]:
    if data is None:
        return None
    if isinstance(data, pd.Series):
        return data
    if data.empty:
        return None
    if data.shape[1] == 1:
        return data.iloc[:, 0]
    logger.warning("Multiple columns for %s data; using first column.", label)
    return data.iloc[:, 0]


def _split_batch_dataframe(
    df: pd.DataFrame,
    tickers: List[str],
) -> Dict[str, pd.DataFrame]:
    if df is None or df.empty:
        return {}
    if not isinstance(df.columns, pd.MultiIndex):
        if len(tickers) == 1:
            return {tickers[0]: df}
        logger.warning("Batch download returned flat columns for multiple tickers.")
        return {}

    level0 = set(df.columns.get_level_values(0))
    level1 = set(df.columns.get_level_values(1))
    results: Dict[str, pd.DataFrame] = {}
    if any(ticker in level0 for ticker in tickers):
        for ticker in tickers:
            if ticker in level0:
                results[ticker] = df[ticker].dropna(how="all")
        return results
    if any(ticker in level1 for ticker in tickers):
        for ticker in tickers:
            if ticker in level1:
                results[ticker] = df.xs(ticker, level=1, axis=1).dropna(how="all")
        return results

    logger.warning("Batch download columns did not match requested tickers.")
    return {}


def _chunked(items: List[str], size: int) -> Iterable[List[str]]:
    if size <= 0:
        yield items
        return
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]
