from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def build_yfinance_variants(base_symbol: str) -> List[str]:
    variants = []
    symbol = base_symbol.strip().upper()
    if not symbol:
        return variants
    variants.append(symbol)
    if "_" in symbol:
        variants.append(symbol.replace("_", "-"))
        variants.append(symbol.replace("_", "."))
        variants.append(symbol.replace("_", "/"))
    if "." in symbol:
        variants.append(symbol.replace(".", "-"))
    if "/" in symbol:
        variants.append(symbol.replace("/", "-"))
        variants.append(symbol.replace("/", "."))
    return list(dict.fromkeys(variants))


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
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            if df is None or df.empty:
                logger.warning("Empty yfinance data for %s on attempt %s", ticker, attempt)
            else:
                df = df.dropna(how="all")
                if df.empty:
                    logger.warning("All-NaN yfinance data for %s on attempt %s", ticker, attempt)
                else:
                    return df
        except Exception as exc:
            logger.warning("Error fetching %s on attempt %s: %s", ticker, attempt, exc)
        time.sleep(retry_sleep_seconds)
    return None


def fetch_history_with_variants(
    base_symbol: str,
    *,
    period: str,
    interval: str,
    retries: int,
    retry_sleep_seconds: float,
) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    variants = build_yfinance_variants(base_symbol)
    for variant in variants:
        df = fetch_history_with_retries(
            variant,
            period=period,
            interval=interval,
            retries=retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        if df is not None and not df.empty:
            logger.info("Resolved %s to yfinance ticker %s", base_symbol, variant)
            return variant, df
        logger.info("No data for yfinance ticker %s (base %s)", variant, base_symbol)
    return None, None


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
