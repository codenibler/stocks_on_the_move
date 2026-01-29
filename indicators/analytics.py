from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress

logger = logging.getLogger(__name__)


def calculate_momentum_score(
    close: pd.Series,
    *,
    annualization_factor: float,
) -> Optional[tuple[float, float, float]]:
    series = close.dropna()
    if series.shape[0] < 2:
        return None
    log_prices = np.log(series.values)
    x = np.arange(len(log_prices))
    regression = linregress(x, log_prices)
    r_squared = regression.rvalue ** 2
    score = regression.slope * annualization_factor * r_squared
    return float(score), float(regression.slope), float(r_squared)


def calculate_atr(df: pd.DataFrame, period: int) -> Optional[float]:
    if df is None or df.empty:
        return None
    if not {"High", "Low", "Close"}.issubset(df.columns):
        logger.warning("ATR calculation missing required columns")
        return None
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    value = atr.iloc[-1]
    if pd.isna(value):
        return None
    return float(value)


def calculate_sma(close: pd.Series, period: int) -> Optional[float]:
    if close is None or close.empty:
        return None
    sma = close.rolling(window=period, min_periods=period).mean()
    value = sma.iloc[-1]
    if pd.isna(value):
        return None
    return float(value)


def find_max_gap_percent(
    df: pd.DataFrame,
    *,
    lookback_days: int,
) -> Optional[float]:
    if df is None or df.empty:
        return None
    if not {"Open", "Close"}.issubset(df.columns):
        return None
    prev_close = df["Close"].shift(1)
    gap = (df["Open"] - prev_close).abs() / prev_close
    gap = gap.replace([np.inf, -np.inf], np.nan)
    if isinstance(df.index, pd.DatetimeIndex):
        idx = pd.to_datetime(df.index, utc=True, errors="coerce")
        gap = gap.copy()
        gap.index = idx.tz_convert(None)
        cutoff = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)).tz_convert(None)
        gap = gap[gap.index >= cutoff]
    else:
        gap = gap.tail(lookback_days)
    if gap.empty:
        return None
    max_gap = gap.max()
    if pd.isna(max_gap):
        return None
    return float(max_gap * 100.0)
