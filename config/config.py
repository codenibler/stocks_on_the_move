from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass(frozen=True)
class Trading212Config:
    api_key: str
    api_secret: str
    base_url: str
    timeout_seconds: float
    extended_hours: bool
    retries: int
    retry_sleep_seconds: float


@dataclass(frozen=True)
class StrategyConfig:
    history_lookback: str
    momentum_lookback: str
    price_interval: str
    annualization_factor: float
    retries: int
    retry_sleep_seconds: float
    batch_size: int
    atr_period: int
    sma_short: int
    sma_long: int
    gap_threshold: float
    gap_lookback_days: int
    top_n: int
    chart_bucket: int
    risk_fraction: float
    sp500_ticker: str
    sp500_lookback: str
    rebalance_threshold: float
    max_position_fraction: float


@dataclass(frozen=True)
class RuntimeConfig:
    log_root: str
    run_log_root: str
    log_level: str
    holdings_pie_delay_seconds: float


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_base_url() -> str:
    base_url = os.getenv("TRADING212_BASE_URL")
    if base_url:
        base_url = base_url.rstrip("/")
        if base_url.endswith("/api/v0"):
            base_url = base_url[:-7]
        return base_url
    env = os.getenv("TRADING212_ENV", "demo").strip().lower()
    if env == "live":
        return "https://live.trading212.com"
    return "https://demo.trading212.com"


def get_trading212_config() -> Trading212Config:
    api_key = os.getenv("TRADING212_API_KEY") or os.getenv("API_KEY")
    api_secret = os.getenv("TRADING212_API_SECRET") or os.getenv("API_SECRET_KEY")
    if not api_key or not api_secret:
        missing = [name for name, value in {
            "TRADING212_API_KEY": api_key,
            "TRADING212_API_SECRET": api_secret,
        }.items() if not value]
        raise ValueError(f"Missing Trading212 credentials: {', '.join(missing)}")

    timeout_seconds = float(os.getenv("TRADING212_TIMEOUT_SECONDS", "30"))
    extended_hours = _env_bool("TRADING212_EXTENDED_HOURS", default=False)
    retries = int(os.getenv("TRADING212_RETRIES", "3"))
    retry_sleep_seconds = float(os.getenv("TRADING212_RETRY_SLEEP", "1"))
    return Trading212Config(
        api_key=api_key,
        api_secret=api_secret,
        base_url=_resolve_base_url(),
        timeout_seconds=timeout_seconds,
        extended_hours=extended_hours,
        retries=retries,
        retry_sleep_seconds=retry_sleep_seconds,
    )


def get_strategy_config() -> StrategyConfig:
    return StrategyConfig(
        history_lookback=os.getenv("HISTORY_LOOKBACK", "6mo"),
        momentum_lookback=os.getenv("MOMENTUM_LOOKBACK", "3mo"),
        price_interval=os.getenv("PRICE_INTERVAL", "1d"),
        annualization_factor=float(os.getenv("ANNUALIZATION_FACTOR", "252")),
        retries=int(os.getenv("YFINANCE_RETRIES", "3")),
        retry_sleep_seconds=float(os.getenv("YFINANCE_RETRY_SLEEP", "1")),
        batch_size=int(os.getenv("YFINANCE_BATCH_SIZE", "100")),
        atr_period=int(os.getenv("ATR_PERIOD", "20")),
        sma_short=int(os.getenv("SMA_SHORT", "100")),
        sma_long=int(os.getenv("SMA_LONG", "200")),
        gap_threshold=float(os.getenv("GAP_THRESHOLD", "0.15")),
        gap_lookback_days=int(os.getenv("GAP_LOOKBACK", "90")),
        top_n=int(os.getenv("TOP_N", "50")),
        chart_bucket=int(os.getenv("CHART_BUCKET", "25")),
        risk_fraction=float(os.getenv("RISK_FRACTION", "0.02")),
        sp500_ticker=os.getenv("SP500_TICKER", "^GSPC"),
        sp500_lookback=os.getenv("SP500_LOOKBACK", "1y"),
        rebalance_threshold=float(os.getenv("REBALANCE_THRESHOLD", "0.01")),
        max_position_fraction=float(os.getenv("MAX_POSITION_FRACTION", "0.10")),
    )


def get_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        log_root=os.getenv("LOG_ROOT", "logs"),
        run_log_root=os.getenv("RUN_LOG_ROOT", "run_logs"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        holdings_pie_delay_seconds=float(os.getenv("HOLDINGS_PIE_DELAY_SECONDS", "15")),
    )
