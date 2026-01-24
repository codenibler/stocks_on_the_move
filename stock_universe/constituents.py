from __future__ import annotations

import logging
import os
import re
import time
import csv
from datetime import date
from io import StringIO
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd
import requests
from dotenv import dotenv_values, load_dotenv

from execution.trading212_client import Trading212Client, Trading212Error

logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=".env", override=True)

_CONSTITUENT_SUFFIX = "_CONSTITUENTS"
_SYMBOL_CLEAN_RE = re.compile(r"\s*\[[^\]]+\]\s*")
_SYMBOL_COLUMNS = {"symbol", "ticker", "ticker symbol"}
_WIKIPEDIA_USER_AGENT = os.getenv(
    "WIKIPEDIA_USER_AGENT",
    "sotm-constituents/0.1 (+https://example.com)",
)
DEFAULT_CONSTITUENT_LINKS = [
    "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
]


def load_constituent_links(env_path: str = ".env") -> List[str]:
    values = dotenv_values(env_path)
    links = _extract_links(values)
    if not links:
        links = _extract_links(os.environ)
    if not links:
        logger.info("No constituent links configured; using default S&P URLs.")
        links = list(DEFAULT_CONSTITUENT_LINKS)
    return links


def scrape_wikipedia_constituents(
    urls: Iterable[str],
    *,
    timeout_seconds: float = 30.0,
) -> Set[str]:
    symbols: Set[str] = set()
    for url in urls:
        logger.info("Fetching constituents from %s", url)
        html = _fetch_url(url, timeout_seconds=timeout_seconds)
        extracted = _extract_symbols_from_html(html, source_url=url)
        logger.info("Extracted %s symbols from %s", len(extracted), url)
        symbols.update(extracted)

    logger.info("Total unique symbols collected: %s", len(symbols))
    return symbols


def fetch_trading212_instruments(client: Trading212Client) -> List[Dict[str, Any]]:
    instruments = client.get_instruments()
    if not isinstance(instruments, list):
        raise Trading212Error("Trading212 instruments response was not a list.")
    return instruments


def filter_tradable_instruments(instruments: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    total = 0
    missing_ticker = 0
    non_stock = 0
    non_usd = 0
    missing_currency = 0
    for inst in instruments:
        total += 1
        ticker = inst.get("ticker")
        inst_type = inst.get("type")
        currency = inst.get("currencyCode")
        if not ticker or not isinstance(ticker, str):
            missing_ticker += 1
            continue
        if inst_type != "STOCK":
            non_stock += 1
            continue
        if currency is None:
            missing_currency += 1
            continue
        if currency != "USD":
            non_usd += 1
            continue
        filtered.append(inst)
    logger.info(
        "Filtered tradable instruments: %s/%s (missing_ticker=%s, non_stock=%s, missing_currency=%s, non_usd=%s)",
        len(filtered),
        total,
        missing_ticker,
        non_stock,
        missing_currency,
        non_usd,
    )
    return filtered


def cross_reference_constituents(
    wiki_symbols: Iterable[str],
    instruments: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    instruments = list(instruments)
    normalized = {normalize_symbol(sym) for sym in wiki_symbols if sym}
    matched: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    base_to_instrument: Dict[str, Dict[str, Any]] = {}
    raw_base_to_instrument: Dict[str, Dict[str, Any]] = {}
    short_to_instrument: Dict[str, Dict[str, Any]] = {}

    logger.info(
        "Cross-referencing %s wiki symbols against %s instruments",
        len(normalized),
        len(instruments),
    )

    for inst in instruments:
        ticker = inst.get("ticker", "")
        base_symbol = extract_trading212_base_symbol(ticker)
        if not base_symbol:
            continue
        key = normalize_symbol(base_symbol)
        base_to_instrument.setdefault(key, inst)
        raw_base_to_instrument.setdefault(base_symbol.upper(), inst)
        short_name = inst.get("shortName")
        if short_name:
            short_to_instrument.setdefault(normalize_symbol(str(short_name)), inst)

    for key in sorted(normalized):
        inst = base_to_instrument.get(key)
        if inst and key not in seen:
            matched.append(inst)
            seen.add(key)

    additional = 0
    for key in sorted(normalized - seen):
        for variant in _dot_variants(key):
            inst = raw_base_to_instrument.get(variant)
            if inst:
                matched.append(inst)
                seen.add(key)
                additional += 1
                break

    short_added = 0
    for key in sorted(normalized - seen):
        inst = short_to_instrument.get(key)
        if inst:
            matched.append(inst)
            seen.add(key)
            short_added += 1

    if additional:
        logger.info("Matched %s additional symbols via dot variants.", additional)
    if short_added:
        logger.info("Matched %s additional symbols via shortName metadata.", short_added)
    logger.info("Matched %s Trading212 instruments to S&P constituents.", len(matched))
    return matched


def get_tradeable_constituents(
    client: Trading212Client,
    *,
    env_path: str = ".env",
    timeout_seconds: float = 30.0,
) -> List[Dict[str, Any]]:
    links = load_constituent_links(env_path)
    wiki_symbols = scrape_wikipedia_constituents(links, timeout_seconds=timeout_seconds)
    instruments = fetch_trading212_instruments(client)
    tradable = filter_tradable_instruments(instruments)
    matched = cross_reference_constituents(wiki_symbols, tradable)
    return matched


def save_universe_snapshot(
    wiki_symbols: Iterable[str],
    tradable_instruments: Iterable[Dict[str, Any]],
    matched_instruments: Iterable[Dict[str, Any]],
    *,
    output_root: str = "stock_universe",
    snapshot_date: Optional[date] = None,
    use_date_subdir: bool = True,
) -> Dict[str, str]:
    snapshot = snapshot_date or date.today()
    output_dir = output_root
    if use_date_subdir:
        output_dir = os.path.join(output_root, snapshot.isoformat())
    os.makedirs(output_dir, exist_ok=True)

    wiki_rows = sorted({normalize_symbol(sym) for sym in wiki_symbols if sym})
    wiki_path = os.path.join(output_dir, "wiki_symbols.csv")
    _write_csv(wiki_path, ["symbol"], [[symbol] for symbol in wiki_rows])

    trading_rows = []
    for inst in tradable_instruments:
        ticker = inst.get("ticker", "")
        trading_rows.append(
            [
                ticker,
                normalize_symbol(extract_trading212_base_symbol(ticker)),
                inst.get("name", ""),
                inst.get("currencyCode", ""),
                inst.get("type", ""),
                inst.get("shortName", ""),
            ]
        )

    trading_path = os.path.join(output_dir, "trading212_symbols.csv")
    _write_csv(
        trading_path,
        ["ticker", "base_symbol", "name", "currency", "type", "short_name"],
        trading_rows,
    )

    matched_rows = []
    for inst in matched_instruments:
        ticker = inst.get("ticker", "")
        matched_rows.append(
            [
                ticker,
                normalize_symbol(extract_trading212_base_symbol(ticker)),
                inst.get("name", ""),
                inst.get("currencyCode", ""),
                inst.get("type", ""),
                inst.get("shortName", ""),
            ]
        )

    matched_path = os.path.join(output_dir, "matched.csv")
    _write_csv(
        matched_path,
        ["ticker", "base_symbol", "name", "currency", "type", "short_name"],
        matched_rows,
    )

    matched_symbols = {
        normalize_symbol(extract_trading212_base_symbol(i.get("ticker", "")))
        for i in matched_instruments
    }
    matched_symbols.update(
        {
            normalize_symbol(str(i.get("shortName")))
            for i in matched_instruments
            if i.get("shortName")
        }
    )
    wiki_norm = {normalize_symbol(sym) for sym in wiki_symbols}
    unmatched = sorted(sym for sym in wiki_norm if sym and sym not in matched_symbols)

    unmatched_path = os.path.join(output_dir, "unmatched.csv")
    _write_csv(unmatched_path, ["symbol"], [[symbol] for symbol in unmatched])

    return {
        "wiki_symbols": wiki_path,
        "trading212_symbols": trading_path,
        "matched": matched_path,
        "unmatched": unmatched_path,
    }


def extract_trading212_base_symbol(ticker: str) -> str:
    if not ticker:
        return ""
    parts = ticker.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:-2])
    return ticker


def normalize_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    cleaned = cleaned.replace(".", "_").replace("-", "_").replace("/", "_").replace(" ", "")
    return cleaned


def _extract_links(mapping: Dict[str, Any]) -> List[str]:
    links: List[str] = []
    for key in sorted(mapping.keys()):
        if not key.endswith(_CONSTITUENT_SUFFIX):
            continue
        value = mapping.get(key)
        if not value:
            continue
        links.append(str(value).strip())
    return links


def _fetch_url(url: str, *, timeout_seconds: float, max_retries: int = 2) -> str:
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(
                url,
                timeout=timeout_seconds,
                headers={"User-Agent": _WIKIPEDIA_USER_AGENT},
            )
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            logger.warning("Failed to fetch %s (%s)", url, exc)
            if attempt >= max_retries:
                raise
            time.sleep(1)
    raise RuntimeError(f"Failed to fetch {url} after retries.")


def _extract_symbols_from_html(html: str, *, source_url: str) -> Set[str]:
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        logger.warning("No tables found in %s", source_url)
        return set()

    symbols: Set[str] = set()
    for df in tables:
        df = _normalize_columns(df)
        symbol_col = _find_symbol_column(df)
        if not symbol_col:
            continue
        symbols.update(_clean_symbol_series(df[symbol_col]))

    if not symbols:
        logger.warning("No symbol columns found in %s", source_url)
    return symbols


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns: List[str] = []
    for col in df.columns:
        if isinstance(col, tuple):
            parts = [str(part) for part in col if part and str(part) != "nan"]
            col = " ".join(parts)
        columns.append(str(col).strip())
    df.columns = columns
    return df


def _find_symbol_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        normalized = str(col).strip().lower()
        if normalized in _SYMBOL_COLUMNS:
            return col
    return None


def _clean_symbol_series(series: pd.Series) -> Set[str]:
    cleaned: Set[str] = set()
    for raw in series:
        if pd.isna(raw):
            continue
        symbol = _SYMBOL_CLEAN_RE.sub("", str(raw))
        symbol = symbol.strip().upper()
        if not symbol or symbol == "NAN":
            continue
        cleaned.add(symbol)
    return cleaned


def _dot_variants(symbol: str) -> List[str]:
    if not symbol:
        return []
    if "_" not in symbol and "-" not in symbol:
        return []
    variants: Set[str] = set()
    variants.add(symbol.replace("_", "."))
    variants.add(symbol.replace("-", "."))
    variants.add(symbol.replace("_", "/"))
    variants.add(symbol.replace("-", "/"))
    return sorted(variants)


def _write_csv(path: str, header: List[str], rows: List[List[Any]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
