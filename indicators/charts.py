from __future__ import annotations

import colorsys
import csv
import logging
import os
from typing import Dict, Iterable, List, Optional, Set

import matplotlib

matplotlib.use("Agg")
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib import colors as mcolors
from matplotlib.patches import Patch

from config.classes import AnalyzedStock
from data_fetching import market_data
from stock_universe.constituents import extract_trading212_base_symbol, normalize_symbol

logger = logging.getLogger(__name__)

load_dotenv(override=True)
_DARK_STYLE_SET = False
_CHART_DPI = 220


def _configure_chart_font(
    *,
    font_family: Optional[str] = None,
    font_path: Optional[str] = None,
) -> None:
    if font_family is None:
        font_family = os.getenv("CHART_FONT", "Cascadia Code").strip()
    if font_path is None:
        font_path = os.getenv("CHART_FONT_PATH", "").strip()
    if font_path and os.path.isfile(font_path):
        try:
            font_manager.fontManager.addfont(font_path)
        except Exception as exc:
            logger.warning("Failed to register chart font at %s: %s", font_path, exc)
    if font_family:
        try:
            font_manager.findfont(font_family, fallback_to_default=False)
        except ValueError:
            logger.warning(
                "Chart font '%s' not found; install it or set CHART_FONT_PATH to a .ttf.",
                font_family,
            )
        matplotlib.rcParams["font.family"] = font_family


def _apply_dark_style() -> None:
    global _DARK_STYLE_SET
    if _DARK_STYLE_SET:
        return
    plt.style.use("dark_background")
    matplotlib.rcParams["figure.facecolor"] = "#0f1117"
    matplotlib.rcParams["axes.facecolor"] = "#0f1117"
    matplotlib.rcParams["savefig.facecolor"] = "#0f1117"
    matplotlib.rcParams["axes.edgecolor"] = "#2a2f3a"
    matplotlib.rcParams["axes.labelcolor"] = "#e5e7eb"
    matplotlib.rcParams["xtick.color"] = "#cbd5f5"
    matplotlib.rcParams["ytick.color"] = "#cbd5f5"
    matplotlib.rcParams["text.color"] = "#e5e7eb"
    _DARK_STYLE_SET = True


def _save_figure(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=_CHART_DPI, bbox_inches="tight")


def _color_with_lightness(base_color: str, lightness: float) -> tuple[float, float, float]:
    r, g, b = mcolors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, lightness))
    return colorsys.hls_to_rgb(h, l, s)


def _lighten_color(base_color: str, amount: float = 0.1) -> tuple[float, float, float]:
    r, g, b = mcolors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l + amount))
    return colorsys.hls_to_rgb(h, l, s)


def _load_matched_index_map(matched_csv_path: str) -> Dict[str, Set[str]]:
    if not os.path.isfile(matched_csv_path):
        logger.warning("Matched symbols file not found for holdings pie: %s", matched_csv_path)
        return {}
    with open(matched_csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "indexes" not in reader.fieldnames:
            logger.warning("Matched symbols file missing indexes column: %s", matched_csv_path)
            return {}
        mapping: Dict[str, Set[str]] = {}
        for row in reader:
            indexes_raw = (row.get("indexes") or "").strip()
            if not indexes_raw:
                continue
            indexes = {item.strip() for item in indexes_raw.split("|") if item.strip()}
            if not indexes:
                continue
            keys = [
                row.get("ticker"),
                row.get("base_symbol"),
                row.get("short_name"),
            ]
            for key in keys:
                if not key:
                    continue
                normalized = normalize_symbol(str(key))
                if not normalized:
                    continue
                mapping.setdefault(normalized, set()).update(indexes)
        return mapping


def plot_momentum_buckets(
    ranked: List[AnalyzedStock],
    *,
    output_dir: str,
    bucket_size: int,
) -> None:
    _apply_dark_style()
    _configure_chart_font()
    if not ranked:
        logger.warning("No ranked stocks available for charting")
        return

    buckets = [
        (0, bucket_size, "top1to25momentum.png"),
        (bucket_size, bucket_size * 2, "top26to50momentum.png"),
        (bucket_size * 2, bucket_size * 3, "top51to75momentum.png"),
        (bucket_size * 3, bucket_size * 4, "top76to100momentum.png"),
    ]
    color_maps = [
        plt.cm.Greens,
        plt.cm.Blues,
        plt.cm.Oranges,
        plt.cm.Reds,
    ]

    charts_dir = os.path.join(output_dir, "momentum_charts", "rankings")
    os.makedirs(charts_dir, exist_ok=True)

    for idx, (start, end, filename) in enumerate(buckets):
        segment = ranked[start:end]
        if not segment:
            logger.info("No stocks available for chart bucket %s", filename)
            continue

        tickers = [stock.base_symbol for stock in segment]
        scores = [stock.score for stock in segment]

        fig, ax = plt.subplots(figsize=(12, 6))
        norm = plt.Normalize(min(scores), max(scores))
        cmap = color_maps[idx % len(color_maps)]
        normalized = norm(scores)
        inverted = 1.0 - normalized
        cmap_scale = 0.35 + 0.65 * inverted
        colors = cmap(cmap_scale)
        alphas = 0.4 + 0.6 * inverted
        colors[:, 3] = np.clip(alphas, 0.4, 1.0)
        ax.bar(range(len(scores)), scores, color=colors)
        ax.set_title(f"Momentum rankings {start + 1}-{min(end, len(ranked))}")
        ax.set_ylabel("Momentum score")
        ax.set_xticks(range(len(tickers)))
        ax.set_xticklabels(tickers, rotation=90, fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3, color="#2a2f3a")
        fig.tight_layout()

        output_path = os.path.join(charts_dir, filename)
        _save_figure(fig, output_path)
        plt.close(fig)
        logger.info("Saved momentum chart: %s", output_path)


def plot_momentum_extremes_summary(
    ranked: List[AnalyzedStock],
    *,
    output_dir: str,
    count: int = 5,
) -> Optional[List[str]]:
    _apply_dark_style()
    _configure_chart_font(font_family="Cascadia Code")
    if not ranked:
        logger.warning("No ranked stocks available for momentum extremes charting")
        return None

    def _select_extremes(values: List[tuple[str, float]]) -> tuple[List[tuple[str, float]], List[tuple[str, float]]]:
        if not values:
            return [], []
        sorted_values = sorted(values, key=lambda item: item[1])
        bottom = sorted_values[:count]
        top = list(reversed(sorted_values[-count:]))
        return top, bottom

    def _plot_panel(
        ax: plt.Axes,
        items: List[tuple[str, float]],
        *,
        title: str,
        color: str,
        number_format: str,
    ) -> None:
        if not items:
            ax.set_visible(False)
            return
        format_str = f"{{:{number_format}}}"
        labels = [item[0] for item in items]
        values = [item[1] for item in items]
        value_strings = [format_str.format(value) for value in values]
        positions = np.arange(len(labels))
        ax.barh(positions, values, color=color, alpha=0.85)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", linestyle="--", alpha=0.3, color="#2a2f3a")
        ax.axvline(0, color="#2a2f3a", linewidth=1)
        min_val = min(values)
        max_val = max(values)
        span = max_val - min_val
        scale = max(abs(max_val), abs(min_val), 1e-6)
        pad = max(span * 0.35, scale * 0.35, 0.05)
        text_pad = pad * 0.4
        ax.set_xlim(min_val - pad, max_val + pad)
        for idx, value in enumerate(values):
            text = value_strings[idx]
            if value >= 0:
                ax.text(value + text_pad, idx, text, va="center", ha="left", fontsize=8)
            else:
                ax.text(value - text_pad, idx, text, va="center", ha="right", fontsize=8)
        ax.invert_yaxis()

    score_values = [(stock.base_symbol, stock.score) for stock in ranked]
    slope_values = [(stock.base_symbol, stock.slope) for stock in ranked]
    r2_values = [(stock.base_symbol, stock.r_squared) for stock in ranked]

    score_top, score_bottom = _select_extremes(score_values)
    slope_top, slope_bottom = _select_extremes(slope_values)
    r2_top, r2_bottom = _select_extremes(r2_values)

    charts_dir = os.path.join(output_dir, "momentum_charts", "regression_metrics")
    os.makedirs(charts_dir, exist_ok=True)

    def _render_metric(
        *,
        title: str,
        top: List[tuple[str, float]],
        bottom: List[tuple[str, float]],
        mean: float,
        median: float,
        number_format: str,
        filename: str,
        top_color: str,
        bottom_color: str,
        footer_label: str,
    ) -> str:
        format_str = f"{{:{number_format}}}"
        label_len = max([len(label) for label, _ in (top + bottom)] or [4])
        value_len = max([len(format_str.format(value)) for _, value in (top + bottom)] or [4])
        width = 11.5 + (label_len * 0.18) + (value_len * 0.45)
        width = max(12.0, min(width, 28.0))
        fig, axes = plt.subplots(1, 2, figsize=(width, 5.2))
        fig.subplots_adjust(wspace=0.35)
        _plot_panel(
            axes[0],
            top,
            title=f"Top 5 {title}",
            color=top_color,
            number_format=number_format,
        )
        _plot_panel(
            axes[1],
            bottom,
            title=f"Bottom 5 {title}",
            color=bottom_color,
            number_format=number_format,
        )
        footer = f"{footer_label} mean/median: {mean:{number_format}} / {median:{number_format}}"
        fig.text(0.5, 0.02, footer, ha="center", va="bottom", fontsize=9, color="#e5e7eb")
        fig.suptitle(f"Momentum {title}", fontsize=13, y=0.99)
        fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.95])
        output_path = os.path.join(charts_dir, filename)
        _save_figure(fig, output_path)
        plt.close(fig)
        logger.info("Saved momentum %s chart: %s", title.lower(), output_path)
        return output_path

    scores = [value for _, value in score_values]
    slopes = [value for _, value in slope_values]
    r2s = [value for _, value in r2_values]
    score_mean = float(np.mean(scores)) if scores else 0.0
    score_median = float(np.median(scores)) if scores else 0.0
    slope_mean = float(np.mean(slopes)) if slopes else 0.0
    slope_median = float(np.median(slopes)) if slopes else 0.0
    r2_mean = float(np.mean(r2s)) if r2s else 0.0
    r2_median = float(np.median(r2s)) if r2s else 0.0

    output_paths = [
        _render_metric(
            title="Scores",
            top=score_top,
            bottom=score_bottom,
            mean=score_mean,
            median=score_median,
            number_format=".4f",
            filename="momentum_scores.png",
            top_color="#22c55e",
            bottom_color="#ef4444",
            footer_label="Score",
        ),
        _render_metric(
            title="Slopes",
            top=slope_top,
            bottom=slope_bottom,
            mean=slope_mean,
            median=slope_median,
            number_format=".6f",
            filename="momentum_slopes.png",
            top_color="#38bdf8",
            bottom_color="#f97316",
            footer_label="Slope",
        ),
        _render_metric(
            title="R^2",
            top=r2_top,
            bottom=r2_bottom,
            mean=r2_mean,
            median=r2_median,
            number_format=".4f",
            filename="momentum_r2.png",
            top_color="#a3e635",
            bottom_color="#f43f5e",
            footer_label="R^2",
        ),
    ]
    return output_paths


def plot_holdings_pie(
    positions: Iterable[dict],
    *,
    output_dir: str,
    filename: str = "holdings_pie.png",
    matched_csv_path: Optional[str] = None,
    cash_value: Optional[float] = None,
) -> None:
    _apply_dark_style()
    _configure_chart_font()
    charts_dir = os.path.join(output_dir, "momentum_charts", "holdings_charts")
    os.makedirs(charts_dir, exist_ok=True)

    holdings: List[tuple[str, float]] = []
    for position in positions:
        instrument = position.get("instrument", {})
        ticker = instrument.get("ticker")
        qty = float(position.get("quantityAvailableForTrading") or 0.0)
        price = position.get("currentPrice")
        if not ticker or qty <= 0 or price is None:
            continue
        value = qty * float(price)
        if value <= 0:
            continue
        holdings.append((ticker, value))

    if cash_value is not None and cash_value > 0:
        holdings.append(("CASH", float(cash_value)))
    if not holdings:
        logger.info("No holdings available for pie chart")
        return

    pie_colors = None
    index_map: Dict[str, Set[str]] = {}
    matched_path = matched_csv_path or os.path.join(output_dir, "symbols", "matched.csv")
    index_map = _load_matched_index_map(matched_path)
    if index_map:
        color_map = {
            "SP500": "#3b82f6",
            "SP400": "#22c55e",
            "SP600": "#f59e0b",
            "UNIDENTIFIED": "#6b7280",
            "CASH": "#9ca3af",
        }
        counts = {"SP500": 0, "SP400": 0, "SP600": 0, "UNIDENTIFIED": 0, "CASH": 0}
        ordered_labels = ["SP500", "SP400", "SP600", "CASH"]
        labeled_holdings: List[tuple[str, float, str]] = []
        for ticker, value in holdings:
            ticker_key = normalize_symbol(ticker)
            base_symbol = normalize_symbol(extract_trading212_base_symbol(ticker))
            if ticker_key == "CASH":
                label = "CASH"
            else:
                indexes = set(index_map.get(ticker_key, set()) or index_map.get(base_symbol, set()))
                if "SP500" in indexes:
                    label = "SP500"
                elif "SP400" in indexes:
                    label = "SP400"
                elif "SP600" in indexes:
                    label = "SP600"
                else:
                    label = "UNIDENTIFIED"
            labeled_holdings.append((ticker, value, label))

        ordered_holdings: List[tuple[str, float, str]] = []
        for label in ordered_labels:
            group = [item for item in labeled_holdings if item[2] == label]
            group.sort(key=lambda item: item[1], reverse=True)
            counts[label] += len(group)
            ordered_holdings.extend(group)
        counts["UNIDENTIFIED"] += sum(1 for item in labeled_holdings if item[2] == "UNIDENTIFIED")
        counts["CASH"] += sum(1 for item in labeled_holdings if item[2] == "CASH")

        labels = [item[0] for item in ordered_holdings]
        values = [item[1] for item in ordered_holdings]
        min_value = min(values)
        max_value = max(values)
        if max_value == min_value:
            normalized = np.full(len(values), 0.5)
        else:
            normalized = (np.array(values) - min_value) / (max_value - min_value)
        colors: List[str] = []
        for (_, _, label), weight in zip(ordered_holdings, normalized):
            lightness = 0.85 - 0.5 * weight
            colors.append(_color_with_lightness(color_map[label], lightness))
        pie_colors = colors
        logger.info(
            "Holdings pie index counts: SP500=%s SP400=%s SP600=%s UNIDENTIFIED=%s CASH=%s",
            counts["SP500"],
            counts["SP400"],
            counts["SP600"],
            counts["UNIDENTIFIED"],
            counts["CASH"],
        )
    else:
        holdings.sort(key=lambda item: item[1], reverse=True)
        labels, values = zip(*holdings)
        min_value = min(values)
        max_value = max(values)
        if max_value == min_value:
            normalized = np.full(len(values), 0.5)
        else:
            normalized = (np.array(values) - min_value) / (max_value - min_value)
        pie_colors = plt.cm.Blues(0.35 + 0.65 * normalized)

    total_value = float(sum(values))
    labels = [
        f"{label}\n{(value / total_value) * 100:.1f}%"
        for label, value in zip(labels, values)
    ]

    fig, ax = plt.subplots(figsize=(8, 8.6))
    ax.pie(
        values,
        labels=labels,
        startangle=90,
        counterclock=False,
        colors=pie_colors,
        textprops={
            "color": "#e5e7eb",
            "ha": "center",
            "va": "center",
            "multialignment": "center",
        },
        wedgeprops={"edgecolor": "#0f1117", "linewidth": 0.5},
    )
    ax.set_title("Holdings allocation")
    ax.axis("equal")
    if index_map:
        legend_handles = [
            Patch(facecolor=color_map["SP500"], label="S&P 500 (Large Caps)"),
            Patch(facecolor=color_map["SP400"], label="S&P 400 (Mid Caps)"),
            Patch(facecolor=color_map["SP600"], label="S&P 600 (Small Caps)"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.06),
            ncol=3,
            frameon=False,
        )
    fig.tight_layout()

    output_path = os.path.join(charts_dir, filename)
    _save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved holdings pie chart: %s", output_path)


def plot_index_exposure_bar(
    positions: Iterable[dict],
    *,
    output_dir: str,
    filename: str = "index_exposure_bar.png",
    matched_csv_path: Optional[str] = None,
    cash_value: Optional[float] = None,
) -> Optional[str]:
    _apply_dark_style()
    _configure_chart_font()
    charts_dir = os.path.join(output_dir, "momentum_charts", "holdings_charts")
    os.makedirs(charts_dir, exist_ok=True)

    holdings: List[tuple[str, float]] = []
    for position in positions:
        instrument = position.get("instrument", {})
        ticker = instrument.get("ticker")
        qty = float(position.get("quantityAvailableForTrading") or 0.0)
        price = position.get("currentPrice")
        if not ticker or qty <= 0 or price is None:
            continue
        value = qty * float(price)
        if value <= 0:
            continue
        holdings.append((ticker, value))

    if not holdings:
        logger.info("No holdings available for index exposure chart")
        return None

    matched_path = matched_csv_path or os.path.join(output_dir, "symbols", "matched.csv")
    index_map = _load_matched_index_map(matched_path)
    totals = {"SP500": 0.0, "SP400": 0.0, "SP600": 0.0, "UNIDENTIFIED": 0.0, "CASH": 0.0}

    for ticker, value in holdings:
        ticker_key = normalize_symbol(ticker)
        base_symbol = normalize_symbol(extract_trading212_base_symbol(ticker))
        indexes = set(index_map.get(ticker_key, set()) or index_map.get(base_symbol, set()))
        if "SP500" in indexes:
            totals["SP500"] += value
        elif "SP400" in indexes:
            totals["SP400"] += value
        elif "SP600" in indexes:
            totals["SP600"] += value
        else:
            totals["UNIDENTIFIED"] += value

    if cash_value is not None and cash_value > 0:
        totals["CASH"] = float(cash_value)

    total_value = sum(totals.values())
    if total_value <= 0:
        logger.info("Index exposure chart skipped due to zero total value")
        return None

    labels = ["S&P 500", "S&P 400", "S&P 600"]
    values = [
        (totals["SP500"] / total_value) * 100.0,
        (totals["SP400"] / total_value) * 100.0,
        (totals["SP600"] / total_value) * 100.0,
    ]
    colors = ["#3b82f6", "#22c55e", "#f59e0b"]
    if totals["UNIDENTIFIED"] > 0:
        labels.append("Unidentified")
        values.append((totals["UNIDENTIFIED"] / total_value) * 100.0)
        colors.append("#6b7280")
    if totals["CASH"] > 0:
        labels.append("CASH")
        values.append((totals["CASH"] / total_value) * 100.0)
        colors.append("#9ca3af")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color=colors, alpha=0.85)
    ax.set_ylabel("Portfolio weight (%)")
    ax.set_title("Holdings by index")
    ax.set_ylim(0, max(values) * 1.25 if values else 1)
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#2a2f3a")

    for idx, value in enumerate(values):
        ax.text(idx, value + (max(values) * 0.03 if values else 0.5), f"{value:.1f}%", ha="center", va="bottom")

    fig.tight_layout()
    output_path = os.path.join(charts_dir, filename)
    _save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved index exposure chart: %s", output_path)
    return output_path


def plot_index_price_charts(
    index_tickers: Dict[str, str],
    *,
    output_dir: str,
    filename: str = "index_price_charts.png",
    period: str = "565d",
    interval: str = "1d",
    retries: int = 3,
    retry_sleep_seconds: float = 1.0,
) -> Optional[str]:
    _apply_dark_style()
    _configure_chart_font()
    charts_dir = os.path.join(output_dir, "momentum_charts", "holdings_charts")
    os.makedirs(charts_dir, exist_ok=True)

    tickers = {label: ticker for label, ticker in index_tickers.items() if ticker}
    if not tickers:
        logger.info("No index tickers configured for price chart")
        return None

    fig, axes = plt.subplots(len(tickers), 1, figsize=(10, 8), sharex=True)
    if len(tickers) == 1:
        axes = [axes]

    plotted = 0
    color_map = {
        "S&P 500": "#3b82f6",
        "S&P 400": "#22c55e",
        "S&P 600": "#f59e0b",
    }
    for ax, (label, ticker) in zip(axes, tickers.items()):
        df = market_data.fetch_history_with_retries(
            ticker,
            period=period,
            interval=interval,
            retries=retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        if df is None or df.empty:
            logger.warning("Index price chart missing data for %s (%s)", label, ticker)
            ax.set_visible(False)
            continue
        df = market_data.normalize_price_frame(df)
        price_series = market_data.select_price_series(df)
        if price_series is None or price_series.dropna().empty:
            logger.warning("Index price chart missing Close series for %s (%s)", label, ticker)
            ax.set_visible(False)
            continue
        line_color = color_map.get(label, "#60a5fa")
        ax.plot(price_series.index, price_series.values, color=line_color, linewidth=1.6, label="Price")
        ema200 = price_series.ewm(span=200, adjust=False, min_periods=1).mean()
        if ema200 is not None and not ema200.dropna().empty:
            ema_color = _lighten_color(line_color, 0.1)
            ax.plot(ema200.index, ema200.values, color=ema_color, linewidth=1.2, label="EMA200")
        ax.set_title(f"{label} ({ticker})")
        ax.grid(axis="y", linestyle="--", alpha=0.3, color="#2a2f3a")
        ax.legend(loc="upper left", frameon=False, fontsize=8)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        logger.warning("No index price charts rendered due to missing data")
        return None

    fig.tight_layout()
    output_path = os.path.join(charts_dir, filename)
    _save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved index price charts: %s", output_path)
    return output_path


def plot_summary_counts(
    summary_counts: Dict[str, float],
    *,
    output_dir: str,
    filename: str = "summary_counts.png",
) -> Optional[str]:
    _apply_dark_style()
    _configure_chart_font(font_family="Cascadia Code")
    charts_dir = os.path.join(output_dir, "momentum_charts", "errors_and_dropouts")
    os.makedirs(charts_dir, exist_ok=True)

    if not summary_counts:
        logger.info("No summary counts available for charting")
        return None

    labels = list(summary_counts.keys())
    values = [float(summary_counts.get(label, 0)) for label in labels]
    if not any(values):
        logger.info("Summary counts chart skipped due to zero values")
        return None

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    if len(values) > 1:
        normalized = (np.array(values) - min(values)) / (max(values) - min(values) or 1)
    else:
        normalized = np.array([0.5])
    colors = plt.cm.PuBu(0.25 + 0.7 * normalized)
    ax.barh(range(len(labels)), values, color=colors, alpha=0.9)
    ax.set_title("Summary counts")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.3, color="#2a2f3a")
    ax.invert_yaxis()

    max_value = max(values) if values else 0
    pad = max_value * 0.03 if max_value else 0.5
    for idx, value in enumerate(values):
        text = f"{value:.0f}" if value.is_integer() else f"{value:.2f}"
        ax.text(value + pad, idx, text, va="center", ha="left", fontsize=9)

    fig.tight_layout()
    output_path = os.path.join(charts_dir, filename)
    _save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved summary counts chart: %s", output_path)
    return output_path


def plot_drop_counts_bar(
    drop_counts: Dict[str, int],
    *,
    output_dir: str,
    filename: str = "dropCountsBarChart.png",
) -> Optional[str]:
    _apply_dark_style()
    _configure_chart_font()
    charts_dir = os.path.join(output_dir, "momentum_charts", "errors_and_dropouts")
    os.makedirs(charts_dir, exist_ok=True)

    if not drop_counts:
        logger.info("No drop counts available for charting")
        return None

    labels = list(drop_counts.keys())
    values = [drop_counts.get(label, 0) for label in labels]
    if not any(values):
        logger.info("Drop counts chart skipped due to zero values")
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.Spectral
    if len(values) > 1:
        normalized = (np.array(values) - min(values)) / (max(values) - min(values) or 1)
    else:
        normalized = np.array([0.5])
    colors = cmap(0.25 + 0.7 * normalized)
    ax.bar(labels, values, color=colors, alpha=0.9)
    ax.set_title("Drop counts by reason")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#2a2f3a")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)

    max_value = max(values) if values else 0
    for idx, value in enumerate(values):
        ax.text(idx, value + max_value * 0.03, str(value), ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    output_path = os.path.join(charts_dir, filename)
    _save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved drop counts chart: %s", output_path)
    return output_path
