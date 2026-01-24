from __future__ import annotations

import logging
import os
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from config.classes import AnalyzedStock

logger = logging.getLogger(__name__)

load_dotenv(override=True)
_DARK_STYLE_SET = False


def _configure_chart_font() -> None:
    font_family = os.getenv("CHART_FONT", "Montserrat").strip()
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
        (0, bucket_size, "top_25"),
        (bucket_size, bucket_size * 2, "top_25to50"),
        (bucket_size * 2, bucket_size * 3, "top50to75"),
        (bucket_size * 3, bucket_size * 4, "top75to100"),
    ]
    color_maps = [
        plt.cm.Greens,
        plt.cm.Blues,
        plt.cm.Oranges,
        plt.cm.Reds,
    ]

    charts_dir = os.path.join(output_dir, "momentum_charts")
    os.makedirs(charts_dir, exist_ok=True)

    for idx, (start, end, label) in enumerate(buckets):
        segment = ranked[start:end]
        if not segment:
            logger.info("No stocks available for chart bucket %s", label)
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

        filename = os.path.join(charts_dir, f"{label}.png")
        fig.savefig(filename)
        plt.close(fig)
        logger.info("Saved momentum chart: %s", filename)


def plot_holdings_pie(
    positions: Iterable[dict],
    *,
    output_dir: str,
    filename: str = "holdings_pie.png",
) -> None:
    _apply_dark_style()
    _configure_chart_font()
    charts_dir = os.path.join(output_dir, "momentum_charts")
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
        logger.info("No holdings available for pie chart")
        return

    holdings.sort(key=lambda item: item[1], reverse=True)
    labels, values = zip(*holdings)
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        normalized = np.full(len(values), 0.5)
    else:
        normalized = (np.array(values) - min_value) / (max_value - min_value)
    inverted = 1.0 - normalized
    pie_colors = plt.cm.Blues(0.35 + 0.65 * inverted)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        colors=pie_colors,
        textprops={"color": "#e5e7eb"},
        wedgeprops={"edgecolor": "#0f1117", "linewidth": 0.5},
    )
    ax.set_title("Holdings allocation")
    ax.axis("equal")
    fig.tight_layout()

    output_path = os.path.join(charts_dir, filename)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved holdings pie chart: %s", output_path)
