from __future__ import annotations

import logging
import os
from typing import List

import matplotlib

matplotlib.use("Agg")
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from config.classes import AnalyzedStock

logger = logging.getLogger(__name__)

load_dotenv(override=True)


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


def plot_momentum_buckets(
    ranked: List[AnalyzedStock],
    *,
    output_dir: str,
    bucket_size: int,
) -> None:
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
        cmap_scale = 0.35 + 0.65 * normalized
        colors = cmap(cmap_scale)
        alphas = 0.4 + 0.6 * normalized
        colors[:, 3] = np.clip(alphas, 0.4, 1.0)
        ax.bar(range(len(scores)), scores, color=colors)
        ax.set_title(f"Momentum rankings {start + 1}-{min(end, len(ranked))}")
        ax.set_ylabel("Momentum score")
        ax.set_xticks(range(len(tickers)))
        ax.set_xticklabels(tickers, rotation=90, fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()

        filename = os.path.join(charts_dir, f"{label}.png")
        fig.savefig(filename)
        plt.close(fig)
        logger.info("Saved momentum chart: %s", filename)
