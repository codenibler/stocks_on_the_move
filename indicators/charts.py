from __future__ import annotations

import logging
import os
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.classes import AnalyzedStock

logger = logging.getLogger(__name__)


def plot_momentum_buckets(
    ranked: List[AnalyzedStock],
    *,
    output_dir: str,
    bucket_size: int,
) -> None:
    if not ranked:
        logger.warning("No ranked stocks available for charting")
        return

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    buckets = [
        (0, bucket_size, "top_1_25"),
        (bucket_size, bucket_size * 2, "top_25_50"),
        (bucket_size * 2, bucket_size * 3, "top_50_75"),
        (bucket_size * 3, bucket_size * 4, "top_75_100"),
    ]

    os.makedirs(output_dir, exist_ok=True)

    for idx, (start, end, label) in enumerate(buckets):
        segment = ranked[start:end]
        if not segment:
            logger.info("No stocks available for chart bucket %s", label)
            continue

        tickers = [stock.base_symbol for stock in segment]
        scores = [stock.score for stock in segment]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(scores)), scores, color=colors[idx % len(colors)])
        ax.set_title(f"Momentum rankings {start + 1}-{min(end, len(ranked))}")
        ax.set_ylabel("Momentum score")
        ax.set_xticks(range(len(tickers)))
        ax.set_xticklabels(tickers, rotation=90, fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()

        filename = os.path.join(output_dir, f"momentum_{label}.png")
        fig.savefig(filename)
        plt.close(fig)
        logger.info("Saved momentum chart: %s", filename)
