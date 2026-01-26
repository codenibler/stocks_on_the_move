from __future__ import annotations

import logging
import os
from datetime import date
from typing import Dict, Iterable, List, Optional, TypeVar

import matplotlib

matplotlib.use("Agg")
from dotenv import load_dotenv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)

load_dotenv(override=True)

BACKGROUND = "#0f1116"
TEXT = "#e5e7eb"
MUTED = "#9ca3af"
ACCENT = "#60a5fa"


def _clean_env(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith(("'", '"')) and cleaned.endswith(("'", '"')):
        cleaned = cleaned[1:-1]
    return cleaned.strip()


def _configure_font() -> None:
    font_family = _clean_env(os.getenv("CHART_FONT", "Cascadia Code"))
    font_path = _clean_env(os.getenv("CHART_FONT_PATH", ""))
    if font_path and os.path.isfile(font_path):
        try:
            font_manager.fontManager.addfont(font_path)
        except Exception as exc:
            logger.warning("Failed to register report font at %s: %s", font_path, exc)
    if font_family:
        try:
            font_manager.findfont(font_family, fallback_to_default=False)
        except ValueError:
            logger.warning(
                "Report font '%s' not found; install it or set CHART_FONT_PATH to a .ttf.",
                font_family,
            )
        matplotlib.rcParams["font.family"] = font_family


def _new_page(title: Optional[str] = None) -> plt.Figure:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor(BACKGROUND)
    if title:
        fig.text(0.06, 0.955, title, color=TEXT, fontsize=18, fontweight="bold")
    return fig


def _draw_lines(fig: plt.Figure, lines: Iterable[str], *, start_y: float, line_height: float, size: int) -> None:
    y = start_y
    for line in lines:
        fig.text(0.06, y, line, color=TEXT, fontsize=size)
        y -= line_height


def _draw_text(
    fig: plt.Figure,
    x: float,
    y: float,
    text: str,
    *,
    size: int,
    color: str = TEXT,
    underline: bool = False,
    underline_pad: float = 0.004,
    underline_width: float = 1.0,
) -> None:
    text_obj = fig.text(x, y, text, color=color, fontsize=size)
    if underline:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = text_obj.get_window_extent(renderer=renderer)
        inv = fig.transFigure.inverted()
        x0, y0 = inv.transform((bbox.x0, bbox.y0))
        x1, _ = inv.transform((bbox.x1, bbox.y0))
        y_line = y0 - underline_pad
        fig.add_artist(
            Line2D(
                [x0, x1],
                [y_line, y_line],
                transform=fig.transFigure,
                color=color,
                linewidth=underline_width,
            )
        )


def _finalize_page(
    fig: plt.Figure,
    pdf: PdfPages,
    *,
    page_index: int,
    page_images_dir: Optional[str],
    page_paths: List[str],
) -> int:
    pdf.savefig(fig)
    if page_images_dir:
        os.makedirs(page_images_dir, exist_ok=True)
        image_path = os.path.join(page_images_dir, f"page_{page_index:02d}.png")
        fig.savefig(image_path, dpi=220, bbox_inches="tight")
        page_paths.append(image_path)
    plt.close(fig)
    return page_index + 1


def _load_image(path: Optional[str]) -> Optional[object]:
    if not path or not os.path.isfile(path):
        if path:
            logger.warning("Report image not found: %s", path)
        return None
    try:
        return mpimg.imread(path)
    except Exception as exc:
        logger.warning("Failed to read image %s: %s", path, exc)
        return None


T = TypeVar("T")


def _chunk(items: List[T], size: int) -> List[List[T]]:
    if size <= 0:
        return [items]
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def _format_order_row(order: dict) -> List[str]:
    qty = order.get("quantity")
    qty_text = f"{qty:.3f}" if isinstance(qty, (int, float)) else str(qty or "")
    error = order.get("error") or ""
    if len(error) > 48:
        error = f"{error[:45]}..."
    return [
        str(order.get("stage", "")),
        str(order.get("side", "")),
        str(order.get("ticker", "")),
        qty_text,
        str(order.get("status") or ""),
        str(order.get("order_id") or ""),
        error,
    ]


def generate_rebalance_report(
    *,
    output_dir: str,
    report_date: date,
    universe_summary: Dict[str, object],
    analysis_summary: Dict[str, object],
    order_results: List[dict],
    regime_summary: Dict[str, object],
    momentum_chart_paths: List[str],
    pre_pie_path: Optional[str],
    post_pie_path: Optional[str],
    index_exposure_path: Optional[str],
    index_price_path: Optional[str],
    drop_counts_chart_path: Optional[str],
    page_images_dir: Optional[str] = None,
) -> str:
    _configure_font()

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "rebalance_report.pdf")

    match_stats = universe_summary.get("match_stats", {}) or {}
    index_counts = universe_summary.get("index_counts", {}) or {}
    drop_counts = analysis_summary.get("drop_counts", {}) or {}
    momentum_stats = analysis_summary.get("momentum_stats", {}) or {}
    ranked_count = analysis_summary.get("ranked_count", 0)
    duplicate_count = analysis_summary.get("duplicate_count", 0)
    regime_risk_on = bool(regime_summary.get("risk_on"))
    regime_price = regime_summary.get("price")
    regime_sma = regime_summary.get("sma")
    regime_window = regime_summary.get("sma_window")
    regime_ticker = regime_summary.get("ticker", "S&P 500")

    error_keys = [
        "no_data",
        "no_momentum_data",
        "nan_close",
        "momentum_failed",
        "atr_missing",
        "missing_close",
        "sma_missing",
        "missing_ticker",
        "missing_base_symbol",
    ]
    error_total = sum(drop_counts.get(key, 0) for key in error_keys)

    page_paths: List[str] = []
    page_index = 1
    with PdfPages(report_path) as pdf:
        fig = _new_page(f"Rebalance Report {report_date.isoformat()}")
        scraped_count = universe_summary.get("scraped_count", 0)
        matched_count = universe_summary.get("matched_count", 0)
        summary_start_y = 0.9
        summary_line_height = 0.028
        summary_lines = [
            ("Universe scraped: {} symbols".format(scraped_count), True),
            (f"  • S&P 500: {index_counts.get('SP500', 0)}", False),
            (f"  • S&P 400: {index_counts.get('SP400', 0)}", False),
            (f"  • S&P 600: {index_counts.get('SP600', 0)}", False),
            ("Matched instruments: {}".format(matched_count), True),
            (f"  • Normalized base ticker matches: {match_stats.get('base_matched', 0)}", False),
            (f"  • Dot/slash variant matches: {match_stats.get('variant_matched', 0)}", False),
            (f"  • ShortName metadata matches: {match_stats.get('short_matched', 0)}", False),
            (f"Unmatched symbols: {universe_summary.get('unmatched_count', 0)}", True),
        ]
        y = summary_start_y
        for line, underline in summary_lines:
            _draw_text(fig, 0.06, y, line, size=11, underline=underline)
            y -= summary_line_height

        regime_text = "Regime Filter: "
        if isinstance(regime_price, (int, float)) and isinstance(regime_sma, (int, float)):
            comparator = ">=" if regime_risk_on else "<"
            regime_text += (
                f"{regime_ticker} last close {regime_price:.2f} {comparator} "
                f"SMA{regime_window} {regime_sma:.2f}"
            )
        else:
            regime_text += f"{regime_ticker} data unavailable; defaulted BELOW SMA{regime_window}"
        fig.text(0.06, 0.64, "●", color="#22c55e", fontsize=12)
        fig.text(0.075, 0.64, regime_text, color=TEXT, fontsize=11)

        if index_price_path:
            img = _load_image(index_price_path)
            ax = fig.add_axes((0.08, 0.08, 0.84, 0.46))
            ax.set_facecolor(BACKGROUND)
            ax.set_axis_off()
            if img is not None:
                ax.imshow(img)
        else:
            fig.text(0.06, 0.12, "Index price chart not available.", color=MUTED, fontsize=10)

        page_index = _finalize_page(
            fig,
            pdf,
            page_index=page_index,
            page_images_dir=page_images_dir,
            page_paths=page_paths,
        )

        fig = _new_page("Momentum Filtering Results")
        fig.text(0.06, 0.88, "Drop Counts", color=ACCENT, fontsize=13, fontweight="bold")
        if drop_counts_chart_path:
            img = _load_image(drop_counts_chart_path)
            ax = fig.add_axes((0.08, 0.5, 0.84, 0.34))
            ax.set_facecolor(BACKGROUND)
            ax.set_axis_off()
            if img is not None:
                ax.imshow(img)
        else:
            drop_lines = [f"{key}: {drop_counts.get(key, 0)}" for key in sorted(drop_counts.keys())]
            if not drop_lines:
                drop_lines = ["No drops recorded"]
            _draw_lines(fig, drop_lines, start_y=0.84, line_height=0.026, size=10)

        fig.text(0.06, 0.44, "Summary", color=ACCENT, fontsize=13, fontweight="bold")
        summary = [
            f"Errors during momentum prep: {error_total}",
            f"Below SMA filter: {drop_counts.get('below_sma', 0)}",
            f"Gap >= 15% filter: {drop_counts.get('gap', 0)}",
            f"Ranked stocks: {ranked_count}",
            f"Duplicate tickers removed: {duplicate_count}",
        ]
        _draw_lines(fig, summary, start_y=0.405, line_height=0.028, size=10)

        score_stats = momentum_stats.get("score")
        slope_stats = momentum_stats.get("slope")
        r2_stats = momentum_stats.get("r_squared")
        if score_stats or slope_stats or r2_stats:
            fig.text(0.06, 0.23, "Momentum Stats", color=ACCENT, fontsize=13, fontweight="bold")
            stats_lines = []
            if score_stats:
                stats_lines.extend(
                    [
                        f"Score min/max: {score_stats.get('min', 0):.4f} / {score_stats.get('max', 0):.4f}",
                        f"Score mean/median: {score_stats.get('mean', 0):.4f} / {score_stats.get('median', 0):.4f}",
                    ]
                )
            if slope_stats:
                stats_lines.extend(
                    [
                        f"Slope min/max: {slope_stats.get('min', 0):.6f} / {slope_stats.get('max', 0):.6f}",
                        f"Slope mean/median: {slope_stats.get('mean', 0):.6f} / {slope_stats.get('median', 0):.6f}",
                    ]
                )
            if r2_stats:
                stats_lines.extend(
                    [
                        f"R^2 min/max: {r2_stats.get('min', 0):.4f} / {r2_stats.get('max', 0):.4f}",
                        f"R^2 mean/median: {r2_stats.get('mean', 0):.4f} / {r2_stats.get('median', 0):.4f}",
                    ]
                )
            _draw_lines(fig, stats_lines, start_y=0.19, line_height=0.026, size=9)
        page_index = _finalize_page(
            fig,
            pdf,
            page_index=page_index,
            page_images_dir=page_images_dir,
            page_paths=page_paths,
        )

        chart_pairs = _chunk(momentum_chart_paths, 2)
        for idx, pair in enumerate(chart_pairs, start=1):
            fig = _new_page(f"Momentum Rankings (Top 100) - Page {idx}")
            slots = [(0.08, 0.53, 0.84, 0.34), (0.08, 0.1, 0.84, 0.34)]
            for slot, path in zip(slots, pair):
                img = _load_image(path)
                ax = fig.add_axes(slot)
                ax.set_facecolor(BACKGROUND)
                ax.set_axis_off()
                if img is not None:
                    ax.imshow(img)
            page_index = _finalize_page(
                fig,
                pdf,
                page_index=page_index,
                page_images_dir=page_images_dir,
                page_paths=page_paths,
            )

        fig = _new_page("Portfolio Before Rebalance")
        img = _load_image(pre_pie_path)
        ax = fig.add_axes((0.14, 0.18, 0.72, 0.72))
        ax.set_facecolor(BACKGROUND)
        ax.set_axis_off()
        if img is not None:
            ax.imshow(img)
        else:
            fig.text(0.06, 0.5, "Pre-rebalance pie chart not available.", color=MUTED, fontsize=12)
        page_index = _finalize_page(
            fig,
            pdf,
            page_index=page_index,
            page_images_dir=page_images_dir,
            page_paths=page_paths,
        )

        orders_per_page = 26
        order_pages = _chunk(order_results, orders_per_page) if order_results else [[]]
        for page_idx, orders in enumerate(order_pages, start=1):
            fig = _new_page(f"Order Submission Summary {page_idx}")
            ax = fig.add_axes((0.04, 0.22, 0.92, 0.7))
            ax.set_facecolor(BACKGROUND)
            ax.set_axis_off()
            if not orders:
                fig.text(0.06, 0.5, "No orders sent.", color=MUTED, fontsize=12)
            else:
                columns = ["Stage", "Side", "Ticker", "Qty", "Status", "Order ID", "Error"]
                rows = [_format_order_row(order) for order in orders]
                table = ax.table(
                    cellText=rows,
                    colLabels=columns,
                    cellLoc="center",
                    loc="center",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(7)
                for (row, col), cell in table.get_celld().items():
                    cell.set_edgecolor("#1f2937")
                    if row == 0:
                        cell.set_facecolor("#1f2937")
                        cell.set_text_props(color=TEXT, weight="bold")
                    else:
                        cell.set_facecolor(BACKGROUND)
                        cell.set_text_props(color=TEXT)
            page_index = _finalize_page(
                fig,
                pdf,
                page_index=page_index,
                page_images_dir=page_images_dir,
                page_paths=page_paths,
            )

        fig = _new_page("Portfolio After Rebalance")
        slots = [(0.1, 0.54, 0.8, 0.38), (0.16, 0.14, 0.68, 0.3)]
        for slot, path in zip(slots, [post_pie_path, index_exposure_path]):
            img = _load_image(path)
            ax = fig.add_axes(slot)
            ax.set_facecolor(BACKGROUND)
            ax.set_axis_off()
            if img is not None:
                ax.imshow(img)
        if post_pie_path is None:
            fig.text(0.06, 0.5, "Post-rebalance pie chart not available.", color=MUTED, fontsize=12)
        page_index = _finalize_page(
            fig,
            pdf,
            page_index=page_index,
            page_images_dir=page_images_dir,
            page_paths=page_paths,
        )

    logger.info("Saved rebalance report: %s", report_path)
    return report_path
