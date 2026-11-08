"""Figures for the notebook and the report.

Every figure is written as a PNG into ``figures/``. The functions take the data
they draw so they can be reused from the notebook; :func:`make_all_figures`
produces the full set from a completed study.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data import REPO_ROOT  # noqa: E402
from .features import daily_channels, window_aggregates  # noqa: E402
from .study import Split, StudyConfig, build_split  # noqa: E402

FIGURES_DIR = REPO_ROOT / "figures"

_MODEL_STYLE = {
    "elastic_net": ("Elastic-net logistic", "#1f77b4"),
    "gradient_boosting": ("Gradient boosting", "#ff7f0e"),
    "lstm": ("LSTM", "#2ca02c"),
}


def _prepare() -> None:
    """Set a clean, consistent Matplotlib style and ensure the output folder."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    FIGURES_DIR.mkdir(exist_ok=True)


def plot_network_structure(path: Path) -> Path:
    """Draw the LSTM network structure used in the study.

    Args:
        path: Output PNG path.

    Returns:
        The written path.
    """
    layers = [
        "Input\n60 days x channels",
        "LSTM layer",
        "Dropout",
        "Dense (1 unit)\nsigmoid",
        "P(up next month)",
    ]
    fig, ax = plt.subplots(figsize=(4.2, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(layers))
    for i, label in enumerate(layers):
        y = len(layers) - 1 - i
        ax.add_patch(
            plt.Rectangle((0.15, y + 0.2), 0.7, 0.6, facecolor="#eaf2fb", edgecolor="#1f77b4")
        )
        ax.text(0.5, y + 0.5, label, ha="center", va="center")
        if i < len(layers) - 1:
            ax.annotate(
                "", xy=(0.5, y + 0.2), xytext=(0.5, y - 0.2 + 1),
                arrowprops={"arrowstyle": "->", "color": "#555"},
            )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_walk_forward(path: Path, split: Split) -> Path:
    """Illustrate the purged walk-forward folds and the held-out test block.

    Args:
        path: Output PNG path.
        split: The study split, providing the folds and test months.

    Returns:
        The written path.
    """
    n_dev = len(split.dev_months)
    n_total = len(split.months)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for row, (train_index, validation_index) in enumerate(split.folds):
        y = len(split.folds) - row
        ax.barh(y, len(train_index), color="#1f77b4", edgecolor="none")
        gap_start = train_index[-1] + 1
        ax.barh(y, validation_index[0] - gap_start, left=gap_start, color="#cccccc")
        ax.barh(
            y, len(validation_index), left=validation_index[0],
            color="#ff7f0e", edgecolor="none",
        )
    ax.barh(0, n_total - n_dev, left=n_dev, color="#777777", edgecolor="none")
    ax.axvline(n_dev, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks([0, *range(1, len(split.folds) + 1)])
    ax.set_yticklabels(["Test", *[f"Fold {k}" for k in range(len(split.folds), 0, -1)]])
    ax.set_xlabel("Month index")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c)
        for c in ("#1f77b4", "#cccccc", "#ff7f0e", "#777777")
    ]
    ax.legend(
        handles, ["Train", "Purge + embargo", "Validation", "Held-out test"],
        loc="upper left", frameon=False, fontsize=9,
    )
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_price_history(path: Path, prices: pd.DataFrame, split: Split) -> Path:
    """Plot the S&P 500 on a log scale with the test period shaded.

    Args:
        path: Output PNG path.
        prices: Daily price data with a ``Close`` column.
        split: The study split, providing the test start date.

    Returns:
        The written path.
    """
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.semilogy(prices.index, prices["Close"], color="#1f77b4", linewidth=0.9)
    test_start = split.test_months[0]
    ax.axvspan(test_start, prices.index[-1], color="#ff7f0e", alpha=0.12)
    ax.text(test_start, prices["Close"].max(), " test period", color="#b35900", va="top")
    ax.set_ylabel("S&P 500 index (log scale)")
    ax.set_xlabel("Year")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_feature_examples(path: Path, prices: pd.DataFrame) -> Path:
    """Show example stationary features over the full sample.

    Args:
        path: Output PNG path.
        prices: Daily price data.

    Returns:
        The written path.
    """
    aggregates = window_aggregates(prices)
    channels = daily_channels(prices)
    panels = [
        (aggregates["ret_12w"], "12-week return"),
        (aggregates["vol_12w"], "12-week volatility"),
        (channels["logvol_rel"], "log volume vs 1-year mean"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 6.0), sharex=True)
    for ax, (series, title) in zip(axes, panels, strict=True):
        ax.plot(series.index, series, color="#1f77b4", linewidth=0.7)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylabel(title, fontsize=9)
    axes[-1].set_xlabel("Year")
    axes[0].set_title("Stationary features (volume detrended over one year)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_auc_by_model(path: Path, classification: pd.DataFrame) -> Path:
    """Plot test AUC with bootstrap intervals for each model and feature set.

    Args:
        path: Output PNG path.
        classification: The classification results table.

    Returns:
        The written path.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    feature_sets = sorted(
        classification.loc[classification["feature_set"] > 0, "feature_set"].unique()
    )
    offsets = np.linspace(-0.22, 0.22, len(_MODEL_STYLE))
    for offset, (model, (label, color)) in zip(offsets, _MODEL_STYLE.items(), strict=True):
        rows = classification[classification["model"] == model].set_index("feature_set")
        rows = rows.loc[feature_sets]
        x = np.array(feature_sets) + offset
        lower = rows["test_auc"] - rows["auc_low"]
        upper = rows["auc_high"] - rows["test_auc"]
        ax.errorbar(
            x, rows["test_auc"], yerr=[lower, upper], fmt="o", color=color,
            label=label, capsize=3, markersize=5,
        )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="No skill (0.5)")
    ax.set_xticks(feature_sets)
    ax.set_xlabel("Feature set")
    ax.set_ylabel("Test AUC (95% block bootstrap)")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_strategy_equity(path: Path, equity: pd.DataFrame) -> Path:
    """Plot the best timing strategy against buy-and-hold, with drawdowns.

    Args:
        path: Output PNG path.
        equity: Equity-curve table with ``month_end``, ``buy_and_hold`` and
            ``best_strategy`` columns.

    Returns:
        The written path.
    """
    dates = pd.to_datetime(equity["month_end"])
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.5, 5.2), sharex=True, height_ratios=[2, 1]
    )
    for column, label, color in (
        ("buy_and_hold", "Buy and hold", "#1f77b4"),
        ("best_strategy", "Best timing strategy", "#ff7f0e"),
    ):
        curve = equity[column]
        top.plot(dates, curve, label=label, color=color)
        drawdown = curve / np.maximum.accumulate(curve) - 1.0
        bottom.fill_between(dates, drawdown, 0, color=color, alpha=0.3)
    top.set_ylabel("Growth of 1 (test)")
    top.legend(frameon=False, fontsize=9)
    bottom.set_ylabel("Drawdown")
    bottom.set_xlabel("Year")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def make_all_figures(
    results: dict, prices: pd.DataFrame, config: StudyConfig | None = None
) -> list[Path]:
    """Generate every figure from a completed study.

    Args:
        results: The dictionary returned by `src.study.run_study`.
        prices: Daily price data.
        config: Study configuration; defaults to `StudyConfig`.

    Returns:
        The list of written figure paths.
    """
    _prepare()
    config = config or StudyConfig()
    split = build_split(prices, config)
    return [
        plot_network_structure(FIGURES_DIR / "network_structure.png"),
        plot_walk_forward(FIGURES_DIR / "walk_forward.png", split),
        plot_price_history(FIGURES_DIR / "price_history.png", prices, split),
        plot_feature_examples(FIGURES_DIR / "feature_examples.png", prices),
        plot_auc_by_model(FIGURES_DIR / "auc_by_model.png", results["classification"]),
        plot_strategy_equity(FIGURES_DIR / "strategy_equity.png", results["equity"]),
    ]
