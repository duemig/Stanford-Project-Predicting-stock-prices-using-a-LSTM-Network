"""Run the whole study and write the results tables and summary.

:func:`run_study` builds the data set, fits every model on every feature set,
evaluates them on the held-out test months, runs the timing strategy and writes
the ``results/`` files that the report, README and figures all read from.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone

from .data import REPO_ROOT, get_prices
from .evaluate import (
    accuracy,
    auc,
    base_rate,
    bootstrap_metric,
    brier,
    buy_and_hold,
    choose_threshold,
    paired_accuracy_vs_always_up,
    timing_strategy,
)
from .features import (
    FEATURE_SETS,
    LSTM_LOOKBACK,
    baseline_matrix,
    lstm_tensor,
    valid_month_ends,
)
from .labels import monthly_labels
from .models import (
    LSTMConfig,
    always_up_proba,
    fit_elastic_net,
    fit_gradient_boosting,
    out_of_fold_proba,
    train_lstm,
)
from .validation import purged_walk_forward_splits

RESULTS_DIR = REPO_ROOT / "results"


@dataclass(frozen=True)
class StudyConfig:
    """Settings for a full run of the study."""

    test_fraction: float = 0.2
    n_splits: int = 5
    embargo: int = 1
    horizon: int = 1
    seeds: tuple[int, ...] = tuple(range(10))
    cost: float = 0.001
    block: int = 12
    n_boot: int = 2000
    lstm: LSTMConfig = field(default_factory=LSTMConfig)


@dataclass(frozen=True)
class Split:
    """The month-end sample split into development and test sets.

    Attributes:
        months: All usable month-end dates.
        dev_months: Development month-ends (the earlier part).
        test_months: Test month-ends (the final `test_fraction`).
        y_dev: Development labels.
        y_test: Test labels.
        return_dev: Development next-month returns.
        return_test: Test next-month returns.
        folds: Purged walk-forward folds over the development set.
        months_total: Count of monthly labels before the feature warm-up is dropped.
    """

    months: pd.DatetimeIndex
    dev_months: pd.DatetimeIndex
    test_months: pd.DatetimeIndex
    y_dev: np.ndarray
    y_test: np.ndarray
    return_dev: np.ndarray
    return_test: np.ndarray
    folds: list[tuple[np.ndarray, np.ndarray]]
    months_total: int


def build_split(prices: pd.DataFrame, config: StudyConfig) -> Split:
    """Build the month-end sample and its development/test split.

    Args:
        prices: Daily price data.
        config: Study configuration controlling the test fraction and folds.

    Returns:
        A `Split` with the month arrays, labels, returns and walk-forward folds.
    """
    labels = monthly_labels(prices)
    months = valid_month_ends(prices, labels.index)
    y = labels.loc[months, "label"].to_numpy()
    next_return = labels.loc[months, "next_return"].to_numpy()
    n = len(months)
    n_test = int(round(config.test_fraction * n))
    n_dev = n - n_test
    folds = list(
        purged_walk_forward_splits(n_dev, config.n_splits, config.embargo, config.horizon)
    )
    return Split(
        months=months,
        dev_months=months[:n_dev],
        test_months=months[n_dev:],
        y_dev=y[:n_dev],
        y_test=y[n_dev:],
        return_dev=next_return[:n_dev],
        return_test=next_return[n_dev:],
        folds=folds,
        months_total=len(labels),
    )


def _classification_row(
    model: str,
    set_id: int,
    set_name: str,
    y_test: np.ndarray,
    proba: np.ndarray,
    dev_accuracy: float,
    config: StudyConfig,
) -> dict:
    """Build one row of the classification results table with bootstrap intervals.

    Args:
        model: Model family name.
        set_id: Feature-set number (``0`` for the feature-independent always-up).
        set_name: Feature-set description.
        y_test: Test labels.
        proba: Test probabilities.
        dev_accuracy: Development (validation) accuracy for the model.
        config: Study configuration.

    Returns:
        A dictionary of metrics and 95% intervals.
    """
    accuracy_ci = bootstrap_metric(y_test, proba, accuracy, config.block, config.n_boot)
    auc_ci = bootstrap_metric(
        y_test, proba, auc, config.block, config.n_boot, need_both_classes=True
    )
    brier_ci = bootstrap_metric(y_test, proba, brier, config.block, config.n_boot)
    difference, diff_low, diff_high, p_value = paired_accuracy_vs_always_up(
        y_test, proba, 0.5, config.block, config.n_boot
    )
    return {
        "model": model,
        "feature_set": set_id,
        "description": set_name,
        "dev_accuracy": round(dev_accuracy, 4),
        "test_accuracy": round(accuracy_ci.point, 4),
        "accuracy_low": round(accuracy_ci.low, 4),
        "accuracy_high": round(accuracy_ci.high, 4),
        "test_auc": round(auc_ci.point, 4),
        "auc_low": round(auc_ci.low, 4),
        "auc_high": round(auc_ci.high, 4),
        "brier": round(brier_ci.point, 4),
        "brier_low": round(brier_ci.low, 4),
        "brier_high": round(brier_ci.high, 4),
        "acc_minus_always_up": round(difference, 4),
        "acc_diff_low": round(diff_low, 4),
        "acc_diff_high": round(diff_high, 4),
        "p_not_better_than_always_up": round(p_value, 4),
    }


def _strategy_row(
    model: str,
    set_id: int,
    set_name: str,
    next_return_test: np.ndarray,
    proba_test: np.ndarray,
    threshold: float,
    config: StudyConfig,
) -> dict:
    """Build one row of the strategy results table.

    Args:
        model: Model family name.
        set_id: Feature-set number.
        set_name: Feature-set description.
        next_return_test: Realised next-month returns on the test set.
        proba_test: Test probabilities.
        threshold: Validation-chosen probability threshold.
        config: Study configuration.

    Returns:
        A dictionary of annualised strategy statistics.
    """
    perf, _ = timing_strategy(next_return_test, proba_test, threshold, config.cost)
    return {
        "model": model,
        "feature_set": set_id,
        "description": set_name,
        "threshold": round(threshold, 3),
        "annual_return": round(perf.annual_return, 4),
        "annual_volatility": round(perf.annual_volatility, 4),
        "sharpe": round(perf.sharpe, 4),
        "max_drawdown": round(perf.max_drawdown, 4),
        "invested_fraction": round(perf.invested_fraction, 4),
    }


def run_study(
    config: StudyConfig | None = None, write: bool = True, verbose: bool = True
) -> dict:
    """Run the full study and optionally write the results files.

    Args:
        config: Study configuration; a default `StudyConfig` is used if ``None``.
        write: If ``True``, write the CSV and JSON results into ``results/``.
        verbose: If ``True``, print progress for each feature set.

    Returns:
        A dictionary with the classification table, strategy table, per-seed
        table, equity curves and a summary, all as plain Python or pandas
        objects.
    """
    config = config or StudyConfig()
    prices, download_info = get_prices()
    split = build_split(prices, config)
    months, dev_months, test_months = split.months, split.dev_months, split.test_months
    y_dev, y_test = split.y_dev, split.y_test
    return_dev, return_test = split.return_dev, split.return_test
    folds = split.folds
    n, n_dev, n_test = len(months), len(dev_months), len(test_months)

    classification_rows: list[dict] = []
    strategy_rows: list[dict] = []
    seed_rows: list[dict] = []

    always_up_test = always_up_proba(y_dev, n_test)
    classification_rows.append(
        _classification_row(
            "always_up", 0, "Always up", y_test, always_up_test, base_rate(y_dev), config
        )
    )
    buy_hold_perf, buy_hold_returns = buy_and_hold(return_test)
    strategy_rows.append(
        {
            "model": "buy_and_hold",
            "feature_set": 0,
            "description": "Buy and hold",
            "threshold": float("nan"),
            "annual_return": round(buy_hold_perf.annual_return, 4),
            "annual_volatility": round(buy_hold_perf.annual_volatility, 4),
            "sharpe": round(buy_hold_perf.sharpe, 4),
            "max_drawdown": round(buy_hold_perf.max_drawdown, 4),
            "invested_fraction": 1.0,
        }
    )

    best_strategy_returns = buy_hold_returns
    best_strategy_label = "buy_and_hold"
    best_strategy_sharpe = buy_hold_perf.sharpe

    for set_id, (set_name, families) in FEATURE_SETS.items():
        if verbose:
            print(f"[{set_id}/7] {set_name}", flush=True)
        X_base_dev = baseline_matrix(prices, dev_months, families).to_numpy()
        X_base_test = baseline_matrix(prices, test_months, families).to_numpy()
        sequences, _ = lstm_tensor(prices, months, families, LSTM_LOOKBACK)
        X_seq_dev, X_seq_test = sequences[:n_dev], sequences[n_dev:]

        elastic_net = fit_elastic_net(X_base_dev, y_dev, folds)
        gradient_boosting = fit_gradient_boosting(X_base_dev, y_dev, folds)
        lstm = train_lstm(X_seq_dev, y_dev, X_seq_test, list(config.seeds), config.lstm)

        proba = {
            "elastic_net": elastic_net.predict_proba(X_base_test)[:, 1],
            "gradient_boosting": gradient_boosting.predict_proba(X_base_test)[:, 1],
            "lstm": lstm.test_proba_mean,
        }
        dev_accuracy = {
            "elastic_net": float(elastic_net.best_score_),
            "gradient_boosting": float(gradient_boosting.best_score_),
            "lstm": float(lstm.validation_accuracy_seeds.mean()),
        }
        oof = {
            "elastic_net": out_of_fold_proba(
                clone(elastic_net.best_estimator_), X_base_dev, y_dev, folds
            ),
            "gradient_boosting": out_of_fold_proba(
                clone(gradient_boosting.best_estimator_), X_base_dev, y_dev, folds
            ),
        }
        n_val = len(lstm.validation_labels)
        thresholds = {
            "elastic_net": choose_threshold(oof["elastic_net"], return_dev, config.cost),
            "gradient_boosting": choose_threshold(
                oof["gradient_boosting"], return_dev, config.cost
            ),
            "lstm": choose_threshold(
                lstm.validation_proba, return_dev[-n_val:], config.cost
            ),
        }

        for model in ("elastic_net", "gradient_boosting", "lstm"):
            classification_rows.append(
                _classification_row(
                    model, set_id, set_name, y_test, proba[model], dev_accuracy[model], config
                )
            )
            strategy_rows.append(
                _strategy_row(
                    model, set_id, set_name, return_test, proba[model],
                    thresholds[model], config,
                )
            )
            perf, strategy_returns = timing_strategy(
                return_test, proba[model], thresholds[model], config.cost
            )
            if np.isfinite(perf.sharpe) and perf.sharpe > best_strategy_sharpe:
                best_strategy_sharpe = perf.sharpe
                best_strategy_returns = strategy_returns
                best_strategy_label = f"{model} (set {set_id})"

        for seed_row, seed in enumerate(config.seeds):
            seed_proba = lstm.test_proba_seeds[seed_row]
            seed_rows.append(
                {
                    "feature_set": set_id,
                    "description": set_name,
                    "seed": int(seed),
                    "test_accuracy": round(accuracy(y_test, seed_proba), 4),
                    "test_auc": round(auc(y_test, seed_proba), 4),
                }
            )

    classification = pd.DataFrame(classification_rows)
    strategy = pd.DataFrame(strategy_rows)
    seeds_table = pd.DataFrame(seed_rows)

    equity = pd.DataFrame(
        {
            "month_end": test_months.strftime("%Y-%m-%d"),
            "buy_and_hold": np.cumprod(1.0 + buy_hold_returns),
            "best_strategy": np.cumprod(1.0 + best_strategy_returns),
        }
    )

    model_families = ("elastic_net", "gradient_boosting", "lstm")
    mask = classification["model"].isin(model_families)
    summary = {
        "download": download_info,
        "sample": {
            "months_total": int(split.months_total),
            "months_used": int(n),
            "dev_months": int(n_dev),
            "test_months": int(n_test),
            "dev_start": dev_months.min().strftime("%Y-%m-%d"),
            "dev_end": dev_months.max().strftime("%Y-%m-%d"),
            "test_start": test_months.min().strftime("%Y-%m-%d"),
            "test_end": test_months.max().strftime("%Y-%m-%d"),
            "dev_up_rate": round(base_rate(y_dev), 4),
            "test_up_rate": round(base_rate(y_test), 4),
        },
        "config": {
            "seeds": len(config.seeds),
            "n_splits": config.n_splits,
            "embargo": config.embargo,
            "cost": config.cost,
            "block": config.block,
            "n_boot": config.n_boot,
            "lookback": LSTM_LOOKBACK,
        },
        "headline": {
            "always_up_accuracy": round(base_rate(y_test), 4),
            "max_test_auc": round(float(classification.loc[mask, "test_auc"].max()), 4),
            "min_test_auc": round(float(classification.loc[mask, "test_auc"].min()), 4),
            "max_test_accuracy": round(float(classification.loc[mask, "test_accuracy"].max()), 4),
            "any_auc_ci_above_half": bool((classification.loc[mask, "auc_low"] > 0.5).any()),
            "any_beats_always_up": bool((classification.loc[mask, "acc_diff_low"] > 0).any()),
            "buy_and_hold_sharpe": round(buy_hold_perf.sharpe, 4),
            "buy_and_hold_annual_return": round(buy_hold_perf.annual_return, 4),
            "best_strategy": best_strategy_label,
            "best_strategy_sharpe": round(float(best_strategy_sharpe), 4),
        },
    }

    if write:
        RESULTS_DIR.mkdir(exist_ok=True)
        classification.to_csv(RESULTS_DIR / "classification.csv", index=False)
        strategy.to_csv(RESULTS_DIR / "strategy.csv", index=False)
        seeds_table.to_csv(RESULTS_DIR / "lstm_seeds.csv", index=False)
        equity.to_csv(RESULTS_DIR / "equity_curves.csv", index=False)
        (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    return {
        "classification": classification,
        "strategy": strategy,
        "seeds": seeds_table,
        "equity": equity,
        "summary": summary,
        "config": asdict(config),
    }
