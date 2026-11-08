"""Evaluation: classification metrics, bootstrap intervals and the timing strategy.

Confidence intervals use a moving-block bootstrap over months so that any
residual serial dependence is preserved. The timing strategy is long the index
next month when the predicted probability of an up month exceeds a threshold
chosen on validation data only, and is otherwise in cash; it is compared with
buying and holding, after transaction costs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

#: Months per year, for annualising returns and volatility.
MONTHS_PER_YEAR = 12


def accuracy(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> float:
    """Classification accuracy at a probability threshold."""
    return float(accuracy_score(y_true, (proba > threshold).astype(int)))


def auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Area under the ROC curve."""
    return float(roc_auc_score(y_true, proba))


def brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Brier score (mean squared error of the probability forecast)."""
    return float(brier_score_loss(y_true, proba))


def base_rate(y: np.ndarray) -> float:
    """Fraction of positive (up) labels."""
    return float(np.mean(y))


@dataclass(frozen=True)
class Interval:
    """A point estimate with a bootstrap confidence interval."""

    point: float
    low: float
    high: float


def _moving_block_index(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a moving-block bootstrap index of length `n`.

    Args:
        n: Length of the series and of the returned index.
        block: Block length in months.
        rng: Random generator.

    Returns:
        An integer index array of length `n`.
    """
    block = min(block, n)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    return np.concatenate([np.arange(start, start + block) for start in starts])[:n]


def bootstrap_metric(
    y_true: np.ndarray,
    proba: np.ndarray,
    metric,
    block: int = 12,
    n_boot: int = 2000,
    seed: int = 0,
    need_both_classes: bool = False,
) -> Interval:
    """Bootstrap a metric with a moving-block resample over months.

    Args:
        y_true: True labels.
        proba: Predicted probabilities.
        metric: Callable ``metric(y_true, proba) -> float``.
        block: Block length in months.
        n_boot: Number of bootstrap resamples.
        seed: Random seed.
        need_both_classes: If ``True``, skip resamples with a single class (for
            metrics such as AUC that are otherwise undefined).

    Returns:
        The point estimate and its 95% percentile interval.
    """
    rng = np.random.default_rng(seed)
    point = float(metric(y_true, proba))
    samples: list[float] = []
    attempts = 0
    while len(samples) < n_boot and attempts < n_boot * 5:
        attempts += 1
        index = _moving_block_index(len(y_true), block, rng)
        if need_both_classes and len(np.unique(y_true[index])) < 2:
            continue
        samples.append(float(metric(y_true[index], proba[index])))
    low, high = np.percentile(samples, [2.5, 97.5])
    return Interval(point, float(low), float(high))


def paired_accuracy_vs_always_up(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float = 0.5,
    block: int = 12,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Compare a model's accuracy with the always-up rule, month by month.

    Args:
        y_true: True labels.
        proba: Predicted probabilities.
        threshold: Probability threshold for the class decision.
        block: Block length in months.
        n_boot: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        A tuple ``(difference, low, high, p_value)`` where difference is the
        model accuracy minus the always-up accuracy, ``[low, high]`` is its 95%
        block-bootstrap interval, and p_value is the share of resamples in which
        the model does no better than always-up.
    """
    model_correct = ((proba > threshold).astype(int) == y_true).astype(float)
    always_up_correct = (y_true == 1).astype(float)
    difference = model_correct - always_up_correct
    rng = np.random.default_rng(seed)
    samples = np.array(
        [
            float(difference[_moving_block_index(len(difference), block, rng)].mean())
            for _ in range(n_boot)
        ]
    )
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(difference.mean()), float(low), float(high), float(np.mean(samples <= 0))


def _strategy_returns(
    next_return: np.ndarray, position: np.ndarray, cost: float
) -> np.ndarray:
    """Net monthly returns of a position series after transaction costs.

    Args:
        next_return: Realised next-month index returns.
        position: Position each month (1 invested, 0 in cash).
        cost: Proportional cost charged on the change in position.

    Returns:
        Net monthly returns.
    """
    previous = np.concatenate([[0.0], position[:-1]])
    turnover = np.abs(position - previous)
    return position * next_return - cost * turnover


def choose_threshold(
    proba_validation: np.ndarray,
    next_return_validation: np.ndarray,
    cost: float,
    grid: np.ndarray | None = None,
) -> float:
    """Pick the probability threshold that maximises validation Sharpe.

    Args:
        proba_validation: Validation probabilities (may contain ``NaN``, which
            are dropped with their returns).
        next_return_validation: Realised next-month returns on the validation set.
        cost: Proportional transaction cost.
        grid: Candidate thresholds; defaults to ``0.30`` to ``0.70`` in ``0.01``
            steps.

    Returns:
        The threshold with the highest validation Sharpe ratio, or ``0.5`` if no
        threshold produces a positive-variance return series.
    """
    if grid is None:
        grid = np.round(np.arange(0.30, 0.7001, 0.01), 2)
    mask = ~np.isnan(proba_validation)
    proba = proba_validation[mask]
    returns = next_return_validation[mask]
    best_threshold, best_sharpe = 0.5, -np.inf
    for threshold in grid:
        position = (proba > threshold).astype(float)
        net = _strategy_returns(returns, position, cost)
        if net.std(ddof=1) == 0:
            continue
        sharpe = net.mean() / net.std(ddof=1)
        if sharpe > best_sharpe:
            best_sharpe, best_threshold = sharpe, float(threshold)
    return best_threshold


@dataclass(frozen=True)
class Performance:
    """Annualised performance of a monthly return series."""

    annual_return: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    invested_fraction: float


def performance(returns: np.ndarray, invested_fraction: float = 1.0) -> Performance:
    """Annualised return, volatility, Sharpe ratio and max drawdown.

    Args:
        returns: Monthly returns.
        invested_fraction: Fraction of months with a non-zero position.

    Returns:
        A `Performance` summary. The Sharpe ratio uses a zero cash rate.
    """
    n = len(returns)
    annual_return = float(np.prod(1.0 + returns) ** (MONTHS_PER_YEAR / n) - 1.0)
    annual_volatility = float(np.std(returns, ddof=1) * np.sqrt(MONTHS_PER_YEAR))
    sharpe = (
        float(np.mean(returns) * MONTHS_PER_YEAR / annual_volatility)
        if annual_volatility > 0
        else float("nan")
    )
    curve = np.cumprod(1.0 + returns)
    max_drawdown = float((curve / np.maximum.accumulate(curve) - 1.0).min())
    return Performance(
        annual_return, annual_volatility, sharpe, max_drawdown, float(invested_fraction)
    )


def timing_strategy(
    next_return_test: np.ndarray,
    proba_test: np.ndarray,
    threshold: float,
    cost: float,
) -> tuple[Performance, np.ndarray]:
    """Evaluate the timing strategy on the test months.

    Args:
        next_return_test: Realised next-month index returns on the test set.
        proba_test: Predicted up-probabilities on the test set.
        threshold: Probability threshold above which the strategy is invested.
        cost: Proportional transaction cost on position changes.

    Returns:
        A tuple ``(performance, net_returns)`` for the strategy.
    """
    position = (proba_test > threshold).astype(float)
    net = _strategy_returns(next_return_test, position, cost)
    return performance(net, invested_fraction=float(position.mean())), net


def buy_and_hold(next_return_test: np.ndarray) -> tuple[Performance, np.ndarray]:
    """Evaluate buying and holding the index over the test months.

    Args:
        next_return_test: Realised next-month index returns on the test set.

    Returns:
        A tuple ``(performance, returns)`` for buy-and-hold.
    """
    return performance(next_return_test, invested_fraction=1.0), next_return_test
