"""Shared synthetic data for the tests.

The fixtures build small, deterministic daily price series so the tests need no
network access and no TensorFlow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def constant_growth_prices() -> pd.DataFrame:
    """Daily prices with a constant log return and constant volume.

    On this series every summed return is a known multiple of the daily return,
    every rolling volatility is zero, and the detrended log volume is zero, which
    makes the feature formulas easy to check.

    Returns:
        A data frame indexed by business day with ``Close`` and ``Volume``.
    """
    index = pd.bdate_range("2013-01-01", "2019-12-31")
    daily_log_return = 0.0004
    close = 100.0 * np.exp(daily_log_return * np.arange(len(index)))
    return pd.DataFrame({"Close": close, "Volume": 1_000_000.0}, index=index)


@pytest.fixture
def monthly_fixture() -> tuple[pd.DataFrame, list, list]:
    """A short series with controlled month-end closes.

    Returns:
        A tuple ``(prices, expected_month_ends, expected_labels)`` where the four
        month-end closes are 100, 110, 99 and 105, so the first three labels are
        up, down, up and the final month is dropped.
    """
    index = pd.bdate_range("2020-01-01", "2020-04-30")
    close = pd.Series(50.0, index=index)
    month_end_values = {
        pd.Timestamp("2020-01-31"): 100.0,
        pd.Timestamp("2020-02-28"): 110.0,
        pd.Timestamp("2020-03-31"): 99.0,
        pd.Timestamp("2020-04-30"): 105.0,
    }
    for date, value in month_end_values.items():
        close.loc[date] = value
    prices = pd.DataFrame({"Close": close, "Volume": 1_000_000.0}, index=index)
    expected_month_ends = list(month_end_values)
    expected_labels = [1, 0, 1]
    return prices, expected_month_ends, expected_labels
