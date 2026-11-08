"""Non-overlapping monthly direction labels.

The original project labelled every trading day by the sign of the next 21-day
return, so consecutive labels shared 20 of their 21 daily returns. Here the
series is reduced to one observation per month (the last trading day) and each
label is the sign of the *next* calendar month's return, so neighbouring labels
use disjoint return windows.
"""

from __future__ import annotations

import pandas as pd


def month_end_dates(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the last trading day of each calendar month.

    Args:
        prices: Daily data indexed by a sorted `pandas.DatetimeIndex`.

    Returns:
        The month-end trading days as a `pandas.DatetimeIndex`.
    """
    close = prices["Close"]
    last_per_month = close.groupby(close.index.to_period("M")).tail(1)
    return pd.DatetimeIndex(last_per_month.index)


def monthly_labels(prices: pd.DataFrame) -> pd.DataFrame:
    """Build non-overlapping monthly labels from daily prices.

    Args:
        prices: Daily data indexed by a sorted `pandas.DatetimeIndex` with a
            ``Close`` column.

    Returns:
        A data frame indexed by month-end date with columns ``next_return``
        (the simple return of the following month, `float`) and ``label``
        (``1`` if that return is positive, else ``0``, `int`). The final month
        has no following month and is dropped.
    """
    close = prices["Close"]
    month_end_close = close.groupby(close.index.to_period("M")).tail(1)
    next_return = month_end_close.shift(-1) / month_end_close - 1.0
    labels = pd.DataFrame(
        {
            "next_return": next_return.to_numpy(),
            "label": (next_return > 0).astype("int64").to_numpy(),
        },
        index=pd.DatetimeIndex(month_end_close.index, name="month_end"),
    )
    return labels.dropna(subset=["next_return"])
