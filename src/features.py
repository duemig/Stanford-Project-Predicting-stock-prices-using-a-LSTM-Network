"""Construct stationary features and model-ready matrices and tensors.

Two representations are built from the same daily series:

* window aggregates at each month-end for the linear and gradient-boosting
  baselines (69 columns: 23 look-back windows for each of volatility, return and
  volume); and
* daily channel sequences covering the most recent `LSTM_LOOKBACK` trading days
  for the LSTM.

All features are stationary. Returns are summed daily log returns, volatility is
the rolling standard deviation of daily log returns, and volume is log volume
relative to its trailing one-year mean, which removes the strong secular growth
in traded volume between 1980 and today.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Trading days per week.
WEEK = 5
#: Look-back windows in weeks (23 windows), matching the original feature grid.
WINDOWS_WEEKS = (
    1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 36,
    40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80,
)
#: Trailing window, in trading days, used to detrend log volume (about one year).
VOLUME_BASELINE = 252
#: Number of trading days of history the LSTM sees for each month-end.
LSTM_LOOKBACK = 60
#: Window, in trading days, for the daily volatility channel.
VOL_SPAN = 20

#: Feature family -> (aggregate column prefix, daily channel name).
_FAMILIES = {
    "vol": ("vol_", "vol20"),
    "return": ("ret_", "ret"),
    "volume": ("volume_", "logvol_rel"),
}

#: Model number -> (description, feature families), reproducing the original seven models.
FEATURE_SETS: dict[int, tuple[str, tuple[str, ...]]] = {
    1: ("Volatility", ("vol",)),
    2: ("Return", ("return",)),
    3: ("Volume", ("volume",)),
    4: ("Volatility + Return", ("vol", "return")),
    5: ("Volatility + Volume", ("vol", "volume")),
    6: ("Return + Volume", ("return", "volume")),
    7: ("Volatility + Return + Volume", ("vol", "return", "volume")),
}


def _clean_volume(volume: pd.Series) -> pd.Series:
    """Replace zero volume with the previous valid value so logs are defined.

    Args:
        volume: Daily trading volume.

    Returns:
        Volume with zeros forward-filled.
    """
    return volume.replace(0, np.nan).ffill()


def daily_channels(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute the daily stationary channels used by the LSTM.

    Args:
        prices: Daily data with ``Close`` and ``Volume`` columns.

    Returns:
        A data frame indexed like `prices` with columns ``ret`` (daily log
        return), ``vol20`` (rolling return volatility) and ``logvol_rel`` (log
        volume minus its trailing one-year mean). Leading rows are ``NaN`` until
        each rolling window is full.

    Note:
        These are the daily series the LSTM reads: the daily analogues of the
        month-end aggregates in `window_aggregates`, not the aggregates themselves.
        In particular ``logvol_rel`` is the instantaneous detrended log volume,
        not a window mean.
    """
    close = prices["Close"]
    ret = np.log(close / close.shift(1))
    log_volume = np.log(_clean_volume(prices["Volume"]))
    return pd.DataFrame(
        {
            "ret": ret,
            "vol20": ret.rolling(VOL_SPAN).std(),
            "logvol_rel": log_volume - log_volume.rolling(VOLUME_BASELINE).mean(),
        },
        index=prices.index,
    )


def window_aggregates(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute the 69 month-end window-aggregate features.

    Args:
        prices: Daily data with ``Close`` and ``Volume`` columns.

    Returns:
        A data frame indexed like `prices` with 69 columns ``ret_<w>w``,
        ``vol_<w>w`` and ``volume_<w>w`` for each look-back window ``w`` in
        `WINDOWS_WEEKS`. Leading rows are ``NaN`` until each window is full.
    """
    close = prices["Close"]
    ret = np.log(close / close.shift(1))
    log_volume = np.log(_clean_volume(prices["Volume"]))
    log_volume_year = log_volume.rolling(VOLUME_BASELINE).mean()
    columns: dict[str, pd.Series] = {}
    for weeks in WINDOWS_WEEKS:
        span = weeks * WEEK
        columns[f"ret_{weeks}w"] = ret.rolling(span).sum()
        columns[f"vol_{weeks}w"] = ret.rolling(span).std()
        columns[f"volume_{weeks}w"] = log_volume.rolling(span).mean() - log_volume_year
    return pd.DataFrame(columns, index=prices.index)


def aggregate_columns(families: tuple[str, ...]) -> list[str]:
    """Return the aggregate column names for the given feature families.

    Args:
        families: Feature families, each a key of `FEATURE_SETS` values.

    Returns:
        The matching ``<prefix><w>w`` column names, in family then window order.
    """
    prefixes = [_FAMILIES[family][0] for family in families]
    return [f"{prefix}{weeks}w" for prefix in prefixes for weeks in WINDOWS_WEEKS]


def channel_names(families: tuple[str, ...]) -> list[str]:
    """Return the daily channel names for the given feature families.

    Args:
        families: Feature families, each a key of `FEATURE_SETS` values.

    Returns:
        The matching daily channel names.
    """
    return [_FAMILIES[family][1] for family in families]


def lstm_tensor(
    prices: pd.DataFrame,
    month_ends: pd.DatetimeIndex,
    families: tuple[str, ...],
    lookback: int = LSTM_LOOKBACK,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the LSTM input tensor of daily sequences ending at each month-end.

    Args:
        prices: Daily data with ``Close`` and ``Volume`` columns.
        month_ends: Month-end trading days to build sequences for.
        families: Feature families whose daily channels are stacked.
        lookback: Number of trading days in each sequence.

    Returns:
        A tuple ``(sequences, valid)`` where sequences has shape
        ``(len(month_ends), lookback, n_channels)`` and valid is a boolean array
        marking the month-ends with a complete, non-``NaN`` window. Rows where
        valid is ``False`` hold undefined values and must be dropped by the
        caller.
    """
    channels = daily_channels(prices)[channel_names(families)]
    values = channels.to_numpy()
    position = {date: i for i, date in enumerate(channels.index)}
    sequences = np.zeros((len(month_ends), lookback, values.shape[1]))
    valid = np.zeros(len(month_ends), dtype=bool)
    for row, date in enumerate(month_ends):
        end = position[date]
        start = end + 1 - lookback
        if start < 0:
            continue
        window = values[start : end + 1]
        if np.isnan(window).any():
            continue
        sequences[row] = window
        valid[row] = True
    return sequences, valid


def valid_month_ends(
    prices: pd.DataFrame,
    month_ends: pd.DatetimeIndex,
    lookback: int = LSTM_LOOKBACK,
) -> pd.DatetimeIndex:
    """Return the month-ends usable by every model and feature set.

    A month-end is kept only if all 69 aggregate features are defined and the
    daily channels of all three families have a complete window, so that every
    feature set is trained and tested on exactly the same months.

    Args:
        prices: Daily data with ``Close`` and ``Volume`` columns.
        month_ends: Candidate month-end trading days.
        lookback: Number of trading days in each LSTM sequence.

    Returns:
        The subset of `month_ends` with complete features, as a
        `pandas.DatetimeIndex`.
    """
    aggregates_ok = ~window_aggregates(prices).loc[month_ends].isna().any(axis=1)
    _, sequences_ok = lstm_tensor(prices, month_ends, ("vol", "return", "volume"), lookback)
    keep = aggregates_ok.to_numpy() & sequences_ok
    return month_ends[keep]


def baseline_matrix(
    prices: pd.DataFrame,
    month_ends: pd.DatetimeIndex,
    families: tuple[str, ...],
) -> pd.DataFrame:
    """Build the month-end aggregate feature matrix for the baselines.

    Args:
        prices: Daily data with ``Close`` and ``Volume`` columns.
        month_ends: Month-end trading days to sample, assumed already valid.
        families: Feature families whose aggregate columns are selected.

    Returns:
        A data frame indexed by `month_ends` with the selected aggregate columns.
    """
    return window_aggregates(prices).loc[month_ends, aggregate_columns(families)]
