"""Tests for the feature construction."""

from __future__ import annotations

import numpy as np

from src.features import (
    LSTM_LOOKBACK,
    WINDOWS_WEEKS,
    aggregate_columns,
    baseline_matrix,
    channel_names,
    daily_channels,
    lstm_tensor,
    valid_month_ends,
    window_aggregates,
)
from src.labels import month_end_dates

DAILY_LOG_RETURN = 0.0004


def test_daily_channels_on_constant_growth(constant_growth_prices) -> None:
    channels = daily_channels(constant_growth_prices).dropna()
    assert np.allclose(channels["ret"], DAILY_LOG_RETURN)
    # Constant returns have zero rolling volatility; constant volume detrends to zero.
    assert np.allclose(channels["vol20"], 0.0, atol=1e-12)
    assert np.allclose(channels["logvol_rel"], 0.0, atol=1e-12)


def test_window_aggregates_sum_and_zero(constant_growth_prices) -> None:
    aggregates = window_aggregates(constant_growth_prices).dropna()
    assert np.isclose(aggregates["ret_1w"].iloc[-1], 5 * DAILY_LOG_RETURN)
    assert np.isclose(aggregates["ret_4w"].iloc[-1], 20 * DAILY_LOG_RETURN)
    assert np.allclose(aggregates.filter(like="vol_"), 0.0, atol=1e-12)
    assert np.allclose(aggregates.filter(like="volume_"), 0.0, atol=1e-10)


def test_feature_counts() -> None:
    assert len(WINDOWS_WEEKS) == 23
    assert aggregate_columns(("vol", "return")) == (
        [f"vol_{w}w" for w in WINDOWS_WEEKS] + [f"ret_{w}w" for w in WINDOWS_WEEKS]
    )
    assert channel_names(("vol", "return", "volume")) == ["vol20", "ret", "logvol_rel"]


def test_lstm_tensor_shape_and_validity(constant_growth_prices) -> None:
    month_ends = month_end_dates(constant_growth_prices)
    valid = valid_month_ends(constant_growth_prices, month_ends)
    sequences, mask = lstm_tensor(constant_growth_prices, valid, ("return", "volume"))
    assert sequences.shape == (len(valid), LSTM_LOOKBACK, 2)
    assert mask.all()
    assert not np.isnan(sequences).any()


def test_valid_month_ends_have_complete_baseline(constant_growth_prices) -> None:
    month_ends = month_end_dates(constant_growth_prices)
    valid = valid_month_ends(constant_growth_prices, month_ends)
    assert len(valid) > 0
    assert set(valid).issubset(set(month_ends))
    matrix = baseline_matrix(constant_growth_prices, valid, ("vol", "return", "volume"))
    assert matrix.shape[1] == 69
    assert not matrix.isna().to_numpy().any()
