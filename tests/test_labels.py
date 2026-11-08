"""Tests for the non-overlapping monthly labels."""

from __future__ import annotations

import pandas as pd
import pytest

from src.labels import month_end_dates, monthly_labels


def test_month_end_dates_picks_last_trading_day(monthly_fixture) -> None:
    prices, expected_month_ends, _ = monthly_fixture
    assert list(month_end_dates(prices)) == expected_month_ends


def test_labels_match_next_month_sign(monthly_fixture) -> None:
    prices, expected_month_ends, expected_labels = monthly_fixture
    labels = monthly_labels(prices)
    # The final month is dropped because it has no following month.
    assert list(labels.index) == expected_month_ends[:-1]
    assert labels["label"].tolist() == expected_labels


def test_next_return_value(monthly_fixture) -> None:
    prices, _, _ = monthly_fixture
    labels = monthly_labels(prices)
    assert labels["next_return"].iloc[0] == pytest.approx(110.0 / 100.0 - 1.0, rel=1e-9)


def test_labels_are_non_overlapping(monthly_fixture) -> None:
    prices, _, _ = monthly_fixture
    labels = monthly_labels(prices)
    periods = labels.index.to_period("M")
    assert len(periods) == len(set(periods))


def test_dropna_removes_only_trailing_month() -> None:
    index = pd.bdate_range("2021-01-01", "2021-06-30")
    close = pd.Series(range(1, len(index) + 1), index=index, dtype=float)
    prices = pd.DataFrame({"Close": close, "Volume": 1.0}, index=index)
    labels = monthly_labels(prices)
    # Six months of data give five labels (the last has no following month).
    assert len(labels) == 5
    assert labels["next_return"].notna().all()
