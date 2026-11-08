"""Tests for the purged walk-forward splitter."""

from __future__ import annotations

import numpy as np
import pytest

from src.validation import purged_walk_forward_splits


def test_folds_are_ordered_and_disjoint() -> None:
    folds = list(purged_walk_forward_splits(100, n_splits=4, embargo=1, horizon=1))
    assert len(folds) == 4
    previous_validation_start = -1
    for train_index, validation_index in folds:
        assert train_index[0] == 0
        assert train_index[-1] < validation_index[0]
        assert validation_index[0] > previous_validation_start
        previous_validation_start = validation_index[0]
        assert not set(train_index) & set(validation_index)


def test_no_label_leakage_across_the_gap() -> None:
    horizon, embargo = 1, 2
    for train_index, validation_index in purged_walk_forward_splits(
        120, n_splits=5, embargo=embargo, horizon=horizon
    ):
        # No training label window may reach into the validation block, and an
        # extra embargo of months is removed on top of that.
        assert train_index[-1] + horizon < validation_index[0]
        assert validation_index[0] - train_index[-1] - 1 >= horizon + embargo


def test_last_fold_reaches_the_end() -> None:
    n = 90
    folds = list(purged_walk_forward_splits(n, n_splits=3, embargo=1, horizon=1))
    assert folds[-1][1][-1] == n - 1


def test_invalid_arguments_raise() -> None:
    with pytest.raises(ValueError, match="embargo"):
        list(purged_walk_forward_splits(100, embargo=0))
    with pytest.raises(ValueError, match="n_splits"):
        list(purged_walk_forward_splits(100, n_splits=0))
    with pytest.raises(ValueError, match="too small"):
        list(purged_walk_forward_splits(3, n_splits=5))


def test_indices_are_integer_arrays() -> None:
    train_index, validation_index = next(purged_walk_forward_splits(50, n_splits=2))
    assert train_index.dtype.kind == "i"
    assert validation_index.dtype.kind == "i"
    assert isinstance(train_index, np.ndarray)
