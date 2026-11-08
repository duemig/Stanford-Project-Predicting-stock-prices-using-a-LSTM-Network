"""Purged walk-forward cross-validation for monthly time-series labels.

Standard k-fold shuffles observations and leaks the future into the past, which
is why it fails on financial series. Walk-forward validation instead uses an
expanding block of past months to predict a later block. Because each label
looks ``horizon`` months into the future, training months whose label window
reaches the validation block are purged, and a further ``embargo`` of months is
dropped just before the validation block.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def purged_walk_forward_splits(
    n_samples: int,
    n_splits: int = 5,
    embargo: int = 1,
    horizon: int = 1,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield expanding-window train/validation index pairs with purge and embargo.

    The samples ``range(n_samples)`` are assumed sorted in time. The series is
    divided into ``n_splits + 1`` equal blocks; validation uses blocks ``1`` to
    ``n_splits`` in turn, and training uses all earlier samples up to a gap of
    ``horizon + embargo`` samples before the validation block.

    Args:
        n_samples: Number of time-ordered samples.
        n_splits: Number of validation folds.
        embargo: Months dropped before each validation block, in addition to the
            purge implied by the label horizon. Must be at least one month.
        horizon: How many months ahead each label looks.

    Yields:
        Tuples ``(train_index, validation_index)`` of integer position arrays.

    Raises:
        ValueError: If the inputs leave no room for a single fold.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be at least 1, got {n_splits}.")
    if embargo < 1:
        raise ValueError(f"embargo must be at least 1 month, got {embargo}.")
    block = n_samples // (n_splits + 1)
    if block == 0:
        raise ValueError(
            f"n_samples={n_samples} is too small for n_splits={n_splits} folds."
        )
    gap = horizon + embargo
    for fold in range(1, n_splits + 1):
        validation_start = fold * block
        validation_stop = n_samples if fold == n_splits else (fold + 1) * block
        train_stop = validation_start - gap
        if train_stop <= 0:
            continue
        yield np.arange(train_stop), np.arange(validation_start, validation_stop)
