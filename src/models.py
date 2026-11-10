"""Model families compared in the study.

Four families predict the probability that next month's return is positive:

* ``always_up`` — the constant base-rate forecast (the naive benchmark);
* an elastic-net logistic regression on the month-end window aggregates;
* a gradient-boosting classifier on the same aggregates; and
* an LSTM on daily sequences, trained over several seeds.

TensorFlow is imported lazily inside the LSTM functions so that the rest of the
package, and the test suite, do not need it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#: Seed for the deterministic scikit-learn estimators.
RANDOM_STATE = 0

ELASTIC_NET_GRID = {
    "sgd__alpha": [1e-4, 1e-3, 1e-2],
    "sgd__l1_ratio": [0.0, 0.5, 1.0],
}
GRADIENT_BOOSTING_GRID = {
    "learning_rate": [0.03, 0.1],
    "max_leaf_nodes": [7, 15],
}


def always_up_proba(y_train: np.ndarray, n: int) -> np.ndarray:
    """Return the constant base-rate probability forecast.

    Args:
        y_train: Training labels used to estimate the up-rate.
        n: Number of forecasts to return.

    Returns:
        An array of length `n`, each entry the training up-rate.
    """
    return np.full(n, float(np.mean(y_train)))


def _elastic_net_pipeline() -> Pipeline:
    """Build the scaled elastic-net logistic-regression pipeline.

    The logistic regression is trained by stochastic gradient descent, as in the
    original project, which keeps the elastic-net penalty available across
    scikit-learn versions. The scaler sits in the pipeline so it is fitted on
    training folds only during cross-validation.

    Returns:
        The scaler-plus-classifier pipeline.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "sgd",
                SGDClassifier(
                    loss="log_loss",
                    penalty="elasticnet",
                    max_iter=5000,
                    tol=1e-3,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def tune_classifier(
    estimator: Pipeline | HistGradientBoostingClassifier,
    grid: dict,
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> GridSearchCV:
    """Tune a classifier by accuracy over the supplied walk-forward folds.

    Args:
        estimator: The estimator to tune.
        grid: The hyper-parameter grid.
        X: Feature matrix for the development set.
        y: Labels for the development set.
        folds: Train/validation index pairs from the purged splitter.

    Returns:
        The fitted `GridSearchCV`, refit on the whole development set.
    """
    search = GridSearchCV(
        estimator, grid, cv=folds, scoring="accuracy", refit=True, n_jobs=-1
    )
    search.fit(X, y)
    return search


def fit_elastic_net(
    X: np.ndarray, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]
) -> GridSearchCV:
    """Tune and fit the elastic-net logistic regression.

    Args:
        X: Development feature matrix.
        y: Development labels.
        folds: Train/validation index pairs from the purged splitter.

    Returns:
        The fitted `GridSearchCV`.
    """
    return tune_classifier(_elastic_net_pipeline(), ELASTIC_NET_GRID, X, y, folds)


def fit_gradient_boosting(
    X: np.ndarray, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]
) -> GridSearchCV:
    """Tune and fit the gradient-boosting classifier.

    Args:
        X: Development feature matrix.
        y: Development labels.
        folds: Train/validation index pairs from the purged splitter.

    Returns:
        The fitted `GridSearchCV`.
    """
    estimator = HistGradientBoostingClassifier(
        max_iter=300,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )
    return tune_classifier(estimator, GRADIENT_BOOSTING_GRID, X, y, folds)


def out_of_fold_proba(
    estimator: Pipeline | HistGradientBoostingClassifier,
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Collect out-of-fold validation probabilities over the walk-forward folds.

    Args:
        estimator: An unfitted estimator exposing ``predict_proba``; cloned per
            fold via a fresh ``fit``.
        X: Development feature matrix.
        y: Development labels.
        folds: Train/validation index pairs from the purged splitter.

    Returns:
        An array of length ``len(y)`` with the validation-fold probabilities and
        ``NaN`` for samples never in a validation fold (the earliest block).
    """
    from sklearn.base import clone

    proba = np.full(len(y), np.nan)
    for train_index, validation_index in folds:
        fitted = clone(estimator).fit(X[train_index], y[train_index])
        proba[validation_index] = fitted.predict_proba(X[validation_index])[:, 1]
    return proba


@dataclass(frozen=True)
class LSTMConfig:
    """Fixed architecture and training settings for the LSTM.

    The data set is small (a few hundred months), so the architecture is kept
    deliberately small and is not tuned; robustness is assessed across seeds
    instead.
    """

    units: int = 16
    dropout: float = 0.3
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 200
    patience: int = 15
    val_fraction: float = 0.2


def _standardise(
    train: np.ndarray, *others: np.ndarray
) -> list[np.ndarray]:
    """Standardise sequence channels using training statistics only.

    Args:
        train: Training sequences, shape ``(n, lookback, channels)``.
        others: Further sequence arrays to transform with the training stats.

    Returns:
        The standardised ``[train, *others]`` arrays.
    """
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return [(array - mean) / std for array in (train, *others)]


def build_lstm(lookback: int, n_channels: int, config: LSTMConfig, seed: int):
    """Build and compile the LSTM model.

    Args:
        lookback: Sequence length in trading days.
        n_channels: Number of daily feature channels.
        config: Architecture and optimiser settings.
        seed: Seed for weight initialisation and training.

    Returns:
        A compiled `keras.Model`.
    """
    from tensorflow import keras

    keras.utils.set_random_seed(seed)
    model = keras.Sequential(
        [
            keras.Input((lookback, n_channels)),
            keras.layers.LSTM(config.units),
            keras.layers.Dropout(config.dropout),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(config.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


@dataclass(frozen=True)
class LSTMResult:
    """Outputs of training the LSTM over several seeds.

    Attributes:
        test_proba_seeds: Test probabilities per seed, shape ``(seeds, n_test)``.
        test_proba_mean: Seed-averaged test probabilities, length ``n_test``.
        validation_proba: Inner-validation probabilities for threshold choice.
        validation_labels: Labels aligned with `validation_proba`.
        validation_accuracy_seeds: Inner-validation accuracy per seed.
    """

    test_proba_seeds: np.ndarray
    test_proba_mean: np.ndarray
    validation_proba: np.ndarray
    validation_labels: np.ndarray
    validation_accuracy_seeds: np.ndarray


def train_lstm(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray,
    seeds: list[int],
    config: LSTMConfig | None = None,
) -> LSTMResult:
    """Train the LSTM over several seeds and collect test and validation outputs.

    The last `config.val_fraction` of the development sequences forms a
    time-ordered inner validation set used for early stopping and for choosing
    the strategy threshold. Channels are standardised with inner-training
    statistics only.

    Args:
        X_dev: Development sequences, shape ``(n_dev, lookback, channels)``.
        y_dev: Development labels.
        X_test: Test sequences, shape ``(n_test, lookback, channels)``.
        seeds: Random seeds to average over.
        config: Architecture and training settings.

    Returns:
        An `LSTMResult` with per-seed and averaged outputs.
    """
    import tensorflow as tf
    from tensorflow import keras

    tf.config.experimental.enable_op_determinism()
    config = config or LSTMConfig()
    n_val = max(1, int(round(config.val_fraction * len(X_dev))))
    split = len(X_dev) - n_val
    X_train, X_val = X_dev[:split], X_dev[split:]
    y_train, y_val = y_dev[:split], y_dev[split:]
    X_train, X_val, X_test = _standardise(X_train, X_val, X_test)

    lookback, n_channels = X_dev.shape[1], X_dev.shape[2]
    test_proba_seeds = np.zeros((len(seeds), len(X_test)))
    val_proba_seeds = np.zeros((len(seeds), len(X_val)))
    val_accuracy = np.zeros(len(seeds))
    for row, seed in enumerate(seeds):
        model = build_lstm(lookback, n_channels, config, seed)
        stopper = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.patience, restore_best_weights=True
        )
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=[stopper],
            verbose=0,
        )
        test_proba_seeds[row] = model.predict(X_test, verbose=0).ravel()
        val_proba = model.predict(X_val, verbose=0).ravel()
        val_proba_seeds[row] = val_proba
        val_accuracy[row] = float(np.mean((val_proba > 0.5).astype(int) == y_val))

    return LSTMResult(
        test_proba_seeds=test_proba_seeds,
        test_proba_mean=test_proba_seeds.mean(axis=0),
        validation_proba=val_proba_seeds.mean(axis=0),
        validation_labels=y_val,
        validation_accuracy_seeds=val_accuracy,
    )
