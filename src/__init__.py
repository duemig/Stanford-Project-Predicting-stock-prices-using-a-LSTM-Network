"""Reusable modules for the S&P 500 monthly-direction study.

The package is deliberately small and import-light: ``data``, ``features``,
``labels`` and ``validation`` depend only on numpy, pandas and scikit-learn, so
the test suite can import them without TensorFlow or a network connection.
TensorFlow is imported lazily inside :mod:`src.models` only when the LSTM is
built.
"""
