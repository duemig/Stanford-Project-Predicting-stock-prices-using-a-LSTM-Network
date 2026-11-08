"""Run the full S&P 500 monthly-direction study.

Downloads the data if needed, fits every model on every feature set, evaluates
them on the held-out test months, runs the timing strategy, and writes the
``results/`` tables and ``figures/`` plots. Run it from the repository root
inside the project environment::

    python run_study.py
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", message=".*Stochastic Average Gradient.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from src.data import get_prices  # noqa: E402
from src.plots import make_all_figures  # noqa: E402
from src.study import StudyConfig, run_study  # noqa: E402


def main() -> None:
    """Run the study with the default configuration and write all outputs."""
    config = StudyConfig()
    results = run_study(config, write=True, verbose=True)
    prices, _ = get_prices()
    make_all_figures(results, prices, config)
    print("\nHeadline numbers:")
    for name, value in results["summary"]["headline"].items():
        print(f"  {name}: {value}")
    print("\nWrote results/ and figures/.")


if __name__ == "__main__":
    main()
