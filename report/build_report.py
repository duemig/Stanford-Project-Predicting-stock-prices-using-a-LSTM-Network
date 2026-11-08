"""Generate the report's numbers and tables from ``results/`` and compile it.

This reads the committed results files, writes the LaTeX fragments in
``report/generated/`` (number macros and the two result tables), then compiles
``report/report.tex`` and copies the PDF to the repository root. Run it from the
repository root after the study::

    python report/build_report.py
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
REPORT_DIR = REPO_ROOT / "report"
GENERATED_DIR = REPORT_DIR / "generated"
FAMILIES = ("elastic_net", "gradient_boosting", "lstm")
FAMILY_LABEL = {
    "elastic_net": "Elastic net",
    "gradient_boosting": "Gradient boosting",
    "lstm": "LSTM",
}


def _percent(value: float) -> str:
    """Format a fraction as a one-decimal LaTeX percentage."""
    return f"{100 * value:.1f}\\%"


def _month_year(iso_date: str) -> str:
    """Format an ISO date as ``Month YYYY``."""
    return dt.date.fromisoformat(iso_date).strftime("%B %Y")


def write_macros(summary: dict) -> None:
    """Write the number macros used in the report prose.

    Args:
        summary: The parsed ``results/summary.json``.
    """
    sample = summary["sample"]
    config = summary["config"]
    headline = summary["headline"]
    download = summary["download"]
    test_fraction = sample["test_months"] / sample["months_used"]
    macros = {
        "dataStart": _month_year(download["start"]),
        "dataEnd": _month_year(download["end"]),
        "nRawRows": f"{download['rows']:,}",
        "nFeatures": "69",
        "lookback": str(config["lookback"]),
        "testFrac": f"{round(100 * test_fraction)}\\%",
        "embargo": str(config["embargo"]),
        "nMonths": str(sample["months_used"]),
        "nDev": str(sample["dev_months"]),
        "nTest": str(sample["test_months"]),
        "devUp": _percent(sample["dev_up_rate"]),
        "testUp": _percent(sample["test_up_rate"]),
        "minAUC": f"{headline['min_test_auc']:.2f}",
        "maxAUC": f"{headline['max_test_auc']:.2f}",
        "maxAcc": _percent(headline["max_test_accuracy"]),
        "nSeeds": str(config["seeds"]),
        "cost": f"{round(config['cost'] * 10000)} basis points",
        "bhReturn": _percent(headline["buy_and_hold_annual_return"]),
        "bhSharpe": f"{headline['buy_and_hold_sharpe']:.2f}",
        "bestSharpe": f"{headline['best_strategy_sharpe']:.2f}",
    }
    lines = [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()]
    (GENERATED_DIR / "macros.tex").write_text("\n".join(lines) + "\n")


def write_classification_table(classification: pd.DataFrame) -> None:
    """Write the test accuracy and AUC table.

    Args:
        classification: The classification results table.
    """
    models = classification[classification["model"].isin(FAMILIES)]
    rows = []
    for set_id in sorted(models["feature_set"].unique()):
        block = models[models["feature_set"] == set_id].set_index("model")
        description = block["description"].iloc[0]
        accuracies = " & ".join(f"{block.loc[m, 'test_accuracy']:.2f}" for m in FAMILIES)
        aucs = " & ".join(f"{block.loc[m, 'test_auc']:.2f}" for m in FAMILIES)
        rows.append(f"{set_id} & {description} & {accuracies} & {aucs} \\\\")

    always_up = classification[classification["model"] == "always_up"].iloc[0]
    always_row = (
        "\\multicolumn{2}{l}{Always up} & "
        f"\\multicolumn{{3}}{{c}}{{{always_up['test_accuracy']:.2f}}} & "
        f"\\multicolumn{{3}}{{c}}{{{always_up['test_auc']:.2f}}} \\\\"
    )

    table = [
        "\\begin{tabular}{clcccccc}",
        "\\toprule",
        " & & \\multicolumn{3}{c}{Test accuracy} & \\multicolumn{3}{c}{Test AUC} \\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
        "Set & Features & EN & GB & LSTM & EN & GB & LSTM \\\\",
        "\\midrule",
        *rows,
        "\\midrule",
        always_row,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    (GENERATED_DIR / "table_classification.tex").write_text("\n".join(table) + "\n")


def write_strategy_table(strategy: pd.DataFrame) -> None:
    """Write the strategy performance table (buy-and-hold and best per family).

    Args:
        strategy: The strategy results table.
    """
    buy_hold = strategy[strategy["model"] == "buy_and_hold"].iloc[0]
    rows = [
        "Buy and hold & "
        f"{_percent(buy_hold['annual_return'])} & {_percent(buy_hold['annual_volatility'])} & "
        f"{buy_hold['sharpe']:.2f} & {_percent(buy_hold['max_drawdown'])} \\\\"
    ]
    for family in FAMILIES:
        block = strategy[strategy["model"] == family]
        best = block.loc[block["sharpe"].idxmax()]
        label = f"{FAMILY_LABEL[family]} (set {int(best['feature_set'])})"
        rows.append(
            f"{label} & {_percent(best['annual_return'])} & "
            f"{_percent(best['annual_volatility'])} & {best['sharpe']:.2f} & "
            f"{_percent(best['max_drawdown'])} \\\\"
        )

    table = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Strategy & Ann. return & Ann. vol & Sharpe & Max drawdown \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    (GENERATED_DIR / "table_strategy.tex").write_text("\n".join(table) + "\n")


def compile_report() -> Path:
    """Compile the report and copy the PDF to the repository root.

    Returns:
        The path of the report PDF at the repository root.

    Raises:
        FileNotFoundError: If no LaTeX engine (latexmk or pdflatex) is available.
    """
    if shutil.which("latexmk"):
        command = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "report.tex"]
        subprocess.run(command, cwd=REPORT_DIR, check=True)
    elif shutil.which("pdflatex"):
        for _ in range(2):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                cwd=REPORT_DIR,
                check=True,
            )
    else:
        raise FileNotFoundError("No LaTeX engine found; install latexmk or pdflatex.")

    destination = REPO_ROOT / "report.pdf"
    shutil.copyfile(REPORT_DIR / "report.pdf", destination)
    return destination


def main() -> None:
    """Build the report fragments and compile the PDF."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads((RESULTS_DIR / "summary.json").read_text())
    classification = pd.read_csv(RESULTS_DIR / "classification.csv")
    strategy = pd.read_csv(RESULTS_DIR / "strategy.csv")

    write_macros(summary)
    write_classification_table(classification)
    write_strategy_table(strategy)
    destination = compile_report()
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
