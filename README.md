# Predicting stock prices with LSTM Networks

Can a long short term memory (LSTM) network predict whether the S&P 500 will be up
or down over the next month, and does it do any better than a regularised logistic
regression or a gradient-boosting model? The short answer is no. Across every
feature set the models score close to the naive rule of always predicting "up",
and a timing strategy built on their forecasts does not beat buying and holding
after costs.

It started as my course project for CS230 Deep Learning at Stanford.

The written report is [report.pdf](report.pdf) and the notebook is
[Final Project.ipynb](Final%20Project.ipynb). Every number below is read from the
files in [results/](results), which are produced by `run_study.py`.

## Method

- **Target.** One non-overlapping observation per month, labelled by the sign of
  the next month's return. This removes the heavy overlap of the original daily
  labelling.
- **Features.** Stationary features over 23 look-back windows for each of three
  families: summed log returns, rolling volatility, and log volume relative to its
  trailing one-year mean (which removes the long-term growth in turnover). The
  three families give the seven feature sets of the original project. The baselines
  use the 69 month-end aggregates; the LSTM reads the most recent 60 trading days
  of the daily channels.
- **Validation.** The last 20% of months are held out as a test set, used once.
  The rest is tuned with purged walk-forward cross-validation: a training month
  whose label reaches a validation month is purged, and a one-month embargo is
  dropped before each validation block. The baselines fit their scaler inside each
  fold; the LSTM standardizes its inputs on the training part of its inner split,
  so no scaling ever sees the data it is evaluated on.
- **Models.** Always-up (the benchmark), an elastic-net logistic regression and a
  gradient-boosting classifier on the aggregates, and an LSTM on the daily
  sequences trained with early stopping over 10 seeds.
- **Evaluation.** Accuracy against the base rate, AUC and Brier score with 95%
  moving-block bootstrap intervals, a paired comparison against always-up, and a
  timing strategy (long the index next month when the predicted up-probability
  exceeds a threshold chosen on validation data only) against buy-and-hold after
  costs.

## Data

The data are daily S&P 500 quotes (`^GSPC`) from 1980 to the end of 2025, about
11,595 trading days. The price data is **not** committed. `run_study.py` downloads
it with `yfinance` on first use and caches it under `data/`; the source, coverage
and download date are recorded in [results/summary.json](results/summary.json). If
Yahoo is blocked, the code falls back to Stooq. To use a file you downloaded
yourself, save it as `data/gspc.csv` with a `Date` index and `Close` and `Volume`
columns.

## Results

The test set is 106 months (February 2017 to November 2025) and is 68.9% up
months, so always predicting "up" already scores 68.9%. Test-set accuracy and AUC:

| Set | Features | Acc EN | Acc GB | Acc LSTM | AUC EN | AUC GB | AUC LSTM |
|----|----|----|----|----|----|----|----|
| 1 | Volatility | 0.69 | 0.60 | 0.68 | 0.48 | 0.47 | 0.49 |
| 2 | Return | 0.66 | 0.56 | 0.69 | 0.52 | 0.47 | 0.54 |
| 3 | Volume | 0.56 | 0.58 | 0.69 | 0.51 | 0.58 | 0.45 |
| 4 | Volatility + Return | 0.68 | 0.58 | 0.69 | 0.55 | 0.48 | 0.53 |
| 5 | Volatility + Volume | 0.69 | 0.61 | 0.69 | 0.51 | 0.55 | 0.44 |
| 6 | Return + Volume | 0.64 | 0.65 | 0.69 | 0.51 | 0.58 | 0.48 |
| 7 | Volatility + Return + Volume | 0.69 | 0.59 | 0.69 | 0.62 | 0.53 | 0.53 |
| — | Always up | | 0.69 | | | 0.50 | |

EN is the elastic-net logistic regression and GB the gradient-boosting model.

Test AUCs run from 0.44 to 0.62 and all but one of the bootstrap intervals contain
0.5. The best test accuracy is 68.9%, equal to the always-up rate, and no model
beats always-up by an accuracy margin whose interval excludes zero. One of the 21
model and feature-set combinations, the elastic net on all three families, has an
AUC interval that just clears 0.5; with 21 combinations that is close to what
chance alone produces, and its accuracy still does not beat always-up.

The timing test tells the same story. Buy-and-hold earns 12.8% a year at a Sharpe
ratio of 0.85 over the test period. The best timing strategy reaches a Sharpe of
1.00, but it is the best of 21 combinations and most strategies simply stay
invested and recover buy-and-hold, so the edge is not evidence of real timing
skill. See [results/strategy.csv](results/strategy.csv) and the figures in
[figures/](figures).

## Running it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python run_study.py              # downloads data, runs everything, writes results/ and figures/
python report/build_report.py    # rebuilds report.pdf from results/ (needs a LaTeX install)
```

Open `Final Project.ipynb` to step through the pipeline; it runs top to bottom.

Tests use small synthetic data and need neither the network nor TensorFlow:

```bash
pip install -r requirements-dev.txt
ruff check src tests run_study.py report/build_report.py
pytest
```

## Layout

| Path | What it is |
|----|----|
| `src/` | data, features, labels, validation, models, evaluation, study and plots |
| `run_study.py` | runs the whole study and writes `results/` and `figures/` |
| `report/` | the LaTeX report and `build_report.py` |
| `tests/` | tests for the labels, the purged splitter and the features |
| `results/`, `figures/` | committed outputs that the report and this README read from |

## Limitations

The test period is a single, strongly rising market, which lifts the base rate and
makes it hard to beat. The monthly sample is small once labels do not overlap, so
confidence intervals are wide. Price and volume may simply not carry enough
information about next month's direction; adding fundamentals or macro data, or
predicting a continuous return rather than a sign, are the natural next steps.
