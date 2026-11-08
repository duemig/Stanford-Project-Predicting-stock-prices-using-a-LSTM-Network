# Predicting the monthly direction of the S&P 500 with LSTM networks

Course project for CS230 Deep Learning, Stanford University (winter 2019).
Written report: [report.pdf](report.pdf). Code and saved outputs: [Final Project.ipynb](Final%20Project.ipynb).

## Question

Can an LSTM network trained on past returns, volatility and trading volume predict whether the S&P 500 will be up or down over the next month better than a regularised logistic regression?

## Short answer

No. Both model families end up close to the naive rule "always predict up". Test AUCs lie between 0.46 and 0.55. The test period is 64% up months, so accuracies of 63% to 66% mostly reflect that base rate. The report reaches the same conclusion.

## Data

| Item | Detail |
|---|---|
| Instrument | S&P 500 index (`^GSPC`), daily OHLCV |
| Source | Yahoo Finance via `pandas_datareader` and `fix_yahoo_finance` |
| Period | 1 January 1980 to 28 February 2019 |
| Rows after feature construction | 9,454 trading days |

## Target

With daily log returns $r_t = \ln(P_t / P_{t-1})$, the label for day $t$ is

$$y_t = \mathbf{1}\left[\sum_{k=1}^{21} r_{t+k} > 0\right],$$

the sign of the cumulative return over the next 21 trading days.

## Features (69)

For each look-back window $w \in \{1, 2, 3, 4, 8, 12, \dots, 80\}$ weeks (23 windows, 5 trading days per week):

| Set | Feature | Definition |
|---|---|---|
| 1 | Volatility | standard deviation of daily returns over the window, scaled by $\sqrt{5w}$ |
| 2 | Return | sum of daily log returns over the window |
| 3 | Volume | average daily trading volume over the window |

Seven models are built from the combinations of the three sets (1, 2, 3, 1+2, 1+3, 2+3, 1+2+3).

## Method

- **Split:** the last 10% of the sample is held out as a test set, without shuffling. It is used once, after all tuning.
- **Validation:** walk-forward validation (`TimeSeriesSplit`, 9 folds) on the first 90%.
- **Scaling:** the baseline standardises features inside the scikit-learn pipeline, so the scaler is fitted on training folds only. The LSTM receives the features unscaled.
- **Baseline:** logistic regression with an elastic net penalty (ridge and lasso as special cases), trained by stochastic gradient descent. The penalty strength and L1 ratio are tuned by grid and random search.
- **LSTM:** one stateful LSTM layer (180 to 400 units, recurrent dropout 0.1 to 0.5) followed by a sigmoid output unit. Batch size 1, one time step per sample, one training epoch, states reset after each epoch.

## Results

Reproduced from Table 1 of the report. Test set: 946 trading days, 64.4% labelled "up".

| Model | Features | LSTM dev / test accuracy | Baseline dev / test accuracy | LSTM test AUC | Baseline test AUC |
|---|---|---|---|---|---|
| 1 | Volatility | 61.0% / 64.4% | 54.5% / 65.0% | 0.52 | 0.52 |
| 2 | Return | 55.6% / 65.0% | 55.4% / 66.1% | 0.52 | 0.54 |
| 3 | Volume | 56.5% / 65.1% | 55.2% / 64.0% | 0.54 | 0.48 |
| 4 | Volatility + return | 54.4% / 63.3% | 52.1% / 60.0% | 0.52 | 0.54 |
| 5 | Volatility + volume | 59.3% / 66.0% | 55.5% / 59.1% | 0.46 | 0.53 |
| 6 | Return + volume | 49.6% / 63.9% | 54.2% / 65.5% | 0.55 | 0.55 |
| 7 | All three | 53.1% / 65.5% | 52.3% / 66.2% | 0.53 | 0.50 |
| Reference | Always predict "up" | | | 64.4% | 0.50 |

LSTM model 1 predicts "up" for every test observation, so its test accuracy equals the base rate exactly.

## Limitations

A reviewer should read the numbers above with these points in mind.

1. **Base rate.** Accuracy has to be compared with 64.4%, not 50%. The best test accuracy (66.2%) is less than two percentage points above the naive rule.
2. **Overlapping labels.** Consecutive labels share 20 of their 21 daily returns. The 946 test days contain only about 45 non-overlapping months, so differences of one or two points are well within noise.
3. **No purging.** The walk-forward folds have no purge or embargo, so labels at the end of each training fold overlap the start of the validation fold. Dev scores are therefore slightly optimistic.
4. **Non-stationary and unscaled inputs.** Volume features are raw levels, which trend strongly between 1980 and 2019. The LSTM receives them unscaled, in share counts, next to returns of order 0.01.
5. **Shallow sequence modelling.** With one time step per sample, the network sees a single day's feature vector. Any memory comes only from the stateful carry-over between batches.
6. **No economic evaluation.** There are no strategy returns, transaction costs or risk measures. This is a classification exercise on a single time series.

## Reproducing

The notebook was run in early 2019 with Python 3, Keras on TensorFlow 1.x and the scikit-learn release of that time. The saved outputs in the notebook are the results quoted above; the notebook has not been re-run since.

- `requirements.txt` lists the packages the notebook imports.
- The data download no longer works because `fix_yahoo_finance` is discontinued. A replacement is `yfinance`: `yf.download("^GSPC", start="1980-01-01", end="2019-03-01")`. Downloaded values may differ slightly from the 2019 pull.
- `keras.wrappers.scikit_learn.KerasClassifier` was removed from later Keras versions; `scikeras.wrappers.KerasClassifier` is the maintained equivalent.

## Contents

| File | Description |
|---|---|
| `Final Project.ipynb` | Data download, feature construction, baselines, LSTM models, evaluation |
| `report.pdf` | Course report |
| `work-flow.png`, `walk-forward.png`, `scaling.png` | Figures used in the notebook |
| `lasso.png`, `ridge.png` | Regularisation illustrations used in the notebook |
| `requirements.txt` | Python packages used |
