"""Download and cache daily S&P 500 index data.

The study uses the S&P 500 price index (Yahoo ticker ``^GSPC``). Yahoo Finance
is the primary source and Stooq is a fallback. Price data is never committed to
the repository; the README explains how to fetch it.
"""

from __future__ import annotations

import datetime as dt
import io
import json
from pathlib import Path

import pandas as pd

#: Yahoo Finance ticker for the S&P 500 price index.
TICKER = "^GSPC"
#: First date downloaded, matching the original project.
START = "1980-01-01"

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PRICE_PATH = DATA_DIR / "gspc.csv"
INFO_PATH = DATA_DIR / "download_info.json"


def default_end() -> str:
    """Return January 1 of the current year as an exclusive end date.

    Downloading up to this date yields data through the end of the latest full
    calendar year.

    Returns:
        The end date as a ``YYYY-01-01`` string.
    """
    return f"{dt.date.today().year}-01-01"


def _download_yahoo(start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance.

    Args:
        start: First date, inclusive.
        end: Last date, exclusive.

    Returns:
        A data frame indexed by date with OHLCV columns.

    Raises:
        RuntimeError: If Yahoo returns no rows.
    """
    import yfinance as yf

    frame = yf.download(
        TICKER,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )
    if frame is None or frame.empty:
        raise RuntimeError("Yahoo Finance returned no data.")
    return frame


def _download_stooq(start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from Stooq as a CSV.

    Stooq often reports zero volume for the index, in which case the volume
    feature set is not usable; the caller should check.

    Args:
        start: First date, inclusive.
        end: Last date, exclusive.

    Returns:
        A data frame indexed by date with OHLCV columns.

    Raises:
        RuntimeError: If Stooq returns no rows in the requested range.
    """
    import requests

    response = requests.get("https://stooq.com/q/d/l/?s=^spx&i=d", timeout=60)
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(response.text), parse_dates=["Date"]).set_index("Date")
    frame = frame.loc[(frame.index >= pd.Timestamp(start)) & (frame.index < pd.Timestamp(end))]
    if frame.empty:
        raise RuntimeError("Stooq returned no data for the requested range.")
    return frame


def download_prices(start: str = START, end: str | None = None) -> tuple[pd.DataFrame, str]:
    """Download daily S&P 500 OHLCV data, trying Yahoo then Stooq.

    Args:
        start: First date to download, inclusive, as ``YYYY-MM-DD``.
        end: (Optional) Exclusive end date. ``None`` uses `default_end`.

    Returns:
        A tuple ``(prices, source)`` where prices is indexed by date with the
        available ``Open``, ``High``, ``Low``, ``Close`` and ``Volume`` columns
        and source names the provider used.

    Raises:
        RuntimeError: If no source returns usable data.
    """
    end = end or default_end()
    errors: dict[str, str] = {}
    for name, fetch in (("yahoo", _download_yahoo), ("stooq", _download_stooq)):
        try:
            frame = fetch(start, end)
        except Exception as error:
            errors[name] = str(error)
            continue
        frame = frame.rename(columns=str.title)
        keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in frame.columns]
        return frame[keep].sort_index(), name
    raise RuntimeError(f"No data source worked. Errors by source: {errors}")


def get_prices(
    start: str = START,
    end: str | None = None,
    force: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Load cached prices, or download and cache them if missing.

    Args:
        start: First date to download, inclusive, as ``YYYY-MM-DD``.
        end: (Optional) Exclusive end date. ``None`` uses `default_end`.
        force: If ``True``, re-download even when a cache exists.

    Returns:
        A tuple ``(prices, info)`` where prices is indexed by date with OHLCV
        columns and info records the source, download date and coverage.
    """
    if PRICE_PATH.exists() and not force:
        prices = pd.read_csv(PRICE_PATH, parse_dates=["Date"]).set_index("Date")
        info = json.loads(INFO_PATH.read_text()) if INFO_PATH.exists() else {}
        return prices, info

    prices, source = download_prices(start, end or default_end())
    DATA_DIR.mkdir(exist_ok=True)
    prices.to_csv(PRICE_PATH, index_label="Date")
    info = {
        "ticker": TICKER,
        "source": source,
        "download_date": dt.date.today().isoformat(),
        "start": str(prices.index.min().date()),
        "end": str(prices.index.max().date()),
        "rows": int(len(prices)),
    }
    INFO_PATH.write_text(json.dumps(info, indent=2))
    return prices, info
