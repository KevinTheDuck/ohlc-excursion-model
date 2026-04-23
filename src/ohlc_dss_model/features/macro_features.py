import polars as pl
import numpy as np
from fredapi import Fred
from datetime import date, timedelta
from typing import Any

_FRED: dict[str, str] = {
    "VIXCLS": "vix",
    "DGS10":  "us10y",
    "DGS2":   "us2y",
    "EFFR":   "effr",
}

def _as_date(value: Any) -> date:
    return value.date() if hasattr(value, "date") else date.fromisoformat(str(value)[:10])

def _rolling_pct_rank(
        values: np.ndarray,
        window: int,
        min_periods: int = 30
    ) -> np.ndarray:

    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if np.isnan(values[i]):
            continue
        lo = max(0, i - window)
        hist = values[lo:i]
        hist = hist[~np.isnan(hist)]
        if len(hist) < min_periods:
            continue
        result[i] = float((hist < values[i]).sum() / len(hist))
    return result

def _add_pct_rank(
        df: pl.DataFrame,
        col: str,
        window: int,
        min_periods: int = 30,
        output_col: str | None = None
    ) -> pl.DataFrame:
    output = output_col or f"{col}_pct_rank_{window}d"
    array = df[col].to_numpy(allow_copy=True).astype(float)
    ranks = _rolling_pct_rank(array, window, min_periods)
    return df.with_columns(pl.Series(output, ranks, dtype=pl.Float64))

def build_fred_macro(
        sessions: pl.DataFrame,
        api_key: str,
        start_date: str,
        end_date: str
    ) -> pl.DataFrame:

    fred = Fred(api_key=api_key)
    start = start_date - timedelta(days=260)
    frames: list[pl.DataFrame] = []

    for id, col_name in _FRED.items():
        try:
            s = fred.get_series(series_id=id, observation_start=start, observation_end=end_date)

            df = (
                pl.DataFrame({
                    "Date": pl.Series(
                        [_as_date(d) for d in s.index], dtype=pl.Date
                    ),
                    col_name: pl.Series(s.values, dtype=pl.Float64),
                })
                .drop_nulls(col_name)
                .sort("Date")
            )
            print(f"[ok] FRED {id:10} → {col_name}: {df.height} observations")
        except Exception as e:
            print(f"[WARN] {id}: {e}")

    if not frames:
        raise RuntimeError("No FRED data could be fetched")
    
    macro = frames[0]
    for frame in frames[1:]:
        macro = macro.join(frame, on="Date", how="outer_coalesce").sort("Date")

    macro = macro.with_columns([
        (pl.col("us10y") - pl.col("us2y")).alias("10y_2y_spread"),
    ])

    macro = macro.with_columns([
        (pl.col("vix") - pl.col("vix").shift(5)).alias("vix_5d_delta"),
        (pl.col("us10y") - pl.col("us10y").shift(5)).alias("us10y_5d_delta"),
    ])

    macro = _add_pct_rank(macro, "vix", window=365, min_periods=50, out_col="vix_pct_rank_1y")

    macro = (
        macro.rename({
            "vix": "vix_t1",
            "us10y": "us10y_t1",
            "us2y": "us2y_t1",
            "effr": "effr_t1",
            "10y_2y_spread": "10y_2y_spread_t1",
            "vix_pct_rank_1y": "vix_pct_rank_1y_t1",
        })
        .with_columns([
            (pl.col("Date") + pl.duration(days=1)).alias("Join_date")
        ])
        .drop("Date")
        .sort("Join_date")
    )

    result = sessions.sort("Session").join_asof(
        macro,
        left_on="Session",
        right_on="Join_date",
        strategy="backward",
    )

    return result

