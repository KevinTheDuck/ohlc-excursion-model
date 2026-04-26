import polars as pl
import numpy as np

from fredapi import Fred
from datetime import date, timedelta
from typing import Any, List, Set

import time

from ohlc_dss_model.features import fetch_fomc_dates

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
        end_date: str,
        max_retries: int = 15,
        base_delay: float = 1.0,
        ratelimit_delay: float = 0.1
    ) -> pl.DataFrame:

    fred = Fred(api_key=api_key)
    start = start_date - timedelta(days=260)
    frames: list[pl.DataFrame] = []

    for idx, (id, col_name) in enumerate(_FRED.items()):
        if idx > 0 and ratelimit_delay:
            time.sleep(ratelimit_delay)

        data_acquired = False
        for attempt in range(1, max_retries + 1):
            try:
                s = fred.get_series(
                    series_id=id,
                    observation_start=start,
                    observation_end=end_date
                )
                data_acquired = True
                break
            except Exception as e:
                if attempt == max_retries:
                    print(f"[FAIL] {id} : {e}  (all {max_retries} attempts exhausted)")
                else:
                    wait = base_delay
                    print(f"[WARN] {id} attempt {attempt}/{max_retries}: {e}  (retrying in {wait:.1f}s)")
                    time.sleep(wait)

        if not data_acquired:
            continue

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
        frames.append(df)
        print(f"[ok] FRED {id:10} → {col_name}: {df.height} observations")

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

    macro = _add_pct_rank(macro, "vix", window=365, min_periods=50, output_col="vix_pct_rank_1y")

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

    macro = macro.with_columns(pl.exclude("Join_date").fill_nan(None))
    macro = macro.fill_null(strategy="forward").fill_null(strategy="backward")

    result = sessions.sort("Session").join_asof(
        macro,
        left_on="Session",
        right_on="Join_date",
        strategy="backward",
    )

    return result.select([
        "Session", 
        "vix_t1", "us10y_t1", "us2y_t1", "effr_t1", "10y_2y_spread_t1", "vix_pct_rank_1y_t1",
        "vix_5d_delta", "us10y_5d_delta"
    ])

def build_individual_event_flags(
    sessions: pl.DataFrame,
    api_key: str,
    start: date,
    end: date,
    max_retries: int = 15,
    base_delay: float = 1.0,
) -> pl.DataFrame:
    fred = Fred(api_key=api_key)
    end = end + timedelta(49)
    fomc_buffer_start = start
    fomc_dates: List[date] = sorted(fetch_fomc_dates(fomc_buffer_start, end))
    fomc_set = set(fomc_dates)
    print(f"  [ok] FOMC dates fetched: {len(fomc_dates)}")
 
    def _vintage_set(series_id: str) -> Set[date]:
        data_acquired = False
        for attempt in range(1, max_retries + 1):
            try:
                vdates = fred.get_series_vintage_dates(series_id)
                data_acquired = True
                break
            except Exception as exc:
                if attempt == max_retries:
                    print(f"  [fail] {series_id}: {exc}  (all {max_retries} attempts exhausted)")
                    return set()
                wait = base_delay
                print(f"  [warn] {series_id} attempt {attempt}/{max_retries}: {exc}  (retrying in {wait:.1f}s)")
                time.sleep(wait)

        s = {_as_date(d) for d in vdates if start <= _as_date(d) <= end}
        print(f"  [ok] {series_id:12} vintage dates: {len(s)}")
        return s
 
    nfp_dates      = _vintage_set("PAYEMS")
    cpi_dates      = _vintage_set("CPIAUCSL")
    core_cpi_dates = _vintage_set("CPILFESL")
 
    session_list: List[date] = sessions["Session"].to_list()
 
    is_fomc_day:      List[bool]        = []
    is_fomc_week:     List[bool]        = []
    days_to_fomc_arr: List[int | None]  = []
    is_nfp:           List[bool]        = []
    is_cpi:           List[bool]        = []
    is_core_cpi:      List[bool]        = []
 
    for sess in session_list:
        is_fomc_day.append(sess in fomc_set)
 
        iso = sess.isocalendar()
        is_fomc_week.append(
            any(
                f.isocalendar().year == iso.year and f.isocalendar().week == iso.week
                for f in fomc_dates
            )
        )
 
        future = [f for f in fomc_dates if f >= sess]
        days_to_fomc_arr.append((future[0] - sess).days if future else None)

        is_nfp.append(sess in nfp_dates)
        is_cpi.append(sess in cpi_dates)
        is_core_cpi.append(sess in core_cpi_dates)
 
    result = sessions.with_columns([
        pl.Series("is_fomc_day",      is_fomc_day,      dtype=pl.Boolean),
        pl.Series("is_fomc_week",     is_fomc_week,     dtype=pl.Boolean),
        pl.Series("days_to_fomc",     days_to_fomc_arr, dtype=pl.Int16),
        pl.Series("is_nfp_day",       is_nfp,           dtype=pl.Boolean),
        pl.Series("is_cpi_day",       is_cpi,           dtype=pl.Boolean),
        pl.Series("is_core_cpi_day",  is_core_cpi,      dtype=pl.Boolean),
    ])
 
    print(
        f"  [info] Individual flags — FOMC days: {sum(is_fomc_day)}, "
        f"NFP: {sum(is_nfp)}, CPI: {sum(is_cpi)}, "
    )
    return result.select(["Session", "is_fomc_day", "is_fomc_week", "days_to_fomc", "is_nfp_day", "is_cpi_day", "is_core_cpi_day"])

def get_calendar_index(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.col("Session").dt.weekday().alias("day_of_week"),
        pl.col("Session").dt.month().alias("month"),
        ((pl.col("Session").dt.day() - 1) // 7 + 1).alias("week_of_month"),
    ])