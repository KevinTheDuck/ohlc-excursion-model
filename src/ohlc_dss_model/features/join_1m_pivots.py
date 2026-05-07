import polars as pl


def join_1m_cumulative_to_pivots(
    pivots_df: pl.DataFrame,
    bars_1m: pl.DataFrame,
) -> pl.DataFrame:
    cum_cols = ["DateTime", "Session", "Intraday_Session"]
    available = [
        c
        for c in ["vwap_cumulative", "_cum_vol", "_cum_ofi", "_cum_abs_ofi"]
        if c in bars_1m.columns
    ]
    cum_cols.extend(available)

    bars_sub = bars_1m.select(cum_cols).sort(
        ["Session", "Intraday_Session", "DateTime"]
    )

    joined = pivots_df.sort(["Session", "Intraday_Session", "DateTime"]).join_asof(
        bars_sub,
        on="DateTime",
        by=["Session", "Intraday_Session"],
        strategy="backward",
    )

    if "_cum_abs_ofi" in joined.columns and "_cum_vol" in joined.columns:
        joined = joined.with_columns(
            (pl.col("_cum_abs_ofi") / (pl.col("_cum_vol") + 1e-9)).alias("_cum_vpin")
        )

    return joined
