import polars as pl

from ohlc_dss_model.config import config


# for sorting data based on column name
def sort_data(
    df: pl.DataFrame, col: str = config.schema.datetime, descending: bool = False
) -> pl.DataFrame:
    return df.sort(col, descending=descending)

# remove days with incomplete session count
def remove_incomplete_days(df: pl.DataFrame) -> pl.DataFrame:
    valid_days = (
        df.group_by("Session")
        .agg(pl.col("Intraday_Session").n_unique().alias("n_sessions"))
        .filter(pl.col("n_sessions") == 4)
        .select("Session")
    )

    return df.join(valid_days, on="Session", how="inner")
