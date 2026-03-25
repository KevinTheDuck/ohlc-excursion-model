import polars as pl

from ohlc_dss_model.config import config


# to convert datetime column into a specified timezone
def convert_to_timezone(
    df: pl.DataFrame,
    dt_col: str = config.schema.datetime,
    current_tz: str = config.timezone.broker,
    target_tz: str = config.timezone.target,
) -> pl.DataFrame:
    return df.with_columns(
        pl.col(dt_col).dt.replace_time_zone(current_tz).dt.convert_time_zone(target_tz)
    )


# we tag the data to seperate define a daily candle
# each session represents a single daily candle
def session_tagging(
    df: pl.DataFrame,
    dt_col: str = config.schema.datetime,
    eod: int = config.timezone.eod_close_hour,
):
    return df.with_columns(
        pl.when(pl.col(dt_col).dt.hour() >= eod)
        .then((pl.col(dt_col) + pl.duration(days=1)).dt.date())
        .otherwise(pl.col(dt_col).dt.date())
        .alias("Session")
    )
