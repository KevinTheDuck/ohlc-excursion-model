import polars as pl

from ohlc_dss_model.config import config
from datetime import time

# we tag the data to seperate define a daily candle
# each session represents a single daily candle
def session_tagging(
    df: pl.DataFrame,
    dt_col: str = config.schema.datetime,
    eod: int = config.timezone.eod_close_hour,
) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col(dt_col).dt.hour() >= eod)
        .then((pl.col(dt_col) + pl.duration(days=1)).dt.date())
        .otherwise(pl.col(dt_col).dt.date())
        .alias("Session")
    )


# to assign candles to its respective sessions (asian/london/new york)
def intraday_session_tagging(
    df: pl.DataFrame,
    dt_col: str = config.schema.datetime,
    asian_interval: tuple = config.session.asian_session,
    london_interval: tuple = config.session.london_session,
    new_york_interval: tuple = config.session.new_york_session,
) -> pl.DataFrame:
    time_col = pl.col(dt_col).dt.time()

    asian_start = time(*asian_interval[0])
    asian_end = time(*asian_interval[1])

    london_start = time(*london_interval[0])
    london_end = time(*london_interval[1])

    ny_start = time(*new_york_interval[0])
    ny_end = time(*new_york_interval[1])

    df = df.with_columns(
        pl.when(time_col.is_between(ny_start, ny_end, closed="left"))
        .then(pl.lit("New York"))
        .when(time_col.is_between(london_start, london_end, closed="left"))
        .then(pl.lit("London"))
        .when((time_col >= asian_start) | (time_col < asian_end))
        .then(pl.lit("Asia"))
        .otherwise(pl.lit("Closed"))
        .alias("Intraday_Session")
    )

    return df.filter(pl.col("Intraday_Session") != "Closed")
