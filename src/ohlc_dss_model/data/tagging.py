import polars as pl

from ohlc_dss_model.config import config
from datetime import time
import exchange_calendars as xcals

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
    pt_1_interval: tuple = config.session.pre_target_split_1,
    pt_2_interval: tuple = config.session.pre_target_split_2,
    target_1_interval: tuple = config.session.target_split_1,
    target_2_interval: tuple = config.session.target_split_2,
) -> pl.DataFrame:
    time_col = pl.col(dt_col).dt.time()

    pre_target_1_start = time(*pt_1_interval[0])
    pre_target_1_end = time(*pt_1_interval[1])

    pre_target_2_start = time(*pt_2_interval[0])
    pre_target_2_end = time(*pt_2_interval[1])

    target_1_start = time(*target_1_interval[0])
    target_1_end = time(*target_1_interval[1])

    target_2_start = time(*target_2_interval[0])
    target_2_end = time(*target_2_interval[1])

    df = df.with_columns(
        pl.when(time_col.is_between(target_1_start, target_1_end, closed="left"))
        .then(pl.lit("Target_1"))
        .when(time_col.is_between(target_2_start, target_2_end, closed="left"))
        .then(pl.lit("Target_2"))
        .when(time_col.is_between(pre_target_2_start, pre_target_2_end, closed="left"))
        .then(pl.lit("Pre_Target_2"))
        .when((time_col >= pre_target_1_start) | (time_col < pre_target_1_end))
        .then(pl.lit("Pre_Target_1"))
        .otherwise(pl.lit("Closed"))
        .alias("Intraday_Session")
    )

    return df.filter(pl.col("Intraday_Session") != "Closed")

# Remove holidays and non trading days
def filter_valid_sessions(df: pl.DataFrame, calendar_name: str = "XNYS") -> pl.DataFrame:
    cal = xcals.get_calendar(calendar_name)
    
    sessions = df["Session"].unique().to_list()
    
    valid_sessions = {
        s for s in sessions
        if cal.is_session(s)
    }
    
    return df.filter(pl.col("Session").is_in(list(valid_sessions)))