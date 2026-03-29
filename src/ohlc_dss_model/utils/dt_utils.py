from datetime import time

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
