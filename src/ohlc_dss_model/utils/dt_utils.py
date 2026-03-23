import polars as pl


# to convert datetime column into a specified timezone
def convert_to_timezone(
    df: pl.DataFrame,
    dt_col: str = "DateTime",
    current_tz: str = "EET",
    target_tz: str = "America/New_York",
    col_suffix: str | None = None,
) -> pl.DataFrame:
    if col_suffix is None:
        col_suffix = target_tz

    return df.with_columns(
        pl.col(dt_col)
        .dt.replace_time_zone(current_tz)
        .dt.convert_time_zone(target_tz)
        .alias(f"{dt_col}_{col_suffix}")
    )
