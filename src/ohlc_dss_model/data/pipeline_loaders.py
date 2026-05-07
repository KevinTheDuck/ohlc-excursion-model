from pathlib import Path

import polars as pl

from ohlc_dss_model.data import (
    load_parquet,
    remove_incomplete_days,
    intraday_session_tagging,
    session_tagging,
    filter_valid_sessions,
)
from ohlc_dss_model.features import (
    aggregate_sessions,
    yang_zhang,
    FULL_DAY_SPEC,
    assign_direction,
    calculate_excursion_bands,
)
from ohlc_dss_model.utils import convert_to_timezone


def load_raw_data(file_path: Path) -> pl.DataFrame:
    raw_data = load_parquet(file_path)
    raw_data = convert_to_timezone(raw_data)
    raw_data = session_tagging(raw_data)
    raw_data = intraday_session_tagging(raw_data)
    raw_data = remove_incomplete_days(raw_data)
    return raw_data.select(
        pl.col(
            [
                "DateTime",
                "Session",
                "Intraday_Session",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]
        )
    )


def load_aggregated_data(df: pl.DataFrame) -> pl.DataFrame:
    aggregated_data = aggregate_sessions(df)
    aggregated_data = filter_valid_sessions(aggregated_data)

    aggregated_data = aggregated_data.with_columns(
        pl.col("O_Pre_Target_1").alias("O_Ref")
    )

    aggregated_data = yang_zhang(aggregated_data, FULL_DAY_SPEC, mode="historical")

    aggregated_data = assign_direction(aggregated_data)
    aggregated_data = calculate_excursion_bands(aggregated_data)
    return aggregated_data
