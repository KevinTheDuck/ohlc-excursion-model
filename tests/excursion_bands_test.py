import math

import polars as pl

from ohlc_dss_model.data import (
	filter_valid_sessions,
	intraday_session_tagging,
	load_parquet,
	remove_incomplete_days,
	session_tagging,
)
from ohlc_dss_model.features.estimator_spec import FULL_DAY_SPEC, PRE_NY_SPEC
from ohlc_dss_model.features.excursion_bands import calculate_excursion_bands
from ohlc_dss_model.features.session_aggregation import aggregate_sessions
from ohlc_dss_model.features.volatility import yang_zhang
from ohlc_dss_model.utils.dt_utils import convert_to_timezone


N = 20
WINDOW_SESSIONS = N + 2
EXCURSION_BAND_COLS = [
	"Band_AE_Neg_Upper",
	"Band_AE_Neg_Lower",
	"Band_AE_Pos_Upper",
	"Band_AE_Pos_Lower",
	"Band_FE_Neg_Upper",
	"Band_FE_Neg_Lower",
	"Band_FE_Pos_Upper",
	"Band_FE_Pos_Lower",
]


def _prepare_real_intraday() -> pl.DataFrame:
	df = load_parquet()
	df = convert_to_timezone(df)
	df = session_tagging(df)
	df = intraday_session_tagging(df)
	df = filter_valid_sessions(df)
	df = remove_incomplete_days(df)

	return df.sort(["Session", "DateTime"])


def _prepare_window_without_last_ny(n_sessions: int = WINDOW_SESSIONS) -> tuple[pl.DataFrame, object]:
	df = _prepare_real_intraday()
	sessions = (
		df.select("Session")
		.unique()
		.sort("Session")
		.tail(n_sessions)
		.get_column("Session")
		.to_list()
	)

	last_session = sessions[-1]
	window_df = df.filter(pl.col("Session").is_in(sessions)).filter(
		~(
			(pl.col("Session") == last_session)
			& (pl.col("Intraday_Session") == "New York")
		)
	)
	return window_df, last_session


def _run_excursion_pipeline(df: pl.DataFrame, n: int = N) -> pl.DataFrame:
	session_df = aggregate_sessions(df)
	session_df = yang_zhang(session_df, FULL_DAY_SPEC, "historical", n=n)
	session_df = yang_zhang(session_df, PRE_NY_SPEC, "session", n=n)

	return calculate_excursion_bands(
		session_df.with_columns([
			pl.col("O_Asia").alias("O_Ref"),
			pl.coalesce([pl.col("Sigma_Historical"), pl.col("Sigma_Today")]).alias("Sigma_Historical"),
		]),
		n=n,
	)


def _assert_finite_bands(last_row: pl.DataFrame) -> None:
	for col in EXCURSION_BAND_COLS:
		value = last_row[col][0]
		assert value is not None
		assert math.isfinite(value)


def test_excursion_bands_last_candle_without_ny_session_has_no_leak():
	window_df, last_session = _prepare_window_without_last_ny()
	result = _run_excursion_pipeline(window_df)
	last_row = result.filter(pl.col("Session") == last_session)

	assert last_row.height == 1
	assert last_row["C_New York"][0] is None
	_assert_finite_bands(last_row)
