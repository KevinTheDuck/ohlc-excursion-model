import polars as pl

from ohlc_dss_model.features.estimator_spec import PRE_NY_SPEC
from ohlc_dss_model.features.volatility import yang_zhang


def calculate_range_structure(df: pl.DataFrame) -> pl.DataFrame:
    sigma_price = pl.col("Sigma_Historical").shift(1) * pl.col("O_Ref")

    range_structure = df.with_columns(
        [
            ((pl.col("H_Pre_Target_2") - pl.col("L_Pre_Target_2")) / sigma_price).alias(
                "ps2_range_norm"
            ),
            ((pl.col("H_Pre_Target_1") - pl.col("L_Pre_Target_1")) / sigma_price).alias(
                "ps1_range_norm"
            ),
            (
                (pl.col("C_Pre_Target_2") - pl.col("L_Pre_Target_2"))
                / (pl.col("H_Pre_Target_2") - pl.col("L_Pre_Target_2"))
            ).alias("ps2_range_ratio"),
            (
                (pl.col("C_Pre_Target_1") - pl.col("L_Pre_Target_1"))
                / (pl.col("H_Pre_Target_1") - pl.col("L_Pre_Target_1"))
            ).alias("ps1_range_ratio"),
            (pl.col("C_Pre_Target_1") > pl.col("O_Pre_Target_1")).alias("ps1_bull"),
            (pl.col("C_Pre_Target_2") > pl.col("O_Pre_Target_2")).alias("ps2_bull"),
        ]
    ).with_columns(
        [
            (pl.col("ps1_range_norm") / pl.col("ps2_range_norm")).alias(
                "ps1_ps2_range_norm_ratio"
            ),
            (pl.col("ps1_bull") == pl.col("ps2_bull")).alias("ps1_ps2_bull_agreement"),
        ]
    )

    range_structure = yang_zhang(range_structure, PRE_NY_SPEC, mode="today")
    return range_structure.drop(["_prior_close"])


def _pct_rank(x):
    if len(x) <= 1:
        return None
    return (x[:-1] < x[-1]).sum() / (len(x) - 1)


def calculate_range_pct_rank(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col("Sigma_Today")
            .rolling_map(_pct_rank, window_size=21)
            .alias("sigma_today_pct_rank_20d"),
            pl.col("Sigma_Today")
            .rolling_map(_pct_rank, window_size=61)
            .alias("sigma_today_pct_rank_60d"),
        ]
    )


def calculate_vpin_range_structure(df: pl.DataFrame) -> pl.DataFrame:
    bullish_breach = pl.col("band_state_ps2").is_in([2, 4, 6])
    bearish_breach = pl.col("band_state_ps2").is_in([3, 5, 7])

    return df.with_columns(
        [
            pl.col("vpin_ps1")
            .rolling_map(_pct_rank, window_size=21)
            .alias("vpin_ps1_pct_rank_20d"),
            pl.col("vpin_ps2")
            .rolling_map(_pct_rank, window_size=21)
            .alias("vpin_ps2_pct_rank_20d"),
        ]
    ).with_columns(
        [
            (pl.col("vpin_ps2") - pl.col("vpin_ps1")).alias("vpin_trend"),
            (pl.col("ofi_cumulative_ps1") * pl.col("ofi_cumulative_ps2") > 0).alias(
                "ofi_ps1_ps2_agree"
            ),
            (
                pl.col("total_volume_ps2")
                / (pl.col("total_volume_ps2").rolling_mean(20) + 1e-9)
                - 1.0
            ).alias("ps2_volume_zscore_20d"),
            (
                pl.col("total_volume_ps1")
                / (pl.col("total_volume_ps1").rolling_mean(20) + 1e-9)
                - 1.0
            ).alias("ps1_volume_zscore_20d"),
            (
                (bullish_breach & (pl.col("ofi_cumulative_ps2") > 0))
                | (bearish_breach & (pl.col("ofi_cumulative_ps2") < 0))
            ).alias("ofi_band_breach_confirm"),
        ]
    )
