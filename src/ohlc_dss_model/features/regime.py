import polars as pl


def get_regime_labels(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.when(
                (pl.col("vix_pct_rank_1y_t1") < 0.5) & (pl.col("us10y_5d_delta") <= 0)
            )
            .then(pl.lit(1))
            .when((pl.col("vix_pct_rank_1y_t1") < 0.5) & (pl.col("us10y_5d_delta") > 0))
            .then(pl.lit(2))
            .when(
                (pl.col("vix_pct_rank_1y_t1") >= 0.5) & (pl.col("us10y_5d_delta") <= 0)
            )
            .then(pl.lit(3))
            .otherwise(pl.lit(4))
            .alias("regime")
        ]
    )


def calculate_vix_vs_realized_spread(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            (pl.col("Sigma_Historical").shift(1).rolling_mean(20)).alias(
                "_sigma_historical_20d"
            )
        )
        .with_columns(
            [
                (
                    pl.col("vix_t1")
                    - pl.col("_sigma_historical_20d") * (252**0.5) * 100.0
                ).alias("vix_vs_realized_spread")
            ]
        )
        .drop(["_sigma_historical_20d"])
    )
