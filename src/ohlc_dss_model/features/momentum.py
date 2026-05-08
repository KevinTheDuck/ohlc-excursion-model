import polars as pl


def calculate_momentum_features(df: pl.DataFrame) -> pl.DataFrame:
    sigma_price = pl.col("Sigma_Historical").shift(1) * pl.col("O_Ref")
    momentum = (
        df.with_columns([pl.col("C_Target_2").shift(1).alias("_prior_close")])
        .with_columns(
            [
                (pl.col("_prior_close").rolling_mean(20).alias("_ma_20")),
                (pl.col("_prior_close").rolling_mean(200).alias("_ma_200")),
                ((pl.col("_prior_close") / pl.col("_prior_close").shift(5)) - 1).alias(
                    "nq_5d_return"
                ),
                ((pl.col("_prior_close") / pl.col("_prior_close").shift(20)) - 1).alias(
                    "nq_20d_return"
                ),
            ]
        )
        .with_columns(
            [
                (pl.col("_prior_close") > pl.col("_ma_20")).alias("above_ma_20"),
                (pl.col("_prior_close") > pl.col("_ma_200")).alias("above_ma_200"),
                ((pl.col("_prior_close") - pl.col("_ma_20")) / sigma_price).alias(
                    "nq_dist_ma_20_norm"
                ),
            ]
        )
        .drop(["_ma_20", "_ma_200"])
    )
    return momentum
