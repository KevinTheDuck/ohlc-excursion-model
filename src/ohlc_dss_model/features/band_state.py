import polars as pl


def get_band_state_on_ps2(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("C_Pre_Target_2") > pl.col("Band_FE_Pos_Upper"))
        .then(pl.lit(6))
        .when(pl.col("C_Pre_Target_2") >= pl.col("Band_FE_Pos_Lower"))
        .then(pl.lit(4))
        .when(pl.col("C_Pre_Target_2") >= pl.col("Band_AE_Pos_Upper"))
        .then(pl.lit(2))
        .when(pl.col("C_Pre_Target_2") > pl.col("Band_AE_Neg_Upper"))
        .then(pl.lit(1))
        .when(pl.col("C_Pre_Target_2") >= pl.col("Band_AE_Neg_Lower"))
        .then(pl.lit(1))
        .when(pl.col("C_Pre_Target_2") >= pl.col("Band_FE_Neg_Upper"))
        .then(pl.lit(3))
        .when(pl.col("C_Pre_Target_2") >= pl.col("Band_FE_Neg_Lower"))
        .then(pl.lit(5))
        .otherwise(pl.lit(7))
        .alias("band_state_ps2")
    )
