import polars as pl


def _drop_ohlc_col(df: pl.DataFrame, idt: str) -> pl.DataFrame:
    return df.drop([("O_" + idt), ("H_" + idt), ("L_" + idt), ("C_" + idt)])


def extract_target_1(df: pl.DataFrame) -> pl.DataFrame:
    target_1 = _drop_ohlc_col(df, "Target_2")
    target_1 = target_1.with_columns(
        [
            pl.col("O_Target_1").alias("O_Target"),
            pl.col("H_Target_1").alias("H_Target"),
            pl.col("L_Target_1").alias("L_Target"),
            pl.col("C_Target_1").alias("C_Target"),
        ]
    )
    target_1 = _drop_ohlc_col(target_1, "Target_1")
    target_1 = target_1.with_columns([(pl.lit(1)).alias("target_stage")])
    return target_1


def extract_target_2(df: pl.DataFrame) -> pl.DataFrame:
    target_2 = df.with_columns(
        [
            pl.when(pl.col("H_Pre_Target_2") > pl.col("H_Pre_Target_1"))
            .then(pl.col("H_Pre_Target_2"))
            .otherwise(pl.col("H_Pre_Target_1"))
            .alias("H_Pre_Target_1"),
            pl.when(pl.col("L_Pre_Target_2") < pl.col("L_Pre_Target_1"))
            .then(pl.col("L_Pre_Target_2"))
            .otherwise(pl.col("L_Pre_Target_1"))
            .alias("L_Pre_Target_1"),
            pl.col("C_Pre_Target_2").alias("C_Pre_Target_1"),
            pl.col("O_Target_1").alias("O_Pre_Target_2"),
            pl.col("H_Target_1").alias("H_Pre_Target_2"),
            pl.col("L_Target_1").alias("L_Pre_Target_2"),
            pl.col("C_Target_1").alias("C_Pre_Target_2"),
            pl.col("O_Target_2").alias("O_Target"),
            pl.col("H_Target_2").alias("H_Target"),
            pl.col("L_Target_2").alias("L_Target"),
            pl.col("C_Target_2").alias("C_Target"),
        ]
    )
    target_2 = _drop_ohlc_col(target_2, "Target_1")
    target_2 = _drop_ohlc_col(target_2, "Target_2")
    target_2 = target_2.with_columns([(pl.lit(2)).alias("target_stage")])
    return target_2


def split_target_pipeline(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    target_1 = extract_target_1(df)
    target_2 = extract_target_2(df)
    return target_1, target_2
