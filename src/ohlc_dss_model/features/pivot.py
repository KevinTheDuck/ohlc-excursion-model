import polars as pl

def detect_pivots(
    df_bars: pl.DataFrame,
    n: int = 2,
) -> pl.DataFrame:

    pre_ny = df_bars.filter(
        pl.col("Intraday_Session").is_in(["Asia", "London"])
    )

    shift_cols = []
    for i in range(1, n + 1):
        shift_cols += [
            pl.col("High").shift(i).over("Session").alias(f"_H_m{i}"),
            pl.col("High").shift(-i).over("Session").alias(f"_H_p{i}"),
            pl.col("Low").shift(i).over("Session").alias(f"_L_m{i}"),
            pl.col("Low").shift(-i).over("Session").alias(f"_L_p{i}"),
        ]

    pre_ny = pre_ny.with_columns(shift_cols)

    ph_condition = pl.lit(True)
    pl_condition = pl.lit(True)

    for i in range(1, n + 1):
        ph_condition = (
            ph_condition
            & pl.col("High").gt(pl.col(f"_H_m{i}"))
            & pl.col("High").gt(pl.col(f"_H_p{i}"))
            & pl.col(f"_H_m{i}").is_not_null()
            & pl.col(f"_H_p{i}").is_not_null()
        )
        pl_condition = (
            pl_condition
            & pl.col("Low").lt(pl.col(f"_L_m{i}"))
            & pl.col("Low").lt(pl.col(f"_L_p{i}"))
            & pl.col(f"_L_m{i}").is_not_null()
            & pl.col(f"_L_p{i}").is_not_null()
        )

    pre_ny = pre_ny.with_columns([
        ph_condition.alias("is_pivot_high"),
        pl_condition.alias("is_pivot_low"),
    ])

    temp_cols = [f"_{x}_{d}{i}" for x in ["H", "L"] for d in ["m", "p"] for i in range(1, n + 1)]
    return pre_ny.drop(temp_cols)