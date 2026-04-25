import polars as pl

def _join_aggregated_data(pivots: pl.DataFrame, aggregated_data: pl.DataFrame) -> pl.DataFrame:
    shifted_aggregated_data = aggregated_data.with_columns([
        pl.col("Sigma_Historical").shift(1).alias("Sigma_Historical_Shifted")
    ])
    return pivots.join(shifted_aggregated_data, on="Session", how="left")

def _calculate_Pi_k(pivots: pl.DataFrame) -> pl.DataFrame:
    return pivots.with_columns([
        ((pl.col("P_k") - pl.col("O_Ref")) / (pl.col("Sigma_Historical_Shifted") * pl.col("O_Ref"))).alias("Pi_k")
    ])

def _calculate_sigma_price(pivots: pl.DataFrame) -> pl.DataFrame:
    return pivots.with_columns([
        (pl.col("Sigma_Historical_Shifted") * pl.col("O_Ref")).alias("Sigma_Price")
    ])

def _calculate_bands_delta(pivots: pl.DataFrame) -> pl.DataFrame:
    return pivots.with_columns([
        ((pl.col("P_k") - pl.col("Band_FE_Pos_Center")) / pl.col("Sigma_Price")).alias("Delta_FE_Pos"),
        ((pl.col("P_k") - pl.col("Band_AE_Pos_Center")) / pl.col("Sigma_Price")).alias("Delta_AE_Pos"),
        ((pl.col("P_k") - pl.col("Band_FE_Neg_Center")) / pl.col("Sigma_Price")).alias("Delta_FE_Neg"),
        ((pl.col("P_k") - pl.col("Band_AE_Neg_Center")) / pl.col("Sigma_Price")).alias("Delta_AE_Neg"),
    ])

def _calculate_band_states(pivots: pl.DataFrame) -> pl.DataFrame:
    return pivots.with_columns([
        ((pl.col("P_k") >= pl.col("Band_AE_Neg_Lower")) & (pl.col("P_k") <= pl.col("Band_AE_Neg_Upper"))).cast(pl.Int8).alias("State_AE_Neg"),
        ((pl.col("P_k") >= pl.col("Band_AE_Pos_Lower")) & (pl.col("P_k") <= pl.col("Band_AE_Pos_Upper"))).cast(pl.Int8).alias("State_AE_Pos"),
        ((pl.col("P_k") >= pl.col("Band_FE_Neg_Lower")) & (pl.col("P_k") <= pl.col("Band_FE_Neg_Upper"))).cast(pl.Int8).alias("State_FE_Neg"),
        ((pl.col("P_k") >= pl.col("Band_FE_Pos_Lower")) & (pl.col("P_k") <= pl.col("Band_FE_Pos_Upper"))).cast(pl.Int8).alias("State_FE_Pos"),
    ])

def _calculate_temporal_dynamics(pivots: pl.DataFrame) -> pl.DataFrame:
    pivots = pivots.with_columns([
        pl.col("Pi_k").shift(1).over("Session").alias("Pi_k_prev"),
        pl.col("DateTime").shift(1).over("Session").alias("DateTime_prev"),
    ])

    pivots = pivots.with_columns([
        (pl.col("Pi_k") - pl.col("Pi_k_prev")).fill_null(0.0).alias("delta_Pi_k"),
        (
            ((pl.col("DateTime") - pl.col("DateTime_prev")).dt.total_minutes() / 30)
            .fill_null(0)
            .cast(pl.Int16)
        ).alias("delta_b_k"),
    ])

    pivots = pivots.with_columns([
        pl.when(pl.col("delta_b_k") > 0)
        .then(pl.col("delta_Pi_k") / pl.col("delta_b_k"))
        .otherwise(0.0)
        .alias("Speed_k"),

        pl.when(pl.col("delta_Pi_k") > 0).then(pl.lit(1))
        .when(pl.col("delta_Pi_k") < 0).then(pl.lit(-1))
        .otherwise(pl.lit(0))
        .alias("Dir_k"),
    ])

    pivots = pivots.with_columns([
        pl.col("Dir_k").shift(1).over("Session").fill_null(0).alias("Dir_prev"),
    ]).with_columns([
        ((pl.col("Dir_k") != pl.col("Dir_prev")) & (pl.col("Dir_k") != 0) & (pl.col("Dir_prev") != 0))
        .cast(pl.Int8)
        .alias("Turn_k")
    ]).drop(["Pi_k_prev", "DateTime_prev", "Dir_prev"])

    return pivots

def detect_pivots(
    df_bars: pl.DataFrame,
    sessions: list = ["Pre_Target_1", "Pre_Target_2"],
    n: int = 2,
) -> pl.DataFrame:

    pre_ny = df_bars.filter(
        pl.col("Intraday_Session").is_in(sessions)
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

def pivot_extraction(df: pl.DataFrame) -> pl.DataFrame:
    def tag(frame: pl.DataFrame, s_k: int, price_col: str, intra_order: int) -> pl.DataFrame:
        return frame.with_columns([
            pl.lit(s_k).alias("s_k"),
            pl.col(price_col).alias("P_k"),
            pl.lit(intra_order).alias("intra_order"),
        ])

    ph = df.filter(pl.col("is_pivot_high") & ~pl.col("is_pivot_low"))
    pl_ = df.filter(pl.col("is_pivot_low") & ~pl.col("is_pivot_high"))
    both = df.filter(pl.col("is_pivot_high") & pl.col("is_pivot_low"))
    bearish = both.filter(pl.col("Close") < pl.col("Open"))
    bullish = both.filter(pl.col("Close") >= pl.col("Open"))

    pivots = pl.concat([
        tag(ph, 1, "High", 0),
        tag(pl_, -1, "Low", 0),
        tag(bearish, 1, "High", 0),
        tag(bearish, -1, "Low", 1),
        tag(bullish, -1, "Low", 0),
        tag(bullish, 1, "High", 1),
    ])

    pivots = pivots.select([
        "DateTime",
        "Session",
        "Intraday_Session",
        "P_k",
        "s_k",
        "intra_order",
    ])

    pivots = pivots.sort(["Session", "DateTime", "intra_order"])
    return pivots.select(["DateTime", "Intraday_Session", "Session", "P_k", "s_k"])

def build_pivot_features(pivots: pl.DataFrame, aggregated_data: pl.DataFrame) -> pl.DataFrame:
    pivots = _join_aggregated_data(pivots, aggregated_data)
    pivots = _calculate_Pi_k(pivots)
    pivots = _calculate_sigma_price(pivots)
    pivots = _calculate_bands_delta(pivots)
    pivots = _calculate_band_states(pivots)
    pivots = _calculate_temporal_dynamics(pivots)
    return pivots.drop_nulls()