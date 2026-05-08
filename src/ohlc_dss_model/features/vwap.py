import polars as pl


def compute_session_vwap(
    bars_1m: pl.DataFrame,
    keep_cumulative: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    bars_1m = (
        bars_1m.sort(["Session", "DateTime"])
        .with_columns(
            [
                ((pl.col("High") + pl.col("Low") + pl.col("Close")) / 3.0).alias(
                    "typical_price"
                )
            ]
        )
        .with_columns([(pl.col("typical_price") * pl.col("Volume")).alias("_tp_vol")])
        .with_columns(
            [
                pl.col("_tp_vol")
                .cum_sum()
                .over(["Session", "Intraday_Session"])
                .alias("_cum_tp_vol"),
                pl.col("Volume")
                .cum_sum()
                .over(["Session", "Intraday_Session"])
                .alias("_cum_vol"),
            ]
        )
        .with_columns(
            (pl.col("_cum_tp_vol") / (pl.col("_cum_vol") + 1e-9)).alias(
                "vwap_cumulative"
            )
        )
    )

    session_rollup = (
        bars_1m.group_by(["Session", "Intraday_Session"])
        .agg(
            [
                pl.col("_tp_vol").sum().alias("_tp_vol_sum"),
                pl.col("Volume").sum().alias("_vol_sum"),
            ]
        )
        .with_columns(
            (pl.col("_tp_vol_sum") / (pl.col("_vol_sum") + 1e-9)).alias("session_vwap")
        )
    )

    session_vwap_wide = session_rollup.pivot(
        on="Intraday_Session", index="Session", values="session_vwap"
    ).rename(
        {
            "Pre_Target_1": "vwap_Pre_Target_1",
            "Pre_Target_2": "vwap_Pre_Target_2",
            "Target_1": "vwap_Target_1",
            "Target_2": "vwap_Target_2",
        }
    )

    pre_combined = (
        bars_1m.filter(
            pl.col("Intraday_Session").is_in(["Pre_Target_1", "Pre_Target_2"])
        )
        .group_by("Session")
        .agg(
            [
                pl.col("_tp_vol").sum().alias("_tp_vol_sum"),
                pl.col("Volume").sum().alias("_vol_sum"),
            ]
        )
        .with_columns(
            (pl.col("_tp_vol_sum") / (pl.col("_vol_sum") + 1e-9)).alias(
                "vwap_Pre_Combined"
            )
        )
        .select(["Session", "vwap_Pre_Combined"])
    )

    session_vwap_wide = session_vwap_wide.join(pre_combined, on="Session", how="left")

    if keep_cumulative:
        return (
            bars_1m.drop(["_tp_vol", "typical_price", "_cum_tp_vol"]),
            session_vwap_wide,
        )
    return (
        bars_1m.drop(["_tp_vol", "typical_price", "_cum_tp_vol", "_cum_vol"]),
        session_vwap_wide,
    )


def get_vwap_position(df: pl.DataFrame) -> pl.DataFrame:
    sigma = pl.col("Sigma_Historical").shift(1) * pl.col("O_Ref") + 1e-9

    vwap_regime = ((pl.col("vwap_ps2") - pl.col("O_Ref")) / sigma).alias(
        "vwap_ps2_regime"
    )
    vwap_regime_ps1 = ((pl.col("vwap_ps1") - pl.col("O_Ref")) / sigma).alias(
        "vwap_ps1_regime"
    )

    fe = pl.col("Band_FE_Pos_Center")
    ae = pl.col("Band_AE_Pos_Center")

    midpoint = (fe + ae) / 2.0

    skew_ps2 = ((pl.col("vwap_ps2") - midpoint) / sigma).alias("vwap_ps2_skew")
    skew_ps1 = ((pl.col("vwap_ps1") - midpoint) / sigma).alias("vwap_ps1_skew")

    return df.with_columns([vwap_regime, vwap_regime_ps1, skew_ps2, skew_ps1])
