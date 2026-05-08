import polars as pl


def compute_ofi_vpin(
    bars_1m: pl.DataFrame,
    window_sessions: int = 20,
    keep_cumulative: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    bars_1m = bars_1m.sort(["Session", "Intraday_Session", "DateTime"])

    bars_1m = (
        bars_1m.with_columns((pl.col("Close") - pl.col("Open")).alias("_bar_move"))
        .with_columns(
            pl.col("_bar_move")
            .rolling_std(window_size=window_sessions * 30)
            .over("Intraday_Session")
            .alias("sigma_bar")
        )
        .with_columns(
            (
                1.0
                / (
                    1.0
                    + (
                        -pl.col("_bar_move") / (pl.col("sigma_bar") + 1e-9) * 1.7025
                    ).exp()
                )
            ).alias("_phi")
        )
        .with_columns(
            [
                (pl.col("Volume") * pl.col("_phi")).alias("buy_volume"),
                (pl.col("Volume") * (1.0 - pl.col("_phi"))).alias("sell_volume"),
            ]
        )
        .with_columns((pl.col("buy_volume") - pl.col("sell_volume")).alias("ofi_bar"))
    )

    bars_1m = bars_1m.with_columns(
        [
            pl.col("ofi_bar")
            .cum_sum()
            .over(["Session", "Intraday_Session"])
            .alias("_cum_ofi"),
            pl.col("ofi_bar")
            .abs()
            .cum_sum()
            .over(["Session", "Intraday_Session"])
            .alias("_cum_abs_ofi"),
        ]
    )

    session_ofi = (
        bars_1m.group_by(["Session", "Intraday_Session"])
        .agg(
            [
                pl.col("ofi_bar").sum().alias("ofi_cumulative"),
                pl.col("ofi_bar").abs().sum().alias("_ofi_abs_sum"),
                pl.col("Volume").sum().alias("total_volume"),
            ]
        )
        .with_columns(
            (pl.col("_ofi_abs_sum") / (pl.col("total_volume") + 1e-9)).alias(
                "vpin_session"
            )
        )
    )

    ofi_wide = session_ofi.pivot(
        on="Intraday_Session",
        index="Session",
        values=["ofi_cumulative", "vpin_session", "total_volume"],
    ).rename(
        {
            "ofi_cumulative_Pre_Target_1": "ofi_cumulative_Pre_Target_1",
            "ofi_cumulative_Pre_Target_2": "ofi_cumulative_Pre_Target_2",
            "ofi_cumulative_Target_1": "ofi_cumulative_Target_1",
            "ofi_cumulative_Target_2": "ofi_cumulative_Target_2",
            "vpin_session_Pre_Target_1": "vpin_Pre_Target_1",
            "vpin_session_Pre_Target_2": "vpin_Pre_Target_2",
            "vpin_session_Target_1": "vpin_Target_1",
            "vpin_session_Target_2": "vpin_Target_2",
            "total_volume_Pre_Target_1": "total_volume_Pre_Target_1",
            "total_volume_Pre_Target_2": "total_volume_Pre_Target_2",
            "total_volume_Target_1": "total_volume_Target_1",
            "total_volume_Target_2": "total_volume_Target_2",
        }
    )

    pre_combined = (
        session_ofi.filter(
            pl.col("Intraday_Session").is_in(["Pre_Target_1", "Pre_Target_2"])
        )
        .group_by("Session")
        .agg(
            [
                pl.col("ofi_cumulative").sum().alias("ofi_cumulative_Pre_Combined"),
                pl.col("_ofi_abs_sum").sum().alias("_ofi_abs_sum_combined"),
                pl.col("total_volume").sum().alias("total_volume_Pre_Combined"),
            ]
        )
        .with_columns(
            (
                pl.col("_ofi_abs_sum_combined")
                / (pl.col("total_volume_Pre_Combined") + 1e-9)
            ).alias("vpin_Pre_Combined")
        )
        .select(
            [
                "Session",
                "ofi_cumulative_Pre_Combined",
                "vpin_Pre_Combined",
                "total_volume_Pre_Combined",
            ]
        )
    )

    ofi_wide = ofi_wide.join(pre_combined, on="Session", how="left")

    if "_cum_vol" not in bars_1m.columns:
        bars_1m = bars_1m.with_columns(
            pl.col("Volume")
            .cum_sum()
            .over(["Session", "Intraday_Session"])
            .alias("_cum_vol")
        )

    if keep_cumulative:
        return (
            bars_1m.drop(
                ["_bar_move", "sigma_bar", "_phi", "buy_volume", "sell_volume"]
            ),
            ofi_wide,
        )
    return (
        bars_1m.drop(
            [
                "_bar_move",
                "sigma_bar",
                "_phi",
                "buy_volume",
                "sell_volume",
                "_cum_ofi",
                "_cum_abs_ofi",
                "_cum_vol",
                "ofi_bar",
            ]
        ),
        ofi_wide,
    )
