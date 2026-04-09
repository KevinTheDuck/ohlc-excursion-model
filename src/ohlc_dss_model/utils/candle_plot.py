import mplfinance as mpf
from datetime import date
import polars as pl

from ohlc_dss_model.config import config

def plot_session(
        session: date,
        raw_data: pl.DataFrame,
        pivot_data: pl.DataFrame,
        aggregated_data: pl.DataFrame,
        figsize: tuple = config.data.plot_fig_size
) -> None:
        
    day_bars = (raw_data.filter(pl.col("Session") == session).to_pandas().set_index("DateTime").sort_index())

    intraday_data = (pivot_data.filter(pl.col("Session") == session).to_pandas().set_index("DateTime").sort_index())

    ph_prices = (
        intraday_data["High"]
        .where(intraday_data["is_pivot_high"])
        .reindex(day_bars.index)
    )

    pl_prices = (
        intraday_data["Low"]
        .where(intraday_data["is_pivot_low"])
        .reindex(day_bars.index)
    )

    offset = intraday_data["High"].std() * config.data.pivot_marker_offset
    ph_markers = ph_prices + offset
    pl_markers = pl_prices - offset

    bands = (
        aggregated_data
        .filter(pl.col("Session") == session)
        .select([
            "O_Ref",
            "Band_AE_Pos_Upper", "Band_AE_Pos_Lower",
            "Band_AE_Neg_Upper", "Band_AE_Neg_Lower",
            "Band_FE_Pos_Upper", "Band_FE_Pos_Lower",
            "Band_FE_Neg_Upper", "Band_FE_Neg_Lower",
        ])
        .to_pandas()
        .iloc[0]
    )

    ap = [
        mpf.make_addplot(
            ph_markers, type="scatter", marker="v",
            markersize=80, color="red", label="Pivot high", alpha=0.5
        ),
        mpf.make_addplot(
            pl_markers, type="scatter", marker="^",
            markersize=80, color="lime", label="Pivot low", alpha=0.5
        ),
    ]

    fig, axes = mpf.plot(
        day_bars,
        type="candle",
        addplot=ap,
        title=f"Session {session}",
        ylabel="Price",
        figsize=figsize,
        returnfig=True,
        hlines=dict(
            hlines=[bands["O_Ref"]],
            colors=["orange"],
            linestyle="--",
            linewidths=1,
        ),
    )

    ax = axes[0]

    ax.axhspan(bands["Band_AE_Pos_Lower"], bands["Band_AE_Pos_Upper"],
               alpha=0.1, color="green",   label="AE Pos")
    ax.axhspan(bands["Band_AE_Neg_Lower"], bands["Band_AE_Neg_Upper"],
               alpha=0.1, color="green",   label="AE Neg")

    ax.axhspan(bands["Band_FE_Pos_Lower"], bands["Band_FE_Pos_Upper"],
               alpha=0.1, color="red", label="FE Pos")
    ax.axhspan(bands["Band_FE_Neg_Lower"], bands["Band_FE_Neg_Upper"],
               alpha=0.1, color="red", label="FE Neg")

    ax.legend(loc="upper left")
    mpf.show()