import polars as pl

def aggregate_sessions(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by(["Session", "Intraday_Session"])
        .agg([
            pl.col("Open").first().alias("O"),
            pl.col("High").max().alias("H"),
            pl.col("Low").min().alias("L"),
            pl.col("Close").last().alias("C"),
        ])
        .pivot(
            index="Session",
            on="Intraday_Session",
            values=["O", "H", "L", "C"],
        )
        .sort("Session")
    )