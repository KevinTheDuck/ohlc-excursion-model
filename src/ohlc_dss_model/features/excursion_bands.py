import polars as pl

from ohlc_dss_model.config import config

def _calculate_z_body(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        ((pl.col("C_New York") / pl.col("O_Ref")).log().abs() / pl.col("Sigma_Historical")).alias("_z_body")
    )

def _calculate_z_sigma(df: pl.DataFrame, n: int = config.excursion_bands.n) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("Sigma_Historical") / pl.col("Sigma_Historical").rolling_mean(n)).alias("_z_sigma")
    )

def _calculate_threshold(
    df: pl.DataFrame,
    tau_0: float = config.excursion_bands.tau_0,
    tau_min: float = config.excursion_bands.tau_min,
    tau_max: float = config.excursion_bands.tau_max,
) -> pl.DataFrame:
    return df.with_columns(
        (tau_0 * (pl.col("_z_sigma") ** -0.5)).clip(tau_min, tau_max).alias("_tau")
    )

def assign_direction(df: pl.DataFrame) -> pl.DataFrame:
    df = _calculate_z_body(df)
    df = _calculate_z_sigma(df)
    df = _calculate_threshold(df)
    
    return df.with_columns(
        pl.when((pl.col("_z_body") > pl.col("_tau")) & (pl.col("C_New York") > pl.col("O_Ref")))
        .then(pl.lit("bullish"))
        .when((pl.col("_z_body") > pl.col("_tau")) & (pl.col("C_New York") < pl.col("O_Ref")))
        .then(pl.lit("bearish"))
        .otherwise(pl.lit("neutral")).alias("Direction")
    )