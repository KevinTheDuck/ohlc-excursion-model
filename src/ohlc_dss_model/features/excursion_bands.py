import polars as pl

from ohlc_dss_model.config import config

# Private 
def _calculate_z_body(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        ((pl.col("C_New York") / pl.col("O_Ref")).log().abs() / pl.col("Sigma_Historical")).alias("Z_Body")
    )

def _calculate_z_sigma(df: pl.DataFrame, n: int = config.excursion_bands.n) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("Sigma_Historical") / pl.col("Sigma_Historical").rolling_mean(n).shift(1)).alias("Z_Sigma")
    )

def _calculate_threshold(
    df: pl.DataFrame,
    tau_0: float = config.excursion_bands.tau_0,
    tau_min: float = config.excursion_bands.tau_min,
    tau_max: float = config.excursion_bands.tau_max,
) -> pl.DataFrame:
    return df.with_columns(
        (tau_0 * (pl.col("Z_Sigma") ** -0.5)).clip(tau_min, tau_max).alias("Tau")
    )

def _get_day_boundaries(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.min_horizontal(pl.col("L_Asia"), pl.col("L_London"), pl.col("L_New York")).alias("L_Day"),
        pl.max_horizontal(pl.col("H_Asia"), pl.col("H_London"), pl.col("H_New York")).alias("H_Day")
    ])

def _calculate_epsilon(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        (
            pl.when(pl.col("Direction") == "bullish")
            .then(pl.col("O_Ref") - pl.col("L_Day"))
            .when(pl.col("Direction") == "bearish")
            .then(pl.col("H_Day") - pl.col("O_Ref"))
            .otherwise(pl.max_horizontal(pl.col("H_Day") - pl.col("O_Ref"), pl.col("O_Ref") - pl.col("L_Day")))
        ).alias("_epsilon_ae"),
        (
            pl.when(pl.col("Direction") == "bullish")
            .then(pl.col("H_Day") - pl.col("O_Ref"))
            .when(pl.col("Direction") == "bearish")
            .then(pl.col("O_Ref") - pl.col("L_Day"))
            .otherwise(pl.max_horizontal(pl.col("H_Day") - pl.col("O_Ref"), pl.col("O_Ref") - pl.col("L_Day")))
        ).alias("_epsilon_fe")
    ])

def _normalize_epsilon(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        (pl.col("_epsilon_ae") / pl.col("Sigma_Historical")).alias("_epsilon_ae_normalized"),
        (pl.col("_epsilon_fe") / pl.col("Sigma_Historical")).alias("_epsilon_fe_normalized")
    ])

def _calculate_mu(df: pl.DataFrame, n: int = config.excursion_bands.n) -> pl.DataFrame:
    return df.with_columns([
        (pl.col("_epsilon_ae_normalized").shift(1).rolling_mean(n)).alias("_mu_ae"),
        (pl.col("_epsilon_fe_normalized").shift(1).rolling_mean(n)).alias("_mu_fe")
    ])

def _calculate_mu_scaled(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        (pl.col("_mu_ae") * pl.col("Sigma_Historical").shift(1)).alias("_mu_ae_scaled"),
        (pl.col("_mu_fe") * pl.col("Sigma_Historical").shift(1)).alias("_mu_fe_scaled")
    ])

def _calculate_delta_t(df: pl.DataFrame, k: float = config.excursion_bands.k) -> pl.DataFrame:
    return df.with_columns([
        (k * pl.col("Sigma_Historical").shift(1) * pl.col("O_Ref")).alias("_delta_t")
    ])

# Public
def assign_direction(df: pl.DataFrame) -> pl.DataFrame:
    df = _calculate_z_body(df)
    df = _calculate_z_sigma(df)
    df = _calculate_threshold(df)
    
    return df.with_columns(
        pl.when((pl.col("Z_Body") > pl.col("Tau")) & (pl.col("C_New York") > pl.col("O_Ref")))
        .then(pl.lit("bullish"))
        .when((pl.col("Z_Body") > pl.col("Tau")) & (pl.col("C_New York") < pl.col("O_Ref")))
        .then(pl.lit("bearish"))
        .otherwise(pl.lit("neutral")).alias("Direction")
    )

def calculate_excursion_bands(df: pl.DataFrame, n: int = config.excursion_bands.n) -> pl.DataFrame:
    df = assign_direction(df)
    df = _get_day_boundaries(df)
    df = _calculate_epsilon(df)
    df = _normalize_epsilon(df)
    df = _calculate_mu(df, n)
    df = _calculate_mu_scaled(df)
    df = _calculate_delta_t(df)

    df = df.with_columns([
        (pl.col("O_Ref") + pl.col("_mu_ae_scaled")).alias("_band_AE_Pos"),
        (pl.col("O_Ref") - pl.col("_mu_ae_scaled")).alias("_band_AE_Neg"),
        (pl.col("O_Ref") + pl.col("_mu_fe_scaled")).alias("_band_FE_Pos"),
        (pl.col("O_Ref") - pl.col("_mu_fe_scaled")).alias("_band_FE_Neg")
    ])

    df = df.with_columns([
        (pl.col("_band_AE_Neg") + pl.col("_delta_t")).alias("Band_AE_Neg_Upper"),
        (pl.col("_band_AE_Neg") - pl.col("_delta_t")).alias("Band_AE_Neg_Lower"),
        (pl.col("_band_AE_Pos") + pl.col("_delta_t")).alias("Band_AE_Pos_Upper"),
        (pl.col("_band_AE_Pos") - pl.col("_delta_t")).alias("Band_AE_Pos_Lower"),
        (pl.col("_band_FE_Neg") + pl.col("_delta_t")).alias("Band_FE_Neg_Upper"),
        (pl.col("_band_FE_Neg") - pl.col("_delta_t")).alias("Band_FE_Neg_Lower"),
        (pl.col("_band_FE_Pos") + pl.col("_delta_t")).alias("Band_FE_Pos_Upper"),
        (pl.col("_band_FE_Pos") - pl.col("_delta_t")).alias("Band_FE_Pos_Lower")
    ])
    
    return df.drop([
            "L_Day", "H_Day",
            "_epsilon_ae", "_epsilon_fe", 
            "_epsilon_ae_normalized", "_epsilon_fe_normalized", 
            "_mu_ae", "_mu_fe",
            "_mu_ae_scaled", "_mu_fe_scaled",
            "_band_AE_Pos", "_band_AE_Neg", "_band_FE_Pos", "_band_FE_Neg",
            # "_delta_t"
        ])