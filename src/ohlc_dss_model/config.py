from dataclasses import dataclass, field
from pathlib import Path
from datetime import date

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Data_cfg:
    # if you use local dataset such as csv or parquet
    raw_folder_path = _PROJECT_ROOT / "data" / "raw"
    processed_folder_path = _PROJECT_ROOT / "data" / "processed"
    file_path = processed_folder_path / "nq_30m.parquet"
    event_path = processed_folder_path / "event_table.parquet"
    csv_separator: str = "\t"

    plot_fig_size: tuple = (12, 8)
    pivot_marker_offset: float = 0.5


# Expected column names from broker
@dataclass(frozen=True)
class Schema_cfg:
    datetime: str = "DateTime"


# Intraday Session Settings
@dataclass(frozen=True)
class Session_cfg:
    pre_target_split_1: tuple = ((18, 0), (3, 0))
    pre_target_split_2: tuple = ((3, 0), (8, 30))
    # including am + pm sessions
    target_split_1: tuple = ((8, 30), (12, 00))
    target_split_2: tuple = ((13, 00), (17, 00))


# Timezone properties to be used in time related operations
@dataclass(frozen=True)
class Timezone_cfg:
    eod_close_hour: int = 17
    broker: str = "UTC"
    # desired data timezone, recommended to leave this be
    target: str = "America/New_York"


@dataclass(frozen=True)
class Volatility_cfg:
    # Rolling window for estimators
    n: int = 7


@dataclass(frozen=True)
class ExcursionBands_cfg:
    # Rolling window for excursion bands
    n: int = 7

    tau_0: float = 0.4
    tau_min: float = 0.26
    tau_max: float = 1.75

    # Scaling factor for bands width
    k: float = 0.1


@dataclass(frozen=True)
class Pivot_transformer_cfg:
    max_pivots: int = 27
    burn_in_buffer = date(2011, 4, 5)

    pivot_numerical_whitelist: list = field(
        default_factory=lambda: [
            "Pi_k",
            "Delta_FE_Pos",
            "Delta_AE_Pos",
            "Delta_FE_Neg",
            "Delta_AE_Neg",
            "State_AE_Neg",
            "State_AE_Pos",
            "State_FE_Neg",
            "State_FE_Pos",
            "delta_Pi_k",
            "delta_b_k",
            "Speed_k",
            "Dir_k",
            "Turn_k",
            "VWAP_Dist",
            "VWAP_Gradient",
            "OFI_Cum_Norm",
            "OFI_Delta",
            "VPIN",
            "VPIN_Delta",
        ]
    )

    pivot_categorical_whitelist: list = field(
        default_factory=lambda: ["s_k", "Intraday_Session"]
    )

    context_whitelist: list = field(
        default_factory=lambda: [
            "Sigma_Today",
            "Sigma_Historical_Shifted",
            "sigma_today_pct_rank_20d",
            "sigma_today_pct_rank_60d",
            "e_yesterday",
            "e_today",
            "e_tomorrow",
            "vix_t1",
            "us10y_t1",
            "us2y_t1",
            "effr_t1",
            "10y_2y_spread_t1",
            "vix_5d_delta",
            "us10y_5d_delta",
            "vix_pct_rank_1y_t1",
            "is_fomc_day",
            "is_fomc_week",
            "days_to_fomc",
            "is_nfp_day",
            "is_cpi_day",
            "is_core_cpi_day",
            "day_of_week",
            "month",
            "week_of_month",
            "ps2_range_norm",
            "ps1_range_norm",
            "ps2_range_ratio",
            "ps1_range_ratio",
            "ps1_ps2_range_norm_ratio",
            "ps1_ps2_bull_agreement",
            "band_state_ps2",
            "nq_5d_return",
            "nq_20d_return",
            "above_ma_20",
            "nq_dist_ma_20_norm",
            "regime",
            "vix_vs_realized_spread",
            "vwap_ps1_regime",
            "vwap_ps2_regime",
            "vwap_ps2_skew",
            "vwap_ps1_skew",
            "ofi_cumulative_ps1",
            "ofi_cumulative_ps2",
            "vpin_ps1",
            "vpin_ps2",
            "ofi_ps1_ps2_agree",
            "vpin_trend",
            "vpin_ps1_pct_rank_20d",
            "vpin_ps2_pct_rank_20d",
            "total_volume_ps1",
            "total_volume_ps2",
            "ps1_volume_zscore_20d",
            "ps2_volume_zscore_20d",
            "ofi_band_breach_confirm",
            "target_stage",
        ]
    )


# Aliases
@dataclass(frozen=True)
class Project:
    data: Data_cfg = Data_cfg()
    schema: Schema_cfg = Schema_cfg()
    timezone: Timezone_cfg = Timezone_cfg()
    session: Session_cfg = Session_cfg()
    volatility: Volatility_cfg = Volatility_cfg()
    excursion_bands: ExcursionBands_cfg = ExcursionBands_cfg()

    pivot_transformer: Pivot_transformer_cfg = Pivot_transformer_cfg()


config = Project()
