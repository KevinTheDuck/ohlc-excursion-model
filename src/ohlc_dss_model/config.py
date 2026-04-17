from dataclasses import dataclass, field
from pathlib import Path
from datetime import date

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Data_cfg:
    # if you use local dataset such as csv or parquet
    raw_folder_path = _PROJECT_ROOT / "data" / "raw"
    processed_folder_path = _PROJECT_ROOT / "data" / "processed"
    file_path = raw_folder_path / "nq_30m.parquet"
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
    asian_session: tuple = ((18, 0), (3, 0))
    london_session: tuple = ((3, 0), (8, 30))
    # including am + pm sessions
    new_york_session: tuple = ((8, 30), (17, 0))


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
    n: int = 20

@dataclass(frozen=True)
class ExcursionBands_cfg:
    # Rolling window for excursion bands
    n: int = 20
    
    tau_0: float = 0.4
    tau_min: float = 0.26
    tau_max: float = 1.75

    # Scaling factor for bands width
    k: float = 0.09

@dataclass(frozen=True)
class Pivot_transformer_cfg:
    max_pivots: int = 27
    burn_in_buffer = date(2016, 4, 5)

    pivot_numerical_whitelist: list = field(default_factory=lambda: [
        "Pi_k",
        'Delta_FE_Pos', 'Delta_AE_Pos', 'Delta_FE_Neg', 'Delta_AE_Neg',
        'State_AE_Neg', 'State_AE_Pos', 'State_FE_Neg', 'State_FE_Pos',
        "delta_Pi_k", "delta_b_k", "Speed_k", "Dir_k", "Turn_k",
    ])

    pivot_categorical_whitelist: list = field(default_factory=lambda: [
        "s_k", "Intraday_Session"
    ])

    context_whitelist: list = field(default_factory=lambda: [
        "Sigma_Today", "Sigma_Historical_Shifted",
        "e_yesterday", "e_today", "e_tomorrow",
        "H_Asia_Normalized", "L_Asia_Normalized", "C_Asia_Normalized",
        "O_London_Normalized", "H_London_Normalized",
        "L_London_Normalized", "C_London_Normalized",
    ])

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
