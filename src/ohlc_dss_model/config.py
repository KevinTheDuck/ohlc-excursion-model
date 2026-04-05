from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Data_cfg:
    # if you use local dataset such as csv or parquet
    raw_folder_path = _PROJECT_ROOT / "data" / "raw"
    file_path = raw_folder_path / "nq_30m.parquet"
    csv_separator: str = "\t"

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
    # Rolling window for z_sigma calculation
    n: int = 20
    
    tau_0: float = 0.4
    tau_min: float = 0.26
    tau_max: float = 1.75

# Aliases
@dataclass(frozen=True)
class Project:
    data: Data_cfg = Data_cfg()
    schema: Schema_cfg = Schema_cfg()
    timezone: Timezone_cfg = Timezone_cfg()
    session: Session_cfg = Session_cfg()
    volatility: Volatility_cfg = Volatility_cfg()
    excursion_bands: ExcursionBands_cfg = ExcursionBands_cfg()

config = Project()
