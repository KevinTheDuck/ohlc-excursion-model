import polars as pl
from ohlc_dss_model.config import config

def load_parquet(file_path: str = config.data.file_path) -> pl.DataFrame:
    try:
        df = pl.read_parquet(file_path)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")

def load_csv(file_path: str = config.data.file_path, separator: str = config.data.csv_separator) -> pl.DataFrame:
    try:
        df = pl.read_csv(file_path, separator=separator)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")