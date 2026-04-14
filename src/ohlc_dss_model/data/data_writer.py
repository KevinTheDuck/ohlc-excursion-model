import polars as pl
from ohlc_dss_model.config import config

def write_parquet(df: pl.DataFrame, name: str, file_path: str = config.data.processed_folder_path) -> None:
    try:
        path = file_path / f"{name}.parquet"
        df.write_parquet(path)
    except Exception as e:
        raise RuntimeError(f"Failed to write data: {e}")
    
def write_csv(df: pl.DataFrame, name: str, file_path: str = config.data.processed_folder_path, separator: str = config.data.csv_separator) -> None:
    try:
        path = file_path / f"{name}.csv"
        df.write_csv(path, separator=separator)
    except Exception as e:
        raise RuntimeError(f"Failed to write data: {e}")