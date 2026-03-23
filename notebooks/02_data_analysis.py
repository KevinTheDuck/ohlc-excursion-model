# %%
import polars as pl

from src.utils.dt_utils import convert_to_timezone

# %%
data_path = "../data/raw/NAS100_30m.parquet"

# %%
df = pl.read_parquet(data_path)
df = convert_to_timezone(df, col_suffix="NY")
print(df.head(5))
