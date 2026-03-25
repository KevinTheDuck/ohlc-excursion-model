# %%
import polars as pl

from ohlc_dss_model.data.integrity import sort_data
from ohlc_dss_model.utils.dt_utils import convert_to_timezone, session_tagging

# %%
data_path = "../data/raw/NAS100_30m.parquet"

# %%
df = pl.read_parquet(data_path)
df = convert_to_timezone(df)
print(df.head(10))

# %%
# The data is inverted so we must resort them so it goes from earliest -> latest
# This will make our job easier later on
df = sort_data(df)
print(df.head(5))

# %%
# We will then check for any duplicated rows
print(df.filter(df.is_duplicated()))
# %%
# Next we will handle session tagging which we will use to seperate different trading days
# Or simply we just define which row belongs to a daily candle
df = session_tagging(df)
print(df.head(5))
