# %%
import polars as pl

# %%
data_path = "../data/raw/NAS100_30m.parquet"

# %%
df = pl.read_parquet(data_path)
print(df.head(5))
# %%
# Before we even start doing anything with data we need to handle timezones and dst changes
sort_time_by_volume = (
    df.with_columns(pl.col("DateTime").dt.time().alias("TimeOfDay"))
    .group_by("TimeOfDay")
    .agg(pl.col("TickVolume").mean().alias("MeanTickVolume"))
    .sort("MeanTickVolume", descending=True)
)

print(sort_time_by_volume.head(3))

# we sorted the time of day based on their mean tick volume
# in order to quickly compute which time had the most volume overall, in this case 16:30
# we assumed this is 09:30 since market open on nasdaq should have the largest volume,
# so we can conclude that this dataset is on EET time so now lets move onto handling the dst changes

# %%
df = df.with_columns(
    pl.col("DateTime")
    .dt.replace_time_zone("EET")
    .dt.convert_time_zone("America/New_York")
    .alias("DateTime_NY")
)

print(df.select(["DateTime", "DateTime_NY"]).head(3))

# %%
dst_test = df.filter(
    (pl.col("DateTime_NY").dt.date() == pl.date(2023, 3, 10))
    | (pl.col("DateTime_NY").dt.date() == pl.date(2023, 3, 15))
).filter(pl.col("DateTime_NY").dt.time() == pl.time(9, 30))

print(dst_test.select(["DateTime", "DateTime_NY"]))
# We see here that we successfully converted EET time to EDT and EST which means we also handled DST time
