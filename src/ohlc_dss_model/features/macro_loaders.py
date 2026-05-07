from pathlib import Path

import polars as pl

from ohlc_dss_model.data import load_parquet, write_parquet


def incremental_loader(
    folder_path: Path,
    file_name: str,
    df: pl.DataFrame,
    build_fn,
) -> pl.DataFrame:
    file_path = folder_path / f"{file_name}.parquet"

    min_current = df.select(pl.col("Session").min()).item()
    max_current = df.select(pl.col("Session").max()).item()

    if file_path.exists():
        print(f"[{file_name}] Loading cached table...")
        table = load_parquet(file_path)

        max_session = table.select(pl.col("Session").max()).item()
        print(f"[{file_name}] Cached max session: {max_session}")

        if max_session < max_current:
            print(f"[{file_name}] Extending to {max_current}...")
            extended = build_fn(max_session + 1, max_current)

            if not extended.is_empty():
                new_sessions = extended.select(pl.col("Session")).unique()
                table = table.join(new_sessions, on="Session", how="anti")
                table = pl.concat([table, extended])

            write_parquet(table, file_name, folder_path)

    else:
        print(f"[{file_name}] No cache found. Building fresh...")
        table = build_fn(min_current, max_current)
        write_parquet(table, file_name, folder_path)

    print(f"[{file_name}] Ready.")
    return table


def load_event_table(folder_path, df, api_key):
    return incremental_loader(
        folder_path,
        "event_table",
        df,
        build_fn=lambda start, end: _build_fn_event_table(start, end, api_key=api_key),
    )


def _build_fn_event_table(start, end, api_key):
    from ohlc_dss_model.features import build_event_table

    return build_event_table(start, end, api_key=api_key)


def load_fred_macro(folder_path, df, api_key):
    return incremental_loader(
        folder_path,
        "fred_macro_table",
        df,
        build_fn=lambda start, end: _build_fn_fred_macro(
            df,
            start,
            end,
            api_key=api_key,
        ),
    )


def _build_fn_fred_macro(df, start, end, api_key):
    from ohlc_dss_model.features import build_fred_macro

    return build_fred_macro(
        df,
        api_key=api_key,
        start_date=start,
        end_date=end,
    )


def load_individual_event_flags(folder_path, df, api_key):
    return incremental_loader(
        folder_path,
        "individual_event_flags",
        df,
        build_fn=lambda start, end: _build_fn_individual_event_flags(
            df,
            start,
            end,
            api_key=api_key,
        ),
    )


def _build_fn_individual_event_flags(df, start, end, api_key):
    from ohlc_dss_model.features import build_individual_event_flags

    return build_individual_event_flags(
        df,
        api_key=api_key,
        start=start,
        end=end,
    )


def get_macro_features(
    folder_path: Path, df: pl.DataFrame, api_key: str
) -> pl.DataFrame:
    from ohlc_dss_model.features import encode_news_context, get_calendar_index

    event_table = load_event_table(folder_path, df, api_key)

    if event_table.is_empty():
        print("[get_macro_features] No events found.")
        return pl.DataFrame()

    print("[get_macro_features] Encoding news context...")
    macro_features = encode_news_context(df, event_table)

    print("[get_macro_features] Adding calendar features...")
    macro_features = get_calendar_index(macro_features)

    print("[get_macro_features] Joining FRED macro...")
    fred_macro = load_fred_macro(folder_path, df, api_key)
    macro_features = macro_features.join(fred_macro, on="Session", how="left")

    print("[get_macro_features] Joining individual event flags...")
    flags = load_individual_event_flags(folder_path, df, api_key)
    macro_features = macro_features.join(flags, on="Session", how="left")

    return macro_features
