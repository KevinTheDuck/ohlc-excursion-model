import re
import requests
from datetime import date, timedelta
from typing import Any
from fredapi import Fred
import polars as pl
import holidays
import json
import os

import time

FRED_SERIES = {
    "CPIAUCSL":      ("US CPI m/m & y/y",                   3),
    "CPILFESL":      ("US Core CPI m/m",                    3),
    "PAYEMS":        ("US Non-Farm Employment Change",       3),
    "ICSA":          ("US Unemployment Claims",              2),
    "CES0500000003": ("US Average Hourly Earnings m/m",      2),
    "WPSFD4131":     ("US Core PPI m/m",                     1),
    "PPIACO":        ("US PPI m/m",                          1),
    "ADPWNUSNERSA":  ("US ADP Non-Farm Employment Change",   1),
    "MANEMP":        ("US ISM Manufacturing PMI",            1),
    "JTSJOL":        ("US JOLTS Job Openings",               1),
    "RSXFS":         ("US Core Retail Sales m/m",            1),
}


def _as_date(value: Any) -> date:
    return value.date() if hasattr(value, "date") else date.fromisoformat(str(value)[:10])

def _collect_fred_records(
    fred: Fred,
    start: date,
    end: date,
    include_metadata: bool,
    print_counts: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    max_retries = 15

    for series_id, (name, weight) in FRED_SERIES.items():
        success = False
        for attempt in range(max_retries):
            try:
                vintage_dates = fred.get_series_vintage_dates(series_id)
                count = 0
                for d in vintage_dates:
                    release_date = _as_date(d)
                    if not (start <= release_date <= end):
                        continue
                    row = {"Session": release_date, "e_weight": weight}
                    if include_metadata:
                        row.update({"series_id": series_id, "name": name})
                    records.append(row)
                    count += 1
                if print_counts:
                    print(f"  [ok] {series_id:20} ({name}): {count} releases")
                success = True
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    if print_counts:
                        print(f"  [warn] {series_id:20}: {e}")
        if not success:
            continue

    return records


def _append_generated_events(
    records: list[dict[str, Any]],
    start: date,
    end: date,
    include_metadata: bool,
    print_ism_count: bool,
) -> None:
    for d in fetch_fomc_dates(start, end):
        if include_metadata:
            records.append({
                "Session": d,
                "series_id": "FOMC",
                "name": "US Federal Funds Rate",
                "e_weight": 3,
            })
        else:
            records.append({"Session": d, "e_weight": 3})

    ism_dates = _generate_ism_services_dates(start, end)
    for d in ism_dates:
        if include_metadata:
            records.append({
                "Session": d,
                "series_id": "ISM_SVC",
                "name": "US ISM Services PMI",
                "e_weight": 1,
            })
        else:
            records.append({"Session": d, "e_weight": 1})

    if print_ism_count:
        print(f"  [ok] {'ISM_SERVICES_GEN':20} (US ISM Services PMI): {len(ism_dates)} releases")

def fetch_fomc_dates(start: date, end: date) -> list[date]:
    local_file = "ne-press.json"
    if os.path.exists(local_file):
        try:
            with open(local_file, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [warn] Could not read local file '{local_file}': {e}")
            return []
    else:
        url = "https://www.federalreserve.gov/json/ne-press.json"
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            data = json.loads(r.content.decode("utf-8-sig"))
        except Exception as e:
            print(f"  [warn] JSON feed failed: {e}")
            return []

    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }

    dates = set()

    for item in data:
        title = item.get("t", "")
        if not ("FOMC" in title or "Federal Open Market Committee" in title):
            continue

        match = re.search(
            r"("
            r"January|February|March|April|May|June|July|August|"
            r"September|October|November|December"
            r")\s+(\d{1,2})\s*[–\-]\s*(\d{1,2}),?\s+(\d{4})",
            title,
        )
        if not match:
            continue

        month_name = match.group(1)
        _ = int(match.group(2))
        day2 = int(match.group(3))
        year = int(match.group(4))
        month = month_map[month_name]

        meeting_final_day = date(year, month, day2)

        if start <= meeting_final_day <= end:
            dates.add(meeting_final_day)

    dates.add(date(2026, 4, 29))

    dates = sorted(dates)
    print(f"  [ok] {'FOMC_SCRAPED':20} (US Federal Funds Rate): {len(dates)} meetings")
    return dates

def _generate_ism_services_dates(start: date, end: date) -> list[date]:
    us_holidays = holidays.US(years=range(start.year, end.year + 1))
    dates = []
    year, month = start.year, start.month

    while date(year, month, 1) <= end:
        d = date(year, month, 1)
        count = 0
        while count < 3:
            if d.weekday() < 5 and d not in us_holidays:
                count += 1
                if count == 3:
                    break
            d += timedelta(days=1)

        if start <= d <= end:
            dates.append(d)

        month += 1
        if month > 12:
            month = 1
            year += 1

    return dates


def build_event_table(
    start: date,
    end: date,
    api_key: str,
) -> pl.DataFrame:
    fred = Fred(api_key=api_key)
    records = _collect_fred_records(
        fred=fred,
        start=start,
        end=end,
        include_metadata=False,
        print_counts=True,
    )
    _append_generated_events(
        records=records,
        start=start,
        end=end,
        include_metadata=False,
        print_ism_count=True,
    )

    if not records:
        raise ValueError("No data fetched, check api_key and network.")

    result = (
        pl.DataFrame(records)
        .with_columns([
            pl.col("Session").cast(pl.Date),
            pl.col("e_weight").cast(pl.Int8),
        ])
        .group_by("Session")
        .agg(pl.col("e_weight").max())
        .sort("Session")
    )

    assert result["Session"].n_unique() == result.height, \
        "duplicate Sessions in event_table after group_by, investigate"
    assert result["e_weight"].max() <= 3, \
        "e_weight exceeded 3, check weight definitions"

    print(f"\n[info] {result.height} event days between {start} and {end}")
    print(f"[info] w=3 ultra-high: {result.filter(pl.col('e_weight')==3).height} days")
    print(f"[info] w=2 high:       {result.filter(pl.col('e_weight')==2).height} days")
    print(f"[info] w=1 medium:     {result.filter(pl.col('e_weight')==1).height} days")

    return result



def encode_news_context(
    sessions: pl.DataFrame,
    event_table: pl.DataFrame,
) -> pl.DataFrame:
    assert event_table["Session"].n_unique() == event_table.height, \
        "event_table has duplicate Sessions, run build_event_table first"

    df = sessions
    join_specs = [
        (0, "e_today"),
        (1, "e_yesterday"),
        (-1, "e_tomorrow"),
    ]
    for shift_days, column_name in join_specs:
        shifted = event_table.with_columns(
            (pl.col("Session") + pl.duration(days=shift_days)).alias("Session")
        ).rename({"e_weight": column_name})
        df = df.join(shifted, on="Session", how="left")

    result = df.with_columns([
        pl.col("e_today").fill_null(0).cast(pl.Int8),
        pl.col("e_yesterday").fill_null(0).cast(pl.Int8),
        pl.col("e_tomorrow").fill_null(0).cast(pl.Int8),
    ])

    assert result["e_today"].max() <= 3,     "e_today exceeded 3"
    assert result["e_yesterday"].max() <= 3, "e_yesterday exceeded 3"
    assert result["e_tomorrow"].max() <= 3,  "e_tomorrow exceeded 3"

    return result

def inspect_event_table(
    api_key: str,
    start: date,
    end: date,
) -> pl.DataFrame:
    fred = Fred(api_key=api_key)
    records = _collect_fred_records(
        fred=fred,
        start=start,
        end=end,
        include_metadata=True,
        print_counts=False,
    )
    _append_generated_events(
        records=records,
        start=start,
        end=end,
        include_metadata=True,
        print_ism_count=False,
    )

    return (
        pl.DataFrame(records)
        .with_columns(pl.col("Session").cast(pl.Date))
        .sort(["Session", "name"])
    )
