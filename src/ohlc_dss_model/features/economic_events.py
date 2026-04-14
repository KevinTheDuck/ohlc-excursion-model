import re
import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta
from fredapi import Fred
import polars as pl
import holidays


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



def fetch_fomc_dates(start: date, end: date) -> list[date]:
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"  [warn] FOMC scrape failed: {e}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")

    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }

    dates = []

    for meeting in soup.find_all("div", class_="fomc-meeting"):
        date_div = meeting.find("div", class_="fomc-meeting__date")
        if not date_div:
            continue

        panel = meeting.find_parent("div", class_=re.compile("panel"))
        year_tag = panel.find("h4") if panel else None
        if not year_tag or not year_tag.get_text(strip=True).isdigit():
            continue
        year = int(year_tag.get_text(strip=True))

        date_text = re.sub(r"[*†‡§#]", "", date_div.get_text(strip=True)).strip()

        m = re.match(
            r"(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+(\d+)(?:-(\d+))?",
            date_text
        )
        if not m:
            continue

        month_str, day1, day2 = m.groups()
        month = month_map[month_str]
        day   = int(day2) if day2 else int(day1)

        try:
            d = date(year, month, day)
            if start <= d <= end:
                dates.append(d)
        except ValueError:
            continue

    dates = sorted(set(dates))
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
    records = []

    for series_id, (name, weight) in FRED_SERIES.items():
        try:
            vintage_dates = fred.get_series_vintage_dates(series_id)
            count = 0
            for d in vintage_dates:
                rd = d.date() if hasattr(d, "date") else date.fromisoformat(str(d)[:10])
                if start <= rd <= end:
                    records.append({"Session": rd, "e_weight": weight})
                    count += 1
            print(f"  [ok] {series_id:20} ({name}): {count} releases")
        except Exception as e:
            print(f"  [warn] {series_id:20}: {e}")

    for d in fetch_fomc_dates(start, end):
        records.append({"Session": d, "e_weight": 3})

    ism_dates = _generate_ism_services_dates(start, end)
    for d in ism_dates:
        records.append({"Session": d, "e_weight": 1})
    print(f"  [ok] {'ISM_SERVICES_GEN':20} (US ISM Services PMI): {len(ism_dates)} releases")

    if not records:
        raise ValueError("No data fetched — check api_key and network.")

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
        "duplicate Sessions in event_table after group_by — investigate"
    assert result["e_weight"].max() <= 3, \
        "e_weight exceeded 3 — check weight definitions"

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
        "event_table has duplicate Sessions — run build_event_table first"

    ev = event_table

    df = sessions.join(
        ev.rename({"e_weight": "e_today"}),
        on="Session", how="left",
    )
    df = df.join(
        ev.with_columns(
            (pl.col("Session") + pl.duration(days=1)).alias("Session")
        ).rename({"e_weight": "e_yesterday"}),
        on="Session", how="left",
    )
    df = df.join(
        ev.with_columns(
            (pl.col("Session") - pl.duration(days=1)).alias("Session")
        ).rename({"e_weight": "e_tomorrow"}),
        on="Session", how="left",
    )

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
    records = []

    for series_id, (name, weight) in FRED_SERIES.items():
        try:
            vintage_dates = fred.get_series_vintage_dates(series_id)
            for d in vintage_dates:
                rd = d.date() if hasattr(d, "date") else date.fromisoformat(str(d)[:10])
                if start <= rd <= end:
                    records.append({
                        "Session":   rd,
                        "series_id": series_id,
                        "name":      name,
                        "e_weight":  weight,
                    })
        except Exception as e:
            print(f"  [warn] {series_id}: {e}")

    for d in fetch_fomc_dates(start, end):
        records.append({"Session": d, "series_id": "FOMC", "name": "US Federal Funds Rate", "e_weight": 3})

    for d in _generate_ism_services_dates(start, end):
        records.append({"Session": d, "series_id": "ISM_SVC", "name": "US ISM Services PMI", "e_weight": 1})

    return (
        pl.DataFrame(records)
        .with_columns(pl.col("Session").cast(pl.Date))
        .sort(["Session", "name"])
    )
