"""
macro_data.py
─────────────
Fetches, aligns, and caches all external data sources used by round2_model.py.

Sources:
  • FRED API       — macro, labor, treasury
  • yfinance       — stock market / ETFs
  • NOAA CDO API   — temperature, precipitation, severe weather (Dallas station)
  • EPA AQS API    — air quality index (Dallas county)
  • FEMA NFIP API  — flood insurance policy counts (proxy for flood risk exposure)
  • USDA Drought   — drought monitor D2+ coverage for TX (weekly → monthly)

All data is resampled to monthly frequency and forward-filled where appropriate.
The final output is a single DataFrame indexed by month-end date.

Usage:
  from macro_data import build_macro_dataset
  df = build_macro_dataset(fred_api_key="YOUR_KEY", noaa_api_key="YOUR_KEY")
"""

import os
import time
import warnings
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

# How far back to pull history (XGBoost needs enough rows to learn cycles)
HISTORY_START = "2010-01-01"
CACHE_FILE    = "macro_cache.parquet"

# Dallas / Collin County identifiers
NOAA_STATION_ID  = "GHCND:USW00003927"   # Dallas Fort Worth Airport
EPA_STATE_CODE   = "48"                   # Texas
EPA_COUNTY_CODE  = "113"                  # Dallas County
FEMA_STATE       = "TX"


# ── FRED ──────────────────────────────────────────────────────────────────────

FRED_SERIES = {
    # Macro
    "case_shiller_dallas":    "DAXRNSA",        # DFW home price index (not seasonally adj)
    "mortgage_rate_30yr":     "MORTGAGE30US",   # weekly → monthly
    "fed_funds_rate":         "FEDFUNDS",
    "cpi_shelter":            "CUSR0000SAH1",   # shelter component of CPI
    "housing_starts_south":   "HOUSTS",         # South region starts (000s units)
    "existing_home_sales":    "EXHOSLUSM495S",  # national existing home sales
    "new_home_sales":         "HSN1F",          # national new home sales

    # Labor — Dallas MSA & Texas
    "unemployment_dallas":    "DALL748URN",     # Dallas-Plano-Irving MSA
    "unemployment_texas":     "TXUR",
    "labor_force_part_texas": "LBSSA48",        # TX labor force participation
    "wage_growth_texas":      "SMU48000000500000003SA",  # TX avg weekly earnings
    "construction_employment":"SMS48000002000000001",    # TX construction jobs
    "tech_employment_dallas": "SMU48191806054200001SA",  # Dallas info sector

    # Treasury / rates
    "treasury_10yr":          "DGS10",          # daily → monthly
    "treasury_2yr":           "DGS2",
    "yield_spread":           None,             # derived: 10yr - 2yr
}


def _fetch_fred_series(series_id: str, api_key: str,
                       start: str = HISTORY_START) -> pd.Series:
    """Fetch a single FRED series and return a monthly pd.Series."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id":         series_id,
        "api_key":           api_key,
        "file_type":         "json",
        "observation_start": start,
        "frequency":         "m",
        "aggregation_method":"avg",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    if not obs:
        return pd.Series(dtype=float, name=series_id)

    s = pd.Series(
        {o["date"]: float(o["value"]) if o["value"] != "." else np.nan
         for o in obs},
        name=series_id,
    )
    s.index = pd.to_datetime(s.index)
    return s.resample("ME").last()


def fetch_fred_data(api_key: str) -> pd.DataFrame:
    """Fetch all FRED series and return a single monthly DataFrame."""
    print("  Fetching FRED data …")
    frames = {}
    for name, sid in FRED_SERIES.items():
        if sid is None:
            continue
        try:
            frames[name] = _fetch_fred_series(sid, api_key)
            time.sleep(0.25)   # be polite to the API
        except Exception as e:
            print(f"    Warning: could not fetch {name} ({sid}): {e}")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)

    # Derived features
    if "treasury_10yr" in df and "treasury_2yr" in df:
        df["yield_spread"] = df["treasury_10yr"] - df["treasury_2yr"]

    # Month-over-month % change for Case-Shiller (this is our training TARGET base)
    if "case_shiller_dallas" in df:
        df["cs_dallas_mom_pct"]  = df["case_shiller_dallas"].pct_change(1)  * 100
        df["cs_dallas_3mo_pct"]  = df["case_shiller_dallas"].pct_change(3)  * 100
        df["cs_dallas_6mo_pct"]  = df["case_shiller_dallas"].pct_change(6)  * 100

    return df.sort_index()


# ── yfinance ──────────────────────────────────────────────────────────────────

TICKERS = {
    "sp500":            "^GSPC",
    "homebuilder_etf":  "ITB",
    "reit_index":       "VNQ",
    "treasury_etf":     "TLT",    # inverse proxy for long rates
    "vix":              "^VIX",
    "oil_wti":          "CL=F",
    "txn_stock":        "TXN",    # Texas Instruments — major Dallas employer
    "att_stock":        "T",      # AT&T — headquartered in Dallas
}


def fetch_market_data(start: str = HISTORY_START) -> pd.DataFrame:
    """Fetch monthly OHLCV data from Yahoo Finance via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        print("    Warning: yfinance not installed. Run: pip install yfinance")
        return pd.DataFrame()

    print("  Fetching market data (yfinance) …")
    frames = {}
    for name, ticker in TICKERS.items():
        try:
            raw = yf.download(ticker, start=start, interval="1mo",
                              progress=False, auto_adjust=True)
            if raw.empty:
                continue
            close = raw["Close"].squeeze()
            close.name = name
            # Monthly return
            ret = close.pct_change() * 100
            ret.name = f"{name}_return"
            frames[name]            = close
            frames[f"{name}_return"] = ret
        except Exception as e:
            print(f"    Warning: could not fetch {ticker}: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp("M")
    return df.sort_index()


# ── NOAA Climate Data ─────────────────────────────────────────────────────────

def fetch_noaa_data(noaa_api_key: str,
                    start: str = HISTORY_START) -> pd.DataFrame:
    """
    Fetch monthly climate normals for DFW from NOAA CDO API.
    Requires a free API token from: https://www.ncdc.noaa.gov/cdo-web/token
    """
    if not noaa_api_key:
        print("  Skipping NOAA (no API key provided).")
        return pd.DataFrame()

    print("  Fetching NOAA climate data …")
    base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    headers  = {"token": noaa_api_key}

    # We fetch GHCND datatypes: TAVG, PRCP, TMAX, TMIN, SNOW
    all_records = []
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.now()

    # NOAA CDO limits to 1-year windows per request
    current = start_dt
    while current < end_dt:
        window_end = min(current + relativedelta(years=1) - timedelta(days=1), end_dt)
        params = {
            "datasetid":  "GHCND",
            "stationid":  NOAA_STATION_ID,
            "datatypeid": "TAVG,PRCP,TMAX,TMIN",
            "startdate":  current.strftime("%Y-%m-%d"),
            "enddate":    window_end.strftime("%Y-%m-%d"),
            "limit":      1000,
            "units":      "standard",
        }
        try:
            r = requests.get(base_url, headers=headers, params=params, timeout=30)
            if r.status_code == 200:
                results = r.json().get("results", [])
                all_records.extend(results)
            time.sleep(0.3)
        except Exception as e:
            print(f"    Warning: NOAA request failed for {current.strftime('%Y-%m')}: {e}")
        current = window_end + timedelta(days=1)

    if not all_records:
        print("    Warning: No NOAA records returned.")
        return pd.DataFrame()

    raw = pd.DataFrame(all_records)
    raw["date"] = pd.to_datetime(raw["date"])
    raw["month"] = raw["date"].dt.to_period("M").dt.to_timestamp("M")

    pivot = raw.pivot_table(index="month", columns="datatype",
                            values="value", aggfunc="mean")
    pivot.columns = [f"noaa_{c.lower()}" for c in pivot.columns]
    pivot.index   = pd.to_datetime(pivot.index)

    # Derive extreme heat days proxy (monthly avg max temp)
    if "noaa_tmax" in pivot:
        pivot["extreme_heat_flag"] = (pivot["noaa_tmax"] > 100).astype(int)

    return pivot.sort_index()


# ── EPA Air Quality ───────────────────────────────────────────────────────────

def fetch_epa_aqi(epa_api_key: str,
                  start_year: int = 2010) -> pd.DataFrame:
    """
    Fetch annual AQI summary for Dallas County from EPA AQS API.
    Free key at: https://aqs.epa.gov/aqsweb/documents/data_api.html
    Annualized data → forward-filled to monthly.
    """
    if not epa_api_key:
        print("  Skipping EPA AQI (no API key provided).")
        return pd.DataFrame()

    print("  Fetching EPA AQI data …")
    url = "https://aqs.epa.gov/data/api/annualData/byCounty"
    records = []

    for year in range(start_year, datetime.now().year + 1):
        params = {
            "email":      "test@example.com",   # EPA AQS requires an email
            "key":        epa_api_key,
            "param":      "88101",               # PM2.5
            "bdate":      f"{year}0101",
            "edate":      f"{year}1231",
            "state":      EPA_STATE_CODE,
            "county":     EPA_COUNTY_CODE,
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json().get("Data", [])
                for row in data:
                    records.append({
                        "year":              year,
                        "aqi_mean":          row.get("arithmetic_mean"),
                        "aqi_99th_pct":      row.get("ninety_nine_percentile"),
                        "aqi_good_days_pct": row.get("observation_count"),
                    })
            time.sleep(0.5)
        except Exception as e:
            print(f"    Warning: EPA AQS {year} failed: {e}")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).groupby("year").mean().reset_index()
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-01-01")
    df = df.set_index("date").drop(columns="year")

    # Upsample annual → monthly via forward fill
    monthly_idx = pd.date_range(
        start=f"{start_year}-01-01",
        end=datetime.now().strftime("%Y-%m-%d"),
        freq="ME"
    )
    df = df.reindex(monthly_idx, method="ffill")
    return df.sort_index()


# ── USDA Drought Monitor ──────────────────────────────────────────────────────

def fetch_drought_data(start: str = HISTORY_START) -> pd.DataFrame:
    """
    Fetch Texas drought coverage (D2+ = severe drought) from the
    USDA Drought Monitor JSON API — no API key required.
    """
    print("  Fetching USDA Drought Monitor data …")
    url = (
        "https://usdmdataservices.unl.edu/api/StateStatistics/GetDroughtSeverityStatisticsByArea"
        f"?aoi=tx&startdate={start}&enddate={datetime.now().strftime('%Y-%m-%d')}"
        "&statisticsType=1"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"    Warning: Drought Monitor fetch failed: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"]           = pd.to_datetime(df["MapDate"])
    df["drought_d2_pct"] = pd.to_numeric(df.get("D2", df.get("d2", 0)),
                                         errors="coerce").fillna(0)
    df["drought_d3_pct"] = pd.to_numeric(df.get("D3", df.get("d3", 0)),
                                         errors="coerce").fillna(0)
    df["drought_severe"] = df["drought_d2_pct"] + df["drought_d3_pct"]

    df = df.set_index("date")[["drought_d2_pct", "drought_d3_pct", "drought_severe"]]
    df = df.resample("ME").mean()
    return df.sort_index()


# ── FEMA NFIP (Flood Risk Proxy) ──────────────────────────────────────────────

def fetch_fema_flood_data(start: str = HISTORY_START) -> pd.DataFrame:
    """
    Fetch FEMA NFIP policy counts for Texas from the OpenFEMA API.
    No API key required.
    Policy count growth = more properties entering flood zones = risk signal.
    """
    print("  Fetching FEMA NFIP data …")
    url = "https://www.fema.gov/api/open/v1/fimaNfipPolicies"
    params = {
        "$filter":  f"state eq '{FEMA_STATE}'",
        "$select":  "policyEffectiveDate,totalInsurancePremiumOfPolicy,countyCode",
        "$top":     10000,
        "$orderby": "policyEffectiveDate asc",
    }
    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json().get("FimaNfipPolicies", [])
    except Exception as e:
        print(f"    Warning: FEMA NFIP fetch failed: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["policyEffectiveDate"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp("M")

    monthly = df.groupby("month").agg(
        fema_policy_count=("policyEffectiveDate", "count"),
        fema_avg_premium=("totalInsurancePremiumOfPolicy", "mean"),
    )
    monthly.index = pd.to_datetime(monthly.index)

    # Normalize: month-over-month change in policy count
    monthly["fema_policy_mom"] = monthly["fema_policy_count"].pct_change() * 100

    start_dt = pd.to_datetime(start)
    return monthly[monthly.index >= start_dt].sort_index()


# ── Master Builder ────────────────────────────────────────────────────────────

def build_macro_dataset(
    fred_api_key:  str,
    noaa_api_key:  str  = "",
    epa_api_key:   str  = "",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch all data sources, align to monthly frequency, and return a single
    merged DataFrame. Results are cached to CACHE_FILE unless force_refresh=True.

    Parameters
    ----------
    fred_api_key  : required — get free key at fred.stlouisfed.org
    noaa_api_key  : optional — get free key at ncdc.noaa.gov/cdo-web/token
    epa_api_key   : optional — get free key at aqs.epa.gov
    force_refresh : if True, ignore cache and re-fetch everything

    Returns
    -------
    pd.DataFrame indexed by month-end date, one row per month.
    """
    if not force_refresh and os.path.exists(CACHE_FILE):
        age_days = (datetime.now() - datetime.fromtimestamp(
            os.path.getmtime(CACHE_FILE))).days
        if age_days < 7:
            print(f"  Loading macro data from cache ({age_days}d old) …")
            return pd.read_parquet(CACHE_FILE)
        else:
            print(f"  Cache is {age_days}d old — refreshing …")

    print("\nFetching macro / environmental / labor / market data …")

    frames = []

    # 1. FRED (macro + labor + rates)
    try:
        frames.append(fetch_fred_data(fred_api_key))
    except Exception as e:
        print(f"  ERROR fetching FRED data: {e}")

    # 2. yfinance (market)
    try:
        frames.append(fetch_market_data())
    except Exception as e:
        print(f"  ERROR fetching market data: {e}")

    # 3. NOAA (climate)
    if noaa_api_key:
        try:
            frames.append(fetch_noaa_data(noaa_api_key))
        except Exception as e:
            print(f"  ERROR fetching NOAA data: {e}")

    # 4. EPA AQI
    if epa_api_key:
        try:
            frames.append(fetch_epa_aqi(epa_api_key))
        except Exception as e:
            print(f"  ERROR fetching EPA data: {e}")

    # 5. USDA Drought (no key needed)
    try:
        frames.append(fetch_drought_data())
    except Exception as e:
        print(f"  ERROR fetching drought data: {e}")

    # 6. FEMA NFIP (no key needed)
    try:
        frames.append(fetch_fema_flood_data())
    except Exception as e:
        print(f"  ERROR fetching FEMA data: {e}")

    if not frames:
        raise RuntimeError("All data sources failed — check API keys and connectivity.")

    # Align all sources to the same monthly index via outer join
    df = frames[0]
    for f in frames[1:]:
        if not f.empty:
            df = df.join(f, how="outer")

    # Standardise index to month-end
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Forward-fill up to 3 months for slow-moving series (e.g. annual AQI)
    df = df.fillna(method="ffill", limit=3)

    # Drop rows before HISTORY_START
    df = df[df.index >= pd.to_datetime(HISTORY_START)]

    # Cache result
    df.to_parquet(CACHE_FILE)
    print(f"  Macro dataset: {len(df)} months × {len(df.columns)} features")
    print(f"  Cached → {CACHE_FILE}")


    # Inside macro_data.py, near the end of build_macro_dataset()
    
    # Existing code: align all dataframes...
    # Keep only data from 2010 onward
    df = df[df.index >= HISTORY_START]
    
    df = df.ffill() # <--- ONLY add this line here
    return df


# ── Convenience: get the latest single row for prediction ─────────────────────

def get_current_macro_snapshot(
    fred_api_key: str,
    noaa_api_key: str = "",
    epa_api_key:  str = "",
) -> pd.Series:
    """
    Return the most recent complete month of macro data as a Series.
    Used by round2_model.py at prediction time.
    """
    df = build_macro_dataset(fred_api_key, noaa_api_key, epa_api_key)
    # Use the last row that has at least the core FRED features populated
    core_cols = ["mortgage_rate_30yr", "fed_funds_rate", "case_shiller_dallas"]
    available = [c for c in core_cols if c in df.columns]
    if available:
        last_valid = df[available].dropna(how="any").index[-1]
        return df.loc[last_valid]
    return df.iloc[-1]


# ── CLI: run standalone to inspect / refresh the cache ────────────────────────

if __name__ == "__main__":
    import sys

    fred_key = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FRED_API_KEY", "")
    noaa_key = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("NOAA_API_KEY", "")
    epa_key  = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("EPA_API_KEY",  "")

    if not fred_key:
        print("Usage: python macro_data.py <FRED_KEY> [NOAA_KEY] [EPA_KEY]")
        sys.exit(1)

    df = build_macro_dataset(fred_key, noaa_key, epa_key, force_refresh=True)

    print("\n── Column list ──────────────────────────────────────────")
    for col in df.columns:
        n_valid = df[col].notna().sum()
        print(f"  {col:<40} {n_valid:>4} valid months")

    print(f"\n── Date range: {df.index[0].date()} → {df.index[-1].date()} ──")
    print("\n── Last 3 rows ──────────────────────────────────────────")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(df.tail(3).to_string())
