"""
Dallas Property Enricher — Step 2 of 2
----------------------------------------
Reads dallas_properties.csv and adds neighborhood and school data.

  Pass A — Community data (~500 API calls total, one per unique neighborhood)
            Adds: crime indices, natural disaster risk, demographics,
                  air quality, climate

  Pass B — School district data (~50 API calls total, one per unique district)
            Adds: district name, rating, enrollment

Total cost: ~550 calls. Runs in about 5 minutes.

Run after scraper.py:
    python enrich.py

Output: dallas_enriched.csv
"""

import csv
import json
import logging
import time
from pathlib import Path
import requests


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

API_KEY     = "e229bd3beb35df9aa1a1b20c118a3aca"
INPUT_FILE  = "NEW_Dallas_Properties.csv"
OUTPUT_FILE = "NEW_dallas_enriched.csv"
CACHE_FILE  = "NEW_enrich_cache.json"
DELAY       = 0.35


# ---------------------------------------------------------------------------
# ATTOM endpoints
# ---------------------------------------------------------------------------

COMMUNITY_URL   = "https://api.gateway.attomdata.com/v4/neighborhood/community"
SCHOOL_DIST_URL = "https://api.gateway.attomdata.com/propertyapi/v1.0.0/school/district"
HEADERS         = {"Accept": "application/json", "apikey": API_KEY}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("enrich.log")],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def load_cache():
    if Path(CACHE_FILE).exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {"community": {}, "school": {}}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def fetch_community(geo_id):
    try:
        resp = requests.get(
            COMMUNITY_URL,
            headers=HEADERS,
            params={"geoIdv4": geo_id},
            timeout=30,
        )
        if resp.status_code != 200:
            log.warning(f"  Community {resp.status_code} for {geo_id[:8]}...")
            return {}
        return resp.json().get("community", {})
    except requests.exceptions.RequestException as e:
        log.warning(f"  Community error {geo_id[:8]}...: {e}")
        return {}


def fetch_school_district(geo_id):
    try:
        resp = requests.get(
            SCHOOL_DIST_URL,
            headers=HEADERS,
            params={"geoIdV4": geo_id},
            timeout=30,
        )
        if resp.status_code != 200:
            log.warning(f"  School {resp.status_code} for {geo_id[:8]}...")
            return {}
        data = resp.json()
        # Try both possible response keys
        result = data.get("schoolDistrict") or data.get("school") or []
        if isinstance(result, list) and result:
            return result[0]
        if isinstance(result, dict):
            return result
        return {}
    except requests.exceptions.RequestException as e:
        log.warning(f"  School error {geo_id[:8]}...: {e}")
        return {}


# ---------------------------------------------------------------------------
# Extract fields from community response
# ---------------------------------------------------------------------------

def extract_community(community):
    if not community:
        return {}

    crime    = community.get("crime",            {}) or {}
    disaster = community.get("naturalDisasters",  {}) or {}
    demo     = community.get("demographics",      {}) or {}
    climate  = community.get("climate",           {}) or {}
    air      = community.get("airQuality",        {}) or {}
    geo      = community.get("geography",         {}) or {}

    return {
        # Geographic context
        "neighborhood_name":            geo.get("geographyName"),
        "neighborhood_type":            geo.get("geographyTypeName"),

        # Crime indices (100 = national average, higher = more crime)
        "crime_index":                  crime.get("crime_Index"),
        "crime_murder":                 crime.get("murder_Index"),
        "crime_rape":                   crime.get("forcible_Rape_Index"),
        "crime_robbery":                crime.get("forcible_Robbery_Index"),
        "crime_assault":                crime.get("aggravated_Assault_Index"),
        "crime_burglary":               crime.get("burglary_Index"),
        "crime_larceny":                crime.get("larceny_Index"),
        "crime_vehicle_theft":          crime.get("motor_Vehicle_Theft_Index"),

        # Natural disaster risk (higher = more risk)
        "disaster_weather":             disaster.get("weather_Index"),
        "disaster_earthquake":          disaster.get("earthquake_Index"),
        "disaster_hail":                disaster.get("hail_Index"),
        "disaster_hurricane":           disaster.get("hurricane_Index"),
        "disaster_tornado":             disaster.get("tornado_Index"),
        "disaster_wind":                disaster.get("wind_Index"),

        # Air quality
        "air_pollution_index":          air.get("air_Pollution_Index"),
        "air_ozone_index":              air.get("ozone_Index"),

        # Demographics
        "pop_density_sq_mi":            demo.get("population_Density_Sq_Mi"),
        "median_age":                   demo.get("median_Age"),
        "pct_in_poverty":               demo.get("population_In_Poverty_Pct"),
        "median_household_income":      demo.get("income_Household_Median"),
        "median_home_value":            demo.get("housing_Owner_Households_Median_Value"),
        "median_rent":                  demo.get("housing_Median_Rent"),
        "pct_owner_occupied":           demo.get("housing_Units_Owner_Occupied_Pct"),
        "pct_vacant":                   demo.get("housing_Units_Vacant_Pct"),
        "avg_commute_mins":             demo.get("median_Travel_Time_To_Work_Mi"),
        "pct_with_mortgage":            demo.get("housing_Owner_Households_With_Mortgage_Pct"),

        # Climate
        "annual_avg_temp":              climate.get("annual_Avg_Temp"),
        "annual_precip_in":             climate.get("annual_Precip_In"),
        "clear_days_per_year":          climate.get("clear_Day_Mean"),
        "rainy_days_per_year":          climate.get("rainy_Day_Mean"),
        "snow_days_per_year":           climate.get("snow_Day_Mean"),
    }


# ---------------------------------------------------------------------------
# Extract fields from school district response
# ---------------------------------------------------------------------------

def extract_school(district):
    if not district:
        return {}

    identifier = district.get("identifier", {}) or {}
    name_info  = district.get("name",       {}) or {}
    ratings    = district.get("ratings",    {}) or {}
    details    = district.get("districtDetails", {}) or {}

    return {
        "school_district_name":        (name_info.get("districtName")
                                        or district.get("districtName", "")),
        "school_district_id":          (identifier.get("districtId")
                                        or district.get("districtId", "")),
        "school_district_rating":      (ratings.get("greatschoolsRating")
                                        or district.get("greatschoolsRating", "")),
        "school_district_grade_range": (details.get("gradeRange")
                                        or district.get("gradeRange", "")),
        "school_district_enrollment":  (details.get("enrollment")
                                        or district.get("enrollment", "")),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not Path(INPUT_FILE).exists():
        log.error(f"'{INPUT_FILE}' not found. Run scraper.py first.")
        return

    cache = load_cache()

    log.info("=" * 60)
    log.info("Dallas Property Enricher  (Step 2 of 2)")
    log.info("=" * 60)
    log.info(f"  Input:   {INPUT_FILE}")
    log.info(f"  Output:  {OUTPUT_FILE}")
    log.info(f"  Cache:   {len(cache['community'])} community + "
             f"{len(cache['school'])} school entries already cached")
    log.info("=" * 60)

    # ── Pass 1: Read all rows and collect unique geoIdV4s ─────────
    log.info("\nReading properties and collecting unique geoIdV4s...")
    rows = []
    unique_community_ids = set()
    unique_school_ids    = set()

    with open(INPUT_FILE, newline="", encoding="utf-8") as f:
        reader       = csv.DictReader(f)
        input_fields = reader.fieldnames or []
        for row in reader:
            rows.append(row)
            # Best available neighborhood ID for community lookup
            community_geo = (row.get("geoIdV4_N2") or
                             row.get("geoIdV4_N1") or
                             row.get("geoIdV4_N4") or
                             row.get("geoIdV4_ZI") or "").strip()
            if community_geo:
                unique_community_ids.add(community_geo)

            school_geo = row.get("geoIdV4_DB", "").strip()
            if school_geo:
                unique_school_ids.add(school_geo)

    log.info(f"  {len(rows):,} properties loaded")
    log.info(f"  {len(unique_community_ids)} unique neighborhoods")
    log.info(f"  {len(unique_school_ids)} unique school districts")

    # ── Pass 2: Fetch community data ──────────────────────────────
    log.info("\nFetching community data (crime + disasters + demographics)...")
    todo_community = [g for g in unique_community_ids if g not in cache["community"]]
    log.info(f"  {len(todo_community)} new lookups "
             f"({len(unique_community_ids) - len(todo_community)} already cached)")

    for i, geo_id in enumerate(todo_community, 1):
        log.info(f"  [{i}/{len(todo_community)}] community {geo_id[:8]}...")
        cache["community"][geo_id] = fetch_community(geo_id)
        if i % 25 == 0:
            save_cache(cache)
        time.sleep(DELAY)
    save_cache(cache)
    log.info("  Community data done.")

    # ── Pass 3: Fetch school district data ────────────────────────
    log.info("\nFetching school district data...")
    todo_schools = [g for g in unique_school_ids if g not in cache["school"]]
    log.info(f"  {len(todo_schools)} new lookups "
             f"({len(unique_school_ids) - len(todo_schools)} already cached)")

    for i, geo_id in enumerate(todo_schools, 1):
        log.info(f"  [{i}/{len(todo_schools)}] school district {geo_id[:8]}...")
        cache["school"][geo_id] = fetch_school_district(geo_id)
        if i % 25 == 0:
            save_cache(cache)
        time.sleep(DELAY)
    save_cache(cache)
    log.info("  School district data done.")

    # ── Pass 4: Write enriched CSV ────────────────────────────────
    log.info(f"\nWriting {OUTPUT_FILE}...")

    new_fields    = list(extract_community({}).keys()) + list(extract_school({}).keys())
    output_fields = input_fields + [f for f in new_fields if f not in input_fields]

    written = 0
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()

        for row in rows:
            community_geo = (row.get("geoIdV4_N2") or
                             row.get("geoIdV4_N1") or
                             row.get("geoIdV4_N4") or
                             row.get("geoIdV4_ZI") or "").strip()
            school_geo    = row.get("geoIdV4_DB", "").strip()

            community_cols = extract_community(cache["community"].get(community_geo, {}))
            school_cols    = extract_school(cache["school"].get(school_geo, {}))

            writer.writerow({**row, **community_cols, **school_cols})
            written += 1

    log.info("=" * 60)
    log.info(f"  Done! {written:,} enriched rows written to {OUTPUT_FILE}")
    log.info(f"  New columns added per property:")
    log.info(f"    Crime:        crime_index, crime_robbery, crime_burglary, ...")
    log.info(f"    Disasters:    disaster_tornado, disaster_hail, disaster_wind, ...")
    log.info(f"    Demographics: median_age, median_household_income, ...")
    log.info(f"    Climate:      annual_avg_temp, rainy_days_per_year, ...")
    log.info(f"    Schools:      school_district_name, school_district_rating, ...")
    log.info("=" * 60)


if __name__ == "__main__":
    main()