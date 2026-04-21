import os
import html
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
import streamlit.components.v1 as components
import joblib
import xgboost as xgb

st.set_page_config(
    page_title="Mollecul | AI Real Estate Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# DATA
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_existing_path(*relative_candidates: str) -> str:
    """Return the first existing path from a list of project-relative candidates."""
    for rel_path in relative_candidates:
        abs_path = os.path.join(BASE_DIR, rel_path)
        if os.path.exists(abs_path):
            return abs_path
    return os.path.join(BASE_DIR, relative_candidates[0])


csv_path = _resolve_existing_path(
    os.path.join("data", "PART1_NEW_Dallas_Properties.csv"),
    os.path.join("data", "dfw_real_estate.csv"),
    "Dallas_properties_combined.csv",
    "Combined_Dallas_Properties.csv",
)
model_path = _resolve_existing_path(
    "VER4_property_valuation_model.joblib",
    "NEW_property_valuation_model.joblib",
    "property_valuation_model.joblib",
)

ATTOM_KEY = os.getenv("ATTOM_API_KEY", "38c0c6bd6a465da7b414bc23a5df9791")
FRED_KEY  = os.getenv("FRED_API_KEY", "3edb96e23b26e2271758689309faee32")
NOAA_KEY  = os.getenv("NOAA_API_KEY", "xyeXCwTZHCBtzcdEHwQpnNvsfTyiWmGq")
EPA_KEY   = os.getenv("EPA_API_KEY", "aquaosprey52")

# Numeric and categorical feature columns the model was trained on
MODEL_NUM_COLS = [
    "lat", "lng", "yearBuilt", "yearBuiltEffective", "sqft", "livingArea",
    "grossSqft", "groundFloorSqft", "lotSqft", "lotAcres", "beds", "bathsFull",
    "bathsHalf", "bathsTotal", "totalRooms", "stories", "garageSize", "garageSpaces",
    "basementSqft", "fireplaces", "assessedTotal", "assessedLand", "assessedImprov",
    "assessedPerSqft", "assessedImprPerSqft", "marketTotal", "marketLand",
    "marketImprov", "taxAmount", "taxYear", "taxPerSqft", "calcTotalValue",
    "calcLandValue", "calcImprValue", "calcValuePerSqft", "salePrice",
    "pricePerSqft", "pricePerBed", "disclosed", "saleYear",
]
MODEL_CAT_COLS = [
    "zip", "city", "county", "municipality", "subdivision", "taxCodeArea",
    "geoIdV4_N2", "geoIdV4_N4", "geoIdV4_DB", "geoIdV4_SB", "propertyType",
    "propSubtype", "propClass", "propLandUse", "ownerOccupied", "pool",
    "basement", "bldgType", "condition", "view", "floors", "garageType",
    "heatingType", "coolingType", "constructionType", "foundationType",
    "roofMaterial", "roofType", "wallType", "frameType", "saleType",
    "saleDocType", "cashOrMortgage", "newConstruction", "interFamily",
    "sellerCarryback",
]
MODEL_ALL_COLS = MODEL_NUM_COLS + MODEL_CAT_COLS

FEATURE_LABELS = {
    "lat": "Latitude",
    "lng": "Longitude",
    "yearBuilt": "Year Built",
    "yearBuiltEffective": "Effective Year Built",
    "sqft": "Living Area",
    "livingArea": "Interior Area",
    "grossSqft": "Gross Area",
    "groundFloorSqft": "Ground Floor Area",
    "lotSqft": "Lot Size",
    "lotAcres": "Lot Acres",
    "beds": "Bedrooms",
    "bathsFull": "Full Baths",
    "bathsHalf": "Half Baths",
    "bathsTotal": "Bathrooms",
    "totalRooms": "Total Rooms",
    "stories": "Stories",
    "garageSize": "Garage Size",
    "garageSpaces": "Garage Spaces",
    "basementSqft": "Basement Area",
    "fireplaces": "Fireplaces",
    "assessedTotal": "Assessed Total",
    "assessedLand": "Assessed Land",
    "assessedImprov": "Assessed Improvements",
    "assessedPerSqft": "Assessed $/Sq Ft",
    "assessedImprPerSqft": "Assessed Improv $/Sq Ft",
    "marketTotal": "Market Total",
    "marketLand": "Market Land",
    "marketImprov": "Market Improvements",
    "taxAmount": "Tax Amount",
    "taxYear": "Tax Year",
    "taxPerSqft": "Tax $/Sq Ft",
    "calcTotalValue": "Calculated Total Value",
    "calcLandValue": "Calculated Land Value",
    "calcImprValue": "Calculated Improvement Value",
    "calcValuePerSqft": "Calculated $/Sq Ft",
    "salePrice": "Sale Price",
    "pricePerSqft": "Price Per Sq Ft",
    "pricePerBed": "Price Per Bed",
    "disclosed": "Sale Disclosure",
    "saleYear": "Sale Year",
    "zip": "ZIP Code",
    "city": "City",
    "county": "County",
    "municipality": "Municipality",
    "subdivision": "Subdivision",
    "taxCodeArea": "Tax Code Area",
    "geoIdV4_N2": "Neighborhood ID",
    "geoIdV4_N4": "Sub-Neighborhood ID",
    "geoIdV4_DB": "District Boundary",
    "geoIdV4_SB": "Sub-Boundary",
    "propertyType": "Property Type",
    "propSubtype": "Property Subtype",
    "propClass": "Property Class",
    "propLandUse": "Land Use",
    "ownerOccupied": "Occupancy",
    "pool": "Pool",
    "basement": "Basement",
    "bldgType": "Building Type",
    "condition": "Condition",
    "view": "View",
    "floors": "Floor Finish",
    "garageType": "Garage Type",
    "heatingType": "Heating",
    "coolingType": "Cooling",
    "constructionType": "Construction",
    "foundationType": "Foundation",
    "roofMaterial": "Roof Material",
    "roofType": "Roof Shape",
    "wallType": "Wall Type",
    "frameType": "Frame Type",
    "saleType": "Sale Type",
    "saleDocType": "Sale Document",
    "cashOrMortgage": "Cash or Mortgage",
    "newConstruction": "New Construction",
    "interFamily": "Inter-Family Transfer",
    "sellerCarryback": "Seller Carryback",
}

MODEL_SOURCE_ALIASES = {
    "lat": ["LATITUDE"],
    "lng": ["LONGITUDE"],
    "yearBuilt": ["YEAR BUILT"],
    "sqft": ["SQUARE FEET"],
    "lotSqft": ["LOT SIZE"],
    "beds": ["BEDS"],
    "bathsTotal": ["BATHS"],
    "salePrice": ["PRICE"],
    "pricePerSqft": ["PRICE_PER_SQFT_FINAL", "$/SQUARE FEET"],
    "zip": ["ZIP", "ZIP OR POSTAL CODE"],
    "city": ["CITY"],
    "propertyType": ["PROPERTY TYPE"],
    "saleType": ["SALE TYPE"],
}

CURRENCY_FEATURES = {
    "assessedTotal", "assessedLand", "assessedImprov", "assessedPerSqft",
    "assessedImprPerSqft", "marketTotal", "marketLand", "marketImprov",
    "taxAmount", "taxPerSqft", "calcTotalValue", "calcLandValue",
    "calcImprValue", "calcValuePerSqft", "salePrice", "pricePerSqft",
    "pricePerBed",
}
AREA_FEATURES = {"sqft", "livingArea", "grossSqft", "groundFloorSqft", "lotSqft", "basementSqft", "garageSize"}
COUNT_FEATURES = {"beds", "bathsFull", "bathsHalf", "bathsTotal", "totalRooms", "stories", "garageSpaces", "fireplaces"}
YEAR_FEATURES = {"yearBuilt", "yearBuiltEffective", "taxYear", "saleYear"}

@st.cache_resource
def load_valuation_model():
    """Load the XGBoost valuation pipeline from disk."""
    if not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.warning(f"Could not load valuation model: {e}")
        return None


def _is_missing_model_value(feature_name: str, value) -> bool:
    if feature_name in MODEL_NUM_COLS:
        return pd.isna(_to_float(value))
    text = _to_text(value).lower()
    return text in {"", "unknown", "__missing__"}


def _get_source_value(source_data: dict, feature_name: str):
    candidates = [feature_name] + MODEL_SOURCE_ALIASES.get(feature_name, [])
    for candidate in candidates:
        if candidate not in source_data:
            continue
        value = source_data.get(candidate)
        if not _is_missing_model_value(feature_name, value):
            return value
    return np.nan if feature_name in MODEL_NUM_COLS else "unknown"


def _extract_sale_year(source_data: dict) -> float:
    raw_sale_year = _get_source_value(source_data, "saleYear")
    if pd.notna(_to_float(raw_sale_year)):
        return _to_float(raw_sale_year)

    for candidate in ("saleDate", "SALE DATE", "saleTransDate", "saleRecDate"):
        if candidate not in source_data:
            continue
        parsed = pd.to_datetime(source_data.get(candidate), errors="coerce")
        if pd.notna(parsed):
            return float(parsed.year)
    return np.nan


def _build_feature_frame_from_source(source_data: dict) -> pd.DataFrame:
    row = {}

    for col in MODEL_NUM_COLS:
        if col == "saleYear":
            row[col] = _extract_sale_year(source_data)
            continue
        row[col] = _to_float(_get_source_value(source_data, col))

    for col in MODEL_CAT_COLS:
        value = _get_source_value(source_data, col)
        row[col] = _to_text(value) or "unknown"

    return pd.DataFrame([row])[MODEL_ALL_COLS]


def _format_feature_display_value(feature_name: str, value) -> str:
    if _is_missing_model_value(feature_name, value):
        return "Unknown"

    if feature_name in CURRENCY_FEATURES:
        return _format_currency(value)

    if feature_name in AREA_FEATURES:
        return f"{_format_metric_value(value)} sq ft"

    if feature_name == "lotAcres":
        return f"{_format_metric_value(value, decimals=2)} acres"

    if feature_name in YEAR_FEATURES:
        return _format_metric_value(value)

    if feature_name in COUNT_FEATURES:
        decimals = 1 if feature_name == "bathsTotal" else 0
        return _format_metric_value(value, decimals=decimals)

    if feature_name in {"lat", "lng"}:
        return _format_metric_value(value, decimals=4)

    return html.escape(_to_text(value))


def _calculate_feature_impacts(property_record: dict) -> dict:
    model = load_valuation_model()
    source_data = property_record.get("MODEL_SOURCE")
    if model is None or source_data is None:
        return {"error": "AI valuation model is unavailable for feature-level impacts."}

    try:
        preprocessor = model.named_steps["preprocessor"]
        estimator = model.named_steps["model"]
        feature_df = _build_feature_frame_from_source(source_data)
        transformed = preprocessor.transform(feature_df[MODEL_ALL_COLS])
        transformed = np.asarray(transformed, dtype=np.float64)

        feature_names = list(preprocessor.get_feature_names_out())
        dmatrix = xgb.DMatrix(transformed, feature_names=feature_names)
        contribs = estimator.get_booster().predict(dmatrix, pred_contribs=True)[0]

        drivers = []
        for encoded_name, impact in zip(feature_names, contribs[:-1]):
            feature_name = encoded_name.split("__", 1)[1] if "__" in encoded_name else encoded_name
            feature_value = feature_df.iloc[0][feature_name]
            if _is_missing_model_value(feature_name, feature_value):
                continue

            drivers.append({
                "feature": feature_name,
                "label": FEATURE_LABELS.get(feature_name, feature_name),
                "value": _format_feature_display_value(feature_name, feature_value),
                "impact": float(impact),
            })

        drivers.sort(key=lambda item: abs(item["impact"]), reverse=True)
        return {
            "baseline": float(contribs[-1]),
            "prediction": float(np.sum(contribs)),
            "drivers": drivers,
        }
    except Exception as e:
        return {"error": f"Could not compute feature impacts: {e}"}


def _render_feature_impact_rows(drivers: list[dict]) -> str:
    if not drivers:
        return (
            "<div style='font-size:11px;color:#8baec8;'>"
            "No populated model features were available for this property."
            "</div>"
        )

    max_impact = max(abs(driver["impact"]) for driver in drivers) or 1.0
    rows = []

    for driver in drivers:
        impact = driver["impact"]
        positive = impact >= 0
        color = "#2ce4df" if positive else "#e7c65a"
        width = max(12.0, (abs(impact) / max_impact) * 100)
        impact_label = f"{'+' if positive else '-'}${abs(impact):,.0f}"

        rows.append(f"""
<div style="padding:10px 0;border-top:1px solid rgba(94,122,146,0.16);">
  <div style="display:flex;align-items:flex-start;gap:10px;">
    <div style="min-width:0;">
      <div style="font-size:12px;font-weight:700;color:#ecf6ff;line-height:1.3;">{html.escape(driver['label'])}</div>
      <div style="font-size:10px;color:#8baec8;margin-top:2px;line-height:1.4;">{driver['value']}</div>
    </div>
    <div style="margin-left:auto;font-family:'Sora',sans-serif;font-size:11px;font-weight:700;color:{color};white-space:nowrap;">{impact_label}</div>
  </div>
  <div style="margin-top:8px;height:7px;background:rgba(94,122,146,0.16);border-radius:999px;overflow:hidden;">
    <div style="height:100%;width:{width:.1f}%;background:{color};box-shadow:0 0 12px {color}55;border-radius:999px;"></div>
  </div>
</div>
""")

    return "".join(rows)


def _render_feature_impact_card(property_record: dict) -> str:
    explanation = _calculate_feature_impacts(property_record)
    if explanation.get("error"):
        return f"""
<div style="background:#0b1c30;border:1px solid rgba(231,198,90,0.18);border-radius:14px;padding:16px 18px;">
  <div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.14em;color:#e7c65a;margin-bottom:6px;">Feature Impact</div>
  <div style="font-size:11px;color:#8baec8;line-height:1.6;">{html.escape(explanation['error'])}</div>
</div>
"""

    drivers = explanation["drivers"]
    preview_drivers = drivers[:6]
    remaining_drivers = drivers[6:]
    baseline_value = _format_currency(explanation.get("baseline"))
    prediction_value = _format_currency(explanation.get("prediction"))
    preview_rows = _render_feature_impact_rows(preview_drivers)
    detail_rows = _render_feature_impact_rows(remaining_drivers)
    detail_html = ""

    if remaining_drivers:
        detail_html = f"""
<details style="margin-top:12px;">
  <summary style="cursor:pointer;font-size:11px;font-weight:700;color:#8baec8;list-style:none;">
    View {len(remaining_drivers)} more observed feature effects
  </summary>
  <div style="margin-top:10px;">
    {detail_rows}
  </div>
</details>
"""

    return f"""
<div style="background:#0b1c30;border:1px solid rgba(94,166,255,0.22);border-radius:14px;padding:16px 18px;">
  <div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:8px;">
    <div>
      <div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.14em;color:#567592;">Feature Impact</div>
      <div style="font-size:11px;color:#8baec8;margin-top:4px;line-height:1.5;">
        How the model is moving this property away from its baseline value.
      </div>
    </div>
    <div style="margin-left:auto;text-align:right;">
      <div style="font-size:10px;color:#567592;text-transform:uppercase;letter-spacing:0.10em;">Baseline</div>
      <div style="font-family:'Sora',sans-serif;font-size:13px;font-weight:700;color:#ecf6ff;">{baseline_value}</div>
    </div>
  </div>
  <div style="font-size:10px;color:#567592;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.10em;">Model Estimate</div>
  <div style="font-family:'Sora',sans-serif;font-size:18px;font-weight:800;color:#5ea6ff;letter-spacing:-0.04em;margin-bottom:10px;">{prediction_value}</div>
  <div style="font-size:10px;color:#567592;margin-bottom:2px;text-transform:uppercase;letter-spacing:0.10em;">Strongest Observed Drivers</div>
  {preview_rows}
  {detail_html}
</div>
"""

def _predict_valuations(model, df_raw: pd.DataFrame) -> np.ndarray:
    """
    Run the valuation model over the raw CSV dataframe.
    Returns an array of predicted sale prices (one per row).
    Missing model columns are filled with 0 / 'unknown'.
    """
    feat = pd.DataFrame(index=df_raw.index)

    for col in MODEL_NUM_COLS:
        if col in df_raw.columns:
            feat[col] = pd.to_numeric(df_raw[col], errors="coerce").fillna(0)
        else:
            feat[col] = 0.0

    for col in MODEL_CAT_COLS:
        if col in df_raw.columns:
            feat[col] = df_raw[col].astype(str).fillna("unknown")
        else:
            feat[col] = "unknown"

    try:
        preds = model.predict(feat[MODEL_ALL_COLS])
        return np.array(preds, dtype=float)
    except Exception as e:
        st.warning(f"Model prediction failed: {e}")
        return np.zeros(len(df_raw))

@st.cache_data(ttl=3600)
def load_real_estate_data():
    df_raw = pd.read_csv(csv_path)
    df_raw.columns = [c.strip() for c in df_raw.columns]

    # ── Run the valuation model ──────────────────────────────────────────────
    model = load_valuation_model()
    if model is not None:
        predicted_values = _predict_valuations(model, df_raw)
    else:
        predicted_values = None  # fall back to heuristic below

    # ── Rename to display columns ────────────────────────────────────────────
    rename_map = {
        "lat":          "LATITUDE",
        "lng":          "LONGITUDE",
        "address":      "ADDRESS",
        "city":         "CITY",
        "salePrice":    "PRICE",
        "beds":         "BEDS",
        "bathsTotal":   "BATHS",
        "sqft":         "SQUARE FEET",
        "yearBuilt":    "YEAR BUILT",
        "propertyType": "PROPERTY TYPE",
        "lotSqft":      "LOT SIZE",
        "pricePerSqft": "$/SQUARE FEET",
    }
    df = df_raw.rename(columns={k: v for k, v in rename_map.items() if k in df_raw.columns})

    numeric_cols = [
        "PRICE", "BEDS", "BATHS", "SQUARE FEET", "LOT SIZE",
        "YEAR BUILT", "$/SQUARE FEET", "LATITUDE", "LONGITUDE"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[\$,]", "", regex=True).str.strip(),
                errors="coerce"
            )

    for col in ["CITY", "PROPERTY TYPE", "ADDRESS"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df = df.dropna(subset=["LATITUDE", "LONGITUDE"])

    # ── PRICE_PER_SQFT ───────────────────────────────────────────────────────
    if "$/SQUARE FEET" in df.columns:
        df["PRICE_PER_SQFT_FINAL"] = df["$/SQUARE FEET"]
    elif "PRICE" in df.columns and "SQUARE FEET" in df.columns:
        df["PRICE_PER_SQFT_FINAL"] = df["PRICE"] / df["SQUARE FEET"].replace(0, np.nan)
    else:
        df["PRICE_PER_SQFT_FINAL"] = 0

    df["DAYS_ON_MARKET_SAFE"] = 0  # not present in this dataset

    # ── Investment score from model predictions ──────────────────────────────
    if predicted_values is not None:
        # Align predictions back to the filtered (dropna) index
        pred_series = pd.Series(predicted_values, index=df_raw.index).reindex(df.index).fillna(0)
        df["PREDICTED_VALUE"] = pred_series

        # Score = how far the predicted value exceeds the asking price (upside)
        # High predicted / actual ratio → better investment opportunity
        actual = df["PRICE"].fillna(0)
        predicted = df["PREDICTED_VALUE"].clip(lower=0)
        ratio = np.where(actual > 0, predicted / actual.replace(0, np.nan), 1.0)
        df["investment_score"] = pd.Series(ratio, index=df.index).fillna(1.0)
    else:
        # Fallback heuristic (no model)
        df["PREDICTED_VALUE"] = df["PRICE"]
        df["investment_score"] = (
            df["PRICE_PER_SQFT_FINAL"].fillna(0) * 0.60 +
            (1 / (df["DAYS_ON_MARKET_SAFE"] + 1)) * 1000 * 0.28 +
            df["BEDS"].fillna(0) * 2.0 * 0.12
        )

    mn = df["investment_score"].min()
    mx = df["investment_score"].max()
    if mx != mn:
        df["investment_score"] = ((df["investment_score"] - mn) / (mx - mn)) * 100
    else:
        df["investment_score"] = 50

    return df

@st.cache_data(ttl=3600)
def get_price_trends(df, city):
    city_df = df[df["CITY"].str.lower() == city.lower()].copy()
    if "YEAR BUILT" in city_df.columns and city_df["YEAR BUILT"].notna().sum() > 0:
        trend_df = (
            city_df.dropna(subset=["YEAR BUILT", "PRICE"])
            .groupby("YEAR BUILT", as_index=False)["PRICE"].mean()
            .sort_values("YEAR BUILT")
        )
        trend_df = trend_df.rename(columns={"YEAR BUILT": "Period", "PRICE": "Value"})
        trend_df["Period"] = trend_df["Period"].astype(int).astype(str)
        return trend_df.tail(20)

    return pd.DataFrame({"Period": ["1", "2", "3"], "Value": [0, 0, 0]})

@st.cache_data(ttl=3600)
def get_roi_by_type(df, city):
    city_df = df[df["CITY"].str.lower() == city.lower()].copy()
    roi_df = (
        city_df.dropna(subset=["PROPERTY TYPE", "PRICE_PER_SQFT_FINAL"])
        .groupby("PROPERTY TYPE", as_index=False)["PRICE_PER_SQFT_FINAL"].mean()
        .sort_values("PRICE_PER_SQFT_FINAL", ascending=False)
        .head(8)
    )
    return roi_df.rename(columns={"PRICE_PER_SQFT_FINAL": "ROI_PROXY"})

@st.cache_data(ttl=3600)
def get_top_listings(df, city):
    city_df = df[df["CITY"].str.lower() == city.lower()].copy()

    cols = [c for c in [
        "ADDRESS", "PROPERTY TYPE", "PRICE", "BEDS", "BATHS",
        "SQUARE FEET", "DAYS ON MARKET", "investment_score"
    ] if c in city_df.columns]

    top_df = city_df.sort_values("investment_score", ascending=False)[cols].head(6).copy()

    if "PRICE" in top_df.columns:
        top_df["PRICE"] = top_df["PRICE"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
    if "BEDS" in top_df.columns:
        top_df["BEDS"] = top_df["BEDS"].apply(lambda x: int(x) if pd.notna(x) else "—")
    if "BATHS" in top_df.columns:
        top_df["BATHS"] = top_df["BATHS"].apply(lambda x: round(float(x), 1) if pd.notna(x) else "—")
    if "SQUARE FEET" in top_df.columns:
        top_df["SQUARE FEET"] = top_df["SQUARE FEET"].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "—")
    if "DAYS ON MARKET" in top_df.columns:
        top_df["DAYS ON MARKET"] = top_df["DAYS ON MARKET"].apply(lambda x: int(x) if pd.notna(x) else "—")
    if "investment_score" in top_df.columns:
        top_df["investment_score"] = top_df["investment_score"].round(2)

    return top_df

# =============================================================================
# EASY_PREDICT INTEGRATION
# =============================================================================
import sys as _sys
_services_dir = os.path.join(BASE_DIR, "services")
if _services_dir not in _sys.path:
    _sys.path.insert(0, _services_dir)


def _to_float(value):
    try:
        if value is None:
            return np.nan
        if isinstance(value, str) and value.strip() in {"", "None", "nan", "NaN"}:
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _to_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "nan"} else text


def _compose_display_address(address, city="", zip_code="") -> str:
    address_text = _to_text(address)
    city_text = _to_text(city)
    zip_text = _to_text(zip_code)

    if not address_text:
        return ", ".join([part for part in [city_text, zip_text] if part])

    if city_text and city_text.lower() not in address_text.lower():
        address_text = f"{address_text}, {city_text}"
    if zip_text and zip_text not in address_text:
        address_text = f"{address_text} {zip_text}"
    return address_text


def _first_valid_number(*values):
    for value in values:
        number = _to_float(value)
        if pd.notna(number):
            return number
    return np.nan


def _build_attom_property_record(address: str, raw_property: dict, prediction: dict | None = None) -> dict:
    sale_price = _to_float(raw_property.get("salePrice"))
    avm_value = _to_float(raw_property.get("avmValue"))
    price_value = sale_price if pd.notna(sale_price) else avm_value
    sqft_value = _first_valid_number(
        raw_property.get("sqft"),
        raw_property.get("livingArea"),
        raw_property.get("grossSqft"),
    )
    price_per_sqft = _to_float(raw_property.get("pricePerSqft"))
    if pd.isna(price_per_sqft) and pd.notna(price_value) and pd.notna(sqft_value) and sqft_value > 0:
        price_per_sqft = price_value / sqft_value

    city = _to_text(raw_property.get("city"))
    zip_code = _to_text(raw_property.get("zip"))
    display_address = _compose_display_address(address, city, zip_code)

    ai_value = np.nan
    if prediction is not None:
        ai_value = _to_float(prediction.get("current"))

    return {
        "ADDRESS": display_address,
        "CITY": city,
        "PRICE": price_value,
        "PRICE_LABEL": "Latest Sale Price" if pd.notna(sale_price) else "ATTOM AVM",
        "SALE_PRICE": sale_price,
        "ATTOM_AVM": avm_value,
        "BEDS": _to_float(raw_property.get("beds")),
        "BATHS": _to_float(raw_property.get("bathsTotal")),
        "SQUARE FEET": sqft_value,
        "YEAR BUILT": _to_float(raw_property.get("yearBuilt")),
        "PROPERTY TYPE": _to_text(raw_property.get("propertyType")) or _to_text(raw_property.get("propSubtype")),
        "LOT SIZE": _to_float(raw_property.get("lotSqft")),
        "$/SQUARE FEET": price_per_sqft,
        "PRICE_PER_SQFT_FINAL": price_per_sqft if pd.notna(price_per_sqft) else 0,
        "LATITUDE": _to_float(raw_property.get("lat")),
        "LONGITUDE": _to_float(raw_property.get("lng")),
        "PREDICTED_VALUE": ai_value,
        "MODEL_ESTIMATE": ai_value,
        "ZIP": zip_code,
        "DATA_SOURCE": "ATTOM",
        "MODEL_SOURCE": raw_property,
    }


def _build_market_property_record(row: pd.Series) -> dict:
    sale_price = _first_valid_number(row.get("PRICE"), row.get("salePrice"))
    avm_value = _to_float(row.get("avmValue"))
    model_estimate = _to_float(row.get("PREDICTED_VALUE"))
    city = _to_text(row.get("CITY") if "CITY" in row else row.get("city"))
    zip_code = _to_text(row.get("ZIP") if "ZIP" in row else row.get("zip"))

    return {
        "ADDRESS": _compose_display_address(row.get("ADDRESS"), city, zip_code),
        "CITY": city,
        "PRICE": sale_price if pd.notna(sale_price) else avm_value,
        "PRICE_LABEL": "Latest Sale Price" if pd.notna(sale_price) else "ATTOM AVM",
        "SALE_PRICE": sale_price,
        "ATTOM_AVM": avm_value,
        "BEDS": _to_float(row.get("BEDS")),
        "BATHS": _to_float(row.get("BATHS")),
        "SQUARE FEET": _to_float(row.get("SQUARE FEET")),
        "YEAR BUILT": _to_float(row.get("YEAR BUILT")),
        "PROPERTY TYPE": _to_text(row.get("PROPERTY TYPE")),
        "LOT SIZE": _to_float(row.get("LOT SIZE")),
        "$/SQUARE FEET": _to_float(row.get("PRICE_PER_SQFT_FINAL")),
        "LATITUDE": _to_float(row.get("LATITUDE")),
        "LONGITUDE": _to_float(row.get("LONGITUDE")),
        "PREDICTED_VALUE": model_estimate,
        "MODEL_ESTIMATE": model_estimate,
        "ZIP": zip_code,
        "DATA_SOURCE": "DATASET",
        "MODEL_SOURCE": row.to_dict(),
    }


def _coords_match(lat1, lng1, lat2, lng2, tol: float = 1e-6) -> bool:
    lat1 = _to_float(lat1)
    lng1 = _to_float(lng1)
    lat2 = _to_float(lat2)
    lng2 = _to_float(lng2)
    return (
        pd.notna(lat1)
        and pd.notna(lng1)
        and pd.notna(lat2)
        and pd.notna(lng2)
        and abs(lat1 - lat2) <= tol
        and abs(lng1 - lng2) <= tol
    )


def _resolve_clicked_property(
    map_state: dict | None,
    map_df: pd.DataFrame,
    searched_property: dict | None,
    searched_prediction: dict | None,
) -> tuple[dict | None, dict | None]:
    if not map_state:
        return None, None

    clicked = map_state.get("last_object_clicked")
    if not clicked:
        return None, None

    click_lat = _to_float(clicked.get("lat"))
    click_lng = _to_float(clicked.get("lng"))
    if pd.isna(click_lat) or pd.isna(click_lng):
        return None, None

    if searched_property is not None and _coords_match(
        click_lat, click_lng, searched_property.get("LATITUDE"), searched_property.get("LONGITUDE")
    ):
        return searched_property, searched_prediction

    matched_rows = map_df[
        (map_df["LATITUDE"].sub(click_lat).abs() <= 1e-6)
        & (map_df["LONGITUDE"].sub(click_lng).abs() <= 1e-6)
    ]
    if matched_rows.empty:
        return None, None

    return _build_market_property_record(matched_rows.iloc[0]), None


def _format_currency(value, fallback: str = "N/A") -> str:
    number = _to_float(value)
    if pd.isna(number):
        return fallback
    return f"${number:,.0f}"


def _format_metric_value(value, decimals: int = 0, fallback: str = "N/A") -> str:
    number = _to_float(value)
    if pd.isna(number):
        return fallback
    if decimals == 0:
        return f"{number:,.0f}"
    return f"{number:,.{decimals}f}"


def _render_value_comparison_chart(property_record: dict) -> str:
    sale_price = _to_float(property_record.get("SALE_PRICE"))
    avm_value = _to_float(property_record.get("ATTOM_AVM"))
    model_estimate = _first_valid_number(
        property_record.get("MODEL_ESTIMATE"),
        property_record.get("PREDICTED_VALUE"),
    )

    metrics = [
        ("Sale Price", sale_price, "#e7c65a"),
        ("ATTOM AVM", avm_value, "#2ce4df"),
        ("Our Model", model_estimate, "#b89eff"),
    ]
    available = [(label, value, color) for label, value, color in metrics if pd.notna(value) and value > 0]
    if not available:
        return ""

    max_value = max(value for _, value, _ in available)
    max_value = max(max_value, 1.0)
    bars = []

    for label, value, color in metrics:
        if pd.notna(value) and value > 0:
            width = max(12.0, min(100.0, (value / max_value) * 100))
            value_label = f"${value:,.0f}"
        else:
            width = 0.0
            value_label = "N/A"

        if label == "Sale Price":
            delta_html = "<span style='color:#567592;'>Baseline</span>"
        elif pd.notna(value) and pd.notna(sale_price) and sale_price > 0:
            diff = value - sale_price
            pct = (diff / sale_price) * 100
            delta_color = "#2ce4df" if diff >= 0 else "#e7c65a"
            delta_html = (
                f"<span style='color:{delta_color};'>"
                f"{'+' if diff >= 0 else '-'}${abs(diff):,.0f} ({pct:+.1f}%) vs sale"
                "</span>"
            )
        else:
            delta_html = "<span style='color:#567592;'>No sale baseline</span>"

        bars.append(f"""
<div style="display:flex;flex-direction:column;gap:6px;">
  <div style="display:flex;align-items:center;gap:8px;">
    <div style="font-size:11px;font-weight:700;color:#ecf6ff;">{html.escape(label)}</div>
    <div style="margin-left:auto;font-family:'Sora',sans-serif;font-size:12px;font-weight:700;color:#ecf6ff;">{value_label}</div>
  </div>
  <div style="height:10px;background:rgba(94,122,146,0.18);border-radius:999px;overflow:hidden;">
    <div style="height:100%;width:{width:.1f}%;background:{color};box-shadow:0 0 14px {color}55;"></div>
  </div>
  <div style="font-size:10px;font-weight:600;letter-spacing:0.02em;">{delta_html}</div>
</div>
""")

    avm_model_summary = ""
    if pd.notna(avm_value) and pd.notna(model_estimate):
        avm_diff = model_estimate - avm_value
        avm_color = "#2ce4df" if avm_diff >= 0 else "#e7c65a"
        avm_model_summary = (
            f"<div style='font-size:11px;color:{avm_color};font-weight:700;'>"
            f"Model vs ATTOM AVM: {'+' if avm_diff >= 0 else '-'}${abs(avm_diff):,.0f}"
            "</div>"
        )

    return f"""
<div style="background:#0b1c30;border:1px solid rgba(184,158,255,0.18);border-radius:14px;padding:16px 18px;">
  <div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.14em;color:#567592;margin-bottom:12px;">Value Comparison</div>
  <div style="display:flex;flex-direction:column;gap:14px;">
    {''.join(bars)}
  </div>
  <div style="margin-top:12px;">
    {avm_model_summary}
  </div>
</div>
"""


def _render_property_panel(property_record: dict, prediction: dict | None = None) -> str:
    address = html.escape(_to_text(property_record.get("ADDRESS")) or "—")
    city = html.escape(_to_text(property_record.get("CITY")))
    prop_type = html.escape(_to_text(property_record.get("PROPERTY TYPE")) or "Residential")
    beds = _format_metric_value(property_record.get("BEDS"))
    baths = _format_metric_value(property_record.get("BATHS"), decimals=1)
    sqft = _format_metric_value(property_record.get("SQUARE FEET"))
    year_built = _format_metric_value(property_record.get("YEAR BUILT"))
    price_label = html.escape(_to_text(property_record.get("PRICE_LABEL")) or "Price Reference")
    price_value = _format_currency(property_record.get("PRICE"))
    comparison_html = _render_value_comparison_chart(property_record)
    feature_impact_html = _render_feature_impact_card(property_record)

    forecast_html = ""
    if prediction is not None:
        if prediction.get("error"):
            forecast_html = f"""
<div style="background:#0b1c30;border:1px solid rgba(231,198,90,0.18);border-radius:14px;padding:16px 18px;">
  <div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.14em;color:#e7c65a;margin-bottom:6px;">Forecast Unavailable</div>
  <div style="font-size:11px;color:#8baec8;">{html.escape(str(prediction.get('error', 'Unknown error'))[:140])}</div>
</div>
"""
        else:
            future = _to_float(prediction.get("future"))
            diff = _to_float(prediction.get("diff"))
            pct = _to_float(prediction.get("pct"))
            forecast_delta = ""
            if pd.notna(diff):
                delta_color = "#2ce4df" if diff >= 0 else "#e7c65a"
                delta_text = (
                    f"{'+' if diff >= 0 else '-'}${abs(diff):,.0f} ({pct:+.2f}%)"
                    if pd.notna(pct)
                    else f"{'+' if diff >= 0 else '-'}${abs(diff):,.0f}"
                )
                forecast_delta = (
                    f"<div style='font-size:11px;color:{delta_color};margin-top:5px;font-weight:700;'>"
                    f"{delta_text}</div>"
                )

            forecast_html = f"""
<div style="background:#0b1c30;border:1px solid rgba(94,166,255,0.25);border-radius:14px;padding:16px 18px;">
  <div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.14em;color:#567592;margin-bottom:4px;">6-Month Forecast</div>
  <div style="font-family:'Sora',sans-serif;font-size:20px;font-weight:800;color:#5ea6ff;letter-spacing:-0.04em;">{_format_currency(future)}</div>
  {forecast_delta}
</div>
"""

    return f"""
<div style="display:flex;flex-direction:column;gap:12px;padding:4px 0;">
  <div style="background:#0b1c30;border:1px solid rgba(44,228,223,0.18);border-radius:14px;padding:16px 18px;">
    <div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.14em;color:#567592;margin-bottom:6px;">Selected Property</div>
    <div style="font-family:'Sora',sans-serif;font-size:13px;font-weight:600;color:#ecf6ff;line-height:1.5;">{address}</div>
    <div style="font-size:11px;color:#8baec8;margin-top:4px;">{city} · {prop_type}</div>
  </div>
  <div style="background:#0b1c30;border:1px solid rgba(44,228,223,0.10);border-radius:14px;padding:16px 18px;">
    <div style="display:grid;grid-template-columns:repeat(2, minmax(0, 1fr));gap:10px 12px;">
      <div>
        <div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.14em;color:#567592;margin-bottom:4px;">Beds</div>
        <div style="font-family:'Sora',sans-serif;font-size:16px;font-weight:700;color:#ecf6ff;">{beds}</div>
      </div>
      <div>
        <div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.14em;color:#567592;margin-bottom:4px;">Baths</div>
        <div style="font-family:'Sora',sans-serif;font-size:16px;font-weight:700;color:#ecf6ff;">{baths}</div>
      </div>
      <div>
        <div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.14em;color:#567592;margin-bottom:4px;">Sq Ft</div>
        <div style="font-family:'Sora',sans-serif;font-size:16px;font-weight:700;color:#ecf6ff;">{sqft}</div>
      </div>
      <div>
        <div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.14em;color:#567592;margin-bottom:4px;">Year Built</div>
        <div style="font-family:'Sora',sans-serif;font-size:16px;font-weight:700;color:#ecf6ff;">{year_built}</div>
      </div>
    </div>
  </div>
  {forecast_html}
  {comparison_html}
  {feature_impact_html}
  <div style="background:#0b1c30;border:1px solid rgba(44,228,223,0.10);border-radius:14px;padding:16px 18px;">
    <div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.14em;color:#567592;margin-bottom:4px;">{price_label}</div>
    <div style="font-family:'Sora',sans-serif;font-size:18px;font-weight:800;color:#ecf6ff;letter-spacing:-0.04em;">{price_value}</div>
  </div>
</div>
"""


def _lookup_attom_property_bundle(address: str) -> dict:
    import predict_address as _predict_address

    _predict_address.API_KEY = ATTOM_KEY
    raw_property = _predict_address.get_property_data(address)
    prediction = _run_easy_predict(raw_property)
    property_record = _build_attom_property_record(address, raw_property, prediction)
    return {
        "property": property_record,
        "prediction": prediction,
        "raw_property": raw_property,
    }


def _run_easy_predict(raw_property: dict) -> dict:
    """
    Run the Stage 1 valuation and Stage 2 forecast from an ATTOM property payload.
    Changes working directory to the project root so relative model paths resolve correctly.
    Returns a dict with keys: current, future, diff, pct, error
    """
    import os as _os
    _orig_cwd = _os.getcwd()
    result = {"current": None, "future": None, "diff": None, "pct": None, "error": None}
    try:
        from model_logic import predict as stage1_predict
        from predict_address import build_feature_row

        stage1_model = load_valuation_model()
        if stage1_model is None:
            result["error"] = f"Valuation model not found at: {model_path}"
            return result

        df_features = build_feature_row(raw_property)
        base_value = float(stage1_predict(stage1_model, df_features)[0])
        result["current"] = base_value

        # round2_model loads artifacts using project-root-relative paths.
        _os.chdir(BASE_DIR)
        from round2_model import predict_forward

        result = predict_forward(
            base_estimate=base_value,
            fred_api_key=FRED_KEY,
            noaa_api_key=NOAA_KEY,
            epa_api_key=EPA_KEY,
            zip_code=_to_text(raw_property.get("zip")) or None,
            property_type=_to_text(raw_property.get("propertyType")) or _to_text(raw_property.get("propSubtype")) or None,
            verbose=False,
        )
        return {
            "current": base_value,
            "future":  result.get("forward_estimate", base_value),
            "diff":    result.get("change_dollars", 0),
            "pct":     result.get("change_pct", 0),
            "error":   None,
        }
    except Exception as e:
        result["error"] = str(e)
        return result
    finally:
        _os.chdir(_orig_cwd)  # always restore original working directory

if not os.path.exists(csv_path):
    st.error(f"CSV not found at: {csv_path}")
    st.stop()

if not os.path.exists(model_path):
    st.warning(
        f"Valuation model not found at: {model_path} — "
        "investment scores will use the fallback heuristic. "
        "Place `VER4_property_valuation_model.joblib` next to `app.py` to enable AI-powered valuations."
    )

df = load_real_estate_data()

# =============================================================================
# GLOBAL STYLES
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

:root {
  --bg: #06111f;
  --panel: #0b1c30;
  --panel-hover: #0e2236;
  --line: rgba(110, 196, 255, 0.09);
  --line-strong: rgba(44, 228, 223, 0.28);
  --text: #ecf6ff;
  --muted: #8baec8;
  --muted-2: #567592;
  --cyan: #2ce4df;
  --blue: #5ea6ff;
  --gold: #e7c65a;
  --purple: #b89eff;
  --shadow: 0 20px 60px rgba(0,0,0,0.35);
  --glow-cyan: 0 0 24px rgba(44,228,223,0.18);
}

/* ── Reset Streamlit chrome ── */
html, body, .stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stAppViewContainer"], [data-testid="stVerticalBlock"],
[data-testid="stMain"], .main > div, section[data-testid="stSidebar"] {
  background: transparent !important;
}
header[data-testid="stHeader"] {
  background: rgba(6,17,31,0.90) !important;
  border-bottom: 1px solid rgba(44,228,223,0.07);
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
}
.block-container {
  padding-top: 0 !important; padding-left: 3rem !important;
  padding-right: 3rem !important; max-width: 100% !important;
}

/* ── Subtle grid bg ── */
.stApp::before {
  content: ""; position: fixed; inset: 0;
  background-image:
    linear-gradient(rgba(44,228,223,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(44,228,223,0.025) 1px, transparent 1px);
  background-size: 64px 64px;
  pointer-events: none; z-index: 0;
}

/* ── Layout shell ── */
.app-shell {
  position: relative; z-index: 2;
  max-width: 1460px; margin: 0 auto;
  padding: 32px 40px 60px;
}
@media (max-width: 900px) { .app-shell { padding: 18px 16px 40px; } }

/* ── Hide native Streamlit metric widget entirely ── */
[data-testid="stMetric"] { display: none !important; }

/* ── Hide native column gaps / decoration ── */
[data-testid="stHorizontalBlock"] > div { gap: 0 !important; }

/* ── Section headers ── */
.dash-section {
  display: flex; align-items: center; gap: 14px;
  margin: 36px 0 22px; padding-bottom: 16px;
  border-bottom: 1px solid var(--line);
}
.dash-section-bar {
  width: 3px; height: 24px; border-radius: 99px;
  background: linear-gradient(180deg, var(--cyan) 0%, rgba(94,166,255,0.15) 100%);
  box-shadow: var(--glow-cyan);
  flex-shrink: 0;
}
.dash-section-label {
  font-family: 'Sora', sans-serif;
  font-size: 13px; font-weight: 700;
  letter-spacing: 0.10em; text-transform: uppercase;
  color: var(--muted); margin: 0;
}
.dash-section-title {
  font-family: 'Sora', sans-serif;
  font-size: 22px; font-weight: 700; letter-spacing: -0.04em;
  color: var(--text); margin: 0;
}

/* ── City selector override ── */
[data-testid="stSelectbox"] label {
  font-size: 10px !important; text-transform: uppercase !important;
  letter-spacing: 0.14em !important; color: var(--muted-2) !important;
  font-weight: 700 !important;
}
[data-testid="stSelectbox"] > div > div {
  background: var(--panel) !important;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important; color: var(--text) !important;
  font-family: 'Sora', sans-serif !important; font-weight: 600 !important;
  box-shadow: none !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
  border-color: var(--line-strong) !important;
  box-shadow: 0 0 0 2px rgba(44,228,223,0.08) !important;
}

/* ── Custom metric cards ── */
.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px; margin: 20px 0 0;
}
@media (max-width: 900px) { .metric-grid { grid-template-columns: repeat(2,1fr); } }

.metric-card {
  position: relative; overflow: hidden;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 22px 24px 20px;
  box-shadow: var(--shadow);
  transition: border-color 0.25s, transform 0.25s;
  cursor: default;
}
.metric-card:hover {
  border-color: rgba(44,228,223,0.22);
  transform: translateY(-3px);
}
.metric-card::after {
  content: ""; position: absolute;
  left: 0; right: 0; bottom: 0; height: 2px;
  background: var(--accent-bar, linear-gradient(90deg,var(--cyan),var(--blue)));
  opacity: 0.9;
}
.metric-card-icon {
  font-size: 22px; margin-bottom: 14px;
  display: block; opacity: 0.85;
}
.metric-card-label {
  font-size: 10px; text-transform: uppercase;
  letter-spacing: 0.16em; font-weight: 700;
  color: var(--muted-2); margin-bottom: 6px;
}
.metric-card-value {
  font-family: 'Sora', sans-serif;
  font-size: 28px; font-weight: 800;
  letter-spacing: -0.05em; color: var(--text);
  line-height: 1;
}
.metric-card-sub {
  font-size: 11px; color: var(--muted-2);
  margin-top: 6px; font-weight: 400;
}

/* ── Chart panel ── */
.chart-panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 22px;
  box-shadow: var(--shadow);
  padding: 20px 20px 10px;
  transition: border-color 0.2s;
}
.chart-panel:hover { border-color: rgba(44,228,223,0.15); }
.chart-panel-header {
  display: flex; align-items: center; gap: 10px; margin-bottom: 4px;
}
.chart-panel-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--dot-color, var(--cyan));
  box-shadow: 0 0 8px var(--dot-color, var(--cyan));
}
.chart-panel-title {
  font-family: 'Sora', sans-serif;
  font-size: 13px; font-weight: 700;
  letter-spacing: -0.01em; color: var(--text);
}
.chart-panel-sub {
  font-size: 11px; color: var(--muted-2);
  margin-left: auto; letter-spacing: 0.04em;
}

/* ── Map & table panels ── */
.map-wrap {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 22px; box-shadow: var(--shadow);
  overflow: hidden; padding: 14px;
}
.table-wrap {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 22px; box-shadow: var(--shadow); padding: 14px;
}

/* ── Legend pills ── */
.legend-row { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:16px; }
.legend-pill {
  display:inline-flex; align-items:center; gap:8px;
  padding:7px 14px; border-radius:999px;
  background:rgba(11,28,48,0.9); border:1px solid var(--line);
  color:var(--muted) !important; font-size:12px; font-weight:600;
}
.legend-dot { width:8px; height:8px; border-radius:50%; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { background: transparent !important; border: 0 !important; }
[data-testid="stDataFrame"] [role="table"] {
  border-radius:14px !important; overflow:hidden !important;
  border:1px solid var(--line) !important;
}
[data-testid="stDataFrame"] thead tr th {
  background:#0d2035 !important; color:var(--muted-2) !important;
  font-size:10px !important; text-transform:uppercase !important;
  letter-spacing:0.12em !important; font-weight:700 !important;
}
[data-testid="stDataFrame"] tbody tr td {
  background:#080f1c !important; color:var(--text) !important;
  font-size:13px !important;
  border-bottom:1px solid rgba(120,200,255,0.05) !important;
}
[data-testid="stDataFrame"] tbody tr:hover td { background:#0e1f33 !important; }

/* ── Footer ── */
.footer-note {
  text-align:center; padding:40px 0 12px;
  color:var(--muted-2); font-size:11px;
  letter-spacing:0.20em; text-transform:uppercase;
}

.hero-spacer { height: 4px; }
h1,h2,h3,h4,h5,h6 {
  font-family:'Sora',sans-serif !important;
  color:var(--text) !important; letter-spacing:-0.04em;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HERO
# =============================================================================
HERO_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap" rel="stylesheet">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body {
    width: 100%; height: 100%;
    overflow: hidden;
    background: #06111f;
    font-family: 'DM Sans', sans-serif;
  }

  .hero {
    position: relative;
    width: 100%; height: 620px;
    overflow: hidden;
    background: #06111f;
  }

  /* Liquid Ether canvas fills the whole hero */
  #liquid-canvas {
    position: absolute;
    inset: 0;
    width: 100%; height: 100%;
    display: block;
  }

  /* Dark vignette so text stays readable */
  .overlay-left {
    position: absolute; inset: 0;
    background: linear-gradient(
      105deg,
      rgba(6,17,31,0.97) 0%,
      rgba(6,17,31,0.82) 30%,
      rgba(6,17,31,0.38) 58%,
      rgba(6,17,31,0.05) 100%
    );
    z-index: 2;
  }
  .overlay-bottom {
    position: absolute; left:0; right:0; bottom:0; height:200px;
    background: linear-gradient(to bottom, transparent, rgba(6,17,31,0.88) 70%, #06111f);
    z-index: 2;
  }
  .overlay-top {
    position: absolute; top:0; left:0; right:0; height:90px;
    background: linear-gradient(to bottom, rgba(6,17,31,0.70), transparent);
    z-index: 2;
  }

  .content {
    position: relative; z-index: 3;
    height: 100%;
    display: flex; align-items: center;
    padding: 80px 60px 0;
  }

  .content-inner { max-width: 660px; }

  /* Eyebrow pill */
  .eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 14px 6px 10px;
    border-radius: 999px;
    background: rgba(44,228,223,0.08);
    border: 1px solid rgba(44,228,223,0.22);
    color: #7de8e4;
    font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase; font-weight: 600;
    margin-bottom: 28px;
    animation: fadeUp 0.7s ease both;
  }
  .eyebrow-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #2ce4df;
    box-shadow: 0 0 10px rgba(44,228,223,0.9);
    animation: pulse 2.4s ease-in-out infinite;
  }
  @keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50% { opacity:0.6; transform:scale(0.8); }
  }

  /* Brand row */
  .brand {
    display: flex; align-items: center; gap: 11px;
    margin-bottom: 22px;
    animation: fadeUp 0.7s 0.08s ease both;
  }
  .brand-name {
    font-family: 'Sora', sans-serif;
    font-size: 26px; font-weight: 700;
    letter-spacing: -0.06em;
    color: #edf7ff;
  }

  /* Hero headline */
  .title {
    font-family: 'Sora', sans-serif;
    font-size: clamp(48px, 6.8vw, 86px);
    line-height: 0.95;
    letter-spacing: -0.06em;
    color: #edf7ff;
    margin-bottom: 20px;
    animation: fadeUp 0.7s 0.14s ease both;
  }
  .title .accent {
    display: block;
    background: linear-gradient(90deg, #2ce4df 0%, #5ea6ff 60%, #b89eff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .title .plain { display: block; }

  /* Subheadline */
  .sub {
    max-width: 500px;
    font-size: 16px; line-height: 1.75; font-weight: 300;
    color: rgba(220,238,252,0.72);
    margin-bottom: 30px;
    animation: fadeUp 0.7s 0.22s ease both;
  }

  /* Stat chips */
  .chip-row {
    display: flex; gap: 10px; flex-wrap: wrap;
    animation: fadeUp 0.7s 0.30s ease both;
  }
  .chip {
    padding: 11px 18px; border-radius: 12px;
    background: rgba(8, 22, 38, 0.55);
    border: 1px solid rgba(110,196,255,0.13);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    transition: border-color 0.2s, transform 0.2s;
  }
  .chip:hover {
    border-color: rgba(44,228,223,0.30);
    transform: translateY(-2px);
  }
  .chip-label {
    font-size: 9px; text-transform: uppercase; letter-spacing: 0.16em;
    color: rgba(140,170,196,0.60); font-weight: 600; margin-bottom: 3px;
  }
  .chip-value {
    font-family: 'Sora', sans-serif;
    font-size: 14px; font-weight: 700;
    color: #edf7ff; letter-spacing: -0.03em;
  }
  .chip-value.teal {
    background: linear-gradient(90deg, #2ce4df, #5ea6ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  @media (max-width: 860px) {
    .hero { height: 560px; }
    .content { padding: 0 22px; }
    .sub { font-size: 14px; }
  }
</style>
</head>
<body>
<div class="hero" id="hero">

  <!-- Liquid Ether fluid simulation canvas -->
  <canvas id="liquid-canvas"></canvas>
  <div class="overlay-left"></div>
  <div class="overlay-top"></div>
  <div class="overlay-bottom"></div>

  <div class="content">
    <div class="content-inner">
      <div class="eyebrow"><span class="eyebrow-dot"></span>AI Real Estate Intelligence</div>

      <div class="brand">
        <svg width="38" height="38" viewBox="0 0 44 44" fill="none" xmlns="http://www.w3.org/2000/svg">
          <circle cx="22" cy="22" r="17" stroke="#2CE4DF" stroke-width="1.3" stroke-dasharray="3 3.5" opacity="0.5"/>
          <circle cx="36" cy="22" r="2.7" fill="#2CE4DF" opacity="0.9"/>
          <rect x="17" y="11" width="6" height="18" rx="1.4" fill="#E7C65A"/>
          <rect x="11" y="16" width="4.8" height="13" rx="1.4" fill="#2CE4DF" opacity="0.85"/>
          <rect x="25" y="14" width="4.8" height="15" rx="1.4" fill="#5EA6FF" opacity="0.8"/>
          <line x1="8" y1="29" x2="31" y2="29" stroke="#2CE4DF" stroke-width="1.0" opacity="0.3"/>
          <circle cx="22" cy="6.5" r="1.8" fill="#5EA6FF" opacity="0.7"/>
          <circle cx="6.5" cy="22" r="2.0" fill="#E7C65A" opacity="0.8"/>
        </svg>
        <div class="brand-name">mollecul</div>
      </div>

      <div class="title">
        <span class="plain">Objective.</span>
        <span class="accent">Data-Driven.</span>
        <span class="plain">Transparent.</span>
      </div>

      <div class="sub">
        Mollecul surfaces high-potential listings with cleaner market signals,
        stronger visual comparison, and faster insight into where opportunity
        actually lives.
      </div>

      <div class="chip-row">
        <div class="chip">
          <div class="chip-label">Markets Covered</div>
          <div class="chip-value teal">DFW Metro</div>
        </div>
        <div class="chip">
          <div class="chip-label">Data Source</div>
          <div class="chip-value">Live MLS</div>
        </div>
        <div class="chip">
          <div class="chip-label">Score Layer</div>
          <div class="chip-value">AI-Powered</div>
        </div>
      </div>
    </div>
  </div>
</div>

<script type="module">
// ── Liquid Ether fluid simulation (Three.js via CDN) ──────────────────────────
import * as THREE from 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.module.js';

const COLORS = ['#0a0f1e', '#1a3a6e', '#2ce4df', '#5ea6ff'];
const container = document.getElementById('hero');

// ── palette texture ──
function makePaletteTex(stops) {
  const data = new Uint8Array(stops.length * 4);
  stops.forEach((hex, i) => {
    const c = new THREE.Color(hex);
    data[i*4]   = Math.round(c.r*255);
    data[i*4+1] = Math.round(c.g*255);
    data[i*4+2] = Math.round(c.b*255);
    data[i*4+3] = 255;
  });
  const t = new THREE.DataTexture(data, stops.length, 1, THREE.RGBAFormat);
  t.magFilter = t.minFilter = THREE.LinearFilter;
  t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
  t.generateMipmaps = false;
  t.needsUpdate = true;
  return t;
}

const paletteTex = makePaletteTex(COLORS);

// ── renderer ──
const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('liquid-canvas'), antialias:false, alpha:false });
renderer.autoClear = false;
renderer.setClearColor(0x06111f, 1);
renderer.setPixelRatio(Math.min(window.devicePixelRatio||1, 2));

let W = container.offsetWidth, H = container.offsetHeight;
renderer.setSize(W, H);

const clock = new THREE.Clock(); clock.start();

// ── FBO helpers ──
function makeFBO(w, h) {
  const type = /iPad|iPhone|iPod/.test(navigator.userAgent) ? THREE.HalfFloatType : THREE.FloatType;
  return new THREE.WebGLRenderTarget(w, h, {
    type, depthBuffer:false, stencilBuffer:false,
    minFilter:THREE.LinearFilter, magFilter:THREE.LinearFilter,
    wrapS:THREE.ClampToEdgeWrapping, wrapT:THREE.ClampToEdgeWrapping
  });
}

const RES = 0.45;
let fw = Math.max(1, Math.round(W*RES)), fh = Math.max(1, Math.round(H*RES));
const fbos = {};
['v0','v1','vv0','vv1','div','p0','p1'].forEach(k => fbos[k] = makeFBO(fw, fh));

const cellScale = new THREE.Vector2(1/fw, 1/fh);
const fboSize   = new THREE.Vector2(fw, fh);

// ── shader source ──
const FACE_VERT = `
attribute vec3 position;
uniform vec2 boundarySpace;
varying vec2 uv;
precision highp float;
void main(){
  vec2 scale = 1.0 - boundarySpace*2.0;
  vec3 pos = position; pos.xy *= scale;
  uv = vec2(0.5)+pos.xy*0.5;
  gl_Position = vec4(pos,1.0);
}`;

const ADV_FRAG = `
precision highp float;
uniform sampler2D velocity; uniform float dt; uniform bool isBFECC;
uniform vec2 fboSize; uniform vec2 px;
varying vec2 uv;
void main(){
  vec2 ratio = max(fboSize.x,fboSize.y)/fboSize;
  if(!isBFECC){
    vec2 v=texture2D(velocity,uv).xy;
    gl_FragColor=vec4(texture2D(velocity,uv-v*dt*ratio).xy,0,0);
  } else {
    vec2 v0=texture2D(velocity,uv).xy;
    vec2 s0=uv-v0*dt*ratio;
    vec2 v1=texture2D(velocity,s0).xy;
    vec2 s2=s0+v1*dt*ratio;
    vec2 err=(s2-uv)/2.0;
    vec2 s3=uv-err;
    vec2 v2=texture2D(velocity,s3).xy;
    gl_FragColor=vec4(texture2D(velocity,s3-v2*dt*ratio).xy,0,0);
  }
}`;

const DIV_FRAG = `
precision highp float;
uniform sampler2D velocity; uniform vec2 px; uniform float dt;
varying vec2 uv;
void main(){
  float x0=texture2D(velocity,uv-vec2(px.x,0)).x;
  float x1=texture2D(velocity,uv+vec2(px.x,0)).x;
  float y0=texture2D(velocity,uv-vec2(0,px.y)).y;
  float y1=texture2D(velocity,uv+vec2(0,px.y)).y;
  gl_FragColor=vec4((x1-x0+y1-y0)/2.0/dt);
}`;

const PSN_FRAG = `
precision highp float;
uniform sampler2D pressure; uniform sampler2D divergence; uniform vec2 px;
varying vec2 uv;
void main(){
  float p0=texture2D(pressure,uv+vec2(px.x*2.0,0)).r;
  float p1=texture2D(pressure,uv-vec2(px.x*2.0,0)).r;
  float p2=texture2D(pressure,uv+vec2(0,px.y*2.0)).r;
  float p3=texture2D(pressure,uv-vec2(0,px.y*2.0)).r;
  float d=texture2D(divergence,uv).r;
  gl_FragColor=vec4((p0+p1+p2+p3)/4.0-d);
}`;

const PRESS_FRAG = `
precision highp float;
uniform sampler2D pressure; uniform sampler2D velocity; uniform vec2 px; uniform float dt;
varying vec2 uv;
void main(){
  float p0=texture2D(pressure,uv+vec2(px.x,0)).r;
  float p1=texture2D(pressure,uv-vec2(px.x,0)).r;
  float p2=texture2D(pressure,uv+vec2(0,px.y)).r;
  float p3=texture2D(pressure,uv-vec2(0,px.y)).r;
  vec2 v=texture2D(velocity,uv).xy;
  gl_FragColor=vec4(v-vec2(p0-p1,p2-p3)*0.5*dt,0,1);
}`;

const FORCE_VERT = `
precision highp float;
attribute vec3 position; attribute vec2 uv;
uniform vec2 center; uniform vec2 scale; uniform vec2 px;
varying vec2 vUv;
void main(){
  vec2 pos=position.xy*scale*2.0*px+center;
  vUv=uv; gl_Position=vec4(pos,0,1);
}`;

const FORCE_FRAG = `
precision highp float;
uniform vec2 force;
varying vec2 vUv;
void main(){
  vec2 c=(vUv-0.5)*2.0; float d=1.0-min(length(c),1.0); d*=d;
  gl_FragColor=vec4(force*d,0,1);
}`;

const COLOR_FRAG = `
precision highp float;
uniform sampler2D velocity; uniform sampler2D palette;
varying vec2 uv;
void main(){
  vec2 v=texture2D(velocity,uv).xy;
  float l=clamp(length(v),0.0,1.0);
  gl_FragColor=vec4(texture2D(palette,vec2(l,0.5)).rgb,1.0);
}`;

// ── pass factory ──
const cam = new THREE.Camera();
function makePass(vSrc, fSrc, uniforms, output) {
  const sc = new THREE.Scene();
  const mat = new THREE.RawShaderMaterial({ vertexShader:vSrc, fragmentShader:fSrc, uniforms });
  sc.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2), mat));
  return { sc, mat, uniforms,
    run(tgt){ renderer.setRenderTarget(tgt||null); renderer.render(sc,cam); renderer.setRenderTarget(null); }
  };
}

const BSpace = new THREE.Vector2(1/fw, 1/fh);

const advPass = makePass(FACE_VERT, ADV_FRAG, {
  boundarySpace:{value:BSpace}, px:{value:cellScale}, fboSize:{value:fboSize},
  velocity:{value:fbos.v0.texture}, dt:{value:0.014}, isBFECC:{value:true}
});

const divPass = makePass(FACE_VERT, DIV_FRAG, {
  boundarySpace:{value:BSpace}, velocity:{value:fbos.v1.texture},
  px:{value:cellScale}, dt:{value:0.014}
});

const psnPass = makePass(FACE_VERT, PSN_FRAG, {
  boundarySpace:{value:BSpace}, px:{value:cellScale},
  pressure:{value:fbos.p0.texture}, divergence:{value:fbos.div.texture}
});

const pressPass = makePass(FACE_VERT, PRESS_FRAG, {
  boundarySpace:{value:BSpace}, px:{value:cellScale}, dt:{value:0.014},
  pressure:{value:fbos.p0.texture}, velocity:{value:fbos.v1.texture}
});

const forcePass = makePass(FORCE_VERT, FORCE_FRAG, {
  px:{value:cellScale}, force:{value:new THREE.Vector2()},
  center:{value:new THREE.Vector2()}, scale:{value:new THREE.Vector2(100,100)}
});
forcePass.mat.blending = THREE.AdditiveBlending;
forcePass.mat.depthWrite = false;

const colorPass = makePass(FACE_VERT, COLOR_FRAG, {
  boundarySpace:{value:new THREE.Vector2()},
  velocity:{value:fbos.v0.texture}, palette:{value:paletteTex}
});

// ── mouse / auto-drive ──
const mouse    = new THREE.Vector2(0,0);
const mouseOld = new THREE.Vector2(0,0);
const mouseDiff= new THREE.Vector2(0,0);
let autoT = 0;
const autoTarget = new THREE.Vector2();
const autoCurrent = new THREE.Vector2();
function pickTarget(){ autoTarget.set((Math.random()*2-1)*0.8,(Math.random()*2-1)*0.8); }
pickTarget();

container.addEventListener('mousemove', e => {
  const r = container.getBoundingClientRect();
  mouse.set((e.clientX-r.left)/r.width*2-1, -((e.clientY-r.top)/r.height*2-1));
});
container.addEventListener('touchmove', e => {
  const t = e.touches[0], r = container.getBoundingClientRect();
  mouse.set((t.clientX-r.left)/r.width*2-1, -((t.clientY-r.top)/r.height*2-1));
}, {passive:true});

// ── resize ──
function onResize() {
  W = container.offsetWidth; H = container.offsetHeight;
  renderer.setSize(W, H);
  fw = Math.max(1, Math.round(W*RES)); fh = Math.max(1, Math.round(H*RES));
  Object.values(fbos).forEach(f => f.setSize(fw, fh));
  cellScale.set(1/fw, 1/fh); fboSize.set(fw, fh);
}
window.addEventListener('resize', onResize);

// ── render loop ──
const DT = 0.014; const ITER = 24;
let frame = 0;

function render() {
  requestAnimationFrame(render);
  const dt = Math.min(clock.getDelta(), 0.04);

  // auto-drive cursor
  autoT += dt * 0.38;
  const d = new THREE.Vector2().subVectors(autoTarget, autoCurrent);
  if (d.length() < 0.02) pickTarget();
  autoCurrent.addScaledVector(d.normalize(), Math.min(d.length(), 0.38*dt));
  mouse.copy(autoCurrent);

  mouseDiff.subVectors(mouse, mouseOld);
  mouseOld.copy(mouse);

  // advection
  advPass.uniforms.velocity.value = fbos.v0.texture;
  advPass.uniforms.dt.value = DT;
  advPass.run(fbos.v1);

  // force
  const fScale = 80;
  forcePass.uniforms.force.value.set(mouseDiff.x*fScale/2, mouseDiff.y*fScale/2);
  const cx = Math.max(-0.98, Math.min(0.98, mouse.x));
  const cy = Math.max(-0.98, Math.min(0.98, mouse.y));
  forcePass.uniforms.center.value.set(cx, cy);
  forcePass.run(fbos.v1);

  // divergence
  divPass.uniforms.velocity.value = fbos.v1.texture;
  divPass.run(fbos.div);

  // poisson pressure
  for (let i=0; i<ITER; i++) {
    const pIn  = i%2===0 ? fbos.p0 : fbos.p1;
    const pOut = i%2===0 ? fbos.p1 : fbos.p0;
    psnPass.uniforms.pressure.value = pIn.texture;
    psnPass.run(pOut);
  }
  const finalP = ITER%2===0 ? fbos.p1 : fbos.p0;

  // pressure projection
  pressPass.uniforms.velocity.value = fbos.v1.texture;
  pressPass.uniforms.pressure.value = finalP.texture;
  pressPass.run(fbos.v0);

  // color output
  colorPass.uniforms.velocity.value = fbos.v0.texture;
  colorPass.run(null);

  frame++;
}
render();
</script>
</body>
</html>
"""

components.html(HERO_HTML, height=610, scrolling=False)

# =============================================================================
# APP BODY
# =============================================================================
st.markdown("<div class='app-shell'>", unsafe_allow_html=True)

available_cities = sorted(df["CITY"].dropna().unique().tolist())
default_city = "Dallas" if "Dallas" in available_cities else available_cities[0]
target_city = default_city
city_df = df[df["CITY"].str.lower() == target_city.lower()].copy()

# ── Search resolution (must happen before charts so searched_property is defined) ──
import re as _re

def _normalize_addr(s):
    return _re.sub(r"\s+", " ", str(s).lower().strip())

# Read from session state (widget value stored from previous render)
address_query = st.session_state.get("map_address_search", "")

searched_property = None
search_feedback   = None
easy_predict_result = None  # will hold {current, future, diff, pct, error}
search_error_message = None

if address_query.strip():
    query_norm = _normalize_addr(address_query)
    cache_key = f"attom_property_{query_norm}"
    if cache_key not in st.session_state:
        with st.spinner("🔎 Looking up the property in ATTOM and running AI valuation…"):
            try:
                st.session_state[cache_key] = _lookup_attom_property_bundle(address_query.strip())
            except Exception as e:
                st.session_state[cache_key] = {"property": None, "prediction": None, "error": str(e)}

    search_bundle = st.session_state[cache_key]
    if search_bundle.get("property") is not None:
        searched_property = search_bundle["property"]
        easy_predict_result = search_bundle.get("prediction")
        search_feedback = "found"
    else:
        search_feedback = "not_found"
        search_error_message = search_bundle.get("error")

# ── Map ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='dash-section'>
  <div class='dash-section-bar'></div>
  <div>
    <div class='dash-section-label'>Explorer</div>
    <div class='dash-section-title'>Interactive Property Map</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='legend-row'>
  <div class='legend-pill'><span class='legend-dot' style='background:#2ce4df;'></span>Above median price</div>
  <div class='legend-pill'><span class='legend-dot' style='background:#e7c65a;'></span>Below median price</div>
</div>
""", unsafe_allow_html=True)

# ── Address search bar (visible here above map; resolved earlier for charts) ────
search_col, _ = st.columns([2, 3])
with search_col:
    st.text_input(
        label="",
        placeholder="🔍  Search an address to zoom in…",
        key="map_address_search",
        label_visibility="collapsed",
    )

# Feedback
if search_feedback == "not_found":
    detail = ""
    if search_error_message:
        safe_error = str(search_error_message).replace("\n", " ").replace("<", "&lt;").replace(">", "&gt;")
        detail = f" {safe_error}"
    st.markdown(
        "<div style='margin-top:-8px;margin-bottom:8px;font-size:12px;color:#e7c65a;'>"
        f"⚠️ ATTOM could not find that property.{detail}</div>",
        unsafe_allow_html=True,
    )
elif search_feedback == "found" and searched_property is not None:
    addr_display  = searched_property.get("ADDRESS", "")
    city_display  = searched_property.get("CITY", "")
    price_label   = searched_property.get("PRICE_LABEL", "Price Reference")
    price_val_sp  = searched_property.get("PRICE")
    price_display = f"${price_val_sp:,.0f}" if pd.notna(price_val_sp) else "N/A"

    if easy_predict_result and easy_predict_result.get("current"):
        pred_display = f"${easy_predict_result['current']:,.0f}"
    else:
        pred_display = "Calculating…"

    st.markdown(
        f"<div style='margin-top:-8px;margin-bottom:8px;font-size:12px;color:#2ce4df;'>"
        f"✓ Found: <b>{addr_display}</b>, {city_display} — "
        f"{price_label}: {price_display} &nbsp;|&nbsp; "
        f"<span style='color:#b89eff;font-weight:700;'>AI Current Value: {pred_display}</span></div>",
        unsafe_allow_html=True,
    )

city_coords = {
    "Dallas": [32.7767, -96.7970], "Fort Worth": [32.7555, -97.3308],
    "Arlington": [32.7357, -97.1081], "Plano": [33.0198, -96.6989],
    "Frisco": [33.1507, -96.8236], "Irving": [32.8140, -96.9489],
    "Richardson": [32.9483, -96.7299], "Garland": [32.9126, -96.6389],
    "Grand Prairie": [32.7459, -96.9978], "Mansfield": [32.5632, -97.1417],
    "Austin": [30.2672, -97.7431], "Houston": [29.7604, -95.3698],
    "Allen": [33.1032, -96.6706],
}

searched_has_coords = (
    searched_property is not None
    and pd.notna(searched_property.get("LATITUDE"))
    and pd.notna(searched_property.get("LONGITUDE"))
)

if searched_property is not None:
    _map_city = searched_property.get("CITY") or target_city
    map_source_df = df[df["CITY"].str.lower() == _map_city.lower()].copy() if _map_city else city_df.copy()
    if map_source_df.empty:
        map_source_df = city_df.copy()
    if searched_has_coords:
        location = [searched_property["LATITUDE"], searched_property["LONGITUDE"]]
        map_zoom = 16
    else:
        location = city_coords.get(_map_city, city_coords.get(target_city, [city_df["LATITUDE"].mean(), city_df["LONGITUDE"].mean()]))
        map_zoom = 11
    map_key_sfx = f"search_{address_query}"
else:
    map_source_df = city_df.copy()
    location      = city_coords.get(target_city, city_coords.get(target_city.title(),
                    [city_df["LATITUDE"].mean(), city_df["LONGITUDE"].mean()]))
    map_zoom      = 11
    map_key_sfx   = target_city

map_df = map_source_df.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()
if len(map_df) > 400:
    map_df = map_df.sample(400, random_state=42)
price_med = map_df["PRICE"].median() if "PRICE" in map_df.columns and map_df["PRICE"].notna().any() else 0

fmap = folium.Map(location=location, zoom_start=map_zoom, tiles="CartoDB dark_matter")

for _, row in map_df.iterrows():
    price_val = row.get("PRICE") if "PRICE" in row else None
    pred_val  = row.get("PREDICTED_VALUE") if "PREDICTED_VALUE" in row else None
    avm_val   = row.get("avmValue")
    price_label = row.get("PRICE_LABEL", "Latest Sale Price" if pd.notna(price_val) else "ATTOM AVM")
    above = (pd.notna(price_val) and price_val >= price_med)
    mc = "#2ce4df" if above else "#e7c65a"
    parts = []
    if "ADDRESS" in row and pd.notna(row["ADDRESS"]):
        parts.append(f"<b style='color:#2ce4df'>{row['ADDRESS']}</b>")
    if pd.notna(price_val):
        parts.append(f"{price_label}: <b>${price_val:,.0f}</b>")
    if pd.notna(avm_val):
        parts.append(f"ATTOM AVM: <b>${avm_val:,.0f}</b>")
    if pd.notna(pred_val):
        parts.append(f"<span style='color:#b89eff;font-weight:600;'>AI Value: ${pred_val:,.0f}</span>")
    if "BEDS"        in row and pd.notna(row["BEDS"]):        parts.append(f"Beds: {int(row['BEDS'])}")
    if "BATHS"       in row and pd.notna(row["BATHS"]):       parts.append(f"Baths: {row['BATHS']}")
    if "SQUARE FEET" in row and pd.notna(row["SQUARE FEET"]): parts.append(f"Sq Ft: {row['SQUARE FEET']:,.0f}")
    popup_html = (
        "<div style='font-family:DM Sans,sans-serif;font-size:13px;"
        "background:#0b1c30;color:#ecf6ff;padding:10px 14px;"
        "border-radius:10px;border:1px solid rgba(44,228,223,0.18);"
        "min-width:170px;line-height:1.7;'>"
        + "<br>".join(parts) + "</div>"
    )
    folium.CircleMarker(
        [row["LATITUDE"], row["LONGITUDE"]],
        radius=5.5,
        popup=folium.Popup(popup_html, max_width=260),
        tooltip=_compose_display_address(row.get("ADDRESS"), row.get("CITY"), row.get("zip")),
        color=mc, fill=True, fill_color=mc,
        fill_opacity=0.88 if above else 0.72, weight=1.4,
    ).add_to(fmap)

# If a searched property exists, add a glowing highlighted marker on top
if searched_property is not None and searched_has_coords:
    sp = searched_property
    price_label = sp.get("PRICE_LABEL", "Price Reference")
    sp_price = sp.get("PRICE")
    sp_parts = []
    if "ADDRESS" in sp and pd.notna(sp["ADDRESS"]):
        sp_parts.append(f"<b style='color:#2ce4df'>{sp['ADDRESS']}</b>")
    if pd.notna(sp_price):
        sp_parts.append(f"{price_label}: <b>${sp_price:,.0f}</b>")
    if pd.notna(sp.get("ATTOM_AVM")):
        sp_parts.append(f"ATTOM AVM: <b>${sp['ATTOM_AVM']:,.0f}</b>")
    if pd.notna(sp.get("PREDICTED_VALUE")):
        sp_parts.append(f"<span style='color:#b89eff;font-weight:600;'>AI Value: ${sp['PREDICTED_VALUE']:,.0f}</span>")
    if "BEDS"        in sp and pd.notna(sp.get("BEDS")):        sp_parts.append(f"Beds: {int(sp['BEDS'])}")
    if "BATHS"       in sp and pd.notna(sp.get("BATHS")):       sp_parts.append(f"Baths: {sp['BATHS']}")
    if "SQUARE FEET" in sp and pd.notna(sp.get("SQUARE FEET")): sp_parts.append(f"Sq Ft: {sp['SQUARE FEET']:,.0f}")
    sp_popup_html = (
        "<div style='font-family:DM Sans,sans-serif;font-size:13px;"
        "background:#0b1c30;color:#ecf6ff;padding:10px 14px;"
        "border-radius:10px;border:2px solid #2ce4df;"
        "min-width:170px;line-height:1.7;'>"
        + "<br>".join(sp_parts) + "</div>"
    )
    # Outer glow ring
    folium.CircleMarker(
        [sp["LATITUDE"], sp["LONGITUDE"]],
        radius=18, color="#2ce4df", fill=True, fill_color="#2ce4df",
        fill_opacity=0.12, weight=2, opacity=0.5,
    ).add_to(fmap)
    # Inner highlighted dot
    folium.CircleMarker(
        [sp["LATITUDE"], sp["LONGITUDE"]],
        radius=9, color="#ffffff", fill=True, fill_color="#2ce4df",
        fill_opacity=1.0, weight=2.5,
        popup=folium.Popup(sp_popup_html, max_width=280),
        tooltip=_to_text(sp.get("ADDRESS")),
    ).add_to(fmap)

map_col, stat_col = st.columns([3, 1], gap="medium")

with map_col:
    st.markdown("<div class='map-wrap'>", unsafe_allow_html=True)
    map_state = st_folium(
        fmap,
        width="100%",
        height=400,
        key=f"map_{map_key_sfx}",
        returned_objects=["last_object_clicked"],
    )
    st.markdown("</div>", unsafe_allow_html=True)

clicked_property, clicked_prediction = _resolve_clicked_property(
    map_state,
    map_df,
    searched_property,
    easy_predict_result,
)

panel_state_key = "active_property_panel_state"
stored_panel_state = st.session_state.get(panel_state_key)
if stored_panel_state and stored_panel_state.get("map_context") != map_key_sfx:
    stored_panel_state = None
    st.session_state.pop(panel_state_key, None)

if clicked_property is not None:
    stored_panel_state = {
        "property": clicked_property,
        "prediction": clicked_prediction,
        "source": "map",
        "map_context": map_key_sfx,
    }
    st.session_state[panel_state_key] = stored_panel_state
elif searched_property is not None and (stored_panel_state is None or stored_panel_state.get("source") == "search"):
    stored_panel_state = {
        "property": searched_property,
        "prediction": easy_predict_result,
        "source": "search",
        "map_context": map_key_sfx,
    }
    st.session_state[panel_state_key] = stored_panel_state
elif searched_property is None and stored_panel_state is None:
    st.session_state.pop(panel_state_key, None)

active_property = stored_panel_state.get("property") if stored_panel_state else None
active_prediction = stored_panel_state.get("prediction") if stored_panel_state else None

with stat_col:
    above_count = int((map_df["PRICE"] >= price_med).sum())
    below_count = int((map_df["PRICE"] < price_med).sum())
    max_price = map_df["PRICE"].max()
    min_price = map_df["PRICE"].min()

    if active_property is not None:
        st.markdown(_render_property_panel(active_property, active_prediction), unsafe_allow_html=True)
    else:
        # Default stats when no property is searched
        st.markdown(f"""
<div style="display:flex;flex-direction:column;gap:12px;padding:4px 0;">
  <div style="background:#0b1c30;border:1px solid rgba(44,228,223,0.10);border-radius:14px;padding:16px 18px;">
    <div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.14em;color:#567592;margin-bottom:6px;">Mapped Listings</div>
    <div style="font-family:'Sora',sans-serif;font-size:26px;font-weight:800;color:#ecf6ff;letter-spacing:-0.04em;">{len(map_df):,}</div>
    <div style="font-size:11px;color:#567592;margin-top:2px;">properties plotted</div>
  </div>
  <div style="background:#0b1c30;border:1px solid rgba(44,228,223,0.10);border-radius:14px;padding:16px 18px;">
    <div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.14em;color:#567592;margin-bottom:10px;">Price Distribution</div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
      <span style="width:8px;height:8px;border-radius:50%;background:#2ce4df;flex-shrink:0;box-shadow:0 0 6px #2ce4df;"></span>
      <span style="font-size:12px;color:#8baec8;">Above median</span>
      <span style="margin-left:auto;font-family:'Sora',sans-serif;font-size:13px;font-weight:700;color:#ecf6ff;">{above_count}</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
      <span style="width:8px;height:8px;border-radius:50%;background:#e7c65a;flex-shrink:0;box-shadow:0 0 6px #e7c65a88;"></span>
      <span style="font-size:12px;color:#8baec8;">Below median</span>
      <span style="margin-left:auto;font-family:'Sora',sans-serif;font-size:13px;font-weight:700;color:#ecf6ff;">{below_count}</span>
    </div>
  </div>
  <div style="background:#0b1c30;border:1px solid rgba(44,228,223,0.10);border-radius:14px;padding:16px 18px;">
    <div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.14em;color:#567592;margin-bottom:10px;">Price Range</div>
    <div style="font-size:11px;color:#567592;margin-bottom:3px;">Highest</div>
    <div style="font-family:'Sora',sans-serif;font-size:13px;font-weight:700;color:#2ce4df;margin-bottom:8px;">${max_price:,.0f}</div>
    <div style="font-size:11px;color:#567592;margin-bottom:3px;">Lowest</div>
    <div style="font-family:'Sora',sans-serif;font-size:13px;font-weight:700;color:#e7c65a;">${min_price:,.0f}</div>
  </div>
  <div style="background:#0b1c30;border:1px solid rgba(44,228,223,0.10);border-radius:14px;padding:16px 18px;">
    <div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.14em;color:#567592;margin-bottom:6px;">Median Price</div>
    <div style="font-family:'Sora',sans-serif;font-size:18px;font-weight:800;color:#ecf6ff;letter-spacing:-0.04em;">${price_med:,.0f}</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='footer-note'>MOLLECUL · AI Real Estate Intelligence · DFW Metro</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
