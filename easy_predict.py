import sys
from predict_address import get_property_data, build_feature_row
from round2_model import predict_forward
from model_logic import load_model, predict as stage1_predict

# ── Keys ───────────────────────────────────────────────────────────────────────
ATTOM_KEY  = "38c0c6bd6a465da7b414bc23a5df9791"
FRED_KEY   = "3edb96e23b26e2271758689309faee32"
NOAA_KEY   = "xyeXCwTZHCBtzcdEHwQpnNvsfTyiWmGq"
EPA_KEY    = "aquaosprey52"
MODEL_PATH = "VER4_property_valuation_model.joblib"

# ── Factor name translations ───────────────────────────────────────────────────
FACTOR_TRANSLATIONS = {
    "mortgage_rate_30yr":         "Mortgage Interest Rates",
    "mortgage_rate_30yr_lag1":    "Recent Mortgage Rate Changes",
    "mortgage_rate_30yr_lag2":    "Mortgage Rate Trend (2mo)",
    "mortgage_rate_30yr_lag3":    "Mortgage Rate Trend (3mo)",
    "mortgage_rate_mom":          "Month-over-Month Rate Change",
    "mortgage_rate_3mo_change":   "3-Month Rate Shock",
    "affordability_stress":       "Overall Affordability Pressure",
    "vix":                        "Stock Market Fear / Volatility",
    "vix_lag1":                   "Recent Market Fear",
    "case_shiller_dallas":        "Dallas Housing Price Level",
    "cs_dallas_mom_pct":          "Dallas Price Momentum (1mo)",
    "cs_dallas_3mo_pct":          "Dallas Price Trend (3mo)",
    "cs_dallas_6mo_pct":          "Dallas Price Trend (6mo)",
    "cs_momentum_divergence":     "Short vs Medium-Term Price Divergence",
    "fed_funds_rate":             "Federal Reserve Rate",
    "fed_funds_rate_lag1":        "Recent Fed Policy Changes",
    "fed_funds_rate_lag2":        "Fed Rate Trend (2mo)",
    "rate_vs_10yr":               "Fed Rate vs Long-Term Bonds",
    "is_summer":                  "Summer Buying Season",
    "is_spring":                  "Spring Buying Season",
    "quarter":                    "Time of Year (Seasonal)",
    "month":                      "Month of Year",
    "housing_starts_south":       "New Homes Being Built (South US)",
    "yield_spread":               "Bond Market Health",
    "treasury_10yr":              "10-Year Treasury Rate",
    "new_home_sales":             "New Home Sales Volume",
    "labor_force_part_texas":     "Texas Labor Market Strength",
    "wage_growth_texas":          "Texas Wage Growth",
    "unemployment_texas":         "Texas Unemployment",
    "cpi_shelter":                "Housing Cost Inflation",
    "sp500_return":               "Stock Market Returns",
    "sp500_return_lag1":          "Recent Stock Market Performance",
    "homebuilder_etf_return":     "Homebuilder Stock Performance",
    "homebuilder_etf_return_lag1":"Recent Homebuilder Momentum",
    "oil_wti":                    "Oil Price (Texas Economy Signal)",
}


def run_simple_prediction(address: str):
    print(f"\n🔍 Looking up property: {address}...\n")

    # ── Stage 1: get base estimate + property details ─────────────────────────
    try:
        raw_data = get_property_data(address)
    except Exception as e:
        print(f"❌ Could not retrieve property data: {e}")
        return

    df_features = build_feature_row(raw_data)
    try:
        pipeline  = load_model(MODEL_PATH)
        base_value = float(stage1_predict(pipeline, df_features)[0])
    except Exception as e:
        print(f"❌ Stage 1 model failed: {e}")
        return

    # Pull property-specific context for Stage 2 beta adjustment
    zip_code      = str(raw_data.get("zip") or "").strip() or None
    property_type = str(raw_data.get("propertyType") or "").strip() or None
    prop_subtype  = str(raw_data.get("propSubtype")  or "").strip() or None
    city          = str(raw_data.get("city") or "").strip()

    # ── Stage 2: macro + property-adjusted forecast ───────────────────────────
    try:
        result = predict_forward(
            base_estimate  = base_value,
            fred_api_key   = FRED_KEY,
            noaa_api_key   = NOAA_KEY,
            epa_api_key    = EPA_KEY,
            zip_code       = zip_code,
            property_type  = property_type,
            verbose        = False,
        )
    except Exception as e:
        print(f"❌ Stage 2 forecast failed: {e}")
        return

    current  = base_value
    future   = result.get("forward_estimate", base_value)
    diff     = result.get("change_dollars", 0)
    pct      = result.get("change_pct", 0)
    macro_pct= result.get("macro_pct", pct)

    # ── Market Weather ────────────────────────────────────────────────────────
    if pct >= 3.0:
        trend = "🔥 HOT MARKET — Prices rising fast. Great for sellers!"
    elif pct >= 1.0:
        trend = "🌤️  WARM MARKET — Prices trending up."
    elif pct >= -1.0:
        trend = "☁️  FLAT MARKET — Prices holding steady."
    elif pct >= -3.0:
        trend = "🌧️  COOLING MARKET — Mild downward pressure."
    else:
        trend = "❄️  COLD MARKET — Significant price softening expected."

    # ── Print Dashboard ───────────────────────────────────────────────────────
    print("=" * 62)
    print("         🏡  MOLLECUL PROPERTY FORECAST  🏡")
    print("=" * 62)
    print(f"📍  Address:       {address}")
    if city:
        prop_label = f"{property_type or 'Property'}" + (f" / {prop_subtype}" if prop_subtype and prop_subtype != property_type else "")
        print(f"🏠  Type:          {prop_label}  |  ZIP: {zip_code or 'N/A'}  |  {city}")
    print()
    print(f"💰  TODAY'S VALUE:  ${current:>12,.0f}")
    print(f"🔮  IN 6 MONTHS:   ${future:>12,.0f}")

    if diff >= 0:
        print(f"📈  EXPECTED:       UP  ${diff:>10,.0f}  (+{pct:.2f}%)")
    else:
        print(f"📉  EXPECTED:       DOWN ${abs(diff):>9,.0f}  ({pct:.2f}%)")

    # Show confidence range if available
    lo = result.get("estimate_low")
    hi = result.get("estimate_high")
    if lo and hi and lo != hi:
        print(f"📊  RANGE (80%):   ${lo:>12,.0f}  –  ${hi:>12,.0f}")

    print()
    print(f"🌡️  MARKET WEATHER: {trend}")
    print("=" * 62)

    # ── Property-specific breakdown ───────────────────────────────────────────
    price_tier    = result.get("price_tier", "")
    zip_tier      = result.get("zip_tier", "")
    combined_beta = result.get("combined_beta", 1.0)

    print(f"\n📐 HOW THIS FORECAST WAS PERSONALISED:")
    print(f"   DFW market signal:    {macro_pct:+.2f}%  (same for all properties today)")
    print(f"   Price tier:           {price_tier}")
    print(f"   ZIP neighbourhood:    {zip_tier}  ({zip_code or 'unknown ZIP'})")
    print(f"   Combined adjustment:  ×{combined_beta:.2f}  →  {pct:+.2f}% (your property)")

    # ── Top 3 macro drivers ───────────────────────────────────────────────────
    factors = result.get("shap_factors", [])
    if factors:
        print(f"\n💡 TOP 3 MARKET FACTORS DRIVING THIS FORECAST:")
        for f in factors[:3]:
            name   = FACTOR_TRANSLATIONS.get(f["feature"], "Economic Conditions")
            impact = f["shap_dollar"]
            arrow  = "🟢" if impact > 0 else "🔴"
            direction = "pushing value UP" if impact > 0 else "pulling value DOWN"
            print(f"   {arrow}  {name} is {direction}.")

    print()


if __name__ == "__main__":
    target_address = "3205 Walker Dr, Richardson, TX, 75082"

    if len(sys.argv) > 1:
        target_address = " ".join(sys.argv[1:])

    run_simple_prediction(target_address)