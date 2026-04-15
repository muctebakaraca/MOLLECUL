import requests
import pandas as pd
import numpy as np
import joblib
from model_logic import load_model, predict

# Configuration
ATTOM_API_KEY = "38c0c6bd6a465da7b414bc23a5df9791"
MODEL_PATH = "property_valuation_model.joblib"

def fetch_attom_detail(address1, address2):
    """Fetches full property details including AVM and Sales history."""
    url = "https://api.gateway.attomdata.com/propertyapi/v1.0.0/property/detail"
    headers = {"accept": "application/json", "apikey": ATTOM_API_KEY}
    params = {"address1": address1, "address2": address2}
    
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

def extract_features_and_meta(attom_json):
    prop = attom_json.get("property", [{}])[0]
    
    def get_path(d, keys, default=None):
        for k in keys:
            if isinstance(d, dict): d = d.get(k, {})
            else: return default
        return d if d != {} and d is not None else default

    # 1. Features (Same as before)
    features = {
        "latitude":   get_path(prop, ["location", "latitude"]),
        "longitude":  get_path(prop, ["location", "longitude"]),
        "year_built": get_path(prop, ["summary", "yearBuilt"], 1990),
        "sqft":       get_path(prop, ["building", "size", "livingSize"]),
        "lot_size":   get_path(prop, ["lot", "lotSize1"]),
        "beds":       get_path(prop, ["building", "rooms", "beds"], 3),
        "baths":      get_path(prop, ["building", "rooms", "bathsFull"], 2)
    }
    df = pd.DataFrame([features])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. Metadata - UPDATED PATHS
    # ATTOM often stores the AVM value under 'avm' -> 'dashboards' or 'indicators'
    avm_val = get_path(prop, ["avm", "indicators", "valueValue"]) or \
              get_path(prop, ["avm", "dashboards", "obv", "value"])

    # Sale price is often under 'sale' -> 'amount' -> 'saleamt'
    sale_val = get_path(prop, ["sale", "amount", "saleamt"]) or \
               get_path(prop, ["sale", "saleTransaction", "saleAmt"])

    meta = {
        "attom_avm":   pd.to_numeric(avm_val, errors='coerce'),
        "sale_price":  pd.to_numeric(sale_val, errors='coerce')
    }

    return df, meta

def run_valuation_comparison(address1, address2):
    """Executes the 3-way comparison logic on a live property."""
    try:
        # 1. Fetch & Extract
        raw_json = fetch_attom_detail(address1, address2)
        X_input, meta = extract_features_and_meta(raw_json)
        
        # 2. Predict using your XGBoost model
        model = load_model(MODEL_PATH)
        model_estimate = predict(model, X_input)[0]

        # 3. Build Comparison (Mirroring model_logic.py)
        # Using .get() to handle cases where ATTOM has no AVM or Sale on record
        avm_val = meta['attom_avm'] or np.nan
        sale_val = meta['sale_price'] or np.nan

        comparison = {
            "Model Estimate": model_estimate,
            "ATTOM AVM":      avm_val,
            "Actual Sale":    sale_val,
            "Model vs AVM":   model_estimate - avm_val if avm_val else np.nan,
            "Model vs Sale":  model_estimate - sale_val if sale_val else np.nan,
            "AVM vs Sale":    avm_val - sale_val if (avm_val and sale_val) else np.nan
        }

        # 4. Display Results
        print(f"\n--- 3-Way Comparison: {address1} ---")
        for key, val in comparison.items():
            if pd.isna(val):
                print(f"{key:<15}: Data Unavailable")
            else:
                print(f"{key:<15}: ${val:,.2f}")
        

        # Add this inside run_valuation_comparison after raw_json = fetch_attom_detail(...)
        import json
        print(json.dumps(raw_json, indent=2))
        return comparison

    except Exception as e:
        print(f"Error processing {address1}: {e}")
        return None

if __name__ == "__main__":
    # Test with a Dallas property (your model's primary training area)
    run_valuation_comparison("4529 Winona Court", "Denver, CO")