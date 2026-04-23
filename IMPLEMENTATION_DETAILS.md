# MOLLECUL Implementation Details

## 1. Project Summary

MOLLECUL is a Python and Streamlit real estate dashboard for the DFW market. It lets users explore properties on a map, search an address, estimate current property value, forecast value six months forward, and view simple explanations for the predictions.

Run the app with:

```bash
streamlit run app.py
```

## 2. Main Features

- Interactive DFW property map.
- Address search using ATTOM property data.
- Current-value estimate using a trained XGBoost model.
- Six-month value forecast using macroeconomic and market data.
- Sale price, ATTOM AVM, and MOLLECUL model comparison.
- Feature impact explanation for the valuation model.
- Forecast driver explanation for the six-month forecast.

## 3. Technology Stack

| Area | Tools |
| --- | --- |
| App | Streamlit |
| Map | Folium, streamlit-folium |
| Data | pandas, numpy |
| Machine learning | scikit-learn, XGBoost |
| Models | joblib |
| APIs | ATTOM, FRED, NOAA, EPA, USDA, FEMA, yfinance |

## 4. Simple Architecture Diagram

```text
User
  |
  v
Streamlit App (app.py)
  |
  +--> Property Map
  |
  +--> Address Search --> ATTOM API
                            |
                            v
                    Property Feature Data
                            |
                            v
              Stage 1: Current Value Model
                            |
                            v
              Stage 2: Six-Month Forecast
                            |
                            v
          Dashboard Results and Explanations
```

## 5. How the System Works

1. The app loads local property CSV files and saved model files.
2. The user clicks a map property or searches an address.
3. If the user searches an address, the app gets property details from ATTOM.
4. The property data is cleaned into the format expected by the model.
5. The Stage 1 model estimates the current property value.
6. The Stage 2 model estimates the value six months ahead.
7. The app displays the property summary, value comparison, forecast, and explanation panels.

## 6. Code Documentation

| File | Purpose |
| --- | --- |
| `app.py` | Main dashboard, UI, map, search, predictions, and explanation sections |
| `model_logic.py` | Builds, trains, saves, and runs the current-value model |
| `round2_model.py` | Builds, trains, saves, and runs the six-month forecast model |
| `macro_data.py` | Collects and caches macroeconomic and market data |
| `property_beta.py` | Adjusts forecasts by price tier, ZIP code, and property type |
| `predict_address.py` | Gets ATTOM property data and converts it into model features |
| `easy_predict.py` | Simple command-line prediction example |
| `scraper.py` | Pulls ATTOM property data by ZIP code |
| `enrich.py` | Adds neighborhood, school, crime, demographic, and climate data |
| `extract_pins_only_properties.py` | Creates a smaller CSV for map pins |

## 7. Data and Model Files

| File | Purpose |
| --- | --- |
| `data/dfw_real_estate.csv` | Smaller fallback property dataset |
| `data/PART1_NEW_Dallas_Properties.csv` | Main ATTOM property dataset |
| `PinsOnlyProperties.csv` | Map-ready property pin file |
| `VER4_property_valuation_model.joblib` | Stage 1 current-value model |
| `VER4_round2_forecast_model.joblib` | Stage 2 forecast model |
| `VER4_round2_scaler.joblib` | Forecast model scaler |
| `VER4_round2_feature_cols.json` | Forecast model feature list |
| `macro_cache.parquet` | Cached macroeconomic data |

## 8. Stage 1: Current Value Model

The Stage 1 model predicts the current value of a property.

Main inputs:

- Location: ZIP, city, latitude, longitude, neighborhood IDs.
- Property details: square feet, lot size, beds, baths, year built.
- Building details: garage, pool, condition, construction type.
- Financial details: assessed value, taxes, sale price, price per square foot.

Basic flow:

```text
Property Data -> Clean Features -> XGBoost Model -> Current Value
```

## 9. Stage 2: Six-Month Forecast Model

The Stage 2 model predicts how the value may change over the next six months.

Main inputs:

- Mortgage rates.
- Fed funds rate.
- Treasury rates.
- Dallas Case-Shiller index.
- Texas unemployment and wage data.
- Housing starts and home sales.
- Stock market and REIT signals.
- Seasonal factors.

The forecast is adjusted for the specific property:

```text
Final Forecast = Market Signal x Price Beta x ZIP Beta x Property Type Beta
```

## 10. UI and Wireframes

The dashboard is designed as a modern real estate intelligence interface with a clear visual hierarchy: brand message first, search and map exploration second, then property insights and explanations.

Desktop layout:

| Screen Area | Layout | Main Content |
| --- | --- | --- |
| Hero Header | Full-width top section | MOLLECUL branding, product message, market focus |
| Search Bar | Full-width below header | Address input, Search button, Clear button |
| Main Explorer | Two-column layout | Large interactive map on the left, property insight panel on the right |
| Property Panel | Right sidebar | Address, beds, baths, square feet, current value, sale/AVM comparison, six-month forecast |
| Valuation Explanation | Full-width card grid | Grouped feature impacts such as location, size, structure, tax, and sale history |
| Forecast Explanation | Full-width card grid | Market drivers such as rates, housing momentum, labor data, and seasonality |

Mobile layout:

| Screen Order | Section | Main Content |
| --- | --- | --- |
| 1 | Header | MOLLECUL branding and short product description |
| 2 | Search | Address input and action buttons |
| 3 | Map | Full-width interactive map |
| 4 | Property Panel | Selected property summary and forecast |
| 5 | Explanation Cards | Valuation and forecast drivers stacked vertically |

Presentation view:

| Panel | Purpose |
| --- | --- |
| Map Panel | Helps users visually explore property opportunities by location |
| Property Insight Panel | Summarizes the selected property's key metrics and valuation |
| Comparison Panel | Shows sale price, ATTOM AVM, and MOLLECUL estimate side by side |
| Explanation Panels | Makes the AI output easier to understand and justify |

## 11. Use Cases

| Use Case | Description |
| --- | --- |
| Explore properties | User views properties on the map |
| Search address | User searches a specific address |
| Estimate value | App predicts current property value |
| Compare values | App compares sale price, ATTOM AVM, and MOLLECUL estimate |
| Forecast value | App predicts six-month future value |
| Explain valuation | App shows the strongest property-level drivers |
| Explain forecast | App shows the strongest market-level drivers |
| Prepare map data | Developer creates `PinsOnlyProperties.csv` |
| Retrain model | Developer retrains valuation or forecast models |

## 12. Test Cases

The following test cases cover the main application workflows and model outputs.

| ID | Test Case | Expected Result |
| --- | --- | --- |
| TC-01 | Start the Streamlit app | App loads successfully |
| TC-02 | Load property CSV | Data loads with expected columns |
| TC-03 | Validate data file handling | App shows a clear message if required data is unavailable |
| TC-04 | Load valuation model | Model loads successfully |
| TC-05 | Search valid address | Property data and prediction appear |
| TC-06 | Search unavailable address | App shows a clear not-found message |
| TC-07 | Click map marker | Correct property panel appears |
| TC-08 | Run current-value model | Model returns a positive numeric value |
| TC-09 | Run forecast model | Forecast returns future value and change |
| TC-10 | Generate pin CSV | Output file has expected map columns |
| TC-11 | Mobile view | Layout remains readable |

Recommended tools: `pytest`, mocked API responses, small sample CSV files, and a Streamlit startup smoke test.

## 13. Setup

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install shap yfinance python-dateutil
```

Run:

```bash
streamlit run app.py
```

Useful commands:

```bash
python easy_predict.py "4529 Wateka Dr, Dallas, TX 75209"
python model_logic.py data/PART1_NEW_Dallas_Properties.csv
python round2_model.py train --fred-key "$FRED_API_KEY"
python extract_pins_only_properties.py -o PinsOnlyProperties.csv
```

## 14. Summary

MOLLECUL combines map-based property exploration, ATTOM property data, machine learning valuation, six-month forecasting, and clear model explanations in one dashboard. The implementation is designed to make real estate analysis faster, more transparent, and easier to understand for DFW property evaluation.
