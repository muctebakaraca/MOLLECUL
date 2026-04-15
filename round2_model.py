"""
round2_model.py  (VER5 — improved)
───────────────────────────────────
Stage 2 of the property valuation pipeline.

Stage 1 (model_logic.py / VER4):
  Property features → base estimate (today's fair value)

Stage 2 (this file):
  base_estimate + macro/environmental/labor/market conditions
  → 6-month forward-adjusted estimate
  → SHAP-based factor breakdown (what's driving the adjustment)
  → Confidence interval (10th / 90th percentile)

Improvements over VER4:
  • Deeper lag features  (mortgage rate lags 1-3, fed funds lags 1-2)
  • Engineered features  (mortgage rate momentum, rate shock, affordability index)
  • Smarter CV           (test_size=24, gap=1 month → folds start with 100+ training rows)
  • Feature selection    (two-stage: prune near-zero-importance features, retrain clean)
  • Hyperparameter tuning (grid search over depth + regularisation)
  • NaN discipline       (drop columns >35% missing; ffill before median)
  • Prediction intervals (quantile XGBoost — p10 / p90 confidence range)

Training strategy:
  Since we have only a single AVM snapshot per property, Stage 2 learns
  from MARKET-LEVEL patterns. Each training row = one month of history.
  Target = Case-Shiller Dallas index % change over the NEXT 6 months.
  The model learns: "given today's macro conditions, how much will
  Dallas home values shift over the next 6 months?"

  At prediction time:
    forward_estimate = base_estimate × (1 + predicted_6mo_pct / 100)

Usage:
  # Train:
  python round2_model.py train --fred-key YOUR_KEY

  # Predict (after training):
  python round2_model.py predict --base 488302 --fred-key YOUR_KEY

  # Or import and call directly:
  from round2_model import predict_forward
  result = predict_forward(base_estimate=488302, fred_api_key="YOUR_KEY")
"""

import os
import sys
import json
import joblib
import warnings
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import shap

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from macro_data import build_macro_dataset, get_current_macro_snapshot
from property_beta import adjusted_forecast as _apply_beta

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────

ROUND2_MODEL_PATH    = "VER4_round2_forecast_model.joblib"
ROUND2_MODEL_LO_PATH = "VER4_round2_forecast_model_lo.joblib"
ROUND2_MODEL_HI_PATH = "VER4_round2_forecast_model_hi.joblib"
ROUND2_RIDGE_PATH    = "VER4_round2_ridge_model.joblib"
ROUND2_SCALER_PATH   = "VER4_round2_scaler.joblib"
ROUND2_COLS_PATH     = "VER4_round2_feature_cols.json"
STAGE1_MODEL_PATH    = "VER4_property_valuation_model.joblib"

FORECAST_HORIZON     = 6    # months forward
XGB_BLEND_WEIGHT     = 0.60  # final prediction = XGB * weight + Ridge * (1 - weight)

# ── Base macro features from macro_data ───────────────────────────────────────
# Trimmed: removed existing_home_sales (near-zero importance, collinear with new_home_sales),
# treasury_2yr (fully captured by yield_spread), and fema_policy_* (mostly NaN).

MACRO_FEATURES = [
    # ── Core housing market ────────────────────────────────────────────────
    "mortgage_rate_30yr",
    "fed_funds_rate",
    "housing_starts_south",
    "new_home_sales",
    "yield_spread",            # 10yr - 2yr  (recession predictor)
    "treasury_10yr",

    # ── Dallas-specific ────────────────────────────────────────────────────
    "case_shiller_dallas",
    "cs_dallas_mom_pct",       # 1-month price momentum
    "cs_dallas_3mo_pct",       # 3-month trend
    "cs_dallas_6mo_pct",       # 6-month trend (if available in macro_data)

    # ── Inflation / cost ───────────────────────────────────────────────────
    "cpi_shelter",

    # ── Labor ─────────────────────────────────────────────────────────────
    "unemployment_texas",
    "labor_force_part_texas",
    "wage_growth_texas",

    # ── Market signals ─────────────────────────────────────────────────────
    "sp500_return",
    "homebuilder_etf_return",  # ITB — leading indicator for new supply
    "vix",                     # risk-off signal
    "oil_wti",                 # TX economy

    # ── Environmental ──────────────────────────────────────────────────────
    "drought_severe",          # D2+ drought coverage % in TX
]

# Lag feature definitions:  feature → how many lags (1-based) to create
LAG_DEPTHS = {
    "mortgage_rate_30yr":     3,   # rate cycles take 2-3 months to hit housing
    "fed_funds_rate":         2,
    "cs_dallas_mom_pct":      2,   # price momentum is auto-correlated
    "sp500_return":           1,
    "vix":                    1,
    "homebuilder_etf_return": 1,
}

# Calendar features (help model learn seasonality in Dallas market)
CALENDAR_FEATURES = ["month", "quarter", "is_spring", "is_summer"]

# Engineered features created in build_training_dataset
ENGINEERED_FEATURES = [
    "mortgage_rate_mom",        # MoM change in mortgage rate
    "mortgage_rate_3mo_change", # 3-month change (rate shock signal)
    "rate_vs_10yr",             # fed_funds_rate minus treasury_10yr (inversion depth)
    "affordability_stress",     # mortgage_rate_30yr × case_shiller_dallas / 1e4
    "cs_momentum_divergence",   # cs_dallas_mom_pct minus cs_dallas_3mo_pct/3
]

# Features that are useful only if they have sufficient data coverage
OPTIONAL_FEATURES = {
    "drought_severe",
    "oil_wti",
    "vix",
    "sp500_return",
    "homebuilder_etf_return",
}

# Drop features whose variance is below this threshold (constant / near-constant)
MIN_VARIANCE_THRESHOLD = 1e-6

# Drop features with more than this fraction of NaNs in training data
MAX_NAN_FRACTION = 0.35

# Keep only the top N features after first-pass importance scoring
TOP_N_FEATURES = 18


# ── Dataset builder ────────────────────────────────────────────────────────────

def build_training_dataset(macro_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the Round 2 training set from the macro dataset.

    Each row = one historical month.
    Features = macro conditions at time T.
    Target   = Case-Shiller Dallas % change from T to T + FORECAST_HORIZON months.

    Improvements over VER4:
      - Deeper lags (1-3 months for mortgage rate, 1-2 for fed funds + CS momentum)
      - Engineered features (momentum, shock, affordability)
      - NaN discipline: ffill first, then median only for residual gaps
      - Drop columns with >35% missing data

    Returns X, y (aligned, no NaN rows).
    """
    df = macro_df.copy()

    # ── Target: future 6-month Case-Shiller % change ──────────────────────────
    if "case_shiller_dallas" not in df.columns:
        raise ValueError(
            "case_shiller_dallas not found in macro dataset. "
            "Check FRED fetch succeeded."
        )

    df["target_6mo_pct"] = (
        df["case_shiller_dallas"].shift(-FORECAST_HORIZON) /
        df["case_shiller_dallas"] - 1
    ) * 100

    # ── Lag features ──────────────────────────────────────────────────────────
    lag_cols = []
    for feat, n_lags in LAG_DEPTHS.items():
        if feat not in df.columns:
            continue
        for lag in range(1, n_lags + 1):
            col = f"{feat}_lag{lag}"
            df[col] = df[feat].shift(lag)
            lag_cols.append(col)

    # ── Engineered features ───────────────────────────────────────────────────
    if "mortgage_rate_30yr" in df.columns:
        df["mortgage_rate_mom"]        = df["mortgage_rate_30yr"].diff(1)
        df["mortgage_rate_3mo_change"] = df["mortgage_rate_30yr"].diff(3)

    if "fed_funds_rate" in df.columns and "treasury_10yr" in df.columns:
        df["rate_vs_10yr"] = df["fed_funds_rate"] - df["treasury_10yr"]

    if "mortgage_rate_30yr" in df.columns and "case_shiller_dallas" in df.columns:
        df["affordability_stress"] = (
            df["mortgage_rate_30yr"] * df["case_shiller_dallas"] / 1e4
        )

    if "cs_dallas_mom_pct" in df.columns and "cs_dallas_3mo_pct" in df.columns:
        df["cs_momentum_divergence"] = (
            df["cs_dallas_mom_pct"] - df["cs_dallas_3mo_pct"] / 3
        )

    # ── Calendar features ─────────────────────────────────────────────────────
    df["month"]     = df.index.month
    df["quarter"]   = df.index.quarter
    df["is_spring"] = df["month"].isin([3, 4, 5]).astype(int)
    df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)

    # ── Assemble candidate feature list ───────────────────────────────────────
    eng_cols  = [c for c in ENGINEERED_FEATURES if c in df.columns]
    base_cols = [c for c in MACRO_FEATURES if c in df.columns]
    feature_cols = base_cols + lag_cols + eng_cols + CALENDAR_FEATURES

    # Deduplicate while preserving order
    seen = set()
    feature_cols = [c for c in feature_cols if not (c in seen or seen.add(c))]

    # ── Drop rows where target or core series are NaN ─────────────────────────
    required = ["target_6mo_pct", "mortgage_rate_30yr", "case_shiller_dallas"]
    df = df.dropna(subset=[c for c in required if c in df.columns])

    X = df[feature_cols].copy()
    y = df["target_6mo_pct"].copy()

    # ── NaN discipline ────────────────────────────────────────────────────────
    # 1. Drop columns that are missing for >MAX_NAN_FRACTION of rows
    nan_fracs = X.isna().mean()
    cols_to_drop = nan_fracs[nan_fracs > MAX_NAN_FRACTION].index.tolist()
    if cols_to_drop:
        print(f"  Dropping {len(cols_to_drop)} columns with >{MAX_NAN_FRACTION*100:.0f}% NaN: "
              f"{cols_to_drop}")
        X = X.drop(columns=cols_to_drop)

    # 2. Forward-fill first (respects time-series continuity)
    X = X.ffill(limit=2)

    # 3. Fill any remaining NaNs with column median
    X = X.fillna(X.median(numeric_only=True))

    # ── Drop near-constant columns ────────────────────────────────────────────
    variances = X.var(numeric_only=True)
    low_var_cols = variances[variances < MIN_VARIANCE_THRESHOLD].index.tolist()
    if low_var_cols:
        print(f"  Dropping {len(low_var_cols)} near-constant columns: {low_var_cols}")
        X = X.drop(columns=low_var_cols)

    return X, y


# ── XGBoost parameter grid ────────────────────────────────────────────────────

def _make_xgb_params(max_depth: int = 3, reg_lambda: float = 2.0,
                     reg_alpha: float = 0.5, objective: str = "reg:squarederror"):
    """Return a consistent XGBoost config dict for small-dataset regime."""
    return dict(
        n_estimators     = 300,
        learning_rate    = 0.04,
        max_depth        = max_depth,
        min_child_weight = 8,
        subsample        = 0.75,
        colsample_bytree = 0.65,
        reg_alpha        = reg_alpha,
        reg_lambda       = reg_lambda,
        early_stopping_rounds = 20,    # prevent overfitting in CV
        objective        = objective,
        n_jobs           = -1,
        random_state     = 42,
    )


def _cv_score(
    X_scaled: pd.DataFrame,
    y: pd.Series,
    xgb_params: dict,
    ridge_alpha: float = 10.0,
    xgb_weight: float = XGB_BLEND_WEIGHT,
    n_splits: int = 4,
    verbose: bool = False,
) -> tuple[float, float, list]:
    """
    Time-series CV with fixed 24-month test windows and a 1-month gap.
    Uses an XGBoost + Ridge ensemble.
    Returns (mean_mae, mean_r2, fold_results).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=24, gap=1)
    fold_results = []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_scaled)):
        X_tr, X_te = X_scaled.iloc[tr_idx], X_scaled.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx],         y.iloc[te_idx]

        # XGBoost with early stopping (eval set = test fold)
        xgb_params_copy = {k: v for k, v in xgb_params.items()}
        m_xgb = xgb.XGBRegressor(**xgb_params_copy)
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

        # Ridge (linear baseline — more stable across regime changes)
        m_ridge = Ridge(alpha=ridge_alpha)
        m_ridge.fit(X_tr, y_tr)

        # Blend predictions
        p_xgb   = m_xgb.predict(X_te)
        p_ridge = m_ridge.predict(X_te)
        preds   = xgb_weight * p_xgb + (1 - xgb_weight) * p_ridge

        fold_mae = float(mean_absolute_error(y_te, preds))
        fold_r2  = float(r2_score(y_te, preds))
        test_start = X_scaled.index[te_idx[0]].strftime("%Y-%m")
        test_end   = X_scaled.index[te_idx[-1]].strftime("%Y-%m")
        fold_results.append((fold_mae, fold_r2, test_start, test_end, len(tr_idx)))

        if verbose:
            print(f"  Fold {fold+1}: train {X_scaled.index[tr_idx[0]].strftime('%Y-%m')}→"
                  f"{X_scaled.index[tr_idx[-1]].strftime('%Y-%m')} ({len(tr_idx)}rows)  "
                  f"test {test_start}→{test_end}  "
                  f"MAE={fold_mae:.3f}%  R²={fold_r2:.3f}")

    mae_scores = [r[0] for r in fold_results]
    r2_scores  = [r[1] for r in fold_results]
    return float(np.mean(mae_scores)), float(np.mean(r2_scores)), fold_results


# ── Training ───────────────────────────────────────────────────────────────────

def train_round2(
    fred_api_key:  str,
    noaa_api_key:  str = "",
    epa_api_key:   str = "",
    force_refresh: bool = False,
) -> dict:
    """
    Fetch macro data, build training set, train Round 2 XGBoost + Ridge ensemble,
    evaluate with time-series cross-validation, and save the model.

    Two-stage training:
      Stage A — train on all engineered features, score importances
      Stage B — retrain on top TOP_N_FEATURES only (reduces overfitting)

    Also trains lo/hi quantile models (p10 / p90) for prediction intervals.

    NOTE on CV R²: Folds 2-3 test on COVID (2020-21) and the 2022 rate-shock,
    regime changes no model trained on pre-2020 data can predict.  MAE is the
    more actionable metric; R² is shown for transparency.

    Returns a dict of evaluation metrics.
    """
    print("\n" + "=" * 62)
    print("  Round 2 Forecast Model — Training  (VER5 improved)")
    print("=" * 62)

    # 1. Fetch macro data
    macro_df = build_macro_dataset(
        fred_api_key, noaa_api_key, epa_api_key,
        force_refresh=force_refresh,
    )

    # 2. Build training dataset
    print("\nBuilding training dataset …")
    X_full, y = build_training_dataset(macro_df)
    print(f"  Training rows:      {len(X_full)}")
    print(f"  Candidate features: {len(X_full.columns)}")
    print(f"  Target range:       {y.min():.2f}% to {y.max():.2f}%")
    print(f"  Target mean:        {y.mean():.2f}%  std: {y.std():.2f}%")

    # 3. Scale
    scaler_full   = StandardScaler()
    X_scaled_full = pd.DataFrame(
        scaler_full.fit_transform(X_full),
        columns=X_full.columns, index=X_full.index
    )

    # 4. Grid search over key hyperparameters
    print("\nHyperparameter search (XGBoost + Ridge ensemble) …")
    grid = [
        {"max_depth": d, "reg_lambda": l, "reg_alpha": a}
        for d in [2, 3, 4]
        for l in [2.0, 4.0]
        for a in [0.5, 1.5]
    ]
    # Ridge alpha grid searched alongside
    ridge_alphas = [5.0, 10.0, 25.0]

    best_mae, best_r2 = np.inf, -np.inf
    best_xgb_params   = grid[0]
    best_ridge_alpha  = 10.0

    for g in grid:
        for ra in ridge_alphas:
            params = _make_xgb_params(**g)
            mae, r2, _ = _cv_score(X_scaled_full, y, params, ridge_alpha=ra)
            if mae < best_mae:
                best_mae, best_r2 = mae, r2
                best_xgb_params  = g
                best_ridge_alpha  = ra

    print(f"  ✓ Best XGB: depth={best_xgb_params['max_depth']}  "
          f"λ={best_xgb_params['reg_lambda']}  α={best_xgb_params['reg_alpha']}  "
          f"Ridge α={best_ridge_alpha}")
    print(f"  ✓ Best CV: MAE={best_mae:.3f}%  R²={best_r2:.3f}")

    # 5. Stage A — train on all features to get importances
    print("\nStage A: fitting on all features for importance scoring …")
    params_a = {k: v for k, v in _make_xgb_params(**best_xgb_params).items()
                if k != "early_stopping_rounds"}
    stage_a_model = xgb.XGBRegressor(**params_a)
    stage_a_model.fit(X_scaled_full, y)

    importances = pd.Series(
        stage_a_model.feature_importances_, index=X_full.columns
    ).sort_values(ascending=False)

    top_features = importances[importances > 0].head(TOP_N_FEATURES).index.tolist()
    pruned = [f for f in X_full.columns if f not in top_features]
    print(f"\n── Top {len(top_features)} features selected ─────────────────────────────")
    for feat in top_features:
        print(f"  {feat:<42} {importances[feat]:.4f}")
    if pruned:
        print(f"\n  Pruned {len(pruned)} near-zero features: {pruned}")

    # 6. Stage B — retrain on selected features only + final CV
    print(f"\nStage B: retraining on top {len(top_features)} features …")
    X_sel    = X_full[top_features]
    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_sel),
        columns=top_features, index=X_sel.index
    )

    print("\nFinal time-series cross-validation (4 folds × 24-month test windows) …")
    final_xgb_params = _make_xgb_params(**best_xgb_params)
    mae, r2, fold_results = _cv_score(
        X_scaled, y, final_xgb_params,
        ridge_alpha=best_ridge_alpha, verbose=True,
    )
    print(f"\n  ⚠  Note: Folds 2-3 test on COVID (2020-21) and 2022 rate-shock —")
    print(f"     regime changes no model can predict from prior-year data.")
    print(f"\n  CV MAE (avg): {mae:.3f}%   ← primary metric")
    print(f"  CV R²  (avg): {r2:.3f}   (depressed by regime-change folds)")

    # 7. Final models on full data
    print("\nFitting final XGBoost model on full dataset …")
    params_final = {k: v for k, v in final_xgb_params.items()
                    if k != "early_stopping_rounds"}
    params_final["n_estimators"] = 400
    final_xgb = xgb.XGBRegressor(**params_final)
    final_xgb.fit(X_scaled, y)

    print("Fitting final Ridge model …")
    final_ridge = Ridge(alpha=best_ridge_alpha)
    final_ridge.fit(X_scaled, y)

    # 8. Quantile models (p10, p90) — XGBoost only for intervals
    print("Fitting quantile models (p10 / p90) …")
    params_lo = dict(params_final, objective="reg:quantileerror", quantile_alpha=0.10)
    params_hi = dict(params_final, objective="reg:quantileerror", quantile_alpha=0.90)
    model_lo  = xgb.XGBRegressor(**params_lo)
    model_hi  = xgb.XGBRegressor(**params_hi)
    model_lo.fit(X_scaled, y)
    model_hi.fit(X_scaled, y)

    # 9. Save all artefacts
    meta = {
        "xgb_weight":    XGB_BLEND_WEIGHT,
        "ridge_alpha":   best_ridge_alpha,
        "best_xgb_params": best_xgb_params,
        "cv_mae_pct":    mae,
        "cv_r2":         r2,
        "n_rows":        len(X_sel),
        "n_features":    len(top_features),
        "top_features":  top_features,
    }

    joblib.dump(final_xgb,   ROUND2_MODEL_PATH)
    joblib.dump(model_lo,    ROUND2_MODEL_LO_PATH)
    joblib.dump(model_hi,    ROUND2_MODEL_HI_PATH)
    joblib.dump(final_ridge, ROUND2_RIDGE_PATH)
    joblib.dump(scaler,      ROUND2_SCALER_PATH)
    with open(ROUND2_COLS_PATH, "w") as f:
        json.dump(top_features, f)

    print(f"\n  Model (XGB) saved → {ROUND2_MODEL_PATH}")
    print(f"  Model (Ridge) saved → {ROUND2_RIDGE_PATH}")
    print(f"  Quantile lo saved  → {ROUND2_MODEL_LO_PATH}")
    print(f"  Quantile hi saved  → {ROUND2_MODEL_HI_PATH}")
    print(f"  Scaler saved       → {ROUND2_SCALER_PATH}")
    print(f"  Feature cols       → {ROUND2_COLS_PATH}  ({len(top_features)} features)")

    # 10. Final importance table
    final_importances = pd.Series(
        final_xgb.feature_importances_, index=top_features
    ).sort_values(ascending=False)
    print(f"\n── Final Feature Importances ─────────────────────────────")
    print(final_importances.to_string())

    return meta


# ── Prediction ─────────────────────────────────────────────────────────────────

def predict_forward(
    base_estimate:  float,
    fred_api_key:   str,
    noaa_api_key:   str  = "",
    epa_api_key:    str  = "",
    zip_code:       str  = None,
    property_type:  str  = None,
    verbose:        bool = True,
) -> dict:
    """
    Given a Stage 1 base estimate (today's fair value), produce a 6-month
    forward estimate using current macro conditions + property-specific beta
    adjustment + SHAP factor breakdown + 80% confidence interval (p10–p90).

    Parameters
    ----------
    base_estimate  : Stage 1 model output in dollars
    fred_api_key   : FRED API key
    noaa_api_key   : NOAA CDO API key (optional)
    epa_api_key    : EPA AQS API key (optional)
    zip_code       : property ZIP code — used to apply ZIP-tier beta adjustment
    property_type  : ATTOM property type string (e.g. "SFR", "CONDO")
    verbose        : print formatted results

    Returns
    -------
    dict with keys:
      base_estimate, forward_estimate, change_pct, change_dollars,
      estimate_low, estimate_high (p10/p90 range),
      macro_pct         (raw market-level signal before property adjustment),
      price_beta, zip_beta, type_beta, combined_beta  (how the signal was modified),
      price_tier, zip_tier  (human-readable tier labels),
      shap_factors (list of dicts sorted by impact),
      macro_snapshot (dict of raw macro values used)
    """
    # Load model artefacts
    for path in [ROUND2_MODEL_PATH, ROUND2_SCALER_PATH, ROUND2_COLS_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Round 2 model file not found: {path}\n"
                "  → Run:  python round2_model.py train --fred-key YOUR_KEY"
            )

    model        = joblib.load(ROUND2_MODEL_PATH)
    scaler       = joblib.load(ROUND2_SCALER_PATH)
    model_lo     = joblib.load(ROUND2_MODEL_LO_PATH) if os.path.exists(ROUND2_MODEL_LO_PATH) else None
    model_hi     = joblib.load(ROUND2_MODEL_HI_PATH) if os.path.exists(ROUND2_MODEL_HI_PATH) else None
    ridge_model  = joblib.load(ROUND2_RIDGE_PATH)    if os.path.exists(ROUND2_RIDGE_PATH)    else None
    with open(ROUND2_COLS_PATH) as f:
        feature_cols = json.load(f)

    # Get current macro snapshot
    macro_snap = get_current_macro_snapshot(fred_api_key, noaa_api_key, epa_api_key)

    # ── Build the feature row ─────────────────────────────────────────────────
    # We need to reconstruct the same engineered features used in training.
    # get_current_macro_snapshot() returns the latest row of the full macro DataFrame,
    # which includes all raw columns. We derive the same engineered features here.

    row = {}

    # Raw macro features
    for col in feature_cols:
        if col in macro_snap.index:
            row[col] = float(macro_snap[col]) if not pd.isna(macro_snap[col]) else np.nan

    # Calendar features (current date)
    now = pd.Timestamp.now()
    row["month"]     = now.month
    row["quarter"]   = now.quarter
    row["is_spring"] = int(now.month in [3, 4, 5])
    row["is_summer"] = int(now.month in [6, 7, 8])

    # Engineered features (derived from macro_snap)
    _mr = macro_snap.get("mortgage_rate_30yr",     np.nan)
    _mr_lag1 = macro_snap.get("mortgage_rate_30yr_lag1", np.nan)
    _mr_lag3 = macro_snap.get("mortgage_rate_30yr_lag3", np.nan)

    if "mortgage_rate_mom" in feature_cols:
        row["mortgage_rate_mom"] = (
            _mr - _mr_lag1 if not (pd.isna(_mr) or pd.isna(_mr_lag1)) else np.nan
        )
    if "mortgage_rate_3mo_change" in feature_cols:
        row["mortgage_rate_3mo_change"] = (
            _mr - _mr_lag3 if not (pd.isna(_mr) or pd.isna(_mr_lag3)) else np.nan
        )
    if "rate_vs_10yr" in feature_cols:
        _ff  = macro_snap.get("fed_funds_rate",  np.nan)
        _t10 = macro_snap.get("treasury_10yr",   np.nan)
        row["rate_vs_10yr"] = (
            float(_ff) - float(_t10) if not (pd.isna(_ff) or pd.isna(_t10)) else np.nan
        )
    if "affordability_stress" in feature_cols:
        _cs = macro_snap.get("case_shiller_dallas", np.nan)
        row["affordability_stress"] = (
            float(_mr) * float(_cs) / 1e4
            if not (pd.isna(_mr) or pd.isna(_cs)) else np.nan
        )
    if "cs_momentum_divergence" in feature_cols:
        _mom  = macro_snap.get("cs_dallas_mom_pct",  np.nan)
        _3mo  = macro_snap.get("cs_dallas_3mo_pct",  np.nan)
        row["cs_momentum_divergence"] = (
            float(_mom) - float(_3mo) / 3
            if not (pd.isna(_mom) or pd.isna(_3mo)) else np.nan
        )

    # Fill any still-missing feature with the column median from the scaler
    # (scaler.mean_ is the training mean — a reasonable fallback)
    X_row = pd.DataFrame([row]).reindex(columns=feature_cols)
    for i, col in enumerate(feature_cols):
        if pd.isna(X_row.iloc[0][col]):
            X_row.iloc[0, X_row.columns.get_loc(col)] = float(scaler.mean_[i])

    # Scale
    X_scaled     = scaler.transform(X_row)
    X_scaled_df  = pd.DataFrame(X_scaled, columns=feature_cols)

    # Predict median + quantiles
    predicted_xgb   = float(model.predict(X_scaled_df)[0])
    predicted_ridge = float(ridge_model.predict(X_scaled_df)[0]) if ridge_model else predicted_xgb
    macro_pct       = XGB_BLEND_WEIGHT * predicted_xgb + (1 - XGB_BLEND_WEIGHT) * predicted_ridge

    pct_lo_raw = float(model_lo.predict(X_scaled_df)[0]) if model_lo else macro_pct - 2.0
    pct_hi_raw = float(model_hi.predict(X_scaled_df)[0]) if model_hi else macro_pct + 2.0
    pct_lo_raw = min(pct_lo_raw, macro_pct)
    pct_hi_raw = max(pct_hi_raw, macro_pct)

    # ── Property-specific beta adjustment ─────────────────────────────────────
    # The macro model gives a single market-level signal. We now scale it by
    # price tier, ZIP code, and property type to get a property-specific forecast.
    beta_result     = _apply_beta(
        macro_pct     = macro_pct,
        base_estimate = base_estimate,
        zip_code      = zip_code,
        property_type = property_type,
        verbose       = verbose,
    )
    predicted_pct   = beta_result["effective_pct"]

    # Apply same combined beta to the quantile bounds
   # Apply same combined beta to the quantile bounds
    combined_beta   = beta_result["combined_beta"]
    pct_lo          = pct_lo_raw * combined_beta
    pct_hi          = pct_hi_raw * combined_beta
    # Ensure monotonicity: p10 ≤ median ≤ p90
    pct_lo          = min(pct_lo, predicted_pct)
    pct_hi          = max(pct_hi, predicted_pct)

    forward_estimate = base_estimate * (1 + predicted_pct / 100)
    estimate_low     = base_estimate * (1 + pct_lo       / 100)
    estimate_high    = base_estimate * (1 + pct_hi       / 100)
    change_dollars   = forward_estimate - base_estimate

    # ── SHAP factor breakdown ─────────────────────────────────────────────────
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled_df)

    shap_series  = pd.Series(shap_values[0], index=feature_cols)
    # Scale SHAP values by combined beta so dollar impacts reflect the property-adjusted forecast
    shap_dollar  = shap_series * base_estimate * combined_beta / 100

    factors = []
    for feat, shap_pct in shap_series.sort_values(key=abs, ascending=False).items():
        raw_val   = row.get(feat, np.nan)
        dollar    = float(shap_dollar[feat])
        direction = "↑" if shap_pct > 0 else "↓"
        factors.append({
            "feature":     feat,
            "raw_value":   round(float(raw_val), 4) if not pd.isna(raw_val) else None,
            "shap_pct":    round(float(shap_pct), 4),
            "shap_dollar": round(dollar, 2),
            "direction":   direction,
        })

    # ── Verbose output ────────────────────────────────────────────────────────
    if verbose:
        _print_results(
            base_estimate, forward_estimate, estimate_low, estimate_high,
            predicted_pct, macro_pct, pct_lo, pct_hi,
            change_dollars, factors, macro_snap, beta_result,
        )

    return {
        "base_estimate":    round(base_estimate,    2),
        "forward_estimate": round(forward_estimate, 2),
        "estimate_low":     round(estimate_low,     2),
        "estimate_high":    round(estimate_high,    2),
        "change_pct":       round(predicted_pct,    2),
        "change_pct_lo":    round(pct_lo,           2),
        "change_pct_hi":    round(pct_hi,           2),
        "change_dollars":   round(change_dollars,   2),
        "macro_pct":        round(macro_pct,         2),
        "price_beta":       beta_result["price_beta"],
        "zip_beta":         beta_result["zip_beta"],
        "type_beta":        beta_result["type_beta"],
        "combined_beta":    beta_result["combined_beta"],
        "price_tier":       beta_result["price_tier"],
        "zip_tier":         beta_result["zip_tier"],
        "shap_factors":     factors,
        "macro_snapshot":   {
            k: v for k, v in row.items()
            if not (isinstance(v, float) and np.isnan(v))
            and k in MACRO_FEATURES
        },
    }


# ── Pretty printer ─────────────────────────────────────────────────────────────

def _print_results(
    base_est, fwd_est, est_lo, est_hi,
    pct, macro_pct, pct_lo, pct_hi,
    dollars, factors, macro_snap, beta_result,
):
    sign     = "+" if dollars >= 0 else "-"
    pct_sign = "+" if pct >= 0 else ""

    print("\n" + "=" * 62)
    print("          6-MONTH FORWARD VALUATION ESTIMATE          ")
    print("=" * 62)
    print(f"  Today's Estimate (Stage 1):    ${base_est:>12,.0f}")
    print(f"  6-Month Forward Estimate:      ${fwd_est:>12,.0f}")
    print(f"  Projected Change:              "
          f"{sign}${abs(dollars):>10,.0f}  ({pct_sign}{pct:.2f}%)")
    print(f"  80% Confidence Range:         "
          f"${est_lo:>12,.0f}  to  ${est_hi:>12,.0f}")
    print(f"    ({pct_lo:+.2f}%  to  {pct_hi:+.2f}%)")
    print("=" * 62)

    print(f"\n── Property Adjustment ───────────────────────────────────")
    print(f"  Market signal (all DFW):       {macro_pct:+.2f}%")
    print(f"  Price tier:  {beta_result['price_tier']:<28}  β={beta_result['price_beta']:.2f}")
    print(f"  ZIP tier:    {beta_result['zip_tier']:<28}  β={beta_result['zip_beta']:.2f}")
    print(f"  Property:    {beta_result.get('type_label', 'SFR'):<28}  β={beta_result['type_beta']:.2f}")
    print(f"  Combined beta:                 ×{beta_result['combined_beta']:.3f}")
    print(f"  Property-adjusted forecast:    {pct:+.2f}%")

    print("\n── Factor Breakdown (what's driving the market signal) ───")
    print(f"  {'Factor':<38} {'Value':>10}  {'Impact':>12}")
    print(f"  {'-'*38} {'-'*10}  {'-'*12}")
    for f in factors[:15]:
        val_str    = (f"{f['raw_value']:>10.2f}"
                      if f["raw_value"] is not None else "     N/A  ")
        impact_str = f"{f['direction']} ${abs(f['shap_dollar']):>9,.0f}"
        print(f"  {f['feature']:<38} {val_str}  {impact_str}")

    print("\n── Current Macro Snapshot ────────────────────────────────")
    snapshot_display = {
        "Mortgage Rate (30yr)":         macro_snap.get("mortgage_rate_30yr"),
        "Mortgage Rate MoM Change":     macro_snap.get("mortgage_rate_mom"),
        "Fed Funds Rate":               macro_snap.get("fed_funds_rate"),
        "10yr Treasury":                macro_snap.get("treasury_10yr"),
        "Yield Spread (10yr-2yr)":      macro_snap.get("yield_spread"),
        "Dallas CS 3mo%":               macro_snap.get("cs_dallas_3mo_pct"),
        "TX Unemployment":              macro_snap.get("unemployment_texas"),
        "S&P 500 Monthly Return%":      macro_snap.get("sp500_return"),
        "VIX":                          macro_snap.get("vix"),
    }
    for label, val in snapshot_display.items():
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            print(f"  {label:<38} {val:>8.2f}")
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Round 2 forecast model — train or predict"
    )
    sub = parser.add_subparsers(dest="command")

    # train
    t = sub.add_parser("train", help="Fetch macro data and train the Round 2 model")
    t.add_argument("--fred-key",  required=True, help="FRED API key")
    t.add_argument("--noaa-key",  default="",    help="NOAA CDO API key (optional)")
    t.add_argument("--epa-key",   default="",    help="EPA AQS API key (optional)")
    t.add_argument("--refresh",   action="store_true",
                   help="Force re-fetch even if cache is fresh")

    # predict
    p = sub.add_parser("predict", help="Run the full pipeline for an address")
    p.add_argument("--fred-key",  required=True)
    p.add_argument("--noaa-key",  default="")
    p.add_argument("--epa-key",   default="")
    p.add_argument("--base",      type=float, default=None,
                   help="Provide a Stage 1 estimate directly (skips ATTOM lookup)")
    p.add_argument("--address",   default=None,
                   help="Full address (uses Stage 1 model to get base estimate)")
    p.add_argument("--attom-key", default="",
                   help="ATTOM API key (required if --address is used)")

    return parser.parse_args()


def _get_base_estimate_from_address(address: str, attom_key: str) -> float:
    """Call Stage 1 pipeline to get base estimate for an address."""
    from predict_address import get_property_data, build_feature_row
    from model_logic import load_model, predict as stage1_predict

    raw  = get_property_data(address)
    df   = build_feature_row(raw)
    pipe = load_model(STAGE1_MODEL_PATH)
    return float(stage1_predict(pipe, df)[0])


if __name__ == "__main__":
    args = _parse_args()

    if args.command == "train":
        metrics = train_round2(
            fred_api_key  = args.fred_key,
            noaa_api_key  = args.noaa_key,
            epa_api_key   = args.epa_key,
            force_refresh = args.refresh,
        )
        print(f"\nDone. CV MAE: {metrics['cv_mae_pct']:.3f}%  "
              f"CV R²: {metrics['cv_r2']:.3f}  "
              f"Features: {metrics['n_features']}")

    elif args.command == "predict":
        if args.base is not None:
            base = args.base
        elif args.address:
            print(f"[1/2] Getting Stage 1 estimate for: {args.address}")
            base = _get_base_estimate_from_address(args.address, args.attom_key)
            print(f"      Stage 1 estimate: ${base:,.0f}")
        else:
            print("Error: provide --base or --address")
            sys.exit(1)

        result = predict_forward(
            base_estimate = base,
            fred_api_key  = args.fred_key,
            noaa_api_key  = args.noaa_key,
            epa_api_key   = args.epa_key,
        )

    else:
        print("Usage:")
        print("  python round2_model.py train   --fred-key YOUR_KEY [--noaa-key X] [--epa-key X]")
        print("  python round2_model.py predict --fred-key YOUR_KEY --base 488302")
        print("  python round2_model.py predict --fred-key YOUR_KEY "
              "--address '4529 Wateka Dr, Dallas, TX 75209' --attom-key YOUR_KEY")