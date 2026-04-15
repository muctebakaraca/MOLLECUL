"""
property_beta.py
────────────────
Applies a property-specific multiplier to the Stage 2 macro forecast.

The Stage 2 model produces a single market-level % change (e.g. -3.5%).
This module differentiates that signal by property characteristics so that
a $200K starter home and a $1.2M luxury home don't get the same forecast.

How it works
────────────
  effective_pct = macro_pct × price_beta × zip_beta × type_beta

Each beta is a multiplier centred on 1.0 (= market average).
  > 1.0  →  amplifies the macro signal  (more volatile than market)
  < 1.0  →  dampens the macro signal    (more stable than market)

So if macro is -3.5%:
  Starter home (<$300K):    -3.5% × 0.75 = -2.6%   (more rate-insensitive)
  Mid-tier ($500K, Allen):  -3.5% × 1.00 = -3.5%   (market average)
  Luxury ($1M+, HP/UP):     -3.5% × 1.55 = -5.4%   (high rate sensitivity)

The betas are calibrated against DFW Case-Shiller sub-index behaviour
and known segment volatility patterns from 2015–2024 market cycles.

Tuning
──────
All three tables (PRICE_TIERS, ZIP_BETAS, TYPE_BETAS) are plain dicts —
easy to adjust as you observe prediction drift vs actual comps.

Usage
─────
  from property_beta import adjusted_forecast

  pct = adjusted_forecast(
      macro_pct      = -3.5,
      base_estimate  = 488302,
      zip_code       = "75205",
      property_type  = "SFR",
  )
"""

from __future__ import annotations

# ── Price Tier Betas ───────────────────────────────────────────────────────────
# Luxury and upper-mid homes have higher beta to mortgage rate cycles because:
#   • More discretionary / move-up demand (buyers can wait)
#   • Larger absolute dollar payment per rate move
#   • More investor/speculative ownership at high end
#
# Starter homes are more stable because:
#   • Owner-occupier demand is less elastic
#   • Rent parity supports a floor
#   • Less investor competition in a down market

PRICE_TIERS: list[tuple[float, float, float]] = [
    # (min_price,  max_price,  beta)
    (0,           250_000,    0.65),   # true starter — very stable
    (250_000,     375_000,    0.80),   # affordable — below-average volatility
    (375_000,     550_000,    0.95),   # core DFW mid-tier
    (550_000,     750_000,    1.10),   # upper-mid — slightly above market
    (750_000,     950_000,    1.25),   # near-luxury
    (950_000,   1_400_000,    1.40),   # luxury
    (1_400_000, float("inf"), 1.55),   # ultra-luxury — highest volatility
]


def _price_beta(base_estimate: float) -> float:
    for lo, hi, beta in PRICE_TIERS:
        if lo <= base_estimate < hi:
            return beta
    return 1.0   # fallback


# ── ZIP Code Betas ─────────────────────────────────────────────────────────────
# Grouped by how much each ZIP amplifies or dampens the DFW market cycle.
# Beta = 1.0 means "moves in line with the Dallas metro average."
#
# Sources: Case-Shiller sub-index patterns, Redfin/Zillow ZIP-level trends,
#          observed DFW cycle behaviour (2018 slowdown, 2020-22 boom, 2023 correction)
#
# Format: "zip_code": beta

ZIP_BETAS: dict[str, float] = {

    # ── Tier A: Premium / High-Beta (1.25–1.35) ────────────────────────────────
    # Luxury enclaves — most sensitive to rate/affordability shocks
    "75205": 1.35,   # Highland Park
    "75225": 1.35,   # University Park
    "75209": 1.30,   # Devonshire / Briarwood (Old Preston Road)
    "75230": 1.25,   # Preston Hollow north
    "75229": 1.20,   # North Dallas / Midway Hollow
    "75240": 1.15,   # Far North Dallas / Galleria corridor
    "76092": 1.25,   # Southlake (luxury Tarrant)
    "76034": 1.20,   # Colleyville

    # ── Tier B: High-Demand Growth Suburbs (1.05–1.15) ─────────────────────────
    # Strong appreciation history, but newer inventory = more supply risk
    "75035": 1.15,   # Frisco NE
    "75033": 1.12,   # Frisco NW / new builds
    "75034": 1.10,   # Frisco central
    "75036": 1.08,   # Frisco SW / The Star area
    "75070": 1.10,   # McKinney east
    "75071": 1.08,   # McKinney west (Allen border)
    "75069": 1.05,   # McKinney core
    "75013": 1.10,   # Allen (main)
    "75002": 1.05,   # Allen east / southern Allen
    "75093": 1.08,   # Plano west (legacy Drive area)
    "75094": 1.05,   # Plano east / Murphy border
    "76210": 1.10,   # Denton / Lake Lewisville area
    "75010": 1.05,   # Carrollton / Colony border

    # ── Tier C: Established Suburbs — Market Average (0.95–1.05) ──────────────
    "75080": 1.00,   # Richardson central
    "75081": 1.00,   # Richardson east
    "75082": 0.98,   # Richardson NE / UTD area
    "75074": 1.00,   # Plano east
    "75075": 1.00,   # Plano central
    "75023": 0.98,   # Plano NW
    "75024": 1.02,   # Plano north
    "75025": 1.00,   # Plano NE
    "75007": 0.97,   # Carrollton
    "75006": 0.97,   # Carrollton west
    "75019": 1.00,   # Coppell
    "75063": 1.00,   # Irving Las Colinas
    "75061": 0.95,   # Irving central
    "75062": 0.95,   # Irving south
    "75038": 0.97,   # Irving / MacArthur
    "76248": 1.00,   # Keller
    "76244": 1.00,   # Keller / Fort Worth NE
    "76262": 1.00,   # Roanoke / Trophy Club
    "76051": 1.02,   # Grapevine
    "75028": 1.00,   # Flower Mound west
    "75022": 1.02,   # Flower Mound east / Grapevine Lake

    # ── Tier D: Inner/Transitional Dallas — Below Average (0.85–0.95) ──────────
    "75201": 0.95,   # Downtown Dallas (condo-heavy, higher rate exposure)
    "75202": 0.90,   # Downtown / Trinity Groves
    "75204": 0.95,   # Uptown / Knox
    "75206": 0.95,   # Lower Greenville / M-Streets
    "75214": 0.95,   # Lakewood
    "75218": 0.93,   # White Rock Lake
    "75228": 0.90,   # East Dallas / Lakeland Heights
    "75243": 0.90,   # Lake Highlands
    "75252": 0.95,   # Far North Dallas (older stock)
    "75287": 0.95,   # Far North Dallas west
    "75231": 0.90,   # NE Dallas / Lake Highlands south
    "75238": 0.90,   # NE Dallas

    # ── Tier E: South/SE Dallas & Outer — Lowest Beta (0.75–0.85) ─────────────
    # More affordable, higher owner-occupier stability, less speculative
    "75208": 0.82,   # Oak Cliff / Kessler Park
    "75211": 0.80,   # Oak Cliff west
    "75212": 0.80,   # West Dallas
    "75216": 0.78,   # South Dallas
    "75217": 0.78,   # South Dallas / Mesquite border
    "75232": 0.78,   # Duncanville border
    "75237": 0.80,   # Duncanville / Redbird
    "75224": 0.82,   # Winnetka Heights / Oak Cliff
    "75233": 0.80,   # Cockrell Hill area
    "75236": 0.80,   # Southwest Dallas
    "75241": 0.78,   # South Dallas
    "75253": 0.80,   # SE Dallas
    "75149": 0.85,   # Mesquite
    "75150": 0.85,   # Mesquite east
    "75040": 0.87,   # Garland central
    "75041": 0.85,   # Garland south
    "75042": 0.85,   # Garland north
    "75043": 0.87,   # Garland east / Lake Ray Hubbard
    "75044": 0.90,   # Garland NE (better submarket)

    # ── Fort Worth metro ───────────────────────────────────────────────────────
    "76109": 1.10,   # TCU / Westover Hills
    "76107": 1.05,   # Fort Worth west / Cultural District
    "76116": 0.95,   # Fort Worth Ridglea
    "76132": 1.00,   # Fort Worth SW / Benbrook
    "76137": 0.95,   # Fort Worth NE
    "76148": 0.90,   # Watauga / Fort Worth NE
    "76118": 0.92,   # Fort Worth east
    "76119": 0.80,   # Fort Worth SE (lower tier)
    "76104": 0.82,   # Fort Worth south central
}

DEFAULT_ZIP_BETA = 1.00   # for any ZIP not in the table


def _zip_beta(zip_code: str | None) -> float:
    if not zip_code:
        return DEFAULT_ZIP_BETA
    z = str(zip_code).strip().split("-")[0]   # handle ZIP+4
    return ZIP_BETAS.get(z, DEFAULT_ZIP_BETA)


# ── Property Type Betas ────────────────────────────────────────────────────────
# Condos and townhomes have higher beta to rate shocks than SFRs because:
#   • Higher HOA fees make them less affordable at high rates (double squeeze)
#   • More investor/short-term buyer demand
#   • Supply of new condos is more elastic — developers can flood the market
# Multi-family has its own rent-market dynamics (partially uncorrelated with SFR)

TYPE_BETAS: dict[str, float] = {
    # Standard SFR
    "SFR":                                  1.00,
    "SINGLE FAMILY RESIDENCE":              1.00,
    "SINGLE FAMILY RESIDENCE / TOWNHOUSE":  1.00,

    # Condos — highest rate sensitivity
    "CONDO":                                1.20,
    "CONDOMINIUM":                          1.20,
    "TOWNHOUSE":                            1.10,
    "TOWNHOME":                             1.10,

    # Multi-family — partially rent-market driven
    "DUPLEX":                               0.90,
    "MULTI-FAMILY":                         0.90,
    "TRIPLEX":                              0.90,
    "QUADPLEX":                             0.90,

    # Land — most speculative, highest beta
    "VACANT LAND":                          1.40,
    "LAND":                                 1.40,

    # Commercial / mixed-use — lower correlation to residential macro
    "COMMERCIAL":                           0.80,
}

DEFAULT_TYPE_BETA = 1.00


def _type_beta(property_type: str | None) -> float:
    if not property_type:
        return DEFAULT_TYPE_BETA
    key = str(property_type).strip().upper()
    # try exact match first, then partial
    if key in TYPE_BETAS:
        return TYPE_BETAS[key]
    for k, v in TYPE_BETAS.items():
        if k in key or key in k:
            return v
    return DEFAULT_TYPE_BETA


# ── Combined Adjustment ────────────────────────────────────────────────────────

def adjusted_forecast(
    macro_pct:      float,
    base_estimate:  float,
    zip_code:       str | None    = None,
    property_type:  str | None    = None,
    verbose:        bool          = False,
) -> dict:
    """
    Apply property-specific betas to the macro market % forecast.

    Parameters
    ----------
    macro_pct      : raw Stage 2 market-level prediction (e.g. -3.5)
    base_estimate  : Stage 1 dollar estimate
    zip_code       : property ZIP code
    property_type  : ATTOM property type string (e.g. "SFR", "CONDO")
    verbose        : print the beta breakdown

    Returns
    -------
    dict with:
      effective_pct    : final adjusted % change
      macro_pct        : original market signal (unchanged)
      price_beta       : multiplier from price tier
      zip_beta         : multiplier from ZIP
      type_beta        : multiplier from property type
      combined_beta    : product of all three
      zip_tier         : human-readable ZIP tier label
      price_tier       : human-readable price tier label
    """
    pb = _price_beta(base_estimate)
    zb = _zip_beta(zip_code)
    tb = _type_beta(property_type)

    combined   = pb * zb * tb
    effective  = macro_pct * combined

    # Human-readable tier labels for display
    if base_estimate < 250_000:
        price_tier = "Starter (<$250K)"
    elif base_estimate < 375_000:
        price_tier = "Affordable ($250K–$375K)"
    elif base_estimate < 550_000:
        price_tier = "Mid-Tier ($375K–$550K)"
    elif base_estimate < 750_000:
        price_tier = "Upper-Mid ($550K–$750K)"
    elif base_estimate < 950_000:
        price_tier = "Near-Luxury ($750K–$950K)"
    elif base_estimate < 1_400_000:
        price_tier = "Luxury ($950K–$1.4M)"
    else:
        price_tier = "Ultra-Luxury ($1.4M+)"

    zb_val = _zip_beta(zip_code)
    if zb_val >= 1.20:
        zip_tier = "Premium (high-beta)"
    elif zb_val >= 1.05:
        zip_tier = "High-demand suburb"
    elif zb_val >= 0.95:
        zip_tier = "Established suburb"
    elif zb_val >= 0.85:
        zip_tier = "Transitional/inner-ring"
    else:
        zip_tier = "Affordable/outer"

    if verbose:
        direction = "market" if macro_pct >= 0 else "market"
        print(f"\n── Property Beta Adjustment ─────────────────────────────")
        print(f"  Market signal (macro):    {macro_pct:+.2f}%")
        print(f"  Price tier:               {price_tier}  →  β={pb:.2f}")
        print(f"  ZIP {zip_code or 'unknown'}:             {zip_tier}  →  β={zb:.2f}")
        print(f"  Property type:            {property_type or 'unknown'}  →  β={tb:.2f}")
        print(f"  Combined beta:            {combined:.3f}")
        print(f"  Effective forecast:       {effective:+.2f}%")

    return {
        "effective_pct":  round(effective, 2),
        "macro_pct":      round(macro_pct, 2),
        "price_beta":     round(pb, 3),
        "zip_beta":       round(zb, 3),
        "type_beta":      round(tb, 3),
        "combined_beta":  round(combined, 3),
        "zip_tier":       zip_tier,
        "price_tier":     price_tier,
    }
