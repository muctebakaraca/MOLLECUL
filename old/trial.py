"""
ATTOM Sale Price Test
----------------------
Tries every sale-related endpoint ATTOM has to see which ones
return price data on the free trial. Run this and paste the
output so we know exactly what we're working with.

Run: python test_sale_price.py
"""

import requests
import json

KEY = "YOUR_KEY"  # replace with your actual key

HEADERS = {"Accept": "application/json", "apikey": KEY}
ZIP     = "75201"  # downtown Dallas — dense area, lots of sales

# Every endpoint that could contain sale price data
ENDPOINTS = [
    "/sale/detail",
    "/sale/snapshot",
    "/saleshistory/detail",
    "/saleshistory/snapshot",
    "/saleshistory/basichistory",
    "/saleshistory/expandedhistory",
    "/allevents/detail",
]

BASE = "https://api.gateway.attomdata.com/propertyapi/v1.0.0"

print("=" * 60)
print("Testing ATTOM sale price endpoints")
print(f"ZIP: {ZIP}  |  Key: ...{KEY[-6:]}")
print("=" * 60)

for endpoint in ENDPOINTS:
    url  = BASE + endpoint
    resp = requests.get(
        url,
        headers=HEADERS,
        params={"postalCode": ZIP, "pageSize": 1, "page": 1},
        timeout=30,
    )

    print(f"\n{'─' * 60}")
    print(f"Endpoint: {endpoint}")
    print(f"HTTP status: {resp.status_code}")

    if resp.status_code != 200:
        print(f"  ✗ Failed — {resp.text[:200]}")
        continue

    data   = resp.json()
    status = data.get("status", {})
    print(f"ATTOM status code: {status.get('code')}  msg: {status.get('msg')}")

    # Try to find a property or sale record
    records = (
        data.get("property")
        or data.get("sale")
        or data.get("saleshistory")
        or data.get("event")
        or []
    )

    if not records:
        print("  ✗ No records returned")
        continue

    record = records[0]

    # Hunt for any key that looks like a price
    def find_prices(obj, path=""):
        """Recursively search for anything that looks like a sale price."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                full_path = f"{path}.{k}" if path else k
                find_prices(v, full_path)
        elif isinstance(obj, (int, float)):
            key_lower = path.lower()
            if any(word in key_lower for word in ["sale", "price", "amt", "amount", "value"]):
                print(f"  ✓ Found: {path} = {obj}")
        elif isinstance(obj, str) and obj.strip():
            key_lower = path.lower()
            if any(word in key_lower for word in ["sale", "price", "amt", "amount"]):
                print(f"  ✓ Found: {path} = {obj}")

    find_prices(record)

    # Also print the raw record so we can see the full structure
    print(f"\n  Raw response (first record):")
    print(json.dumps(record, indent=4))

print("\n" + "=" * 60)
print("Done. Paste this output to see which endpoints work.")
print("=" * 60)