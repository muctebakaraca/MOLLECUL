"""
Dallas Property + Sale Price Scraper — ATTOM API
-------------------------------------------------
Uses the /sale/detail endpoint which returns both full property
characteristics AND sale price data in a single call.

Some properties will have no sale price — this is normal in Texas
since it's a non-disclosure state. Those rows will just have a
blank salePrice column. Collect everything and filter later.

    5%  of 500 =  25 calls ~  2,500 properties  (about 30 seconds)
   10%  of 500 =  50 calls ~  5,000 properties  (about 1 minute)
   25%  of 500 = 125 calls ~ 12,500 properties  (about 3 minutes)
   50%  of 500 = 250 calls ~ 25,000 properties  (about 6 minutes)
  100%  of 500 = 500 calls ~ 50,000 properties  (about 11 minutes)

Setup:  pip install requests
Run:    python attom_dallas_scraper.py

NOTE: Delete progress.json and dallas_properties.csv before running
so you get a clean pull with the corrected endpoint and fields.
"""

import csv
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests


# ---------------------------------------------------------------------------
# Settings — change these
# ---------------------------------------------------------------------------

API_KEYS = [
    "e229bd3beb35df9aa1a1b20c118a3aca",
    # "YOUR_KEY_2",
    # "YOUR_KEY_3",
]

# How much of your daily 500-call limit to use per key, per run.
# 0.10 = 10% = 50 calls. Good for testing. Set to 1.0 for full runs.
DAILY_LIMIT_PCT = 0.05

OUTPUT_FILE   = "dallas_properties.csv"
PROGRESS_FILE = "progress.json"


# ---------------------------------------------------------------------------
# Constants — don't change these
# ---------------------------------------------------------------------------

ATTOM_DAILY_CAP = 500
PAGE_SIZE       = 100
DELAY           = 0.35

# Using /sale/detail instead of /property/detail —
# this is the only endpoint that returns sale price alongside property info
ATTOM_URL = "https://api.gateway.attomdata.com/propertyapi/v1.0.0/sale/detail"

CALLS_PER_KEY = max(1, min(ATTOM_DAILY_CAP, round(ATTOM_DAILY_CAP * DAILY_LIMIT_PCT)))


# ---------------------------------------------------------------------------
# Dallas metro ZIP codes (no Fort Worth / Tarrant County)
# ---------------------------------------------------------------------------

DALLAS_ZIPS = sorted(set([
    # Dallas city core
    "75201","75202","75203","75204","75205","75206","75207","75208","75209","75210",
    "75211","75212","75214","75215","75216","75217","75218","75219","75220","75223",
    "75224","75225","75226","75227","75228","75229","75230","75231","75232","75233",
    "75234","75235","75236","75237","75238","75240","75241","75243","75244","75246",
    "75247","75248","75249","75251","75252","75253","75254",
    # Garland
    "75040","75041","75042","75043","75044","75046",
    # Mesquite
    "75149","75150","75181",
    # Irving
    "75038","75039","75061","75062","75063",
    # Grand Prairie (Dallas County portion)
    "75050","75051","75052","75053",
    # Duncanville, Cedar Hill, DeSoto, Lancaster
    "75116","75104","75115","75134","75146",
    # Rowlett, Sunnyvale, Balch Springs, Seagoville
    "75030","75089","75180","75182",
    # Farmers Branch, Addison, Coppell
    "75001","75244",
    # Hutchins, Wilmer
    "75141","75172",
    # Plano
    "75023","75024","75025","75074","75075","75093","75094",
    # Frisco
    "75033","75034","75035",
    # McKinney
    "75069","75070","75071","75072",
    # Allen
    "75002","75013",
    # Richardson
    "75080","75081","75082","75083",
    # Wylie, Sachse, Murphy
    "75098","75048",
    # Prosper, Little Elm, Celina, Anna, Melissa
    "75078","75068","75009","75409","75454",
    # Princeton, Lavon, Royse City (Collin portion)
    "75407","75166","75189",
    # Lewisville
    "75029","75056","75057","75067",
    # Carrollton
    "75006","75007","75010","75019",
    # Flower Mound, Highland Village
    "75022","75028","75077",
    # The Colony
    "75056",
    # Denton city
    "76201","76202","76205","76207","76208","76209","76210",
    # Argyle, Northlake, Justin, Sanger, Aubrey
    "76226","76247","76249","76266","76227",
    # Corinth, Lake Dallas
    "76065",
    # Rockwall, Heath, Fate
    "75032","75087","75088",
    # Forney, Terrell, Kaufman, Crandall
    "75126","75160","75142","75114",
    # Waxahachie, Midlothian, Ennis, Red Oak
    "75165","75154","75119","75125",
]))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scraper.log"),
    ]
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def load_progress():
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE) as f:
            data = json.load(f)
        log.info(f"Resuming — {len(data.get('done_zips', []))} ZIPs done, "
                 f"{len(data.get('seen_ids', []))} properties already saved.")
        return data
    return {"done_zips": [], "seen_ids": [], "key_usage": {}}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


# ---------------------------------------------------------------------------
# Key rotation
# ---------------------------------------------------------------------------

class Keys:
    def __init__(self, raw_keys, progress):
        self.keys = [k for k in raw_keys if k and not k.startswith("YOUR_KEY")]
        if not self.keys:
            raise ValueError(
                "No API keys found! Replace YOUR_KEY_1 with your actual ATTOM key."
            )
        self.usage  = progress.setdefault("key_usage", {})
        self._index = 0

    def _today(self):
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _used_today(self, key):
        entry = self.usage.get(key, {})
        return entry.get("calls", 0) if entry.get("date") == self._today() else 0

    def _has_quota(self, key):
        return self._used_today(key) < CALLS_PER_KEY

    def _tick(self, key):
        today = self._today()
        entry = self.usage.get(key, {})
        if entry.get("date") != today:
            self.usage[key] = {"date": today, "calls": 1}
        else:
            self.usage[key]["calls"] = entry["calls"] + 1

    def get(self):
        for _ in range(len(self.keys)):
            key = self.keys[self._index % len(self.keys)]
            self._index += 1
            if self._has_quota(key):
                return key
        return None

    def used_up(self, key):
        self.usage[key] = {"date": self._today(), "calls": CALLS_PER_KEY}
        remaining = sum(1 for k in self.keys if self._has_quota(k))
        log.info(f"Key ...{key[-6:]} exhausted. {remaining} key(s) remaining.")

    def record(self, key):
        self._tick(key)

    def sleep_until_reset(self):
        now   = datetime.now(timezone.utc)
        reset = (now + timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
        secs  = (reset - now).total_seconds()
        log.info(f"All keys exhausted. Sleeping {secs/3600:.1f}h until midnight UTC...")
        while secs > 0:
            time.sleep(min(600, secs))
            secs -= 600
            if secs > 0:
                log.info(f"  Still waiting — {secs/3600:.1f}h left.")
        log.info("Midnight passed, quotas reset. Resuming.")

    def status(self):
        return " | ".join(
            f"...{k[-6:]}: {self._used_today(k)}/{CALLS_PER_KEY}" for k in self.keys
        )


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_attom(key, zip_code, page):
    try:
        resp = requests.get(
            ATTOM_URL,
            headers={"Accept": "application/json", "apikey": key},
            params={"postalCode": zip_code, "pageSize": PAGE_SIZE, "page": page},
            timeout=30,
        )
        if resp.status_code in (401, 403, 429):
            return None, "exhausted"
        resp.raise_for_status()
        return resp.json(), "ok"
    except requests.exceptions.RequestException as e:
        log.warning(f"  Request failed for ZIP {zip_code} page {page}: {e}")
        return None, "error"


# ---------------------------------------------------------------------------
# Flatten property
#
# Field paths verified against real ATTOM /sale/detail responses.
# Sale price lives at sale.amount.saleamt (not sale.saleAmountData).
# All building/construction fields are lowercase.
# ---------------------------------------------------------------------------

def flatten(prop):
    ident  = prop.get("identifier",  {})
    addr   = prop.get("address",     {})
    loc    = prop.get("location",    {})
    lot    = prop.get("lot",         {})
    area   = prop.get("area",        {})
    bldg   = prop.get("building",    {})
    rooms  = bldg.get("rooms",       {})
    constr = bldg.get("construction",{})
    utils  = bldg.get("utilities",   {})
    bsize  = bldg.get("size",        {})
    bsumm  = bldg.get("summary",     {})
    park   = bldg.get("parking",     {})
    inter  = bldg.get("interior",    {})
    asmt   = prop.get("assessment",  {})
    asd    = asmt.get("assessed",    {})
    tax    = asmt.get("tax",         {})
    summ   = prop.get("summary",     {})

    # Sale data — correct paths from verified real response
    sale      = prop.get("sale",     {}) or {}
    amt       = sale.get("amount",       {}) or {}   # sale price lives here
    calc      = sale.get("calculation",  {}) or {}

    # Owner
    owner  = prop.get("owner", {}) or {}
    owner1 = owner.get("owner1", {}) or {}

    return {
        # Identity
        "attomId":              ident.get("attomId"),
        "apn":                  ident.get("apn"),
        "fips":                 ident.get("fips"),

        # Address
        "address":              addr.get("line1"),
        "city":                 addr.get("locality"),
        "state":                addr.get("countrySubd"),
        "zip":                  addr.get("postal1"),

        # Location
        "lat":                  loc.get("latitude"),
        "lng":                  loc.get("longitude"),

        # Property type
        "propertyType":         summ.get("proptype"),
        "propSubtype":          summ.get("propsubtype"),
        "yearBuilt":            summ.get("yearbuilt"),
        "ownerOccupied":        summ.get("absenteeInd"),

        # Size
        "sqft":                 bsize.get("universalsize"),
        "livingArea":           bsize.get("livingsize"),
        "grossSqft":            bsize.get("grosssize"),

        # Lot
        "lotSqft":              lot.get("lotsize2"),
        "lotAcres":             lot.get("lotsize1"),
        "pool":                 lot.get("pooltype"),

        # Rooms
        "beds":                 rooms.get("beds"),
        "bathsFull":            rooms.get("bathsfull"),
        "bathsHalf":            rooms.get("bathspartial"),
        "bathsTotal":           rooms.get("bathstotal"),
        "totalRooms":           rooms.get("roomsTotal"),

        # Building
        "stories":              bsumm.get("levels"),
        "bldgType":             bsumm.get("bldgType"),
        "condition":            constr.get("condition"),

        # Parking
        "garageType":           park.get("prkgType"),
        "garageSize":           park.get("prkgSize"),
        "garageSpaces":         park.get("prkgSpaces"),

        # Interior features
        "basement":             inter.get("bsmttype"),
        "basementSqft":         inter.get("bsmtsize"),
        "fireplaces":           inter.get("fplccount"),

        # Systems
        "heatingType":          utils.get("heatingtype"),
        "coolingType":          utils.get("coolingtype"),

        # Construction
        "constructionType":     constr.get("constructiontype"),
        "foundationType":       constr.get("foundationtype"),
        "roofMaterial":         constr.get("roofcover"),
        "roofType":             constr.get("roofShape"),
        "wallType":             constr.get("wallType"),

        # Area / neighborhood
        "subdivision":          area.get("subdname"),
        "county":               area.get("countrysecsubd"),
        "municipality":         area.get("munname"),

        # Assessment & tax
        "assessedTotal":        asd.get("assdttlvalue"),
        "assessedLand":         asd.get("assdlandvalue"),
        "assessedImprov":       asd.get("assdimprvalue"),
        "taxAmount":            tax.get("taxamt"),
        "taxYear":              tax.get("taxyear"),

        # Sale price — the whole reason we switched endpoints
        # Will be blank for non-disclosed TX sales — that's normal
        "salePrice":            amt.get("saleamt"),
        "saleDate":             sale.get("salesearchdate"),
        "saleTransDate":        sale.get("saleTransDate"),
        "saleType":             amt.get("saletranstype"),
        "saleDocType":          amt.get("saledoctype"),
        "disclosed":            amt.get("saledisclosuretype"),  # 0=disclosed, 1=non-disclosed
        "pricePerSqft":         calc.get("pricepersizeunit"),
        "pricePerBed":          calc.get("priceperbed"),
        "interFamily":          sale.get("interfamily"),
        "newConstruction":      sale.get("resaleornewconstruction"),
        "cashOrMortgage":       sale.get("cashormortgagepurchase"),

        # Owner
        "ownerName":            owner1.get("fullname"),
    }

FIELDS = list(flatten({}).keys())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    total_keys  = len([k for k in API_KEYS if k and not k.startswith("YOUR_KEY")])
    total_calls = total_keys * CALLS_PER_KEY
    est_props   = total_calls * PAGE_SIZE
    est_mins    = (total_calls * (DELAY + 0.75)) / 60

    log.info("=" * 55)
    log.info("Dallas Property + Sale Price Scraper")
    log.info("=" * 55)
    log.info(f"  Keys loaded:       {total_keys}")
    log.info(f"  Daily limit:       {DAILY_LIMIT_PCT*100:.0f}% = {CALLS_PER_KEY} calls per key")
    log.info(f"  Total calls:       {total_calls:,}")
    log.info(f"  Est. properties:   ~{est_props:,}")
    log.info(f"  Est. runtime:      ~{est_mins:.0f} minutes")
    log.info(f"  Endpoint:          /sale/detail (property info + sale price)")
    log.info(f"  Note:              TX non-disclosure = some salePrice will be blank")
    log.info("=" * 55)

    progress  = load_progress()
    done_zips = set(progress.get("done_zips", []))
    seen_ids  = set(progress.get("seen_ids",  []))
    keys      = Keys(API_KEYS, progress)

    csv_exists = Path(OUTPUT_FILE).exists()
    out    = open(OUTPUT_FILE, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out, fieldnames=FIELDS, extrasaction="ignore")
    if not csv_exists:
        writer.writeheader()

    total_written  = 0
    total_with_price = 0
    todo = [z for z in DALLAS_ZIPS if z not in done_zips]
    log.info(f"{len(todo)} ZIPs left to process.\n")

    try:
        for zip_code in todo:
            log.info(f"ZIP {zip_code}  [{keys.status()}]")
            page        = 1
            zip_total   = 0
            total_pages = None

            while True:
                key = keys.get()
                if key is None:
                    save_progress(progress)
                    out.flush()
                    keys.sleep_until_reset()
                    key = keys.get()
                    if key is None:
                        log.error("No keys available after reset. Check your keys.")
                        return

                data, outcome = call_attom(key, zip_code, page)
                keys.record(key)

                if outcome == "exhausted":
                    keys.used_up(key)
                    time.sleep(1)
                    continue

                if outcome == "error" or data is None:
                    log.warning(f"  Skipping page {page} of ZIP {zip_code}.")
                    break

                status = data.get("status", {})
                code   = status.get("code")

                if code == 400 or status.get("msg") == "SuccessWithoutResult":
                    break

                if code != 0:
                    log.warning(f"  Status {code}: {status.get('msg')}")
                    break

                if total_pages is None:
                    total       = status.get("total", 0)
                    total_pages = -(-total // PAGE_SIZE)
                    log.info(f"  {total:,} properties across {total_pages} pages")

                new_count   = 0
                price_count = 0
                for prop in data.get("property", []):
                    row      = flatten(prop)
                    attom_id = str(row.get("attomId", ""))
                    if attom_id and attom_id in seen_ids:
                        continue
                    if attom_id:
                        seen_ids.add(attom_id)
                    writer.writerow(row)
                    new_count += 1
                    if row.get("salePrice"):
                        price_count += 1

                zip_total        += new_count
                total_written    += new_count
                total_with_price += price_count

                pct = (total_with_price / total_written * 100) if total_written else 0
                log.info(f"  Page {page}/{total_pages} — +{new_count} saved  "
                         f"({price_count} had sale price)  "
                         f"[{pct:.0f}% of all have price]  "
                         f"total: {total_written:,}")

                if page >= (total_pages or 1):
                    break

                page += 1
                time.sleep(DELAY)

            done_zips.add(zip_code)
            progress["done_zips"] = list(done_zips)
            progress["seen_ids"]  = list(seen_ids)
            save_progress(progress)
            log.info(f"  Done — ZIP {zip_code} ({zip_total:,} properties)\n")

    finally:
        out.flush()
        out.close()
        save_progress(progress)
        pct = (total_with_price / total_written * 100) if total_written else 0
        log.info("=" * 55)
        log.info(f"  {total_written:,} properties written to {OUTPUT_FILE}")
        log.info(f"  {total_with_price:,} had a sale price ({pct:.1f}%)")
        log.info(f"  Key usage: {keys.status()}")
        log.info("=" * 55)


if __name__ == "__main__":
    main()