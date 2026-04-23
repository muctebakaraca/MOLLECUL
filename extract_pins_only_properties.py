#!/usr/bin/env python3
"""Extract map-ready property fields from Dallas property CSV exports."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

OUTPUT_HEADERS = [
    "address",
    "latitude",
    "longitude",
    "last_sale_price",
    "beds",
    "baths",
    "sq_ft",
    "attom_avm",
]

DISCOVERY_PATTERNS = [
    "PART1_NEW_Dallas_Properties.csv",
    "data/PART1_NEW_Dallas_Properties.csv",
    "*PART1*Dallas*Properties*.csv",
    "data/*PART1*Dallas*Properties*.csv",
]

FALLBACK_PATTERNS = [
    "PART1*.csv",
    "data/PART1*.csv",
]

COLUMN_ALIASES = {
    "address": ["address", "propertyAddress", "streetAddress", "property_address"],
    "city": ["city"],
    "state": ["state"],
    "zip": ["zip", "zipcode", "zipCode", "postalCode"],
    "latitude": ["lat", "latitude"],
    "longitude": ["lng", "lon", "longitude", "longtitude"],
    "sale_price": ["salePrice", "lastSalePrice", "sale_price", "last_sale_price"],
    "beds": ["beds", "bedrooms"],
    "baths_total": ["bathsTotal", "baths", "bathrooms"],
    "baths_full": ["bathsFull", "fullBaths", "full_baths"],
    "baths_half": ["bathsHalf", "halfBaths", "half_baths"],
    "sqft": ["sqft", "sq_ft", "squareFeet", "square_feet", "livingArea", "grossSqft"],
    "avm": ["avmValue", "attomAVM", "attomAvm", "avm", "attom_avm"],
}

HEADERLESS_INDEXES = {
    "address": 3,
    "city": 4,
    "state": 5,
    "zip": 6,
    "latitude": 7,
    "longitude": 8,
    "sqft": 22,
    "beds": 29,
    "baths_full": 30,
    "baths_half": 31,
    "baths_total": 32,
    "sale_price": 57,
    "avm": 71,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract address, coordinates, sale price, beds, baths, square footage, "
            "and ATTOM AVM into a new CSV."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Input CSV files. If omitted, the script auto-discovers the Part1 Dallas property CSV.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="PinsOnlyProperties.csv",
        help="Output CSV path. Defaults to PinsOnlyProperties.csv in the current directory.",
    )
    return parser.parse_args()


def normalize_name(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def clean_text(value: str | None) -> str:
    return (value or "").strip()


def looks_like_header(values: list[str]) -> bool:
    normalized = {normalize_name(value) for value in values if value}
    return {"address", "lat", "lng"}.issubset(normalized) or {
        "address",
        "latitude",
        "longitude",
    }.issubset(normalized)


def parse_number(value: str | None) -> float | None:
    text = clean_text(value).replace("$", "").replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def format_number(value: str | float | None, integer_if_whole: bool = False) -> str:
    numeric = parse_number(value if isinstance(value, str) or value is None else str(value))
    if numeric is None:
        return clean_text(value if isinstance(value, str) else "")
    if integer_if_whole and numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.12g}"


def first_non_empty(row: dict[str, str], columns: list[str | None]) -> str:
    for column in columns:
        if column:
            value = clean_text(row.get(column))
            if value:
                return value
    return ""


def resolve_columns(fieldnames: list[str]) -> dict[str, str | None]:
    normalized = {normalize_name(name): name for name in fieldnames if name}
    resolved: dict[str, str | None] = {}
    for key, aliases in COLUMN_ALIASES.items():
        resolved[key] = None
        for alias in aliases:
            column = normalized.get(normalize_name(alias))
            if column:
                resolved[key] = column
                break

    required = ["address", "latitude", "longitude"]
    missing = [name for name in required if not resolved.get(name)]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    return resolved


def build_full_address(row: dict[str, str], columns: dict[str, str | None]) -> str:
    street = clean_text(row.get(columns["address"])) if columns["address"] else ""
    city = clean_text(row.get(columns["city"])) if columns["city"] else ""
    state = clean_text(row.get(columns["state"])) if columns["state"] else ""
    zip_code = clean_text(row.get(columns["zip"])) if columns["zip"] else ""

    parts = [street]
    if city:
        parts.append(city)

    state_zip = " ".join(part for part in [state, zip_code] if part)
    if state_zip:
        parts.append(state_zip)

    return ", ".join(part for part in parts if part)


def build_address_from_parts(
    street: str | None, city: str | None, state: str | None, zip_code: str | None
) -> str:
    parts = [clean_text(street)]
    if clean_text(city):
        parts.append(clean_text(city))

    state_zip = " ".join(part for part in [clean_text(state), clean_text(zip_code)] if part)
    if state_zip:
        parts.append(state_zip)

    return ", ".join(part for part in parts if part)


def build_baths(row: dict[str, str], columns: dict[str, str | None]) -> str:
    total = clean_text(row.get(columns["baths_total"])) if columns["baths_total"] else ""
    if total:
        return format_number(total, integer_if_whole=True)

    full_baths = parse_number(row.get(columns["baths_full"])) if columns["baths_full"] else None
    half_baths = parse_number(row.get(columns["baths_half"])) if columns["baths_half"] else None

    if full_baths is None and half_baths is None:
        return ""

    total_baths = (full_baths or 0.0) + ((half_baths or 0.0) * 0.5)
    return format_number(total_baths, integer_if_whole=True)


def build_baths_from_parts(
    total: str | None, full_baths: str | None, half_baths: str | None
) -> str:
    if clean_text(total):
        return format_number(total, integer_if_whole=True)

    full_value = parse_number(full_baths)
    half_value = parse_number(half_baths)

    if full_value is None and half_value is None:
        return ""

    total_baths = (full_value or 0.0) + ((half_value or 0.0) * 0.5)
    return format_number(total_baths, integer_if_whole=True)


def value_at(values: list[str], index: int) -> str:
    if index >= len(values):
        return ""
    return clean_text(values[index])


def build_output_row(row: dict[str, str], columns: dict[str, str | None]) -> dict[str, str]:
    return {
        "address": build_full_address(row, columns),
        "latitude": format_number(row.get(columns["latitude"])),
        "longitude": format_number(row.get(columns["longitude"])),
        "last_sale_price": format_number(
            row.get(columns["sale_price"]), integer_if_whole=True
        ),
        "beds": format_number(row.get(columns["beds"]), integer_if_whole=True),
        "baths": build_baths(row, columns),
        "sq_ft": format_number(
            first_non_empty(row, [columns["sqft"]]), integer_if_whole=True
        ),
        "attom_avm": format_number(row.get(columns["avm"]), integer_if_whole=True),
    }


def build_output_row_from_headerless(values: list[str]) -> dict[str, str]:
    return {
        "address": build_address_from_parts(
            value_at(values, HEADERLESS_INDEXES["address"]),
            value_at(values, HEADERLESS_INDEXES["city"]),
            value_at(values, HEADERLESS_INDEXES["state"]),
            value_at(values, HEADERLESS_INDEXES["zip"]),
        ),
        "latitude": format_number(value_at(values, HEADERLESS_INDEXES["latitude"])),
        "longitude": format_number(value_at(values, HEADERLESS_INDEXES["longitude"])),
        "last_sale_price": format_number(
            value_at(values, HEADERLESS_INDEXES["sale_price"]), integer_if_whole=True
        ),
        "beds": format_number(
            value_at(values, HEADERLESS_INDEXES["beds"]), integer_if_whole=True
        ),
        "baths": build_baths_from_parts(
            value_at(values, HEADERLESS_INDEXES["baths_total"]),
            value_at(values, HEADERLESS_INDEXES["baths_full"]),
            value_at(values, HEADERLESS_INDEXES["baths_half"]),
        ),
        "sq_ft": format_number(
            value_at(values, HEADERLESS_INDEXES["sqft"]), integer_if_whole=True
        ),
        "attom_avm": format_number(
            value_at(values, HEADERLESS_INDEXES["avm"]), integer_if_whole=True
        ),
    }


def natural_file_key(path: Path) -> tuple[int, int | str, str]:
    match = re.search(r"part(\d+)", path.name, re.IGNORECASE)
    if match:
        return (0, int(match.group(1)), path.name.lower())
    return (1, path.name.lower(), path.as_posix().lower())


def discover_input_files(base_dir: Path, output_path: Path) -> list[Path]:
    files: list[Path] = []

    for pattern in DISCOVERY_PATTERNS:
        files.extend(path for path in base_dir.glob(pattern) if path.is_file())

    if not files:
        for pattern in FALLBACK_PATTERNS:
            files.extend(path for path in base_dir.glob(pattern) if path.is_file())

    unique_files = []
    seen: set[Path] = set()
    for path in sorted(files, key=natural_file_key):
        resolved = path.resolve()
        if resolved == output_path.resolve() or resolved in seen:
            continue
        seen.add(resolved)
        unique_files.append(path)

    return unique_files


def load_input_files(input_args: list[str], output_path: Path) -> list[Path]:
    if input_args:
        files = [Path(value).expanduser() for value in input_args]
    else:
        files = discover_input_files(Path.cwd(), output_path)

    if not files:
        raise FileNotFoundError(
            "No matching input CSV files found. Pass one or more files explicitly."
        )

    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input file(s) not found: {', '.join(missing)}")

    return files


def extract_rows(input_files: list[Path]) -> list[dict[str, str]]:
    extracted_rows: list[dict[str, str]] = []

    for input_file in input_files:
        with input_file.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            first_row = next(reader, None)
            if not first_row:
                raise ValueError(f"{input_file} does not contain a header row.")

            if looks_like_header(first_row):
                columns = resolve_columns(first_row)
                for values in reader:
                    row = {
                        column: values[index] if index < len(values) else ""
                        for index, column in enumerate(first_row)
                    }
                    extracted_rows.append(build_output_row(row, columns))
                continue

            extracted_rows.append(build_output_row_from_headerless(first_row))
            for values in reader:
                extracted_rows.append(build_output_row_from_headerless(values))

    return extracted_rows


def write_output(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_path = Path(args.output).expanduser()

    try:
        input_files = load_input_files(args.inputs, output_path)
        rows = extract_rows(input_files)
        write_output(output_path, rows)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Wrote {len(rows)} properties from {len(input_files)} file(s) to {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
