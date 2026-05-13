"""
Download NOAA CO-OPS monthly mean sea level for 8 East Coast stations.

Usage:
    python scripts/download_tide_gauges.py [--dry-run]
"""
import argparse
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "data" / "raw" / "tide_gauges"

STATIONS = [
    {"id": "8410140", "name": "Eastport_ME"},
    {"id": "8443970", "name": "Boston_MA"},
    {"id": "8534720", "name": "Atlantic_City_NJ"},
    {"id": "8518750", "name": "Battery_NY"},
    {"id": "8638610", "name": "Sewells_Point_VA"},
    {"id": "8665530", "name": "Charleston_SC"},
    {"id": "8720218", "name": "Mayport_FL"},
    {"id": "8724580", "name": "Key_West_FL"},
]

BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"


def download_station(station: dict, begin_year: int, end_year: int, dry_run: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{station['id']}_{station['name']}.csv"
    if out_path.exists():
        print(f"  [skip] {out_path.name}")
        return
    if dry_run:
        print(f"  [dry-run] would download {out_path.name}")
        return

    params = (
        f"?product=monthly_mean"
        f"&application=ssh_diffusion"
        f"&begin_date={begin_year}0101"
        f"&end_date={end_year}1231"
        f"&datum=MSL"
        f"&station={station['id']}"
        f"&time_zone=GMT"
        f"&units=metric"
        f"&format=json"
    )
    url = BASE_URL + params
    print(f"  Downloading {out_path.name} ...")
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read())

    if "error" in data:
        print(f"  WARNING: {data['error']['message']}")
        return

    # monthly_mean records contain year, month, and MSL in meters
    rows = ["date,msl_m"]
    for rec in data.get("data", []):
        yr = int(rec["year"])
        mo = int(rec["month"])
        msl = rec.get("MSL") or rec.get("msl") or rec.get("v")
        if msl is None or msl == "":
            continue
        rows.append(f"{yr:04d}-{mo:02d}-01,{msl}")

    out_path.write_text("\n".join(rows))
    print(f"  Saved {len(rows)-1} records → {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NOAA CO-OPS tide gauge data")
    parser.add_argument("--begin-year", type=int, default=1993)
    parser.add_argument("--end-year",   type=int, default=2024)
    parser.add_argument("--dry-run",    action="store_true")
    args = parser.parse_args()

    for station in STATIONS:
        print(f"Station {station['id']} ({station['name']})")
        download_station(station, args.begin_year, args.end_year, args.dry_run)

    print("Done.")


if __name__ == "__main__":
    main()
