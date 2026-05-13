"""
Download ERA5 fields needed for NAO and wind predictors.

Downloads two products per year:
  1. MSLP over the North Atlantic (20-80N, 90W-40E) — for NAO EOF
  2. 10m u/v winds over the East Coast domain (5-50N, 98-50W) — for wind stress EOF

Requires ~/.cdsapirc with your CDS API key:
    url: https://cds.climate.copernicus.eu/api/v2
    key: <your-uid>:<your-api-key>

Usage:
    python scripts/download_era5_slp.py [--year YYYY] [--product mslp|wind|all] [--dry-run]
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "data" / "raw" / "era5"

PRODUCTS = {
    "mslp": {
        "variable": ["mean_sea_level_pressure"],
        "area": [80, -90, 20, 40],   # N, W, S, E  (North Atlantic for NAO)
        "file_prefix": "era5_mslp",
    },
    "wind": {
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "area": [50, -98, 5, -50],   # East Coast + Caribbean (same as GLORYS domain)
        "file_prefix": "era5_wind_east_coast",
    },
}


def _download(year: int, product: str, dry_run: bool) -> None:
    cfg = PRODUCTS[product]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{cfg['file_prefix']}_{year}.nc"

    if out_path.exists():
        print(f"    [skip] {out_path.name}")
        return
    if dry_run:
        print(f"    [dry-run] would download {out_path.name}")
        return

    import cdsapi
    c = cdsapi.Client()
    print(f"    Downloading {out_path.name} ...")
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": cfg["variable"],
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day":   [f"{d:02d}" for d in range(1, 32)],
            "time":  "12:00",
            "area":  cfg["area"],
            "format": "netcdf",
        },
        str(out_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ERA5 for NAO and wind predictors")
    parser.add_argument("--year",       type=int,   default=None)
    parser.add_argument("--start-year", type=int,   default=1993)
    parser.add_argument("--end-year",   type=int,   default=2024)
    parser.add_argument("--product",    choices=["mslp", "wind", "all"], default="all")
    parser.add_argument("--dry-run",    action="store_true")
    args = parser.parse_args()

    products = list(PRODUCTS) if args.product == "all" else [args.product]
    years    = [args.year] if args.year else range(args.start_year, args.end_year + 1)

    for year in years:
        print(f"Year {year}")
        for p in products:
            _download(year, p, dry_run=args.dry_run)

    print("Done.")


if __name__ == "__main__":
    main()
