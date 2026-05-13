"""
Download GLORYS12 daily SST and SSH for 1993-2024.

Usage:
    python scripts/download_glorys_bulk.py [--year YYYY] [--region east_coast|nino34] [--dry-run]

Credentials are read from env vars CMEMS_USER and CMEMS_PASS.
"""
import argparse
import os
import sys
from pathlib import Path

import copernicusmarine

ROOT = Path(__file__).parent.parent

# GLORYS dataset IDs
# MY covers 1993–present (confirmed covering through 2024 as of 2026-04).
# NRT retained as fallback for future years beyond MY coverage.
DATASET_MY  = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
DATASET_NRT = "cmems_mod_glo_phy_nrt_0.083deg_P1D-m"

MY_END_YEAR = 2024


def _pick_dataset(year: int) -> str:
    return DATASET_MY if year <= MY_END_YEAR else DATASET_NRT

REGIONS = {
    "east_coast": dict(
        minimum_latitude=5.0,
        maximum_latitude=50.0,
        minimum_longitude=-98.0,
        maximum_longitude=-50.0,
        variables=["thetao", "zos"],
        out_dir=ROOT / "data" / "raw" / "glorys",
        file_prefix="glorys",
        file_suffix="east_coast",
    ),
    "nino34": dict(
        minimum_latitude=-5.0,
        maximum_latitude=5.0,
        minimum_longitude=-170.0,
        maximum_longitude=-120.0,
        variables=["thetao"],
        out_dir=ROOT / "data" / "raw" / "glorys_nino34",
        file_prefix="glorys_nino34",
        file_suffix="sst",
    ),
}


def download_year(year: int, region: str, dry_run: bool = False) -> None:
    cfg = REGIONS[region]
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)

    dataset_id = _pick_dataset(year)
    start = f"{year}-01-01T00:00:00"
    end   = f"{year}-12-31T23:59:59"

    variables = cfg["variables"]

    # For east_coast, download SST and SSH into separate files
    if region == "east_coast":
        for var, varname in [("thetao", "sst"), ("zos", "ssh")]:
            out_path = cfg["out_dir"] / f"glorys_{varname}_{year}_{cfg['file_suffix']}.nc"
            if out_path.exists():
                print(f"  [skip] {out_path.name} already exists")
                continue
            if dry_run:
                print(f"  [dry-run] would download {out_path.name} from {dataset_id}")
                continue
            print(f"  Downloading {out_path.name} ...")
            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=[var],
                minimum_latitude=cfg["minimum_latitude"],
                maximum_latitude=cfg["maximum_latitude"],
                minimum_longitude=cfg["minimum_longitude"],
                maximum_longitude=cfg["maximum_longitude"],
                start_datetime=start,
                end_datetime=end,
                minimum_depth=0.0,
                maximum_depth=1.0,
                output_filename=out_path.name,
                output_directory=str(cfg["out_dir"]),
                force_download=True,
            )
    else:
        out_path = cfg["out_dir"] / f"{cfg['file_prefix']}_{year}_{cfg['file_suffix']}.nc"
        if out_path.exists():
            print(f"  [skip] {out_path.name} already exists")
            return
        if dry_run:
            print(f"  [dry-run] would download {out_path.name} from {dataset_id}")
            return
        print(f"  Downloading {out_path.name} ...")
        copernicusmarine.subset(
            dataset_id=dataset_id,
            variables=cfg["variables"],
            minimum_latitude=cfg["minimum_latitude"],
            maximum_latitude=cfg["maximum_latitude"],
            minimum_longitude=cfg["minimum_longitude"],
            maximum_longitude=cfg["maximum_longitude"],
            start_datetime=start,
            end_datetime=end,
            minimum_depth=0.0,
            maximum_depth=1.0,
            output_filename=out_path.name,
            output_directory=str(cfg["out_dir"]),
            force_download=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk GLORYS download")
    parser.add_argument("--year", type=int, default=None, help="Download a single year")
    parser.add_argument("--region", choices=list(REGIONS), default="east_coast")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-year", type=int, default=1993)
    parser.add_argument("--end-year",   type=int, default=2024)
    args = parser.parse_args()

    user = os.environ.get("CMEMS_USER")
    pwd  = os.environ.get("CMEMS_PASS")
    if not user or not pwd:
        sys.exit("Set CMEMS_USER and CMEMS_PASS environment variables before running.")

    copernicusmarine.login(username=user, password=pwd, force_overwrite=True)

    years = [args.year] if args.year else range(args.start_year, args.end_year + 1)
    for year in years:
        print(f"Year {year} | region={args.region}")
        download_year(year, args.region, dry_run=args.dry_run)

    print("Done.")


if __name__ == "__main__":
    main()
