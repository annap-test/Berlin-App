from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from berlin_labels.io import load_neighborhoods, load_csv, save_dataframe
from berlin_labels.geo import ensure_wgs84, compute_area_km2
from berlin_labels.labels_parks import compute_parks_labels, label_green_share


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute parks green share and labels")
    ap.add_argument("neighborhoods", help="Path to neighborhoods GeoJSON/CSV")
    ap.add_argument("parks_csv", help="CSV with columns: district_id, neighborhood, size_sqm")
    ap.add_argument("--out", default="outputs/parks_features.csv", help="Output CSV path")
    args = ap.parse_args()

    gdf_nei = load_neighborhoods(args.neighborhoods)
    gdf_nei = ensure_wgs84(gdf_nei)
    gdf_nei = compute_area_km2(gdf_nei)
    area_keys = gdf_nei[["district_id", "neighborhood", "neighborhood_canon", "area_km2"]].copy()
    if "neighborhood_canon" not in area_keys.columns:
        area_keys["neighborhood_canon"] = area_keys["neighborhood"].str.lower()

    df_parks = load_csv(args.parks_csv)
    parks = compute_parks_labels(df_parks)
    merged = parks.merge(area_keys, on=["district_id", "neighborhood_canon"], how="left")
    out = label_green_share(merged)
    save_dataframe(out, args.out)
    print(f"Wrote {args.out} with {len(out)} rows")


if __name__ == "__main__":
    main()

