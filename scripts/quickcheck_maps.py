from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from berlin_labels.io import load_neighborhoods
from berlin_labels.geo import ensure_wgs84
from berlin_labels.viz import make_quickcheck_maps, save_maps


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate quick Folium maps to inspect labels")
    ap.add_argument("neighborhoods", help="Path to neighborhoods GeoJSON/CSV")
    ap.add_argument("merged_csv", help="neighborhood_labels_with_venues.csv (canonical)")
    ap.add_argument("--out_dir", default="outputs/maps", help="Directory for HTML maps")
    args = ap.parse_args()

    gdf = load_neighborhoods(args.neighborhoods)
    gdf = ensure_wgs84(gdf)
    df = pd.read_csv(args.merged_csv)

    # Prefer join on IDs if present; fallback to neighborhood name
    keys = [c for c in ["district_id", "neighborhood_id"] if c in df.columns and c in gdf.columns]
    if keys:
        j = gdf.merge(df, on=keys, how="left")
    else:
        j = gdf.merge(df, on=["district_id", "neighborhood"], how="left")

    maps = make_quickcheck_maps(j)
    save_maps(maps, args.out_dir)
    print(f"Saved maps to {args.out_dir}")


if __name__ == "__main__":
    main()

