from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path

from berlin_labels.io import load_neighborhoods, load_csv, save_dataframe
from berlin_labels.labels_mobility import compute_mobility_labels
from berlin_labels.geo import compute_area_km2, ensure_wgs84
from berlin_labels.text import canon_nh


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute mobility labels from U-Bahn and Bus/Tram stops")
    ap.add_argument("neighborhoods", help="Path to neighborhoods GeoJSON/CSV")
    ap.add_argument("ubahn_csv", help="CSV of U-Bahn stations with lat/lon or geometry")
    ap.add_argument("bus_tram_csv", help="CSV of Bus/Tram stops with lat/lon or geometry")
    ap.add_argument("--out", default="outputs/mobility_labels.csv", help="Output CSV path")
    args = ap.parse_args()

    gdf_nei = load_neighborhoods(args.neighborhoods)
    gdf_nei = ensure_wgs84(gdf_nei)
    gdf_nei = compute_area_km2(gdf_nei)

    df_ubahn = load_csv(args.ubahn_csv)
    df_bus = load_csv(args.bus_tram_csv)

    labels = compute_mobility_labels(gdf_nei, df_ubahn, df_bus)
    save_dataframe(labels, args.out)
    print(f"Wrote {args.out} with {len(labels)} rows")


if __name__ == "__main__":
    main()

