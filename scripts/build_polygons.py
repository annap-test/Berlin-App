from __future__ import annotations

import argparse
from pathlib import Path

from berlin_labels.geo import compute_area_km2, ensure_wgs84
from berlin_labels.io import load_neighborhoods, save_geodataframe
from berlin_labels.text import canon_nh


def main() -> None:
    ap = argparse.ArgumentParser(description="Build neighborhoods polygons with area_km2")
    ap.add_argument("neighborhoods", help="Path to neighborhoods GeoJSON/CSV")
    ap.add_argument("--out", default="outputs/neighborhoods.geojson", help="Output GeoJSON path")
    args = ap.parse_args()

    gdf = load_neighborhoods(args.neighborhoods)
    gdf = ensure_wgs84(gdf)
    gdf = compute_area_km2(gdf)
    if "neighborhood" not in gdf.columns:
        raise ValueError("neighborhoods file must have a 'neighborhood' column")
    gdf["neighborhood_canon"] = gdf["neighborhood"].map(canon_nh)
    save_geodataframe(gdf, args.out)
    print(f"Wrote {args.out} with {len(gdf)} neighborhoods")


if __name__ == "__main__":
    main()

