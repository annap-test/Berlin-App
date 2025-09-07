from __future__ import annotations

import argparse

from berlin_labels.io import load_neighborhoods, load_csv, save_dataframe
from berlin_labels.geo import ensure_wgs84, compute_area_km2
from berlin_labels.labels_playgrounds import compute_playgrounds_labels, label_playgrounds_density


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute playground densities and labels")
    ap.add_argument("neighborhoods", help="Path to neighborhoods GeoJSON/CSV")
    ap.add_argument("playgrounds_csv", help="CSV with columns: district_id, neighborhood, green_area_type")
    ap.add_argument("--out", default="outputs/playgrounds_features.csv", help="Output CSV path")
    args = ap.parse_args()

    gdf_nei = load_neighborhoods(args.neighborhoods)
    gdf_nei = ensure_wgs84(gdf_nei)
    gdf_nei = compute_area_km2(gdf_nei)
    area_keys = gdf_nei[["district_id", "neighborhood", "neighborhood_canon", "area_eff_km2"]].copy()
    if "neighborhood_canon" not in area_keys.columns:
        area_keys["neighborhood_canon"] = area_keys["neighborhood"].str.lower()

    df_play = load_csv(args.playgrounds_csv)
    play = compute_playgrounds_labels(df_play)
    merged = play.merge(area_keys, on=["district_id", "neighborhood_canon"], how="left")
    out = label_playgrounds_density(merged)
    save_dataframe(out, args.out)
    print(f"Wrote {args.out} with {len(out)} rows")


if __name__ == "__main__":
    main()

