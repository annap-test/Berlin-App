from __future__ import annotations

import argparse

from berlin_labels.io import load_neighborhoods, load_csv, save_dataframe
from berlin_labels.geo import ensure_wgs84, compute_area_km2
from berlin_labels.labels_venues import compute_venues_features, compute_vibrancy_scores
from berlin_labels.text import canon_nh


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute venues national cuisines and vibrancy scores")
    ap.add_argument("neighborhoods", help="Path to neighborhoods GeoJSON/CSV")
    ap.add_argument("venues_csv", help="CSV with columns: district_id, neighborhood, cuisine")
    ap.add_argument("--out_features", default="outputs/features_venues_nationals.csv", help="Output CSV for national cuisines features")
    ap.add_argument("--out_vibrancy", default="outputs/features_venues_vibrancy.csv", help="Output CSV for vibrancy scores")
    args = ap.parse_args()

    gdf_nei = load_neighborhoods(args.neighborhoods)
    gdf_nei = ensure_wgs84(gdf_nei)
    gdf_nei = compute_area_km2(gdf_nei)
    area = gdf_nei[["district_id", "neighborhood", "neighborhood_canon", "neighborhood_id", "area_eff_km2"]].copy()
    if "neighborhood_canon" not in area.columns:
        area["neighborhood_canon"] = area["neighborhood"].map(canon_nh)

    df_venues = load_csv(args.venues_csv)
    feats = compute_venues_features(df_venues)
    save_dataframe(feats, args.out_features)

    vib = compute_vibrancy_scores(feats, area)
    save_dataframe(vib, args.out_vibrancy)
    print(f"Wrote {args.out_features} and {args.out_vibrancy}")


if __name__ == "__main__":
    main()

