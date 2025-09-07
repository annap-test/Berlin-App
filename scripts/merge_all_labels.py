from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from berlin_labels.io import load_neighborhoods, save_dataframe
from berlin_labels.geo import ensure_wgs84, compute_area_km2
from berlin_labels.text import canon_nh


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge all label outputs into final CSVs for the app")
    ap.add_argument("neighborhoods", help="Path to neighborhoods GeoJSON/CSV")
    ap.add_argument("--mobility", default="outputs/mobility_labels.csv")
    ap.add_argument("--parks", default="outputs/parks_features.csv")
    ap.add_argument("--playgrounds", default="outputs/playgrounds_features.csv")
    ap.add_argument("--venues_features", default="outputs/features_venues_nationals.csv")
    ap.add_argument("--venues_vibrancy", default="outputs/features_venues_vibrancy.csv")
    ap.add_argument("--out_minimal", default="outputs/neighborhood_labels_minimal.csv")
    ap.add_argument("--out_merged", default="outputs/neighborhood_labels_with_venues.csv")
    args = ap.parse_args()

    gdf_nei = load_neighborhoods(args.neighborhoods)
    gdf_nei = ensure_wgs84(gdf_nei)
    gdf_nei = compute_area_km2(gdf_nei)
    base = gdf_nei[[
        "district_id",
        "district" if "district" in gdf_nei.columns else "district_id",
        "neighborhood_id",
        "neighborhood",
        "neighborhood_canon" if "neighborhood_canon" in gdf_nei.columns else "neighborhood",
        "area_km2",
    ]].copy()
    base.columns = ["district_id", "district", "neighborhood_id", "neighborhood", "neighborhood_canon", "area_km2"]
    base["neighborhood_canon"] = base["neighborhood"].map(canon_nh)

    # Load features
    mob = pd.read_csv(args.mobility) if Path(args.mobility).exists() else pd.DataFrame()
    parks = pd.read_csv(args.parks) if Path(args.parks).exists() else pd.DataFrame()
    play = pd.read_csv(args.playgrounds) if Path(args.playgrounds).exists() else pd.DataFrame()
    venF = pd.read_csv(args.venues_features) if Path(args.venues_features).exists() else pd.DataFrame()
    venV = pd.read_csv(args.venues_vibrancy) if Path(args.venues_vibrancy).exists() else pd.DataFrame()

    # Minimal: mobility + parks + playgrounds
    minimal = base.copy()
    if not mob.empty:
        minimal = minimal.merge(
            mob.drop(columns=[c for c in ["district", "area_eff_km2"] if c in mob.columns]),
            on=["district_id", "neighborhood_id", "neighborhood", "neighborhood_canon"],
            how="left",
        )
    if not parks.empty:
        minimal = minimal.merge(
            parks[["district_id", "neighborhood_canon", "green_area_km2", "green_share", "green_share_label"]],
            on=["district_id", "neighborhood_canon"],
            how="left",
        )
    if not play.empty:
        minimal = minimal.merge(
            play[["district_id", "neighborhood_canon", "n_playgrounds", "playgrounds_per_km2", "playgrounds_density_label"]],
            on=["district_id", "neighborhood_canon"],
            how="left",
        )

    save_dataframe(minimal.drop(columns=["neighborhood_canon"]), args.out_minimal)

    # With venues merged
    merged = minimal.copy()
    if not venF.empty:
        merged = merged.merge(
            venF,
            on=["district_id", "neighborhood_canon"],
            how="left",
        )
    if not venV.empty:
        merged = merged.merge(
            venV[[
                "district_id",
                "neighborhood_canon",
                "venues_per_km2",
                "n_cuisine_types",
                "V_score",
                "C_score",
                "VV_index",
                "vibrancy_label",
            ]],
            on=["district_id", "neighborhood_canon"],
            how="left",
        )

    save_dataframe(merged.drop(columns=["neighborhood_canon"]), args.out_merged)
    print(f"Wrote {args.out_minimal} and {args.out_merged}")


if __name__ == "__main__":
    main()

