from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def _out_dir(base: Path | None = None) -> Path:
    if base is not None:
        return base
    # Prefer notebooks' outputs directory
    cand = Path("labels_with_visualization") / "outputs"
    return cand if cand.exists() else Path("outputs")


def _single_tag(col: str, val: str) -> str | None:
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if col == "mobility_label":
        m = {"well-connected": "#well_connected_mobility", "moderate": "#moderate_mobility", "remote": "#remote_mobility"}
        return m.get(s)
    if col == "vibrancy_label":
        m = {"vibrant": "#vibrant_venues", "average": "#average_venues", "sparse": "#sparse_venues"}
        return m.get(s)
    if col == "playgrounds_density_label":
        m = {"below average": "#low_playground_density", "average": "#average_playground_density", "above average": "#high_playground_density"}
        return m.get(s)
    if col == "green_share_label":
        # Accept both neighborhood style (above/below/average) and district style (low_/average_/high_green_share)
        m = {
            "below average": "#low_green_share",
            "average": "#average_green_share",
            "above average": "#high_green_share",
            "low_green_share": "#low_green_share",
            "average_green_share": "#average_green_share",
            "high_green_share": "#high_green_share",
        }
        return m.get(s)
    # Generic: accept existing hashtags from wide tables
    if s.startswith("#"):
        return s
    # Otherwise normalize value into hashtag
    norm = "#" + s.replace(" ", "_").replace("-", "_")
    return norm


def rebuild_neighborhoods(out: Path) -> int:
    nw_path = out / "berlin_neighborhoods_labels_wide.csv"
    if not nw_path.exists():
        return 0
    NW = pd.read_csv(nw_path)
    rows: list[dict] = []
    for _, r in NW.iterrows():
        key = {"district": r.get("district"), "neighborhood": r.get("neighborhood")}
        for col in ("mobility_label", "vibrancy_label", "playgrounds_density_label", "green_share_label"):
            if col in NW.columns:
                tag = _single_tag(col, r.get(col))
                if tag:
                    theme = col.replace("_label", "")
                    rows.append({**key, "hashtags": tag, "source": f"{theme}: rule-based"})
    if not rows:
        return 0
    df = pd.DataFrame(rows).drop_duplicates(subset=["district", "neighborhood", "hashtags", "source"]).reset_index(drop=True)
    df.to_csv(out / "berlin_neighborhoods_labels_long_v2.csv", index=False)
    return len(df)


def rebuild_districts(out: Path) -> int:
    dw_path = out / "berlin_districts_labels_wide.csv"
    if not dw_path.exists():
        return 0
    DW = pd.read_csv(dw_path)
    rows: list[dict] = []
    for _, r in DW.iterrows():
        key = {"district": r.get("district")}
        # Include base/composite labels created by districts_with_income
        dist_cols = (
            "income_label",
            "safety_label",
            "unemployment_label",
            "density_label",
            "diversity_label",
            "income_safety_label",
            "urbanity_label",
            # plus other themes
            "mobility_label",
            "vibrancy_label",
            "playgrounds_density_label",
            "green_share_label",
        )
        for col in dist_cols:
            if col in DW.columns:
                tag = _single_tag(col, r.get(col))
                if tag:
                    theme = col.replace("_label", "")
                    rows.append({**key, "hashtags": tag, "source": f"{theme}: rule-based"})
    if not rows:
        return 0
    df = pd.DataFrame(rows).drop_duplicates(subset=["district", "hashtags", "source"]).reset_index(drop=True)
    df.to_csv(out / "berlin_districts_labels_long_v2.csv", index=False)
    return len(df)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild unified long_v2 label tables from wide tables")
    ap.add_argument("--out_dir", help="Outputs directory (defaults to labels_with_visualization/outputs or outputs)")
    args = ap.parse_args()

    out = _out_dir(Path(args.out_dir)) if args.out_dir else _out_dir()
    out.mkdir(parents=True, exist_ok=True)
    n_nei = rebuild_neighborhoods(out)
    n_dist = rebuild_districts(out)
    print(f"Wrote long_v2: neighborhoods={n_nei} rows, districts={n_dist} rows -> {out}")


if __name__ == "__main__":
    main()
