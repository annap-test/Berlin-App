from __future__ import annotations

import argparse
from pathlib import Path

from berlin_labels.pipeline import Paths, run_all_preprocessing, make_readable_columns
from berlin_labels.io import save_dataframe


def main() -> None:
    ap = argparse.ArgumentParser(description="Run all preprocessing and write final CSVs")
    # Option A: provide a raw directory with standard filenames
    ap.add_argument("--raw_dir", help="Directory containing standard input files (neighborhoods.geojson, ubahns.csv, bus_tram_stops.csv, parks.csv, playgrounds.csv, venues.csv)")
    # Option B: explicit paths (remain compatible). Mark as optional when using --raw_dir
    ap.add_argument("neighborhoods", nargs="?", help="Path to neighborhoods GeoJSON/CSV")
    ap.add_argument("ubahn_csv", nargs="?")
    ap.add_argument("bus_tram_csv", nargs="?")
    ap.add_argument("parks_csv", nargs="?")
    ap.add_argument("playgrounds_csv", nargs="?")
    ap.add_argument("venues_csv", nargs="?")
    ap.add_argument("--out_dir", default="outputs", help="Directory to write outputs")
    ap.add_argument("--readable_csv", default="outputs/neighborhood_labels_readable.csv", help="Readable single CSV path")
    args = ap.parse_args()

    if args.raw_dir:
        base = Path(args.raw_dir)
        neighborhoods = base / "neighborhoods.geojson"
        ubahn_csv = base / "ubahns.csv"
        bus_tram_csv = base / "bus_tram_stops.csv"
        parks_csv = base / "parks.csv"
        playgrounds_csv = base / "playgrounds.csv"
        venues_csv = base / "venues.csv"
        missing = [p for p in [neighborhoods, ubahn_csv, bus_tram_csv, parks_csv, playgrounds_csv, venues_csv] if not p.exists()]
        if missing:
            files = "\n".join(str(p) for p in missing)
            raise SystemExit(
                "Some expected files were not found under --raw_dir.\n" \
                "Expected: neighborhoods.geojson, ubahns.csv, bus_tram_stops.csv, parks.csv, playgrounds.csv, venues.csv\n" \
                f"Missing:\n{files}"
            )
    else:
        # Fallback to explicit arguments
        required = [args.neighborhoods, args.ubahn_csv, args.bus_tram_csv, args.parks_csv, args.playgrounds_csv, args.venues_csv]
        if any(v is None for v in required):
            ap.error("Provide either --raw_dir or all six explicit paths.")
        neighborhoods = Path(args.neighborhoods)
        ubahn_csv = Path(args.ubahn_csv)
        bus_tram_csv = Path(args.bus_tram_csv)
        parks_csv = Path(args.parks_csv)
        playgrounds_csv = Path(args.playgrounds_csv)
        venues_csv = Path(args.venues_csv)

    paths = Paths(
        neighborhoods=neighborhoods,
        ubahn_csv=ubahn_csv,
        bus_tram_csv=bus_tram_csv,
        parks_csv=parks_csv,
        playgrounds_csv=playgrounds_csv,
        venues_csv=venues_csv,
        out_dir=Path(args.out_dir),
    )

    minimal, merged, gdf = run_all_preprocessing(paths, write_intermediate=True)
    readable = make_readable_columns(merged.drop(columns=["neighborhood_canon"]) if "neighborhood_canon" in merged.columns else merged)
    save_dataframe(readable, args.readable_csv)
    print(f"Wrote canonical CSVs under {args.out_dir} and readable CSV {args.readable_csv}")


if __name__ == "__main__":
    main()
