Berlin Neighborhoods — Preprocessing and Streamlit App

Overview
- Scripts convert raw datasets into tidy CSV/GeoJSON for the Streamlit app (`app.py`).
- A small package (`src/berlin_labels`) provides reusable helpers and label computations.

Folder layout
- data/: put raw CSVs/GeoJSON
- outputs/: generated files (fed to app.py)
- scripts/: CLIs to build each set of features
- src/berlin_labels/: implementation modules
- tests/: basic unit tests (pytest)

Quick start
1) Build polygons with area
   python scripts/build_polygons.py data/neighborhoods.geojson --out outputs/neighborhoods.geojson

2) Mobility (U-Bahn + Bus/Tram)
   python scripts/build_mobility.py outputs/neighborhoods.geojson data/ubahns.csv data/bus_tram_stops.csv --out outputs/mobility_labels.csv

3) Parks
   python scripts/build_parks.py outputs/neighborhoods.geojson data/parks.csv --out outputs/parks_features.csv

4) Playgrounds
   python scripts/build_playgrounds.py outputs/neighborhoods.geojson data/playgrounds.csv --out outputs/playgrounds_features.csv

5) Venues (national cuisines + vibrancy)
   python scripts/build_venues.py outputs/neighborhoods.geojson data/venues.csv --out_features outputs/features_venues_nationals.csv --out_vibrancy outputs/features_venues_vibrancy.csv

6) Merge
   python scripts/merge_all_labels.py outputs/neighborhoods.geojson --out_minimal outputs/neighborhood_labels_minimal.csv --out_merged outputs/neighborhood_labels_with_venues.csv

Run the app
   streamlit run app.py

Notes
- Joins use `(district_id, neighborhood_canon)` where applicable; spatial joins for point datasets.
- Areas use EPSG:25833 for accuracy; densities use an area floor of 0.20 km².
- Percentile scaling uses p10–p90 → 0–100; terciles yield labels.

