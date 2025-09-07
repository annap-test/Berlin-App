Berlin Explorer — App Documentation

Overview
- Purpose: Explore Berlin neighborhoods and districts with weighted suitability scoring across multiple features, rendered on an interactive map.
- Entry point: `app.py` (Streamlit).
- Data sources: `data/neighborhoods.geojson`, wide label tables in `labels_with_visualization/outputs/` or `outputs/`.

Inputs
- Geometry: `data/neighborhoods.geojson` as WGS84 (EPSG:4326).
- Neighborhood-wide metrics (optional, loaded if present): `labels_with_visualization/outputs/berlin_neighborhoods_labels_wide.csv`.
- District-wide metrics (optional, loaded if present): `labels_with_visualization/outputs/berlin_districts_labels_wide.csv`.

Main Components
- `find_paths()`: Resolves root and picks outputs directory (`labels_with_visualization/outputs` if exists, else `outputs`).
- `load_sources()`: Loads polygons (fixes CRS to EPSG:4326), and wide tables if available. Result: `nei_geo`, `nei_wide`, `dist_wide`, `out_dir`.
- `percentile_score(series, lo=10, hi=90)`: Percentile scaling to 0–100 using p10 and p90 as anchors; robust against outliers.
- `render_map(gdf, value_col)`: Builds a Folium map and styles polygons by `value_col` using a 3-color linear colormap (approx. p5→p95 range).
- `weight_ui(container, items, defaults, key_prefix)`: Renders checkbox + slider per feature and returns weights (0–100).

User Flow
1) Pick level (Neighborhoods | Districts).
2) Select features to include and adjust importance (0–100).
3) Click “Show” to compute suitability and update the map and top list.

Features and Columns
Neighborhoods (if present in `nei_wide`):
- Venues vibrancy: `VV_index` (or fallback `venues_per_km2`).
- Mobility: `mobility_score` (or fallback `connectivity_density`).
- Playgrounds: `playgrounds_per_km2`.
- Green share: `green_share`.

Districts (if present in `dist_wide`):
- Income: `income_value_eur` (higher is better).
- Safety: `crimes_per_1000` (inverted; lower is better).
- Unemployment: `unemployment_per_1000` (inverted; lower is better).
- Density: `density_per_km2`.
- Diversity: `diversity_share`.
- Green share: `green_share`.
- Vibrancy: `VV_index` (or `venues_per_km2`).
- Mobility: `mobility_score` (or `connectivity_density`).
- Playgrounds: `playgrounds_per_km2`.

Scoring
- For each selected feature, values are converted to 0–100 via `percentile_score` (p10→0, p90→100). For “inverse” metrics (safety, unemployment), the series is negated before scaling.
- Suitability = weighted sum of scaled components, with each weight in [0, 100] normalized to a fraction (w/100).
- The result is stored in a new `suitability` column and displayed.

Mapping
- `render_map` creates a `folium.Map` centered on Berlin.
- Styles: a 3-stop `LinearColormap` between approx. p5 and p95 of the selected column; polygons are colored by the scaled value or grey if missing.
- Tooltip: shows district/neighborhood and the selected value column when available.
- Display: Map is embedded via `streamlit_folium.st_folium`.

Outputs
- No files are written by the app. It only reads prepared CSV/GeoJSON files.

Running Locally
- `pip install -r requirements.txt`
- `streamlit run app.py`

Deploying (Streamlit Community Cloud)
- New app → Repo `annap-test/Berlin-App`, Branch `main`, File `app.py` → Deploy.

Notes
- If geospatial reads fail on some platforms, consider adding `pyogrio` and using `gpd.options.io_engine = "pyogrio"` before reading.
- The app tolerates missing columns; unavailable features simply don’t appear in the UI.
