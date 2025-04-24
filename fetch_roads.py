import os
import time
import gc
import requests
import geopandas as gpd
import pandas as pd
import osmnx as ox
from shapely.geometry import box, Point

# ----------------------------
# Configuration & Setup
# ----------------------------
ox.settings.use_cache = True
ox.settings.log_console = False

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TILE_DIR = os.path.join(OUTPUT_DIR, "tile_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TILE_DIR, exist_ok=True)
FINAL_DATAFILE = os.path.join(OUTPUT_DIR, "roads.gpkg")
OVERPASS_URL = "http://overpass-api.de/api/interpreter"

# Grid and simplification parameters
GRID_SIZE = 40  # 40x40 grid for fine tiling
OVERLAP = 0.10  # 10% overlap
SIMPLIFY_TOLERANCE = 4000  # meters (for a coarse overview)


# ----------------------------
# Helper: Overpass Request with Retries
# ----------------------------
def overpass_request(query, retries=1, delay=1):
    for attempt in range(retries):
        try:
            response = requests.get(OVERPASS_URL, params={"data": query}, timeout=30)
            data = response.json()
            return data
        except Exception as e:
            print(f"Overpass request failed (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(delay)
    return None


# ----------------------------
# Helper: Quiet Count Query
# ----------------------------
def count_motorways_in_bbox(bbox):
    total = 0
    queries = [
        f"""
        [out:json][timeout:10];
        (way["highway"="motorway"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}););
        out count;
        """,
        f"""
        [out:json][timeout:10];
        (way["highway"="trunk"]["motorroad"="yes"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}););
        out count;
        """,
        f"""
        [out:json][timeout:10];
        (way["highway"="trunk"]["oneway"="yes"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}););
        out count;
        """
    ]
    for query in queries:
        data = overpass_request(query)
        if data and data.get("elements") and "tags" in data["elements"][0]:
            try:
                total += int(data["elements"][0]["tags"]["total"])
            except Exception:
                continue
    return total


# ----------------------------
# Load Landmass (for quick skipping)
# ----------------------------
def load_landmass_3035():
    import cartopy.io.shapereader as shpreader
    shp_path = shpreader.natural_earth(resolution='50m', category='physical', name='land')
    land = gpd.read_file(shp_path)
    land_3035 = land.to_crs("EPSG:3035")
    return land_3035.unary_union.buffer(1000)


# ----------------------------
# Define Europe's Rectangular Frame in EPSG:3035
# ----------------------------
def define_europe_rectangle_3035():
    poly_4326 = box(-16, 26.8, 40, 72)
    gdf_4326 = gpd.GeoDataFrame(geometry=[poly_4326], crs="EPSG:4326")
    gdf_3035 = gdf_4326.to_crs("EPSG:3035")
    minx, miny, maxx, maxy = gdf_3035.total_bounds
    rect_3035 = box(minx, miny, maxx, maxy)
    return gpd.GeoDataFrame(geometry=[rect_3035], crs="EPSG:3035")


# ----------------------------
# Create Grid Over the Rectangle in EPSG:3035
# ----------------------------
def create_grid(rect_gdf, grid_size=GRID_SIZE, overlap=OVERLAP):
    minx, miny, maxx, maxy = rect_gdf.total_bounds
    cell_width = (maxx - minx) / grid_size
    cell_height = (maxy - miny) / grid_size
    cells = []
    for i in range(grid_size):
        for j in range(grid_size):
            x_min = max(minx + i * cell_width - cell_width * overlap, minx)
            x_max = min(minx + (i + 1) * cell_width + cell_width * overlap, maxx)
            y_min = max(miny + j * cell_height - cell_height * overlap, miny)
            y_max = min(miny + (j + 1) * cell_height + cell_height * overlap, maxy)
            cell_geom = box(x_min, y_min, x_max, y_max)
            if not cell_geom.is_valid:
                cell_geom = cell_geom.buffer(0)
            cells.append({"id": f"cell_{i}_{j}", "geometry": cell_geom})
    grid_gdf = gpd.GeoDataFrame(cells, crs="EPSG:3035")
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    center_point = box(center_x, center_y, center_x, center_y)
    grid_gdf["distance"] = grid_gdf.centroid.distance(gpd.GeoSeries([center_point], crs="EPSG:3035").iloc[0])
    grid_gdf = grid_gdf.sort_values("distance").reset_index(drop=True)
    return grid_gdf


# ----------------------------
# Fetch and Classify Road Features for a Tile
# ----------------------------
def fetch_road_features_for_tile(tile_geom_3035, tile_id):
    empty_df = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    tile_geom_4326 = gpd.GeoSeries([tile_geom_3035], crs="EPSG:3035").to_crs("EPSG:4326").iloc[0]

    # 1. Motorway
    try:
        df_motorway = ox.features_from_polygon(tile_geom_4326, tags={"highway": "motorway"})
    except Exception:
        df_motorway = empty_df.copy()
    if not df_motorway.empty:
        df_motorway["road_type"] = "motorway"

    # 2. Trunk Motorroads (strictly motorroad=yes)
    try:
        df_trunk_motorroad = ox.features_from_polygon(tile_geom_4326, tags={"highway": "trunk", "motorroad": "yes"})
    except Exception:
        df_trunk_motorroad = empty_df.copy()

    if not df_trunk_motorroad.empty and "highway" in df_trunk_motorroad.columns:
        df_trunk_motorroad = df_trunk_motorroad[df_trunk_motorroad["highway"] == "trunk"]

        if "oneway" in df_trunk_motorroad.columns:
            df_trunk_motorroad_oneway = df_trunk_motorroad[df_trunk_motorroad["oneway"] == "yes"].copy()
            df_trunk_motorroad_other = df_trunk_motorroad[df_trunk_motorroad["oneway"] != "yes"].copy()
        else:
            df_trunk_motorroad_oneway = empty_df.copy()
            df_trunk_motorroad_other = df_trunk_motorroad.copy()

        if not df_trunk_motorroad_oneway.empty:
            df_trunk_motorroad_oneway["road_type"] = "trunk_motorroad_oneway"
        if not df_trunk_motorroad_other.empty:
            df_trunk_motorroad_other["road_type"] = "trunk_motorroad_other"
    else:
        df_trunk_motorroad_oneway = empty_df.copy()
        df_trunk_motorroad_other = empty_df.copy()

    # 3. Trunk Dual Carriageways (NOT motorroad, but one-way)
    try:
        df_trunk_dual = ox.features_from_polygon(tile_geom_4326, tags={"highway": "trunk", "oneway": "yes"})
    except Exception:
        df_trunk_dual = empty_df.copy()

    # Remove motorroad=yes if it slipped through
    if not df_trunk_dual.empty and "highway" in df_trunk_dual.columns:
        df_trunk_dual = df_trunk_dual[df_trunk_dual["highway"] == "trunk"]
        if "motorroad" in df_trunk_dual.columns:
            df_trunk_dual = df_trunk_dual[df_trunk_dual["motorroad"] != "yes"]

        if "lanes" in df_trunk_dual.columns:
            df_trunk_dual = df_trunk_dual[df_trunk_dual["lanes"].isin(["2", "3", "4"])]

        if not df_trunk_dual.empty:
            df_trunk_dual["road_type"] = "dual_carriageway"
    else:
        df_trunk_dual = empty_df.copy()

    # 4. Construction Motorway
    try:
        df_construction = ox.features_from_polygon(tile_geom_4326, tags={"construction": "motorway"})
    except Exception:
        df_construction = empty_df.copy()
    if not df_construction.empty:
        if "highway" in df_construction.columns:
            df_construction = df_construction[df_construction["highway"].notnull()]
        else:
            df_construction = empty_df.copy()
        if not df_construction.empty:
            df_construction["road_type"] = "construction_motorway"

    layers = []
    for df in [df_motorway, df_trunk_motorroad_oneway, df_trunk_motorroad_other, df_trunk_dual, df_construction]:
        if not df.empty:
            layers.append(df)
    if not layers:
        return None
    combined = pd.concat(layers, ignore_index=True).copy()
    combined = combined.to_crs("EPSG:3035")
    combined["geometry"] = combined["geometry"].simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)
    combined["tile_id"] = tile_id
    combined = combined[combined.geometry.type.isin(['LineString', 'MultiLineString'])]

    counts = combined["road_type"].value_counts().to_dict()
    print(f"Tile {tile_id}: Found {len(combined)} features: {counts}")
    return combined


# ----------------------------
# Save Data (with column cleanup)
# ----------------------------
def save_data(gdf, filename):
    if not gdf.empty:
        gdf.columns = gdf.columns.str.lower()
        gdf = gdf.loc[:, ~gdf.columns.duplicated()]
        important_cols = ["geometry", "road_type", "tile_id", "name", "highway", "oneway", "motorroad", "dual_carriageway", "lanes", "construction"]
        gdf = gdf[[col for col in important_cols if col in gdf.columns]]
        gdf.to_file(filename, driver="GPKG")
        print(f"Saved data to {filename}")


# ----------------------------
# Merge Tile Files and Update Final Merged File
# ----------------------------
def merge_tiles():
    all_tiles = []
    for fname in os.listdir(TILE_DIR):
        if fname.endswith(".gpkg"):
            tile_path = os.path.join(TILE_DIR, fname)
            try:
                gdf = gpd.read_file(tile_path)
                all_tiles.append(gdf)
            except Exception as e:
                print(f"Error reading {tile_path}: {e}")
    if all_tiles:
        combined = pd.concat(all_tiles, ignore_index=True).copy()
        unique = combined.drop_duplicates(subset="geometry").copy()
        save_data(unique, FINAL_DATAFILE)
        print(f"Merged {len(unique)} unique features into final file.")
    else:
        print("No tile files to merge.")


# ----------------------------
# Loop Over Grid and Fetch Data with Checkpointing and Regular Merge
# ----------------------------
def fetch_all_roads(rect_gdf):
    grid = create_grid(rect_gdf, GRID_SIZE, OVERLAP)

    # Save grid for debugging purposes.
    grid_file = os.path.join(OUTPUT_DIR, "grid.gpkg")
    grid.to_file(grid_file, driver="GPKG")
    print(f"Grid saved to {grid_file}")

    # Check which tile covers the provided Mallorca coordinates (lon: 2.9128, lat: 39.6049).
    mallorca_point_ll = gpd.GeoSeries([Point(2.9128, 39.6049)], crs="EPSG:4326")
    mallorca_point_3035 = mallorca_point_ll.to_crs("EPSG:3035").iloc[0]
    tile_for_mallorca = grid[grid.contains(mallorca_point_3035)]
    if not tile_for_mallorca.empty:
        tile_id = tile_for_mallorca.iloc[0]["id"]
        print(f"Mallorca is in tile: {tile_id}")
    else:
        print("No tile found covering the provided Mallorca coordinates.")

    land_union = load_landmass_3035()
    total_tiles = len(grid)
    processed_tiles = 0

    # Identify already processed tiles
    processed_ids = set()
    for fname in os.listdir(TILE_DIR):
        if fname.startswith("tile_") and fname.endswith(".gpkg"):
            processed_ids.add(fname.replace("tile_", "").replace(".gpkg", ""))

    print(f"Total tiles: {total_tiles}")
    for idx, row in grid.iterrows():
        tile_id = row["id"]
        if tile_id in processed_ids:
            print(f"Tile {tile_id} already processed. Skipping.")
            processed_tiles += 1
            continue

        if not row["geometry"].intersects(land_union):
            print(f"Tile {tile_id}: No land, skipping quickly.")
            processed_tiles += 1
            continue

        tile_geom_4326 = gpd.GeoSeries([row["geometry"]], crs="EPSG:3035").to_crs("EPSG:4326").iloc[0]
        bounds = tile_geom_4326.bounds
        bbox_for_count = (bounds[1], bounds[0], bounds[3], bounds[2])
        highway_count = count_motorways_in_bbox(bbox_for_count)
        if highway_count == 0:
            print(f"Tile {tile_id}: No highways detected via count, skipping quickly.")
            processed_tiles += 1
            continue

        feats = fetch_road_features_for_tile(row["geometry"], tile_id)
        if feats is not None:
            tile_file = os.path.join(TILE_DIR, f"tile_{tile_id}.gpkg")
            save_data(feats, tile_file)
            processed_tiles += 1
            print(f"Tile {tile_id}: Data found and saved.")
        else:
            print(f"Tile {tile_id}: Skipped.")
            processed_tiles += 1

        # Regularly update merged file every 10 processed tiles
        if processed_tiles % 10 == 0:
            merge_tiles()

        gc.collect()
        time.sleep(1)

    print(f"Processed {total_tiles} tiles.")
    merge_tiles()
    # Finally, load and return the merged final data.
    try:
        final_gdf = gpd.read_file(FINAL_DATAFILE)
        return final_gdf
    except Exception as e:
        print(f"Error reading final file: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:3035")


# ----------------------------
# Main Function
# ----------------------------
def main():
    print("Defining Europe's rectangular frame in EPSG:3035...")
    rect = define_europe_rectangle_3035()
    print("Fetching road features from Overpass using a 40x40 grid, starting from the center...")
    roads = fetch_all_roads(rect)
    print("Data fetching complete.")


if __name__ == "__main__":
    main()
