import geopandas as gpd
from shapely.geometry import box, Point


def define_europe_rectangle_3035():
    # Define Europe's bounding box in EPSG:4326 and reproject to EPSG:3035.
    poly_4326 = box(-16, 26.8, 40, 72)
    gdf_4326 = gpd.GeoDataFrame(geometry=[poly_4326], crs="EPSG:4326")
    gdf_3035 = gdf_4326.to_crs("EPSG:3035")
    minx, miny, maxx, maxy = gdf_3035.total_bounds
    rect_3035 = box(minx, miny, maxx, maxy)
    return gpd.GeoDataFrame(geometry=[rect_3035], crs="EPSG:3035")


def create_grid(rect_gdf, grid_size=40, overlap=0.10):
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
    center_point = Point(center_x, center_y)
    grid_gdf["distance"] = grid_gdf.centroid.distance(
        gpd.GeoSeries([center_point], crs="EPSG:3035").iloc[0]
    )
    grid_gdf = grid_gdf.sort_values("distance").reset_index(drop=True)
    return grid_gdf


def main():
    # Define the European rectangle and build the grid.
    rect = define_europe_rectangle_3035()
    grid = create_grid(rect, grid_size=40, overlap=0.10)

    # Given Mallorca coordinates (lat, lon) in EPSG:4326.
    mallorca_lat, mallorca_lon = 39.6049, 2.9128
    mallorca_point = Point(mallorca_lon, mallorca_lat)  # note: Point(lon, lat)

    # Convert the point to EPSG:3035.
    gdf_point = gpd.GeoDataFrame(geometry=[mallorca_point], crs="EPSG:4326")
    gdf_point = gdf_point.to_crs("EPSG:3035")
    mallorca_point_3035 = gdf_point.geometry.iloc[0]

    # Find and print the tile(s) that intersect this point.
    matching_tiles = grid[grid.intersects(mallorca_point_3035)]
    if not matching_tiles.empty:
        print("Tile(s) for Mallorca:")
        for _, row in matching_tiles.iterrows():
            print(row["id"])
    else:
        print("No tile found for the given coordinates.")


if __name__ == "__main__":
    main()
