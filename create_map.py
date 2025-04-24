import os
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union, linemerge
import matplotlib.font_manager as fm

# Set the Inter font for everything.
font_path = r"D:\QGIS\dual_carriegeways\font\Inter_28pt-Regular.ttf"
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Inter'

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATAFILE = os.path.join(OUTPUT_DIR, "roads.gpkg")
MAPFILE = os.path.join(OUTPUT_DIR, "final_map.png")


def define_europe_rectangle_3035():
    poly_4326 = box(-15, 28, 40, 72)
    gdf_4326 = gpd.GeoDataFrame(geometry=[poly_4326], crs="EPSG:4326")
    gdf_3035 = gdf_4326.to_crs("EPSG:3035")
    minx, miny, maxx, maxy = gdf_3035.total_bounds
    rect_3035 = box(minx, miny, maxx, maxy)
    return gpd.GeoDataFrame(geometry=[rect_3035], crs="EPSG:3035")


def create_map():
    roads = gpd.read_file(DATAFILE)
    roads = roads.copy()

    # --- Debug Reclassification for trunk roads ---
    if "motorroad" in roads.columns:
        def reclassify(row):
            current_fid = row["fid"] if "fid" in row.index else row.name
            highway_val = str(row.get("highway")).lower()
            road_type_val = row.get("road_type")
            motorroad_val = str(row.get("motorroad")).lower()
            oneway_val = str(row.get("oneway")).lower()
            if highway_val == "trunk" and road_type_val in ["trunk_motorroad_oneway", "trunk_motorroad_other"]:
                if motorroad_val != "yes":
                    new_type = "dual_carriageway" if oneway_val == "yes" else None
                    print(f"Debug: FID {current_fid} reclassified from {road_type_val} to {new_type}")
                    return new_type
            return road_type_val

        roads["road_type"] = roads.apply(reclassify, axis=1)
        roads = roads[roads["road_type"].notnull()]

        if "fid" in roads.columns:
            for fid in [50896, 75858]:
                row = roads[roads["fid"] == fid]
                if not row.empty:
                    print(f"Post-reclassification FID {fid}: {row['road_type'].values}")
                else:
                    print(f"Post-reclassification FID {fid}: removed from dataset")
        else:
            print("No 'fid' column in roads; skipping specific FID debug output.")

    rect = define_europe_rectangle_3035()

    fig, ax = plt.subplots(figsize=(12, 11))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Load the countries shapefile, reproject, clip to the defined rectangle,
    # and dissolve by the "SOVEREIGNT" field so that each country is separate.
    countries = gpd.read_file(os.path.join(BASE_DIR, "data", "countries.shp"))
    countries = countries.to_crs("EPSG:3035")
    countries = gpd.clip(countries, rect)
    countries = countries.dissolve(by="SOVEREIGNT").reset_index()

    # Compute the overall landmass (for coastline extraction).
    land_union = countries.unary_union
    rect_geom = rect.geometry.iloc[0]
    ocean_geom = rect_geom.difference(land_union)
    ocean = gpd.GeoDataFrame(geometry=[ocean_geom], crs="EPSG:3035")

    # Plot ocean (background) first.
    ocean.plot(ax=ax, color="lightsteelblue", edgecolor="none", alpha=0.7, zorder=0)

    # Plot landmass fill with gainsboro (here lightgray) and transparency.
    landmass = gpd.GeoDataFrame(geometry=[land_union], crs="EPSG:3035")
    landmass.plot(ax=ax, color="lightgray", edgecolor="none", alpha=0.5, zorder=1)

    # Compute the coastline from the union of all countries.
    coastline = land_union.boundary

    # Compute internal boundaries: for every pair of countries, find the common boundary.
    internal_boundaries_list = []
    n = len(countries)
    for i in range(n):
        for j in range(i + 1, n):
            common = countries.geometry.iloc[i].boundary.intersection(countries.geometry.iloc[j].boundary)
            if not common.is_empty:
                internal_boundaries_list.append(common)
    if internal_boundaries_list:
        internal_boundaries = unary_union(internal_boundaries_list)
        # Remove borders that coincide with the coastline.
        internal_boundaries = internal_boundaries.difference(coastline)
        # Extract only LineString and MultiLineString geometries.
        if internal_boundaries.geom_type == 'GeometryCollection':
            lines = [geom for geom in internal_boundaries if geom.geom_type in ['LineString', 'MultiLineString']]
            if lines:
                internal_boundaries = linemerge(lines)
        else:
            internal_boundaries = linemerge(internal_boundaries)
        # Simplify to smooth out choppiness.
        internal_boundaries = internal_boundaries.simplify(100, preserve_topology=True)
    else:
        internal_boundaries = None

    # Plot the internal boundaries as blurred, transparent light green lines.
    if internal_boundaries is not None and not internal_boundaries.is_empty:
        borders_gdf = gpd.GeoDataFrame(geometry=[internal_boundaries], crs="EPSG:3035")
        # First layer: a thicker, very transparent line.
        borders_gdf.plot(ax=ax, color="white", linewidth=0.5, alpha=1.00, zorder=2)
        # Second layer: a thinner, more visible line.
        borders_gdf.plot(ax=ax, color="white", linewidth=2, alpha=0.35, zorder=3)

    # Adjust the extent of the map to the rectangle bounds.
    minx, miny, maxx, maxy = rect.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # Draw the rectangle boundary for reference (without legend entry).
    rect.boundary.plot(ax=ax, edgecolor="dimgray", linewidth=1)

    # Plot roads on top.
    if not roads.empty:
        # Plot dual carriageways ("Dwujezdniowe kolizyjne")


        dual = roads[roads["road_type"] == "dual_carriageway"]
        if not dual.empty:
            union_geom = dual.geometry.unary_union
            if union_geom.geom_type == "MultiLineString":
                parts = list(union_geom.geoms)
            elif union_geom.geom_type == "LineString":
                parts = [union_geom]
            else:
                parts = []
            dual_merged = gpd.GeoDataFrame(
                geometry=[part for part in parts if part.length >= 200],
                crs=dual.crs
            )
            if not dual_merged.empty:
                dual_merged.plot(ax=ax, color="silver", linewidth=0.5, label="Główne drogi\ndwujezdniowe")

        # Plot construction roads ("W budowie")
        construction = roads[roads["road_type"] == "construction_motorway"]
        if not construction.empty:
            if "highway" in construction.columns:
                construction = construction[~construction["highway"].str.lower().isin(["proposed", "unclassified", "path", "track"])]
            if not construction.empty:
                construction.plot(ax=ax, color="red", linewidth=0.5, label="Autostrady/\ndrogi ekspresowe\nw budowie")

        # Plot autostrady (trunk motorroads and motorways) separated into one-way and two-way.
        # One-way (thicker) and two-way (thinner) are plotted separately,
        # but only one legend entry ("Autostrady") is added.
        legend_added = False
        trunk_oneway = roads[roads["road_type"] == "trunk_motorroad_oneway"]
        trunk_other = roads[roads["road_type"] == "trunk_motorroad_other"]
        motorways = roads[roads["road_type"] == "motorway"]

        if not trunk_oneway.empty:
            trunk_oneway.plot(ax=ax, color="black", linewidth=0.5, label="Autostrady/\ndrogi ekspresowe")
            legend_added = True
        if not trunk_other.empty:
            label_val = "Autostrady" if not legend_added else "_nolegend_"
            trunk_other.plot(ax=ax, color="black", linewidth=0.3, label=label_val)
            legend_added = True
        if not motorways.empty:
            label_val = "Autostrady" if not legend_added else "_nolegend_"
            motorways.plot(ax=ax, color="black", linewidth=0.5, label=label_val)

    plt.title("Drogi szybkiego ruchu w Europie")
    # Add an annotation with the data source in the bottom of the map frame (inside the map)
    ax.text(0.5, 0.01, "Źródło: OpenStreetMap, własne opracowanie", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6, color="gray")
    plt.axis("off")
    leg = plt.legend(frameon=False)
    for handle in leg.legend_handles:
        handle.set_linewidth(3)
    plt.tight_layout()
    plt.savefig(MAPFILE, dpi=400)
    plt.show()
    print(f"Final map saved to {MAPFILE}")


if __name__ == "__main__":
    create_map()
