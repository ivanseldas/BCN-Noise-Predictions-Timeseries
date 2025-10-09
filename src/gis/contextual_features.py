# ======================================================
# 🌍 Generate contextual GIS layers for Barcelona
# using OSMnx + GeoPandas
# ======================================================

import os
import osmnx as ox
import geopandas as gpd

def download_contextual_layers(city_name="Barcelona, Spain", output_dir="data/gis_layers/"):
    os.makedirs(output_dir, exist_ok=True)
    ox.settings.log_console = True

    print(f"🌆 Downloading OSM layers for {city_name}...")

    # 1️⃣ Green areas
    tags_green = {"leisure": ["park", "garden"], "landuse": ["grass", "forest"]}
    gdf_green = ox.geometries_from_place(city_name, tags_green)
    gdf_green.to_file(os.path.join(output_dir, "green_areas.geojson"), driver="GeoJSON")
    print(f"✅ Green areas: {len(gdf_green)} features")

    # 2️⃣ Roads (main highways)
    tags_roads = {"highway": ["motorway", "primary", "secondary", "tertiary"]}
    gdf_roads = ox.geometries_from_place(city_name, tags_roads)
    gdf_roads.to_file(os.path.join(output_dir, "main_roads.geojson"), driver="GeoJSON")
    print(f"✅ Main roads: {len(gdf_roads)} features")

    # 3️⃣ Buildings
    tags_build = {"building": True}
    gdf_buildings = ox.geometries_from_place(city_name, tags_build)
    gdf_buildings.to_file(os.path.join(output_dir, "buildings.geojson"), driver="GeoJSON")
    print(f"✅ Buildings: {len(gdf_buildings)} features")

    return gdf_green, gdf_roads, gdf_buildings

if __name__ == "__main__":
    download_contextual_layers()