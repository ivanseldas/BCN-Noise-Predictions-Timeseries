import geopandas as gpd
from shapely.ops import nearest_points
import os
import pandas as pd
from shapely.geometry import Point

# ------------------------------------------------------
# 🔧 Helper: Convert CSV to GeoDataFrame if needed
# ------------------------------------------------------
def csv_to_geojson(csv_path, output_path):
    if os.path.exists(output_path):
        print("ℹ️ GeoJSON already exists. Skipping conversion.")
        return gpd.read_file(output_path)

    print("📥 Converting CSV to GeoJSON...")
    df = pd.read_csv(csv_path)

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["Longitud"], df["Latitud"])],
        crs="EPSG:4326"
    )
    gdf.to_file(output_path, driver="GeoJSON")
    print(f"✅ GeoJSON created: {output_path}")
    return gdf

# ======================================================
# 📊 Enrich sensor data with contextual features
# ======================================================

def sjoin_nearest(gdf_base, gdf_ref, ref_name):
    """Adds distance to nearest feature from gdf_ref."""
    gdf_base = gdf_base.to_crs(epsg=3857)
    gdf_ref = gdf_ref.to_crs(epsg=3857)
    gdf_base[f"dist_to_{ref_name}"] = gdf_base.geometry.apply(
        lambda geom: gdf_ref.distance(geom).min() if not gdf_ref.empty else None
    )
    return gdf_base.to_crs(epsg=4326)

def enrich_sensors_with_context():
    gdf_sensors = gpd.read_file("data/gis_layers/sensor_summary.geojson")
    gdf_green = gpd.read_file("data/gis_layers/green_areas.geojson")
    gdf_roads = gpd.read_file("data/gis_layers/main_roads.geojson")

    gdf_sensors = sjoin_nearest(gdf_sensors, gdf_green, "park")
    gdf_sensors = sjoin_nearest(gdf_sensors, gdf_roads, "mainroad")

    gdf_sensors.to_file("data/gis_layers/bcn_sensor_enriched.geojson", driver="GeoJSON")
    print("✅ Sensors enriched with contextual distances.")
    return gdf_sensors

if __name__ == "__main__":
    
    csv_to_geojson("data/processed/sensor_summary.csv", "data/gis_layers/sensor_summary.geojson")
    enrich_sensors_with_context()