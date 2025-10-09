# src/gis/enrich_context.py
"""
Main orchestrator: enrich noise sensors with spatial, network and local features.
"""

import geopandas as gpd
from src.gis.spatial_features import load_osm_layers, add_distance_features
from src.gis.network_features import load_graph, compute_betweenness, attach_centrality
from src.gis.local_features import add_local_mean
import os


def enrich_with_context(sensors_path, output_path):
    print("🚀 Enriching sensors with GIS context ...")
    sensors_gdf = gpd.read_file(sensors_path)

    # --- Spatial distances
    gdf_green, gdf_edges = load_osm_layers()
    gdf = add_distance_features(sensors_gdf, gdf_green, gdf_edges)

    # --- Network centrality
    G = load_graph()
    C = compute_betweenness(G)
    gdf = attach_centrality(gdf, G, C)

    # --- Local mean noise
    gdf = add_local_mean(gdf)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdf.to_file(output_path, driver="GeoJSON")

    print(f"✅ Enriched layer saved to {output_path}")
    return gdf


if __name__ == "__main__":
    enrich_with_context(
        "data/gis_layers/sensor_summary.geojson",
        "data/gis_layers/Barcelona_Sensors_Enriched.geojson"
    )
