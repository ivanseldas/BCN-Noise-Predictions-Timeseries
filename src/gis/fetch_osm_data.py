# src/gis/fetch_osm_data.py
import os
import osmnx as ox
import geopandas as gpd

def ensure_landuse_data(
    place_name: str = "Barcelona, Spain",
    out_path: str = "data/gis_layers/landuse.geojson"
) -> str:
    """
    Download land-use polygons (landuse, leisure, natural) from OSM if not cached.

    Args:
        place_name: City or region name.
        out_path: Local file path for saving GeoJSON.

    Returns:
        Path to the saved GeoJSON file.
    """
    # Create output directory if missing
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # If file exists, reuse it
    if os.path.exists(out_path):
        print(f"✅ Landuse file already exists at: {out_path}")
        return out_path

    print(f"⬇️  Downloading landuse polygons for {place_name} from OpenStreetMap...")

    # Query OSM for landuse-related features
    tags = {"landuse": True, "leisure": True, "natural": True, "amenity": "park"}
    gdf = ox.features_from_place(place_name, tags=tags)

    # Keep only polygons
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

    # Combine relevant tags into a single column
    gdf["landuse_type"] = (
        gdf["landuse"]
        .fillna(gdf["leisure"])
        .fillna(gdf["natural"])
        .fillna("unknown")
    )

    # Simplify columns
    gdf = gdf[["landuse_type", "geometry"]].to_crs(4326)

    # Save locally
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"✅ Landuse data saved to: {out_path}")

    return out_path


def ensure_roads_data(
    place_name: str = "Barcelona, Spain",
    out_path: str = "data/gis_layers/roads_primary.geojson"
) -> str:
    """
    Download primary roads (motorway, trunk, primary) from OSM if not cached.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        print(f"✅ Roads file already exists at: {out_path}")
        return out_path

    print(f"⬇️  Downloading road network for {place_name} from OpenStreetMap...")
    G = ox.graph_from_place(place_name, network_type="drive")
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    edges = edges[edges.geometry.type.isin(["LineString", "MultiLineString"])]

    # Keep only major roads
    edges = edges[
        edges["highway"]
        .astype(str)
        .str.contains("motorway|trunk|primary", case=False, na=False)
    ]
    edges.to_crs(4326).to_file(out_path, driver="GeoJSON")
    print(f"✅ Road network saved to: {out_path}")

    return out_path

# -------------------------------------------------------------------------
# ✅ Run directly from the terminal
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("🌍 Fetching GIS data for Barcelona...")
    landuse_path = ensure_landuse_data("Barcelona, Spain")
    roads_path = ensure_roads_data("Barcelona, Spain")
    print("\n✅ Download complete!")
    print(f" - Landuse: {landuse_path}")
    print(f" - Roads:   {roads_path}")