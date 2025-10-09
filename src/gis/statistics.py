# src/gis/statistics.py
"""
Spatial statistics for noise sensors:
- Global Moran's I (autocorrelation)
- Local Getis-Ord Gi* (hotspot detection)
- Local Moran's I (LISA clusters)

These methods validate spatial dependencies in the enriched dataset.
"""

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from libpysal.weights import DistanceBand
from esda.moran import Moran, Moran_Local
from esda.getisord import G_Local


# ============================================================
# 🔹 1. Global Moran's I
# ============================================================
def compute_global_moran(gdf, value_col="Avg_Noise_dB", threshold=1000):
    """
    Compute global Moran's I to test spatial autocorrelation.
    """
    gdf_proj = gdf.to_crs(3857)
    w = DistanceBand(gdf_proj.geometry, threshold=threshold, binary=True)
    moran = Moran(gdf_proj[value_col], w)

    print(f"🌍 Global Moran's I = {moran.I:.3f}")
    print(f"p-value = {moran.p_norm:.4f}")
    return moran


# ============================================================
# 🔹 2. Local Getis-Ord Gi*
# ============================================================
def compute_hotspots(gdf, value_col="Avg_Noise_dB", threshold=1000):
    """
    Identify statistically significant hot and cold spots using Getis-Ord Gi*.
    """
    gdf_proj = gdf.to_crs(3857)
    w = DistanceBand(gdf_proj.geometry, threshold=threshold, binary=True)
    g = G_Local(gdf_proj[value_col], w)

    gdf_proj["GiZScore"] = g.Zs
    gdf_proj["Gi_Bin"] = gdf_proj["GiZScore"].apply(
        lambda z: 3 if z > 2.58 else (2 if z > 1.96 else (-2 if z < -1.96 else 0))
    )

    print("🔥 Hotspots computed successfully.")
    return gdf_proj.to_crs(4326)


def plot_hotspots(gdf, column="Gi_Bin", title="Getis-Ord Gi* Hotspots (1 km)"):
    """
    Plot the resulting hotspot map (red = hotspots, blue = coldspots).
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    gdf.plot(column=column, cmap="RdBu_r", legend=True, ax=ax)
    plt.title(title)
    plt.axis("off")
    plt.show()


# ============================================================
# 🔹 3. Local Moran’s I (LISA)
# ============================================================
def compute_local_moran(gdf, value_col="Avg_Noise_dB", threshold=1000):
    """
    Compute Local Moran’s I (LISA) clusters: High-High, Low-Low, etc.
    """
    gdf_proj = gdf.to_crs(3857)
    w = DistanceBand(gdf_proj.geometry, threshold=threshold, binary=True)
    lisa = Moran_Local(gdf_proj[value_col], w)

    gdf_proj["I_local"] = lisa.Is
    gdf_proj["p_value"] = lisa.p_sim
    gdf_proj["cluster_type"] = np.select(
        [
            (lisa.q == 1),
            (lisa.q == 2),
            (lisa.q == 3),
            (lisa.q == 4),
        ],
        ["High-High", "Low-Low", "Low-High", "High-Low"],
        default="Not Significant",
    )

    print("📊 Local Moran's I computed successfully.")
    return gdf_proj.to_crs(4326)


def plot_lisa_clusters(gdf, column="cluster_type", title="Local Moran's I (LISA) Clusters"):
    """
    Plot LISA clusters with distinct colors for each type.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    gdf.plot(column=column, categorical=True, legend=True, ax=ax)
    plt.title(title)
    plt.axis("off")
    plt.show()


# ============================================================
# 🚀 Example run
# ============================================================
if __name__ == "__main__":
    gdf = gpd.read_file("data/gis_layers/Barcelona_Sensors_Enriched.geojson")

    # 1. Global autocorrelation
    moran = compute_global_moran(gdf)

    # 2. Hotspots
    gdf_hot = compute_hotspots(gdf)
    plot_hotspots(gdf_hot)

    # 3. LISA
    gdf_lisa = compute_local_moran(gdf)
    plot_lisa_clusters(gdf_lisa)

    # Export optional results
    gdf_hot.to_file("data/gis_layers/Barcelona_Hotspots.geojson", driver="GeoJSON")
    gdf_lisa.to_file("data/gis_layers/Barcelona_LISA.geojson", driver="GeoJSON")

    print("✅ Spatial statistics completed and exported.")