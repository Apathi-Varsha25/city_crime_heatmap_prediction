import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import numpy as np

df = pd.read_csv("crime_dataset_india_with_coordinates.csv")

print("CSV Loaded")
print(df.head())

df = df.dropna(subset=["Latitude", "Longitude"])

K = 5

kmeans = KMeans(n_clusters=K, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['Latitude', 'Longitude']])

print("ML Clustering Complete")
print(df['cluster'].value_counts())

colormap = cm.get_cmap('Set1', K)
colors = [colormap(i) for i in range(K)]

def to_hex(rgba):
    return "#{:02x}{:02x}{:02x}".format(
        int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
    )

cluster_colors = [to_hex(c) for c in colors]

india_map = folium.Map(
    location=[22.9734, 78.6569],
    zoom_start=5,
    tiles="CartoDB positron"
)

cluster_group = MarkerCluster().add_to(india_map)

for i, row in df.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=5,
        color=cluster_colors[row["cluster"]],
        fill=True,
        fill_color=cluster_colors[row["cluster"]],
        fill_opacity=0.8,
        popup=f"City: {row['City']}<br>Crime: {row['Crime Description']}<br>Cluster: {row['cluster']}"
    ).add_to(cluster_group)

heat_data = df[['Latitude', 'Longitude']].values.tolist()

HeatMap(
    heat_data,
    radius=18,
    blur=30,
    min_opacity=0.4,
    gradient={
        0.2: "blue",
        0.4: "lime",
        0.6: "yellow",
        1.0: "red"
    }
).add_to(india_map)

india_map.save("crime_hotspot_heatmap_ml.html")

print("Heatmap and ML Hotspots Created")
print("Open file: crime_hotspot_heatmap_ml.html")
