import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster, MiniMap, MeasureControl
from sklearn.cluster import KMeans
import matplotlib.cm as cm

df = pd.read_csv("crime_dataset_india_with_coordinates.csv")
print("CSV Loaded")

df = df.dropna(subset=["Latitude", "Longitude"])

K = 5
kmeans = KMeans(n_clusters=K, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['Latitude', 'Longitude']])
print("ML Clustering Complete")
print(df['cluster'].value_counts())

colormap = cm.get_cmap('Set1', K)

def to_hex(rgba):
    return "#{:02x}{:02x}{:02x}".format(
        int(rgba[0] * 255),
        int(rgba[1] * 255),
        int(rgba[2] * 255)
    )

cluster_colors = [to_hex(colormap(i)) for i in range(K)]

india_map = folium.Map(
    location=[22.9734, 78.6569],
    zoom_start=5,
    tiles="CartoDB positron"
)

MiniMap(toggle_display=True).add_to(india_map)
MeasureControl(primary_length_unit='kilometers').add_to(india_map)

marker_cluster = MarkerCluster(name="All Crimes").add_to(india_map)

crime_types = df['Crime Description'].unique()
crime_layers = {crime: folium.FeatureGroup(name=crime) for crime in crime_types}

for crime_layer in crime_layers.values():
    crime_layer.add_to(india_map)

for _, row in df.iterrows():
    popup_html = f"""
    <b>City:</b> {row['City']}<br>
    <b>Crime:</b> {row['Crime Description']}<br>
    <b>Victim:</b> {row['Victim Age']} y/o {row['Victim Gender']}<br>
    <b>Weapon:</b> {row['Weapon Used']}<br>
    <b>Police Deployed:</b> {row['Police Deployed']}<br>
    <b>Case Closed:</b> {row['Case Closed']}<br>
    <b>Cluster:</b> {row['cluster']}
    """

    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=6,
        color=cluster_colors[row["cluster"]],
        fill=True,
        fill_color=cluster_colors[row["cluster"]],
        fill_opacity=0.8,
        popup=popup_html
    ).add_to(marker_cluster)

    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=5,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.7,
        popup=popup_html
    ).add_to(crime_layers[row['Crime Description']])

heat_data = df[['Latitude', 'Longitude']].values.tolist()

HeatMap(
    heat_data,
    radius=15,
    blur=25,
    min_opacity=0.4,
    gradient={
        0.2: "blue",
        0.4: "lime",
        0.6: "yellow",
        1.0: "red"
    }
).add_to(india_map)

folium.LayerControl(collapsed=False).add_to(india_map)

india_map.save("crime_hotspot_advanced_map.html")
print("Advanced Heatmap and ML Hotspots Created")
print("Open file: crime_hotspot_advanced_map.html")
