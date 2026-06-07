import folium
from folium.plugins import HeatMap, MarkerCluster
import matplotlib.cm as cm
from urllib.parse import urlencode

def generate_crime_map(df, show_popup_links=True):
    m = folium.Map(location=[22.9734, 78.6569], zoom_start=5, tiles="CartoDB positron")

    heat_data = df[["Latitude", "Longitude"]].values.tolist()
    HeatMap(heat_data, radius=15, blur=20, min_opacity=0.4).add_to(m)

    cluster_summary = df.groupby("cluster").agg({
        "Latitude": "mean",
        "Longitude": "mean",
        "City": "count"
    }).rename(columns={"City": "count"}).reset_index()

    K = cluster_summary.shape[0]
    colormap = cm.get_cmap("Set1", K)
    cluster_colors = [
        "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b, _ in colormap.colors
    ]

    for i, row in cluster_summary.iterrows():
        if show_popup_links:
            query = urlencode({"cluster": int(row["cluster"])})
            link_html = f'<br><a href="/cluster?{query}" target="_blank" style="font-weight:bold;color:#0d6efd;">View crimes</a>'
        else:
            link_html = ""

        popup_html = f"""
        <b>Cluster {int(row['cluster'])}</b><br>
        Crimes: {row['count']:,}
        {link_html}
        """

        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=15,
            color=cluster_colors[i],
            fill=True,
            fill_color=cluster_colors[i],
            fill_opacity=0.9,
            popup=popup_html
        ).add_to(m)

    return m
