from flask import Flask, render_template, request
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import io
import base64

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml_utils import apply_kmeans
from map_utils import generate_crime_map

app = Flask(__name__)

df = pd.read_csv("data/crime_dataset_india_with_coordinates.csv")
df = df.dropna(subset=["Latitude", "Longitude"])

df = apply_kmeans(df, k=5)

df['Date'] = pd.to_datetime(df['Date of Occurrence'], errors='coerce', dayfirst=True)

def generate_crime_growth_plot(filtered_df):
    if filtered_df.empty or 'Date' not in filtered_df.columns:
        return "<p>No data for graph</p>"
    
    monthly = filtered_df.groupby(pd.Grouper(key='Date', freq='M')).size()
    plt.figure(figsize=(8,4))
    monthly.plot(marker='o', color='crimson')
    plt.title("Crime Growth Over Time")
    plt.ylabel("Incidents")
    plt.xlabel("Month")
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return f'<img src="data:image/png;base64,{img_base64}" class="img-fluid"/>'

def calculate_safety_index(filtered_df):
    if filtered_df.empty:
        return "N/A"
    max_incidents = df.shape[0]
    safety_score = max(0, 100 - int(len(filtered_df)/max_incidents*100))
    return f"{safety_score}/100"

def preventive_recommendations(filtered_df):
    if filtered_df.empty:
        return "No recommendations due to lack of data."
    top_crimes = filtered_df['Crime Description'].value_counts().head(3)
    recs = [f"Increase vigilance for {crime}" for crime in top_crimes.index]
    return "; ".join(recs)

@app.route("/", methods=["GET"])
def index():
    city = request.args.get("city", "All")
    crime_type = request.args.getlist("crime_type")

    filtered_df = df.copy()
    if city != "All":
        filtered_df = filtered_df[filtered_df["City"] == city]
    if crime_type:
        filtered_df = filtered_df[filtered_df["Crime Description"].isin(crime_type)]

    if filtered_df.empty:
        record_count = 0
        hotspot_count = 0
        crime_map = "<p>No data available for selected filters</p>"
        crime_growth_plot = "<p>No data available for graph</p>"
        safety_index = "N/A"
        high_risk_zones = "N/A"
        future_hotspots = "N/A"
        recommendations = "No recommendations available."
    else:
        record_count = len(filtered_df)
        hotspot_count = filtered_df['cluster'].nunique()
        crime_map = generate_crime_map(filtered_df)._repr_html_()
        crime_growth_plot = generate_crime_growth_plot(filtered_df)
        safety_index = calculate_safety_index(filtered_df)

        # High-Risk Zones = clusters with most incidents
        cluster_counts = filtered_df['cluster'].value_counts().head(3)
        high_risk_zones = ", ".join([f"Cluster {c}" for c in cluster_counts.index])
        
        # Predicted Future Hotspots = clusters with highest growth in last month
        if 'Date' in filtered_df.columns:
            last_month = filtered_df['Date'].max() - pd.DateOffset(months=1)
            recent_counts = filtered_df[filtered_df['Date'] > last_month]['cluster'].value_counts()
            future_hotspots = ", ".join([f"Cluster {c}" for c in recent_counts.index])
        else:
            future_hotspots = "N/A"

        recommendations = preventive_recommendations(filtered_df)

    return render_template(
        "index.html",
        cities=sorted(df["City"].unique()),
        crimes=sorted(df["Crime Description"].unique()),
        selected_city=city,
        selected_crime=crime_type,
        record_count=record_count,
        hotspot_count=hotspot_count,
        crime_map=crime_map,
        crime_growth_plot=crime_growth_plot,
        safety_index=safety_index,
        high_risk_zones=high_risk_zones,
        future_hotspots=future_hotspots,
        recommendations=recommendations
    )

@app.route("/cluster", methods=["GET"])
def cluster_details():
    cluster_id = int(request.args.get("cluster", 0))
    cluster_df = df[df["cluster"] == cluster_id]

    if cluster_df.empty:
        return "<h2>No crimes found for this cluster.</h2><p><a href='/'>Back to main map</a></p>"

    m = generate_crime_map(cluster_df, show_popup_links=False)

    marker_cluster = MarkerCluster().add_to(m)
    for _, row in cluster_df.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=6,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.8,
            popup=f"<b>{row['City']}</b><br>{row['Crime Description']}<br>Victim: {row['Victim Age']} ({row['Victim Gender']})"
        ).add_to(marker_cluster)

    return f"""
    <h2 style='text-align:center;'>Cluster {cluster_id} Crimes</h2>
    {m._repr_html_()}
    <p style='text-align:center;'><a href='/'>Back to main map</a></p>
    """

if __name__ == "__main__":
    app.run(debug=True)
