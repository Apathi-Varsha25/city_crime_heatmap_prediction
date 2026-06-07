from sklearn.cluster import KMeans

def apply_kmeans(df, k=5):
    """
    Apply KMeans clustering safely based on available data
    """
    unique_points = df[["Latitude", "Longitude"]].drop_duplicates().shape[0]
    k = min(k, unique_points)
    if k < 2:
        df["cluster"] = 0
        return df
    kmeans = KMeans(n_clusters=k, random_state=42)
    df["cluster"] = kmeans.fit_predict(df[["Latitude", "Longitude"]])
    return df
