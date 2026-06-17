# data_processing/heatpump.py

"""
Heat pump load profile generation.

This module creates synthetic heat pump electricity consumption profiles
based on outdoor temperature patterns.

Workflow:
1. Convert temperature time series into daily profiles.
2. Cluster daily temperature patterns using KMeans.
3. Generate heat pump load profiles using temperature demand,
   daily usage patterns, COP, and stochastic variation.
"""


import numpy as np
from sklearn.cluster import KMeans


def generate_hp_profile(cluster_id, day_temp, kmeans, cop=3.0):
    """
    Generate heat pump electricity load profile for one day.

    Parameters
    ----------
    cluster_id : int
        Assigned temperature cluster.

    day_temp : pandas.Series
        Temperature profile for one day.

    kmeans : sklearn KMeans
        Trained clustering model.

    cop : float
        Coefficient of performance of heat pump.

    Returns
    -------
    numpy.ndarray
        Heat pump electricity profile.
    """

    temp = day_temp.values
    n = len(temp)

    # Calculate temperature-dependent heating demand
    demand = np.clip(18 - temp, 0, None)

    if demand.max() > 0:
        demand = demand / demand.max()

    # Create typical daily operation pattern
    hours = np.linspace(0, 24, n)

    morning = np.exp(-0.5 * ((hours - 7) / 2) ** 2)
    evening = np.exp(-0.5 * ((hours - 19) / 3) ** 2)

    shape = morning + evening
    shape = shape / shape.max()

    # Adjust demand according to temperature cluster characteristics
    cluster_temp = kmeans.cluster_centers_[cluster_id].mean()
    cluster_factor = np.clip((18 - cluster_temp) / 10, 0.5, 2.0)

    # Convert thermal demand into electricity consumption
    power = (demand * shape * cluster_factor) / cop

    # Add random variation
    noise = np.random.normal(1.0, 0.05, n)
    power = power * noise

    # Convert power to energy for 30-minute timestep
    profile = power * 0.5

    return profile



def build_daily_matrix(temp_series):
    """
    Convert full temperature time series into daily profiles.

    Output:
    Each row represents one day with 48 half-hourly values.
    """

    days = []

    n_days = len(temp_series) // 48

    for i in range(n_days):

        day = temp_series.iloc[i*48:(i+1)*48]

        if len(day) == 48:
            days.append(day.values)

    return np.stack(days)



def train_kmeans(X_train, n_clusters=6, seed=42):
    """
    Train KMeans model on daily temperature patterns.
    """

    model = KMeans(
        n_clusters=n_clusters,
        random_state=seed
    )

    model.fit(X_train)

    return model



def predict_daily_clusters(model, X):
    """
    Assign daily temperature profiles to clusters.
    """

    return model.predict(X)