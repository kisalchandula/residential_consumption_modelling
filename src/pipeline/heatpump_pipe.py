import numpy as np
import pandas as pd

from data_access.api_reader import fetch_era5_temperature, get_grid_id
from data_processing.heatpump import build_daily_matrix, train_kmeans, predict_daily_clusters, generate_hp_profile

def generate_heatpump_profiles(
    household_df,
    start_year="2023-01-01",
    end_year="2025-01-01",
    kmeans=None,
    token=None,
    plot=False
):
    """
    Generate heat pump load profiles per household using ERA5 temperature clustering.
    """

    # =====================================================
    # INTERNAL API CONFIG 
    # =====================================================
    params = {
        "crs": "EPSG:4326",
        "spatial_interp": "nearest",
        "spatial_interp_samples": 3,
        "height": 100,
        "height_interp": "nearest",
        "begin": start_year,
        "end": end_year,
        "resample": "",
        "resample_method": "nearest",
        "t2m:var": "t2m",
        "format": "application/json",
        "model": "era5"
    }

    url = "https://maps.iee.fraunhofer.de/antsapi/api/data/v1/query"

    headers = {
        "Authorization": f"Bearer {token}"
    }

    # =====================================================
    # GRID ASSIGNMENT
    # =====================================================
    household_df["grid_id"] = household_df.apply(
        lambda r: get_grid_id(r["lat"], r["lon"]),
        axis=1
    )

    unique_grids = household_df["grid_id"].unique()
    grid_temperature_data = {}

    # =====================================================
    # FETCH WEATHER DATA
    # =====================================================
    for lat, lon in unique_grids:
        print(f"Fetching {lat}, {lon}")

        ts = fetch_era5_temperature(lat, lon, params, headers)
        grid_temperature_data[(lat, lon)] = ts

    # =====================================================
    # TRAIN KMEANS IF NOT PROVIDED
    # =====================================================
    if kmeans is None:
        all_days = []

        for ts in grid_temperature_data.values():
            X = build_daily_matrix(ts)
            all_days.append(X)

        X_train = np.vstack(all_days)
        kmeans = train_kmeans(X_train, n_clusters=6)

    # =====================================================
    # GENERATE HEAT PUMP PROFILES
    # =====================================================
    all_hp_profiles = []

    for _, row in household_df.iterrows():

        lat, lon = row["lat"], row["lon"]
        annual_kwh = row["W_WP"]
        grid_id = get_grid_id(lat, lon)

        ts = grid_temperature_data[grid_id]
        day_splits = build_daily_matrix(ts)

        clusters = predict_daily_clusters(kmeans, day_splits)

        hp_load = []
        time_index = []

        for i, cluster_id in enumerate(clusters):

            profile = generate_hp_profile(
                cluster_id,
                pd.Series(day_splits[i]),
                kmeans
            )

            hp_load.extend(profile)

            start = i * 48
            time_index.extend(ts.index[start:start + 48])

        hp_load = np.array(hp_load)

        # scale to annual energy
        if hp_load.sum() > 0:
            hp_load *= (annual_kwh / hp_load.sum())

        df = pd.DataFrame({
            "time": time_index,
            "hp_load": hp_load,
            "household_id": row["AN_FID"]
        })

        all_hp_profiles.append(df)


        final_df = pd.concat(all_hp_profiles, ignore_index=True)

        final_wide = final_df.pivot(
            index="time",
            columns="household_id",
            values="hp_load"
        )


    # =====================================================
    # OPTIONAL PLOT
    # =====================================================
    if plot:

        import matplotlib.pyplot as plt

        total_hp = final_wide.sum(axis=1)

        plt.figure(figsize=(12, 4))

        plt.plot(
            total_hp.index,
            total_hp.values,
            linewidth=0.8,
            label="Total Heat Pump Load"
        )

        plt.xlabel("Time")
        plt.ylabel("Load [kWh]")
        plt.title("Aggregated Heat Pump Load")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.show()

    return final_wide