import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from data_access.api_reader import get_sarah3_timeseries
from data_processing.pv_feedIn import compute_pv_power


GRID_RES = 0.05


def simulate_pv_clustered_parallel(
    household_df,
    start_date="2024-01-01",
    end_date="2025-01-01",
    tz="Europe/Berlin",
    n_jobs=-1,
    token=None
):

    df = household_df.copy()

    # fill defaults
    df["ea_p_pv"] = df["ea_p_pv"].fillna(0)
    df["slope"] = df["slope"].fillna(30)
    df["orientation"] = df["orientation"].fillna(180)

    # clustering
    df["lat_grid"] = np.floor(df["lat"] / GRID_RES) * GRID_RES
    df["lon_grid"] = np.floor(df["lon"] / GRID_RES) * GRID_RES

    df["cluster_id"] = df["lat_grid"].astype(str) + "_" + df["lon_grid"].astype(str)

    grouped = df.groupby("cluster_id")

    print("Clusters:", df["cluster_id"].nunique())

    def process_cluster(cluster_id, group):

        lat = group["lat_grid"].iloc[0]
        lon = group["lon_grid"].iloc[0]

        # WEATHER LAYER (external)
        ghi = get_sarah3_timeseries(lat, lon, start_date, end_date, token)

        if ghi is None:
            return None

        # PV MODEL LAYER
        pv = compute_pv_power(
            ghi_series=ghi,
            lat=lat,
            lon=lon,
            metadata_df=group,
            tz=tz
        )

        return pv

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_cluster)(cid, group)
        for cid, group in grouped
    )

    results = [r for r in results if r is not None]

    results_df = pd.concat(results, axis=1)

    results_df["total_pv"] = results_df.sum(axis=1)

    return results_df