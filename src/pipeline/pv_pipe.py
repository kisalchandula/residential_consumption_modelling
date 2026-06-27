import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from data_access.api_reader import get_sarah3_timeseries
from data_processing.pv_feedIn import compute_pv_power


def simulate_pv_feedin(
    household_df,
    start_date="2024-01-01",
    end_date="2025-01-01",
    pv_efficiency=0.20,
    tz="Europe/Berlin",
    n_jobs=-1,
    token=None,
    GRID_RES = 0.05,
    plot=False
):

    df = household_df.copy()

    # fill defaults
    df["ea_p_pv"] = df["ea_p_pv"].fillna(0)
    df["slope"] = df["slope"].fillna(30)
    df["orientation"] = df["orientation"].fillna(180)

    # grouping
    df["lat_grid"] = np.floor(df["lat"] / GRID_RES) * GRID_RES
    df["lon_grid"] = np.floor(df["lon"] / GRID_RES) * GRID_RES

    df["group_id"] = df["lat_grid"].astype(str) + "_" + df["lon_grid"].astype(str)

    grouped = df.groupby("group_id")

    print("household groups:", df["group_id"].nunique())

    def process_group(group_id, group):

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
            tz=tz,
            pv_efficiency=pv_efficiency
        )

        return pv

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_group)(cid, group)
        for cid, group in grouped
    )

    results = [r for r in results if r is not None]

    results_df = pd.concat(results, axis=1)

    results_df["total_pv"] = results_df.sum(axis=1)


    # --------------------------------------------------
    # OPTIONAL PLOT
    # --------------------------------------------------
    if plot:

        plt.figure(figsize=(12, 4))

        plt.plot(
            results_df.index,
            results_df["total_pv"],
            linewidth=0.8,
            label="Total PV Generation"
        )

        plt.xlabel("Time")
        plt.ylabel("Power [kW]")
        plt.title("Aggregated PV Generation")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.show()

    return results_df


