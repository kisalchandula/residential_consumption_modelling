import pandas as pd
import numpy as np
import requests
import time
from functools import lru_cache

TOKEN = None  # set from config or environment

URL = "https://maps.iee.fraunhofer.de/antsapi/api/v1/query"


def _fetch_sarah_chunk(start, end, lat, lon, token, retries=3):
    params = {
        "x": lon,
        "y": lat,
        "crs": "EPSG:4326",
        "spatial_interp": "nearest",
        "spatial_interp_samples": 3,
        "begin": start,
        "end": end,
        "SIS:var": "SIS",
        "format": "application/json",
        "model": "sarah3"
    }

    headers = {"Authorization": f"Bearer {token}"}

    for _ in range(retries):
        try:
            r = requests.get(URL, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()

            times = pd.to_datetime(data["data"]["coords"]["time/0"])
            ghi = np.array(data["data"]["data"][0]["data"]).squeeze()

            if len(times) == 0:
                return None

            return pd.Series(ghi, index=times)

        except Exception:
            time.sleep(2)

    return None


def get_sarah3_timeseries(lat, lon, start_date, end_date, token):
    """
    Fetch full SARAH-3 time series for a grid cell.
    """

    date_ranges = pd.date_range(start_date, end_date, freq="MS")

    chunks = []

    for i in range(len(date_ranges) - 1):
        start = date_ranges[i].strftime("%Y-%m-%dT00:00:00")
        end = date_ranges[i + 1].strftime("%Y-%m-%dT00:00:00")

        chunk = _fetch_sarah_chunk(start, end, lat, lon, token)

        if chunk is not None:
            chunks.append(chunk)

    if not chunks:
        return None

    series = pd.concat(chunks).sort_index()
    series = series[~series.index.duplicated()]
    series = series.astype(float).fillna(0)

    return series


def fetch_era5_temperature(lat, lon, params, headers, sleep=1):
    """
    Fetch ERA5 temperature time series for one grid cell.
    """

    params_local = params.copy()
    params_local["x"] = lon
    params_local["y"] = lat

    response = requests.get(URL, params=params_local, headers=headers)
    response.raise_for_status()

    data = response.json()

    times = pd.to_datetime(data["data"]["coords"]["time/0"])
    t2m = np.array(data["data"]["data"][0]["data"]).squeeze() - 273.15

    ts = pd.Series(t2m, index=times)

    ts = ts.resample("30min").mean().interpolate()

    time.sleep(sleep)

    return ts


def get_grid_id(lat, lon, resolution=0.25):
    return (
        round(lat / resolution) * resolution,
        round(lon / resolution) * resolution
    )



def build_temperature_cache(df, params, headers):
    """
    Fetch ERA5 temperature per grid cell (like HP model).
    """

    df = df.copy()

    df["grid_id"] = df.apply(
        lambda r: get_grid_id(r["lat"], r["lon"]),
        axis=1
    )

    unique_grids = df["grid_id"].unique()

    grid_temperature_data = {}

    for lat, lon in unique_grids:
        print(f"Fetching temperature: {lat}, {lon}")

        ts = fetch_era5_temperature(
            lat, lon,
            params=params,
            headers=headers
        )

        grid_temperature_data[(lat, lon)] = ts

    return grid_temperature_data, df