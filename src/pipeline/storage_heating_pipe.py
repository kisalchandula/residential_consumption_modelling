"""
Storage heating profile generation pipeline.

Workflow:
1. Group households by ERA5 grid location.
2. Fetch temperature data for each grid.
3. Convert temperature into heat demand.
4. Convert heat demand into electricity consumption using storage model.
5. Return household electricity profiles.
"""

import pandas as pd

from data_access.api_reader import fetch_era5_temperature, get_grid_id
from data_processing.storage_heating import temperature_to_heat_demand, heat_to_electric_load


def simulate_storage_heating(
    household_df,
    token,
    start_date="2024-01-01",
    end_date="2025-01-01",
    heater_efficiency=0.60,
    T_base=15,
    storage_capacity=10
):
    """
    Generate storage heating electricity profiles.

    Parameters
    ----------
    household_df : pandas.DataFrame
        Household metadata containing:
        - lat
        - lon
        - W_SH
        - AN_FID

    token : str
        API authentication token.

    start_date : str
        Simulation start date.

    end_date : str
        Simulation end date.

    heater_efficiency : float
        Heater heater_efficiency.

    T_base : float
        Heating base temperature.

    storage_capacity : float
        Thermal storage capacity [kWh].

    Returns
    -------
    pandas.DataFrame
        Hourly electricity demand profile per household.
    """

    df = household_df.copy()

    df["W_SH"] = df["W_SH"].fillna(0)


    # --------------------------------------------------
    # Create ERA5 grid IDs
    # --------------------------------------------------

    df["grid_id"] = df.apply(
        lambda x: get_grid_id(
            x["lat"],
            x["lon"]
        ),
        axis=1
    )


    # --------------------------------------------------
    # ERA5 API configuration
    # --------------------------------------------------

    params = {
        "crs": "EPSG:4326",
        "spatial_interp": "nearest",
        "spatial_interp_samples": 3,
        "height": 100,
        "height_interp": "nearest",
        "begin": start_date,
        "end": end_date,
        "resample": "",
        "resample_method": "nearest",
        "t2m:var": "t2m",
        "format": "application/json",
        "model": "era5"
    }


    headers = {
        "Authorization": f"Bearer {token}"
    }


    # --------------------------------------------------
    # Temperature fetching
    # One request per grid
    # --------------------------------------------------

    temperature_cache = {}

    for grid in df["grid_id"].unique():

        temperature_cache[grid] = fetch_era5_temperature(
            grid[0],
            grid[1],
            params,
            headers
        )


    # --------------------------------------------------
    # Household simulation
    # --------------------------------------------------

    profiles = []

    for _, row in df.iterrows():

        temperature = temperature_cache[
            row["grid_id"]
        ]


        heat = temperature_to_heat_demand(
            temperature_series=temperature,
            annual_heat_demand_kwh=row["W_SH"] * heater_efficiency,
            T_base=T_base
        )


        electric = heat_to_electric_load(
            heat,
            heater_efficiency=heater_efficiency,
            storage_capacity=storage_capacity
        )


        electric.name = str(
            row["AN_FID"]
        )

        profiles.append(
            electric
        )


    # --------------------------------------------------
    # Combine household profiles
    # --------------------------------------------------

    result = pd.concat(
        profiles,
        axis=1
    )


    result["total_storage_heating"] = (
        result.sum(axis=1)
    )


    return result