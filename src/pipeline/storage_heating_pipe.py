import numpy as np
import pandas as pd

from data_access.api_reader import fetch_era5_temperature, get_grid_id, build_temperature_cache
from data_processing.storage_heating import temperature_to_heat_demand, heat_to_electric_load


def generate_storage_heating_profiles(
    household_df,
    token,
    start_year="2024-01-01",
    end_year="2024-12-31",
    heater_efficiency=0.9,
    T_base=15
):

    # =====================================================
    # API CONFIG
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

    headers = {
        "Authorization": f"Bearer {token}"
    }

    # =====================================================
    # GRID TEMPERATURE CACHE
    # =====================================================
    grid_temperature_data, household_df = build_temperature_cache(
        household_df, params, headers
    )

    all_profiles = []

    # =====================================================
    # MAIN LOOP
    # =====================================================
    for _, row in household_df.iterrows():

        lat = row["lat"]
        lon = row["lon"]

        annual_electricity = row["W_SH"]

        grid_id = get_grid_id(lat, lon)

        temperature_series = grid_temperature_data[grid_id]

        # -------------------------------------------------
        # Electricity → Heat
        # -------------------------------------------------
        annual_heat = annual_electricity * heater_efficiency

        heat_ts = temperature_to_heat_demand(
            temperature_series=temperature_series,
            annual_heat_demand_kwh=annual_heat,
            T_base=T_base
        )

        # -------------------------------------------------
        # Convert heat → electric load (your model)
        # -------------------------------------------------
        electric_profile = heat_to_electric_load(heat_ts)

        # -------------------------------------------------
        # Store result
        # -------------------------------------------------
        tmp = pd.DataFrame({
            "time": electric_profile.index,
            "electric_load": electric_profile.values,
            "household_id": row["AN_FID"],
            "lat": lat,
            "lon": lon
        })

        all_profiles.append(tmp)

    # =====================================================
    # FINAL OUTPUT (OUTSIDE LOOP - IMPORTANT FIX)
    # =====================================================
    final_df = pd.concat(all_profiles, ignore_index=True)

    final_wide = final_df.pivot(
        index="time",
        columns="household_id",
        values="electric_load"
    )

    return final_wide