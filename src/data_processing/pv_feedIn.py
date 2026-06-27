# data_processing/pv_feedIn.py

"""
PV feed-in profile generation.

This module converts solar irradiance data into PV power generation
profiles for multiple households.

Workflow:
1. Calculate solar position from location and time.
2. Estimate DNI and DHI from GHI.
3. Calculate plane-of-array irradiance for each PV system.
4. Convert solar irradiance into electrical PV power output.

The calculation uses household-specific PV parameters:
- roof slope (tilt)
- roof orientation (azimuth)
- installed PV capacity
"""


import numpy as np
import pandas as pd
import pvlib

def compute_pv_power(ghi_series, lat, lon, metadata_df, pv_efficiency=0.20, tz="Europe/Berlin"):
    """
    Convert irradiance into PV power generation profiles.

    Parameters
    ----------
    ghi_series : pandas.Series
        Global horizontal irradiance time series (W/m²).

    lat : float
        Latitude of PV systems.

    lon : float
        Longitude of PV systems.

    metadata_df : pandas.DataFrame
        PV metadata for households.
        Required columns:
        - slope
        - orientation
        - ea_p_pv
        - an_fid

    tz : str
        Timezone.

    Returns
    -------
    pandas.DataFrame
        PV power generation profiles (kW) per household.
    """

    # Create pvlib location object
    location = pvlib.location.Location(lat, lon, tz=tz)

    # Calculate sun position for each timestep
    solar_position = location.get_solarposition(ghi_series.index)

    zenith_rad = np.radians(solar_position["apparent_zenith"])

    ghi = ghi_series

    # Estimate diffuse horizontal irradiance
    dhi = 0.2 * ghi

    # Calculate direct normal irradiance
    cos_zenith = np.cos(zenith_rad)
    cos_zenith[cos_zenith <= 0] = np.nan

    dni = (ghi - dhi) / cos_zenith
    dni = np.maximum(dni, 0)

    dni = pd.Series(dni, index=ghi.index).fillna(0)


    # Extract household PV parameters
    tilt = metadata_df["slope"].values
    azimuth = metadata_df["orientation"].values
    capacity = metadata_df["ea_p_pv"].values


    # Expand dimensions: rows = timesteps / columns = households
    tilt_2d = tilt[np.newaxis, :]
    az_2d = azimuth[np.newaxis, :]
    cap_2d = capacity[np.newaxis, :]


    dni_2d = dni.values[:, np.newaxis]
    ghi_2d = ghi.values[:, np.newaxis]
    dhi_2d = dhi.values[:, np.newaxis]

    zenith = solar_position["apparent_zenith"].values[:, np.newaxis]
    azimuth_sun = solar_position["azimuth"].values[:, np.newaxis]

    # Calculate irradiance received by tilted PV panels 
    # poa(Plane of Array)
    poa = pvlib.irradiance.get_total_irradiance( 
        surface_tilt=tilt_2d,
        surface_azimuth=az_2d,
        dni=dni_2d,
        ghi=ghi_2d,
        dhi=dhi_2d,
        solar_zenith=zenith,
        solar_azimuth=azimuth_sun
    )

    # Convert solar irradiance into electrical PV power
    power = poa["poa_global"] * cap_2d * pv_efficiency / 1000 # w-->kw

    # Use household IDs as output columns
    cols = metadata_df["an_fid"].astype(str).values

    return pd.DataFrame(
        power,
        index=ghi.index,
        columns=cols
    )