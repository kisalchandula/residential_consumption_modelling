# data_processing/storage_heating.py

"""
Storage heating load generation.

This module converts annual heat demand into an hourly heat demand profile
and uses a PyPSA optimization model to calculate the corresponding
electricity consumption of storage heating systems.

Workflow:
1. Convert temperature profile into heat demand distribution.
2. Model heat storage operation with PyPSA.
3. Calculate electricity demand required for heating.
"""


import numpy as np
import pandas as pd
import pypsa


def temperature_to_heat_demand(temperature_series, annual_heat_demand_kwh, T_base=15.0):
    """
    Convert outdoor temperature into hourly heat demand.

    Parameters
    ----------
    temperature_series : pandas.Series
        Outdoor temperature time series.

    annual_heat_demand_kwh : float
        Annual heat demand in kWh.

    T_base : float
        Heating base temperature.

    Returns
    -------
    pandas.Series
        Hourly heat demand profile.
    """

    # Calculate heating intensity based on temperature
    heating_signal = np.maximum(T_base - temperature_series, 0)

    if heating_signal.sum() == 0:
        return pd.Series(0, index=temperature_series.index)

    # Scale hourly signal to annual heat demand
    heat_demand = heating_signal / heating_signal.sum() * annual_heat_demand_kwh

    return pd.Series(heat_demand, index=temperature_series.index)



def heat_to_electric_load(heat_ts, heater_efficiency=0.9, storage_capacity=50):
    """
    Convert heat demand into electricity consumption using PyPSA.

    Parameters
    ----------
    heat_ts : pandas.Series
        Hourly heat demand profile.

    heater_efficiency : float
        Efficiency of electric heating system.

    storage_capacity : float
        Thermal storage capacity.

    Returns
    -------
    pandas.Series
        Electricity consumption profile.
    """

    # Create PyPSA network
    n = pypsa.Network()

    n.set_snapshots(heat_ts.index)

    # Create energy buses
    n.add("Bus", "electricity")
    n.add("Bus", "heat")

    # Add heat demand
    n.add("Load", "heat_demand", bus="heat", p_set=heat_ts)

    # Add electricity supply
    n.add("Generator", "grid_supply", bus="electricity", p_nom=10000, marginal_cost=0.2)

    # Add thermal storage
    n.add("Store", "water_tank", bus="heat", e_nom=storage_capacity, e_initial=storage_capacity / 2, e_cyclic=True)

    # Add electric heater
    n.add("Link", "power_to_heat", bus0="electricity", bus1="heat", efficiency=heater_efficiency, p_nom=10000)

    # Optimize storage operation
    n.optimize()

    # Extract electricity consumption
    electric_load = n.links_t.p0["power_to_heat"].copy()

    return electric_load