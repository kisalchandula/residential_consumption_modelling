"""
Storage heating model.

This module converts outdoor temperature data into hourly thermal
heating demand and then converts the thermal demand into electricity
consumption of a storage heating system.

Workflow:
1. Calculate hourly heat demand from outdoor temperature.
2. Simulate thermal storage charging and discharging.
3. Convert thermal energy supplied by the heater into electricity demand.

The storage model represents a simple night-storage heating system:
- Storage is charged during low-demand night hours.
- Stored heat is discharged during the day.
- Additional electricity is consumed when storage is insufficient.

The numerical storage dispatch is accelerated using Numba.
"""


import numpy as np
import pandas as pd
from numba import njit



def temperature_to_heat_demand(
    temperature_series,
    annual_heat_demand_kwh,
    T_base=15
):
    """
    Convert outdoor temperature into hourly heating demand.

    A heating degree approach is used. When the outdoor temperature
    falls below the heating base temperature, heating demand increases
    proportionally.

    Parameters
    ----------
    temperature_series : pandas.Series
        Hourly outdoor temperature profile [°C].

    annual_heat_demand_kwh : float
        Annual thermal heating demand [kWh].

    T_base : float, default=15
        Heating base temperature [°C].
        Above this temperature no heating demand is assumed.

    Returns
    -------
    pandas.Series
        Hourly thermal heating demand [kWh].
    """

    # Heating demand signal based on temperature difference
    heating_signal = np.maximum(
        T_base - temperature_series,
        0
    )

    # Avoid division by zero for locations without heating demand
    if heating_signal.sum() == 0:
        return pd.Series(
            0,
            index=temperature_series.index
        )

    # Normalize profile to match annual heat demand
    return (
        heating_signal /
        heating_signal.sum() *
        annual_heat_demand_kwh
    )



@njit
def storage_dispatch_numba(
    heat_values,
    hours,
    days,
    efficiency,
    storage_capacity
):
    """
    Numba accelerated storage heating simulation.

    The algorithm simulates:
    - charging of thermal storage during night hours
    - storage state of charge (SOC)
    - daytime heat supply from storage
    - backup electricity consumption if storage is empty

    Parameters
    ----------
    heat_values : numpy.ndarray
        Hourly thermal demand [kWh].

    hours : numpy.ndarray
        Hour of each timestep.

    days : numpy.ndarray
        Day number of each timestep.

    efficiency : float
        Electrical-to-thermal conversion efficiency.

    storage_capacity : float
        Maximum thermal storage capacity [kWh].

    Returns
    -------
    numpy.ndarray
        Hourly electricity consumption [kWh].
    """

    n = len(heat_values)

    electric = np.zeros(n)

    # Initial storage state of charge
    soc = storage_capacity * 0.5

    current_day = days[0]
    day_start = 0


    for i in range(n + 1):

        # Process one complete day
        if i == n or days[i] != current_day:

            day_end = i

            # Calculate following day heat requirement
            next_day_heat = 0.0

            if i < n:

                next_day = days[i]

                for j in range(i, n):

                    if days[j] == next_day:
                        next_day_heat += heat_values[j]
                    else:
                        break


            # Count available charging hours
            charge_hours = 0

            for j in range(day_start, day_end):

                h = hours[j]

                if h == 22 or h == 23 or h <= 5:
                    charge_hours += 1


            # Charge storage during night hours
            if charge_hours > 0:

                charge_energy = max(
                    next_day_heat - soc,
                    0
                )

                charge_energy = min(
                    charge_energy,
                    storage_capacity - soc
                )

                charge_per_hour = (
                    charge_energy /
                    charge_hours
                )


                for j in range(day_start, day_end):

                    h = hours[j]

                    if h == 22 or h == 23 or h <= 5:

                        electric[j] += (
                            charge_per_hour /
                            efficiency
                        )

                        soc += charge_per_hour

                        if soc > storage_capacity:
                            soc = storage_capacity


            # Supply heat demand from storage
            for j in range(day_start, day_end):

                demand = heat_values[j]

                supplied = min(
                    soc,
                    demand
                )

                soc -= supplied

                remaining = demand - supplied


                # Direct electric heating if storage is empty
                if remaining > 0:

                    electric[j] += (
                        remaining /
                        efficiency
                    )


            if i < n:

                current_day = days[i]
                day_start = i


    return electric



def heat_to_electric_load(
    heat,
    heater_efficiency=0.60,
    storage_capacity=50
):
    """
    Convert thermal heating demand into electricity demand.

    This function applies the storage heating model and returns the
    electricity required by the electric heater.

    Parameters
    ----------
    heat : pandas.Series
        Hourly thermal heating demand [kWh].

    efficiency : float, default=0.9
        Heater efficiency.

    storage_capacity : float, default=50
        Thermal storage capacity [kWh].

    Returns
    -------
    pandas.Series
        Hourly electricity consumption [kWh].
    """

    heat_values = heat.values.astype(
        np.float64
    )

    hours = heat.index.hour.values

    days = heat.index.dayofyear.values


    electric = storage_dispatch_numba(
        heat_values,
        hours,
        days,
        heater_efficiency,
        storage_capacity
    )


    return pd.Series(
        electric,
        index=heat.index
    )