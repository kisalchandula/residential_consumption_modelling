import pandas as pd
import matplotlib.pyplot as plt


def load_measured_profile(
    metadata_df,
    measured_ts_df,
    station_id="42001369",
    voltage_level="110kV",
    output_resolution="30min",
    plot=False,
):
    """
    Load and aggregate measured substation profile.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Metadata containing measurement information.

    measured_ts_df : pd.DataFrame
        Raw measured time series data.

    station_id : str
        Station_FID used for filtering.

    voltage_level : str
        Voltage level used for filtering.

    output_resolution : str
        Output resampling frequency (e.g. "30min", "1H").

    plot : bool
        If True, plot resulting measured profile.

    Returns
    -------
    pd.Series
        Aggregated measured load profile [kWh].
    """

    # --------------------------------------------------
    # 1. FILTER METADATA
    # --------------------------------------------------
    meta_filtered = metadata_df[
        metadata_df["Messung"].str.contains(
            "wirkl.",
            regex=False,
            na=False,
            case=False,
        )
        &
        metadata_df["MittelMinMaxWert"].str.contains(
            "Wert",
            regex=False,
            na=False,
            case=False,
        )
        &
        metadata_df["Station_FID"].astype(str).str.contains(
            str(station_id),
            regex=False,
            na=False,
        )
        &
        metadata_df["Spannungsebene"].str.contains(
            voltage_level,
            regex=False,
            na=False,
        )
    ]

    if meta_filtered.empty:
        raise ValueError(
            "No matching metadata found."
        )

    # --------------------------------------------------
    # 2. EXTRACT MEASUREMENT IDS
    # --------------------------------------------------
    ids = (
        meta_filtered["ZeitreihenID"]
        .astype(str)
        .tolist()
    )

    selected_cols = [
        col
        for col in measured_ts_df.columns
        if str(col) in ids
    ]

    if not selected_cols:
        raise ValueError(
            "No matching measurement IDs found in time series data."
        )

    # --------------------------------------------------
    # 3. CONVERT VALUES AND AGGREGATE
    # --------------------------------------------------
    profile = (
        measured_ts_df[selected_cols]
        .apply(
            lambda x:
            x.astype(str)
             .str.split()
             .str[0]
             .astype(float)
        )
        .sum(axis=1)
    )

    # --------------------------------------------------
    # 4. TIME INDEX
    # --------------------------------------------------
    profile.index = pd.to_datetime(profile.index)

    profile = profile.sort_index()

    # --------------------------------------------------
    # 5. DETERMINE INPUT RESOLUTION AUTOMATICALLY
    # --------------------------------------------------
    time_step = (
        profile.index
        .to_series()
        .diff()
        .dropna()
        .mode()[0]
    )

    time_step_hours = (
        time_step.total_seconds() / 3600
    )

    if time_step_hours <= 0:
        raise ValueError(
            "Invalid timestep detected."
        )

    # --------------------------------------------------
    # 6. REMOVE OUTLIERS
    # --------------------------------------------------
    mean = profile.mean()
    std = profile.std()

    profile = profile[
        (profile > mean - 4 * std)
        &
        (profile < mean + 2 * std)
    ]

    # --------------------------------------------------
    # 7. CONVERT MW -> kWh
    # --------------------------------------------------
    profile_kwh = (
        profile
        * 1000
        * time_step_hours
    )

    # --------------------------------------------------
    # 8. RESAMPLE OUTPUT RESOLUTION
    # --------------------------------------------------
    profile_kwh = (
        profile_kwh
        .resample(output_resolution)
        .sum()
    )

    profile_kwh.name = "measured_load_kwh"


    # --------------------------------------------------
    # 9. OPTIONAL PLOT
    # --------------------------------------------------
    if plot:

        plt.figure(figsize=(12, 4))

        plt.plot(
            profile_kwh.index,
            profile_kwh.values,
            linewidth=0.8,
            label="Measured Load"
        )

        plt.xlabel("Time")
        plt.ylabel("Energy [kWh]")
        plt.title(
            f"Measured Profile - "
            f"Station {station_id} ({voltage_level})"
        )

        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.show()


    return profile_kwh



def get_rlm_aggregate_for_uw(
    rlm_ts: pd.DataFrame,
    rlm_meta: pd.DataFrame,
    uw_fid: str,
    voltage_level: str = "NS",
    aggregation: str = "sum"
):
    """
    Aggregate RLM time series for a given UW station.

    Parameters
    ----------
    rlm_ts : pd.DataFrame
        Time series dataframe containing 'Zeit' and metering code columns.

    rlm_meta : pd.DataFrame
        Metadata dataframe.

    uw_fid : str
        UW station ID (e.g. "42001361").

    voltage_level : str, default="NS"
        Voltage level filter.

    aggregation : str, default="sum"
        Either 'sum' or 'mean'.

    Returns
    -------
    pd.DataFrame
        DataFrame with Zeit and aggregated load.
    """

    # Filter metadata
    meta_filtered = rlm_meta[
        rlm_meta["Station_UW_FID"]
        .astype(str)
        .str.contains(str(uw_fid), regex=False, na=False)
        &
        (rlm_meta["rlm_zeitreihen_SPANNUNGSEBENE_S"] == voltage_level)
    ]

    # Get metering IDs
    metering_ids = meta_filtered[
        "rlm_zeitreihen_METERINGCODE_S"
    ].tolist()

    # Keep only available IDs
    available = [c for c in metering_ids if c in rlm_ts.columns]
    missing = [c for c in metering_ids if c not in rlm_ts.columns]

    if missing:
        print(f"Warning: {len(missing)} metering IDs not found.")

    if len(available) == 0:
        raise ValueError("No matching metering IDs found in rlm_ts.")

    # Aggregate
    result = rlm_ts[["Zeit"]].copy()

    if aggregation.lower() == "sum":
        result["rlm_aggregate"] = rlm_ts[available].sum(axis=1)

    elif aggregation.lower() == "mean":
        result["rlm_aggregate"] = rlm_ts[available].mean(axis=1)

    else:
        raise ValueError("aggregation must be 'sum' or 'mean'")

    return result

