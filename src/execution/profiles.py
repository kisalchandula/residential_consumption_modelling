import pandas as pd

def create_net_load(load_df, pv_df, hp_df, sh_df=None, resample_rule="30T"):

    load_df = load_df.copy()
    pv_df = pv_df.copy()
    hp_df = hp_df.copy()

    if sh_df is not None:
        sh_df = sh_df.copy()

    # datetime index
    load_df.index = pd.to_datetime(load_df.index)
    pv_df.index = pd.to_datetime(pv_df.index)
    hp_df.index = pd.to_datetime(hp_df.index)

    if sh_df is not None:
        sh_df.index = pd.to_datetime(sh_df.index)

    # timezone fix
    for df in [load_df, pv_df, hp_df]:
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

    if sh_df is not None and sh_df.index.tz is not None:
        sh_df.index = sh_df.index.tz_convert(None)

    # resample
    load_df = load_df.resample(resample_rule).sum()
    pv_df = pv_df.resample(resample_rule).sum()
    hp_df = hp_df.resample(resample_rule).sum()

    if sh_df is not None:
        sh_df = sh_df.resample(resample_rule).sum()

    # ---- SAFE CONVERSION ----
    load = load_df.sum(axis=1) if isinstance(load_df, pd.DataFrame) else load_df
    pv = pv_df.sum(axis=1) if isinstance(pv_df, pd.DataFrame) else pv_df
    hp = hp_df.sum(axis=1) if isinstance(hp_df, pd.DataFrame) else hp_df

    if sh_df is not None:
        sh = sh_df.sum(axis=1) if isinstance(sh_df, pd.DataFrame) else sh_df
    else:
        sh = 0

    # align indices
    idx = load.index.intersection(pv.index).intersection(hp.index)

    if sh_df is not None:
        idx = idx.intersection(sh.index)

    load = load.loc[idx]
    pv = pv.loc[idx]
    hp = hp.loc[idx]

    if sh_df is not None:
        sh = sh.loc[idx]

    # ---- NET LOAD ----
    net = load + hp + sh - pv

    return net


def create_residual_load(
    rlm_df: pd.DataFrame,
    measured_series: pd.Series,
    rlm_column: str = "selected_aggregate_sum",
    measured_factor: float = 2.0,
    resample_freq: str = "30min",
):
    """
    Calculate residual load from measured and RLM data.

    Parameters
    ----------
    rlm_df : pd.DataFrame
        DataFrame containing columns ['Zeit', rlm_column].

    measured_series : pd.Series
        Measured load time series with DatetimeIndex.

    rlm_column : str
        Name of the RLM column to use.

    measured_factor : float
        Divisor applied to measured values before subtraction.
        Example:
            measured_factor=2 -> measured/2 - rlm
            measured_factor=1 -> measured - rlm

    resample_freq : str
        Resampling frequency (e.g. '30min', '1h').

    Returns
    -------
    pd.Series
        Residual load time series.

    pd.DataFrame
        Intermediate dataframe containing:
        ['measured', 'rlm', 'residual_load']
    """

    # -------------------------
    # RLM
    # -------------------------
    rlm = rlm_df.copy()
    rlm["Zeit"] = pd.to_datetime(rlm["Zeit"])

    if rlm["Zeit"].dt.tz is None:
        rlm["Zeit"] = rlm["Zeit"].dt.tz_localize("UTC")
    else:
        rlm["Zeit"] = rlm["Zeit"].dt.tz_convert("UTC")

    rlm = rlm.set_index("Zeit")[rlm_column]

    # -------------------------
    # Measured
    # -------------------------
    meas = measured_series.copy()

    if meas.index.tz is None:
        meas.index = meas.index.tz_localize("UTC")
    else:
        meas.index = meas.index.tz_convert("UTC")

    # -------------------------
    # Resample
    # -------------------------
    rlm_resampled = rlm.resample(resample_freq).mean()
    meas_resampled = meas.resample(resample_freq).mean()

    # -------------------------
    # Align
    # -------------------------
    df = pd.concat(
        [meas_resampled, rlm_resampled],
        axis=1,
        join="inner"
    )

    df.columns = ["measured", "rlm"]

    # -------------------------
    # Residual Load
    # -------------------------
    df["residual_load"] = (
        df["measured"] / measured_factor
        - df["rlm"]
    )

    return df["residual_load"], df




def normalize_profile(profile, target_kwh, dt=0.5):
    """
    Scale kW profile to match annual energy target (kWh).
    """
    current_kwh = profile.sum() * dt

    if current_kwh == 0:
        return profile

    return profile * (target_kwh / current_kwh)


import pandas as pd
import glob
import os
import re


def normalize_uw_profiles(
    base_folder,
    yearly_energy_mapping,
    start_date,
    end_date,
    timestep_hours=0.5
):
    """
    Aggregate and normalize UW synthetic profiles.

    Parameters
    ----------
    base_folder : str
        Folder containing generated_chunksX folders.

    yearly_energy_mapping : dict
        Dictionary {hh_id: yearly energy in kWh}
        Example:
        {
            "10001": 3500,
            "10002": 4200
        }

    start_date : str
        Start date of the period to normalize.
        Example: "2024-01-01"

    end_date : str
        End date of the period to normalize.
        Example: "2024-12-31 23:30"

    timestep_hours : float
        Duration of each timestep.
        For 30-min profiles use 0.5.

    Returns
    -------
    None
    """

    # ---------------------------------
    # FIND GENERATED CHUNK FOLDERS
    # ---------------------------------
    all_folders = glob.glob(
        os.path.join(base_folder, "generated_chunks*")
    )

    def extract_number(path):
        match = re.search(r"(\d+)$", path)
        return int(match.group(1)) if match else -1

    all_folders = sorted(
        all_folders,
        key=extract_number
    )


    # ---------------------------------
    # PROCESS EACH CHUNK FOLDER
    # ---------------------------------
    for folder in all_folders:

        print(f"\nProcessing: {folder}")

        files = sorted(
            glob.glob(
                os.path.join(folder, "*.parquet")
            )
        )

        if len(files) == 0:
            print(" → No parquet files found")
            continue


        # Load profiles
        dfs = [
            pd.read_parquet(f)
            for f in files
        ]

        profiles = pd.concat(
            dfs,
            axis=1
        )

        # Remove duplicated HH
        profiles = profiles.loc[
            :,
            ~profiles.columns.duplicated()
        ]


        # Ensure datetime index
        profiles.index = pd.to_datetime(
            profiles.index
        )


        # ---------------------------------
        # SELECT REQUIRED PERIOD
        # ---------------------------------
        profiles = profiles.loc[
            start_date:end_date
        ]


        # ---------------------------------
        # NORMALIZE EACH HOUSEHOLD
        # ---------------------------------
        normalized = profiles.copy()


        for hh_id in profiles.columns:

            # skip if no mapping available
            if hh_id not in yearly_energy_mapping:
                continue


            target_energy = yearly_energy_mapping[hh_id]


            # Current generated energy
            current_energy = (
                profiles[hh_id].sum()
                * timestep_hours
            )


            if current_energy == 0:
                print(
                    f"Warning: {hh_id} has zero energy"
                )
                continue


            # Scaling factor
            factor = (
                target_energy /
                current_energy
            )


            normalized[hh_id] = (
                profiles[hh_id]
                * factor
            )


        # ---------------------------------
        # CREATE UW AGGREGATE PROFILE
        # ---------------------------------
        uw_sum_profile = pd.DataFrame(
            {
                "uw_sum_profile":
                normalized.sum(axis=1)
            }
        )


        # Save
        output_path = os.path.join(
            folder,
            "uw_sum_profile_normalized.parquet"
        )

        uw_sum_profile.to_parquet(
            output_path
        )


        print(
            " → Saved:",
            output_path
        )