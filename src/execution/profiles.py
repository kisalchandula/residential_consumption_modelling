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