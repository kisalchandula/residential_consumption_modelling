import pandas as pd
import numpy as np

from scipy.stats import wasserstein_distance, entropy
from scipy.spatial.distance import jensenshannon
from statsmodels.tsa.stattools import acf, pacf
from hurst import compute_Hc


def compare_load_profiles(
    series,
    net_gan,
    acf_lags=48,
    normalize=True
):
    """
    Compare measured and synthetic load profiles using:

    - Wasserstein Distance
    - Jensen-Shannon Divergence
    - KL Divergence
    - ACF Distance
    - PACF Distance
    - Hurst Exponent Difference
    - FFT Spectral Distance
    """

    # ===================================================
    # TIMEZONE HANDLING
    # ===================================================
    def to_utc(s):
        s = s.copy()

        if s.index.tz is None:
            s.index = s.index.tz_localize("UTC")
        else:
            s.index = s.index.tz_convert("UTC")

        return s

    series = to_utc(series)
    net_gan = to_utc(net_gan)

    # ===================================================
    # RESOLUTION MATCHING
    # ===================================================
    def get_resolution_minutes(index):
        freq = index.to_series().diff().dropna().median()
        return int(freq.total_seconds() / 60)

    res_series = get_resolution_minutes(series.index)
    res_gan = get_resolution_minutes(net_gan.index)

    target_res = max(res_series, res_gan)
    rule = f"{target_res}min"

    if res_series != target_res:
        series = series.resample(rule).mean()

    if res_gan != target_res:
        net_gan = net_gan.resample(rule).mean()

    # ===================================================
    # ALIGN + CLEAN
    # ===================================================
    df = pd.concat([series, net_gan], axis=1, join="inner")
    df.columns = ["meas", "gan"]

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(df) == 0:
        raise ValueError(
            "No overlapping valid timestamps after alignment."
        )

    meas = df["meas"].astype(float)
    gan = df["gan"].astype(float)

    # ===================================================
    # NORMALIZATION
    # ===================================================
    def minmax(x):

        denom = x.max() - x.min()

        if denom == 0:
            return pd.Series(
                np.zeros(len(x)),
                index=x.index
            )

        return (x - x.min()) / denom

    meas_proc = minmax(meas) if normalize else meas.copy()
    gan_proc = minmax(gan) if normalize else gan.copy()

    # ===================================================
    # WASSERSTEIN DISTANCE
    # ===================================================
    wass_distance = wasserstein_distance(
        meas_proc,
        gan_proc
    )

    # ===================================================
    # HISTOGRAMS FOR JS + KL
    # ===================================================
    bins = 50

    bins_edges = np.linspace(
        min(meas_proc.min(), gan_proc.min()),
        max(meas_proc.max(), gan_proc.max()),
        bins + 1
    )

    px, _ = np.histogram(
        meas_proc,
        bins=bins_edges,
        density=True
    )

    py, _ = np.histogram(
        gan_proc,
        bins=bins_edges,
        density=True
    )

    px += 1e-12
    py += 1e-12

    px /= px.sum()
    py /= py.sum()

    # ===================================================
    # JS DIVERGENCE
    # ===================================================
    js_distance = (
        jensenshannon(
            px,
            py,
            base=2
        ) ** 2
    )

    # ===================================================
    # KL DIVERGENCE
    # ===================================================
    kl_distance = entropy(
        px,
        py
    )

    # ===================================================
    # ACF
    # ===================================================
    meas_acf = acf(
        meas_proc,
        nlags=acf_lags,
        fft=True
    )

    gan_acf = acf(
        gan_proc,
        nlags=acf_lags,
        fft=True
    )

    acf_distance = np.sqrt(
        np.mean(
            (meas_acf - gan_acf) ** 2
        )
    )

    # ===================================================
    # PACF
    # ===================================================
    meas_pacf = pacf(
        meas_proc,
        nlags=acf_lags,
        method="ywm"
    )

    gan_pacf = pacf(
        gan_proc,
        nlags=acf_lags,
        method="ywm"
    )

    pacf_distance = np.sqrt(
        np.mean(
            (meas_pacf - gan_pacf) ** 2
        )
    )

    # ===================================================
    # HURST EXPONENT
    # ===================================================
    try:

        hurst_meas, _, _ = compute_Hc(
            meas_proc.values,
            kind="change",
            simplified=True
        )

        hurst_gan, _, _ = compute_Hc(
            gan_proc.values,
            kind="change",
            simplified=True
        )

        hurst_distance = abs(
            hurst_meas - hurst_gan
        )

    except Exception:

        hurst_distance = np.nan

    # ===================================================
    # FFT DISTANCE
    # ===================================================
    x = meas_proc.values - meas_proc.values.mean()
    y = gan_proc.values - gan_proc.values.mean()

    fx = np.abs(
        np.fft.rfft(x)
    )

    fy = np.abs(
        np.fft.rfft(y)
    )

    fx /= fx.sum() + 1e-12
    fy /= fy.sum() + 1e-12

    fft_distance = np.sqrt(
        np.mean(
            (fx - fy) ** 2
        )
    )

    # ===================================================
    # OUTPUT
    # ===================================================
    metrics = {
        "Wasserstein": wass_distance,
        "JS_divergence": js_distance,
        "KL_divergence": kl_distance,
        "ACF_distance": acf_distance,
        "PACF_distance": pacf_distance,
        "Hurst_distance": hurst_distance,
        "FFT_distance": fft_distance
    }

    return pd.Series(metrics)