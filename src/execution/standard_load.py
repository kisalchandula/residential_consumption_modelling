#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime
from pathlib import Path

import pandas as pd
import geopandas as gpd
import holidays


def make_year_index(year: int, freq: str, tz):
    year_start = pd.Timestamp(str(year), tz="UTC")
    year_end = pd.Timestamp(str(year + 1), tz="UTC")

    return (
        pd.date_range(start=year_start, end=year_end, freq=freq)[:-1]
        .tz_convert(tz)
    )

def get_timezone(alpha2code):

    """
      getting timezone of country in Europe
    """

    timezonemap = {

        'AT': 'Europe/Berlin', 'BE': 'Europe/Berlin', 'BG': 'Europe/Sofia', 'BA' : 'Europe/Sarajevo', 'CH': 'Europe/Zurich', 'CY': 'Europe/Sofia', 
        
        'CZ': 'Europe/Sofia', 'DE': 'Europe/Berlin',

        'DK': 'Europe/Berlin', 'EE': 'Europe/Sofia', 'GB': 'Europe/London', 'GR': 'Europe/Sofia', 'ES': 'Europe/Berlin', 'FI': 'Europe/Sofia', 
        
        'FR': 'Europe/Berlin', 'HR': 'Europe/Berlin', 'HU': 'Europe/Berlin', 'IE': 'Europe/London', 'IS': 'Atlantic/Reykjavik', 'IT': 'Europe/Berlin', 'LT': 'Europe/Sofia', 
        
        'LU': 'Europe/Berlin', 'LV': 'Europe/Sofia', 'ME': 'Europe/Podgorica', 'MK': 'Europe/Skopje', 'MT': 'Europe/Sofia', 'NL': 'Europe/Berlin', 'NO': 'Europe/Oslo', 'PL': 'Europe/Berlin', 'PT': 'Europe/London', 
        
        'RO': 'Europe/Berlin', 'RS': 'Europe/Belgrade', 'SE': 'Europe/Berlin', 'SI': 'Europe/Berlin', 'SK': 'Europe/Berlin', 'UK': 'Europe/London'

    }


    return timezonemap.get(alpha2code)



# Load profiles
def load_power_load_profile(profile: str) -> pd.DataFrame:
    """
    Retuns the power load profiles for the given profile.
    DISS: "4.2.5.2 Standardlastprofile" -> Tabelle A.9
    """

    raw_file = f"C:/Users/kwijeyasekera/Documents/IEE-Kisal/Masterarbeit - Kisal/data/Standard Household profiles/power_load_profiles/39_VDEW_Strom_Repräsentative_Profile_{profile}.xlsx"
    load_profiles = pd.read_excel(raw_file)

    return load_profiles

# =========================================================
# Helper
# =========================================================

def Leistung(Tag_Zeit, mask, df, df_SLP):
    u = pd.merge(df[mask], df_SLP[["Hour", Tag_Zeit]], on=["Hour"], how="left")
    v = pd.merge(df, u[["Date", Tag_Zeit]], on=["Date"], how="left")
    v = v.infer_objects(copy=False).fillna(0.0)
    return v[Tag_Zeit]


# =========================================================
# Standard profile builder
# =========================================================

def build_standard_profile(state: str,
    year: int,
    profile: str,
    resolution: str = "15min") -> pd.Series:

    profile = profile.upper()

    tz = get_timezone("DE")
    idx = make_year_index(year, resolution, tz)

    df = pd.DataFrame({"Date": idx})
    df["Day"] = pd.DatetimeIndex(df["Date"]).date
    df["Hour"] = pd.DatetimeIndex(df["Date"]).time
    df["DayOfYear"] = pd.DatetimeIndex(df["Date"]).dayofyear.astype(int)

    # -------------------------
    # Holidays
    # -------------------------
    de_holidays = holidays.DE(state=state, years=year)
    hd = df["Day"].isin(list(de_holidays.keys()))

    df["WD"] = df["Date"].apply(lambda x: x.weekday() < 5) & (~hd)
    df["SA"] = df["Date"].apply(lambda x: x.weekday() == 5) & (~hd)
    df["SU"] = df["Date"].apply(lambda x: x.weekday() == 6) | hd

    # Special days
    mask_special = df["Day"].isin([
        datetime.date(year, 12, 24),
        datetime.date(year, 12, 31)
    ])
    df.loc[mask_special, ["WD", "SU"]] = False
    df.loc[mask_special, "SA"] = True

    # -------------------------
    # Seasons
    # -------------------------
    wiz1 = df[df["Date"] < f"{year}-03-21"]
    wiz2 = df[df["Date"] >= f"{year}-11-01"]

    soz = df[(df["Date"] >= f"{year}-05-15") & (df["Date"] < f"{year}-09-15")]
    uez1 = df[(df["Date"] >= f"{year}-03-21") & (df["Date"] < f"{year}-05-15")]
    uez2 = df[(df["Date"] >= f"{year}-09-15") & (df["Date"] <= f"{year}-10-31")]

    df["WIZ"] = df["Day"].isin(wiz1["Day"]) | df["Day"].isin(wiz2["Day"])
    df["SOZ"] = df["Day"].isin(soz["Day"])
    df["UEZ"] = df["Day"].isin(uez1["Day"]) | df["Day"].isin(uez2["Day"])

    # -------------------------
    # Load SLP
    # -------------------------
    df_load = load_power_load_profile(profile)

    df_load.columns = [
        "Hour",
        "SA_WIZ", "SU_WIZ", "WD_WIZ",
        "SA_SOZ", "SU_SOZ", "WD_SOZ",
        "SA_UEZ", "SU_UEZ", "WD_UEZ",
    ]

    df_SLP = df_load.reset_index()

    # -------------------------
    # Combine profiles
    # -------------------------
    Summe = (
        Leistung("WD_WIZ", (df.WD & df.WIZ), df, df_SLP) +
        Leistung("WD_SOZ", (df.WD & df.SOZ), df, df_SLP) +
        Leistung("WD_UEZ", (df.WD & df.UEZ), df, df_SLP) +
        Leistung("SA_WIZ", (df.SA & df.WIZ), df, df_SLP) +
        Leistung("SA_SOZ", (df.SA & df.SOZ), df, df_SLP) +
        Leistung("SA_UEZ", (df.SA & df.UEZ), df, df_SLP) +
        Leistung("SU_WIZ", (df.SU & df.WIZ), df, df_SLP) +
        Leistung("SU_SOZ", (df.SU & df.SOZ), df, df_SLP) +
        Leistung("SU_UEZ", (df.SU & df.UEZ), df, df_SLP)
    )

    # -------------------------
    # Seasonal correction (H0 only)
    # -------------------------
    if profile == "H0":
        dofy = df["DayOfYear"].astype(float)
        Ft = (
            -3.92e-10 * dofy**4 +
            3.2e-7 * dofy**3 -
            7.02e-5 * dofy**2 +
            2.1e-3 * dofy +
            1.24
        )
        values = Summe * Ft
    else:
        values = Summe

    # -------------------------
    # Normalize
    # -------------------------
    values = values.rename(profile)

    values.index = pd.date_range(
        start=str(year),
        end=str(year + 1),
        freq=resolution
    )[:-1]

    # Keep profile normalized
    values = values / values.sum()

    return values


# =========================================================
# Scaling
# =========================================================

def scale_profile_to_yearly(
    std_profile: pd.Series,
    yearly_kwh: float,
    output_unit: str,
    resolution: str
):

    curve = std_profile * yearly_kwh

    if output_unit.lower() == "kw":

        timestep_hours = (
            pd.Timedelta(resolution).total_seconds() / 3600
        )

        return curve / timestep_hours

    return curve


# =========================================================
# GPKG processing
# =========================================================

def generate_curves_from_gpkg(
    gpkg_path,
    layer,
    id_col,
    annual_col,
    std_profile,
    output_unit
):

    gdf = gpd.read_file(gpkg_path, layer=layer)

    if id_col not in gdf.columns:
        gdf = gdf.reset_index(drop=True)
        gdf[id_col] = gdf.index

    df_in = gdf[[id_col, annual_col]].dropna()

    curves = {}

    for _, r in df_in.iterrows():
        hid = str(r[id_col])
        yearly = float(r[annual_col])

        if yearly <= 0:
            continue

        curves[hid] = scale_profile_to_yearly(std_profile, yearly, output_unit, "30min")

    return pd.DataFrame(curves, index=std_profile.index)


# =========================================================
# CLI
# =========================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", required=True)
    p.add_argument("--annual-col", required=True)
    p.add_argument("--id-col", required=True)
    p.add_argument(
    "--resolution",
    default="15min",
    choices=["15min", "30min", "1h"],
    help="Output time resolution")
    p.add_argument("--gpkg", required=True)
    p.add_argument("--layer", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    state = "HE"
    year = 2024

    print(f"[INFO] Building profile {args.profile}")

    std = build_standard_profile(
        state=state,
        year=year,
        profile=args.profile,
        resolution=args.resolution
    )

    print("[INFO] Generating curves...")

    curves = generate_curves_from_gpkg(
        args.gpkg,
        args.layer,
        args.id_col,
        args.annual_col,
        std,
        "kW"
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    curves.to_parquet(args.out)

    print(f"[DONE] Saved: {args.out}")


if __name__ == "__main__":
    main()

    """
===========================================================
HOUSEHOLD LOAD PROFILE GENERATION (SLP CURVES)
===========================================================

This script generates normalized household load profiles
based on standard load profiles (H0, G0–G6, L0–L2)
and scales them to annual consumption values from a GeoPackage.

-----------------------------------------------------------
HOW TO RUN (Windows PowerShell / CLI)
-----------------------------------------------------------

Using Python launcher (recommended on Windows):

py household_profiles.py ^
  --profile H0 ^
  --annual-col W_H0 ^
  --id-col AN_FID ^
  --gpkg "C:\\path\\to\\Kisky.gpkg" ^
  --layer Kundenstruktur_NS_Anschluesse ^
  --out "C:\\path\\to\\curves.parquet"

OR using python:

python household_profiles.py \
  --profile H0 \
  --annual-col W_H0 \
  --id-col AN_FID \
  --gpkg /path/to/Kisky.gpkg \
  --layer Kundenstruktur_NS_Anschluesse \
  --out /path/to/curves.parquet

-----------------------------------------------------------
INPUTS
-----------------------------------------------------------
--profile       Standard load profile type (H0, G0-G6, L0-L2)
--annual-col    Column containing annual energy demand (kWh)
--id-col        Unique household identifier
--gpkg          Input GeoPackage file
--layer         Layer name inside GeoPackage
--out           Output parquet file path

-----------------------------------------------------------
OUTPUT
-----------------------------------------------------------
- Parquet file with time series columns per household ID
- Index: 15-minute resolution for full year (UTC-naive)

===========================================================
"""