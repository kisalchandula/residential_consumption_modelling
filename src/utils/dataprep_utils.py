import pandas as pd
import numpy as np

#assumes utc
def getSummerStart(year):
    dt = pd.Timestamp(year=year, month=4, day=1) 
    dt += pd.Timedelta(days=6-dt.dayofweek)
    dt -= pd.Timedelta(days=7)
    dt += pd.Timedelta(hours=2)
    return dt

#assumes utc
def getSummerEnd(year):
    dt = pd.Timestamp(year=year, month=11, day=1) 
    dt += pd.Timedelta(days=6-dt.dayofweek)
    dt -= pd.Timedelta(days=7)
    dt += pd.Timedelta(hours=2)
    return dt

def getSummertimes(dates:pd.DatetimeIndex):
    summerStarts = {}
    summerEnds = {}
    years = np.unique(dates.year)
    for year in years:
        summerStarts[f"{year}"] = getSummerStart(year)
        summerEnds[f"{year}"] = getSummerEnd(year)
    summertime = np.zeros(len(dates))
    for idx, date in enumerate(dates):
        summertime[idx] = 1 if (date > summerStarts[f"{date.year}"] and date < summerEnds[f"{date.year}"]) else 0
    return summertime

def getDayPeriodicEmbedding(dates:pd.DatetimeIndex):
    day_sin = np.sin((dates.hour + dates.minute / 60) * (2. * np.pi/24))
    day_cos = np.cos((dates.hour + dates.minute / 60) * (2. * np.pi/24))
    return np.stack((day_sin, day_cos), axis=-1)

def getWeekPeriodicEmbedding(dates:pd.DatetimeIndex):
    week_sin = np.sin(dates.dayofweek * (2 * np.pi / 7))
    week_cos = np.cos(dates.dayofweek * (2 * np.pi / 7))
    return np.stack((week_sin, week_cos), axis=-1)

def getYearPeriodicEmbedding(dates:pd.DatetimeIndex):
    daysInYear = np.array([365 + (1 if date.is_leap_year else 0 ) for date in dates])
    year_sin = np.sin(dates.dayofyear * (2 * np.pi / daysInYear))
    year_cos = np.cos(dates.dayofyear * (2 * np.pi / daysInYear))
    return np.stack((year_sin, year_cos), axis=-1)