# ------------------------ IMPORTS ------------------------

import pandas as pd
import numpy as np
import os.path
from sklearn.preprocessing import StandardScaler

# ------------------------ LOADING DATASET ------------------------
dataset_link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz'
dataset_name = "mitv.csv"

if not os.path.isfile(dataset_name):
    dataframe = pd.read_csv(dataset_link)
    dataframe.to_csv(dataset_name, index=False)
else:
    dataframe = pd.read_csv(dataset_name)


# ------------------------ UTILITY FUNCTION ------------------------

def print_section(title: str):
    print(f'------------------------ {title.upper()} ------------------------')


# ------------------------ SHOW INFO ------------------------
print_section('info')

"""
The data set includes hourly Minneapolis-St Paul traffic volume for westbound I-94 from 2012-2018. 
It contains 48204 instances with 9 attributes:

- holiday: categorical US national holidays plus regional holiday;
- temp: Numeric average temperature in kelvin;
- rain_1h: numeric amount in mm of rain that occurred in the hour;
- snow_1h: numeric amount in mm of snow that occurred in the hour;
- clouds_all: numeric percentage of cloud cover;
- weather_main: categorical short textual description of the current weather;
- weather_description: Categorical longer textual description of the current weather;
- date_time: date and hour of the data collected in local CST time;
- traffic_volume: Numeric hourly I-94 ATR 301 reported westbound traffic volume.
"""
print(dataframe.head())
print(dataframe.info())
print(dataframe.describe().transpose())

# ------------------------ PREPROCESSING ------------------------

print_section('preprocessing')

# drop u