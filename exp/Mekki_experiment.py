"""
Experiment summary
------------------
Compare the recovery/death ratios for each province/country
Use this to then predict the chance of recovering for a given location (latitude, longitude)
"""

import sys
sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
from sklearn.neighbors import (
    KNeighborsClassifier,
    DistanceMetric
)
import json

# ------------ HYPERPARAMETERS -------------
BASE_PATH = '../COVID-19/csse_covid_19_data/'
N_NEIGHBORS = 5
MIN_CASES = 1000
NORMALIZE = True
# ------------------------------------------

confirmed = os.path.join(
    BASE_PATH,
    'csse_covid_19_time_series',
    'time_series_covid19_confirmed_global.csv')
confirmed = data.load_csv_data(confirmed)

deaths = os.path.join(
    BASE_PATH,
    'csse_covid_19_time_series',
    'time_series_covid19_deaths_global.csv')
deaths = data.load_csv_data(deaths)

recoveries = os.path.join(
    BASE_PATH,
    'csse_covid_19_time_series',
    'time_series_covid19_recovered_global.csv')
recoveries = data.load_csv_data(recoveries)

features = []
targets = []
#print(len(confirmed["Country/Region"]))
#print(len(deaths["Country/Region"]))
#print(len(recoveries["Country/Region"]))
N = len(confirmed["Country/Region"])

for i in range(0, N):
    example = []
    if isinstance(confirmed["Province/State"][i], float):
        example.append(confirmed["Country/Region"][i])
    else:
        example.append(confirmed["Province/State"][i] + ' ' + confirmed["Country/Region"][i])
    example.append(confirmed["Lat"][i])
    example.append(confirmed["Long"][i])
    features.append(example)

death_rates = []
for i in range(0, N):
    j = 0
    while True:
        if j == len(deaths["Country/Region"]):
            death_rates.append('N/A')
            break

        if isinstance(deaths["Province/State"][j], float):
            the_name = deaths["Country/Region"][j]
        else:
            the_name = deaths["Country/Region"][j] + ' ' + deaths["Country/Region"][j]

        if features[i][0] == the_name:
            death_rates.append(deaths["5/19/2020"][j])
            break
        else:
            j += 1

recovery_rates = []
for i in range(0, N):
    j = 0
    while True:
        if j == len(recoveries["Country/Region"]):
            recovery_rates.append('N/A')
            break

        if isinstance(recoveries["Province/State"][j], float):
            the_name = recoveries["Country/Region"][j]
        else:

            the_name = recoveries["Country/Region"][j] + ' ' + recoveries["Country/Region"][j]
        if features[i][0] == the_name:
            recovery_rates.append(recoveries["5/19/2020"][j])
            break
        else:
            j += 1

survival_rates = []
for i in range(0, N):
    if death_rates[i] == 'N/A' or recovery_rates[i] == 'N/A':
        survival_rates.append('N/A')
    elif death_rates[i] == 0 and recovery_rates[i] == 0:
        survival_rates.append(1)
    elif death_rates[i] == 0:
        survival_rates.append(1000)
    elif recovery_rates[i] == 0:
        survival_rates.append(0)
    else:
        survival_rates.append(recovery_rates[i] / death_rates[i])

f = np.array(features)
t = np.array(survival_rates)

print(f)
print(t)







