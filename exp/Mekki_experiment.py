"""
Experiment summary
------------------
Compare the recovery/death ratios for each province/country
Use this to then predict the chance of recovering for a given location (latitude, longitude)
"""

import sys
sys.path.insert(0, '..')

from utils import data
from src import distances, k_nearest_neighbor
import os
import sklearn
import numpy as np
from sklearn.neighbors import (
    KNeighborsClassifier,
    DistanceMetric
)
import json
import matplotlib.pyplot as plt
import seaborn as sns

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

total_cases = []
for i in range(0, N):
    j = 0
    while True:
        if j == len(confirmed["Country/Region"]):
            total_cases.append('N/A')
            break

        if isinstance(confirmed["Province/State"][j], float):
            the_name = confirmed["Country/Region"][j]
        else:
            the_name = confirmed["Country/Region"][j] + ' ' + confirmed["Country/Region"][j]

        if features[i][0] == the_name:
            total_cases.append(confirmed["5/19/2020"][j])
            break
        else:
            j += 1

total_deaths = []
for i in range(0, N):
    j = 0
    while True:
        if j == len(deaths["Country/Region"]):
            total_deaths.append('N/A')
            break

        if isinstance(deaths["Province/State"][j], float):
            the_name = deaths["Country/Region"][j]
        else:
            the_name = deaths["Country/Region"][j] + ' ' + deaths["Country/Region"][j]

        if features[i][0] == the_name:
            total_deaths.append(deaths["5/19/2020"][j])
            break
        else:
            j += 1

total_recoveries = []
for i in range(0, N):
    j = 0
    while True:
        if j == len(recoveries["Country/Region"]):
            total_recoveries.append('N/A')
            break

        if isinstance(recoveries["Province/State"][j], float):
            the_name = recoveries["Country/Region"][j]
        else:

            the_name = recoveries["Country/Region"][j] + ' ' + recoveries["Country/Region"][j]
        if features[i][0] == the_name:
            total_recoveries.append(recoveries["5/19/2020"][j])
            break
        else:
            j += 1

death_rates = []
for i in range(0, N):
    if total_deaths[i] == 'N/A' or total_cases[i] == 'N/A':
        death_rates.append('N/A')
    elif total_cases[i] == 0:
        death_rates.append('N/A')
    elif total_deaths[i] == 0:
        death_rates.append(0)
    else:
        death_rates.append(total_deaths[i] / total_cases[i])

recovery_rates = []
for i in range(0, N):
    if total_recoveries[i] == 'N/A' or total_cases[i] == 'N/A':
        recovery_rates.append('N/A')
    elif total_cases[i] == 0:
        recovery_rates.append('N/A')
    elif total_recoveries[i] == 0:
        recovery_rates.append(0)
    else:
        recovery_rates.append(total_recoveries[i] / total_cases[i])

survival_rates = []
for i in range(0, N):
    if total_deaths[i] == 'N/A' or total_recoveries[i] == 'N/A':
        survival_rates.append('N/A')
    elif total_deaths[i] == 0 and total_recoveries[i] == 0:
        survival_rates.append(1)
    elif total_deaths[i] == 0:
        survival_rates.append(1000)
    elif total_recoveries[i] == 0:
        survival_rates.append(0)
    else:
        survival_rates.append(total_recoveries[i] / total_deaths[i])

#input_latitude = 30
#input_longitude = -160

#the_model = k_nearest_neighbor.KNearestNeighbor(N_NEIGHBORS, distance_measure='manhattan', aggregator='mean')

#the_model.fit(features, death_rates)

#test_features = np.array([[input_latitude, input_longitude]])
#the_target_values, the_targets = the_model.predict(test_features)

#print("\n")
#print("Closest Locations: ")
#for target in the_targets:
    #print(features[target][0])
#print("\n")
#print("Death Rate: ", the_target_values[0])

#the_model.fit(features, recovery_rates)

#test_features = np.array([[input_latitude, input_longitude]])
#the_target_values, the_targets = the_model.predict(test_features)

#print("\n")
#print("Closest Locations: ")
#for target in the_targets:
    #print(features[target][0])
#print("\n")
#print("Recovery Rate: ", the_target_values[0])

#the_model.fit(features, survival_rates)

#test_features = np.array([[input_latitude, input_longitude]])
#the_target_values, the_targets = the_model.predict(test_features)

#print("\n")
#print("Closest Locations: ")
#for target in the_targets:
    #print(features[target][0])
#print("\n")
#print("Survival Rate: ", the_target_values[0])

the_model_d = k_nearest_neighbor.KNearestNeighbor(N_NEIGHBORS, distance_measure='manhattan', aggregator='mean')
the_model_r = k_nearest_neighbor.KNearestNeighbor(N_NEIGHBORS, distance_measure='manhattan', aggregator='mean')
the_model_s = k_nearest_neighbor.KNearestNeighbor(N_NEIGHBORS, distance_measure='manhattan', aggregator='mean')
the_model_d.fit(features, death_rates)
the_model_r.fit(features, recovery_rates)
the_model_s.fit(features, survival_rates)
death_values = np.zeros((13, 19))
recovery_values = np.zeros((13, 19))
survival_values = np.zeros((13, 19))

for lat in range(0, 13):
    for lon in range(0, 19):
        test_feature = np.array([[(lat - 6) * 15, (lon - 9) * 20]])

        value_d, d = the_model_d.predict(test_feature)
        death_values[lat][lon] = value_d

        value_r, r = the_model_r.predict(test_feature)
        recovery_values[lat][lon] = value_r

        value_s, s = the_model_s.predict(test_feature)
        survival_values[lat][lon] = value_s

#print("Death Values: ", death_values)
#print("Recovery Values: ", recovery_values)
#print("Survival Values: ", survival_values)
plt.figure(figsize=(20,15))
sns_plot = sns.heatmap(death_values)
sns_plot.figure.savefig('cv_death_values')
plt.figure(figsize=(20,15))
sns_plot = sns.heatmap(recovery_values)
sns_plot.figure.savefig('cv_recovery_values')
plt.figure(figsize=(20,15))
sns_plot = sns.heatmap(survival_values)
sns_plot.figure.savefig('cv_survival_values')








