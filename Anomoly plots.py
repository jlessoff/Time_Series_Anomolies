#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pip install --upgrade --user matrixprofile

import pandas as pd
import math
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
import ruptures as rpt
from statsmodels import robust
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import matrixprofile as mp
import datetime
from sklearn import cluster
from sklearn import preprocessing
from sklearn import ensemble
from dtaidistance import dtw
from sklearn import manifold


df = pd.read_csv('TOTALSA.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.set_index('DATE').sort_index()

df.head()
df.describe()

df.plot(figsize=(20,7), legend=None, \
        title='Scheduled passengers (in thousands)')
plt.xlabel('Pickup Datetime')
plt.ylabel('Passengers')
plt.show()

# Point anomalies
sb.histplot(df, stat="density", bins=100)
plt.show()

sb.distplot(df, bins=10, kde=True, rug=True)
plt.show()

# Only the smallest 3 values are considered as anomalies
plt.figure(figsize=(20,10))
plt.plot(df,'k',label='Data')
plt.plot(df.nsmallest(3, 'ENPLANED'),'ro',label='Anomalies')
plt.legend()
plt.show()

# Compute the emprical 5% quantile and take all smaller 
# values as anomalies
anomalies = df < df.quantile(0.05)
plt.figure(figsize=(20,10))
plt.plot(df,'k',label='Data')
plt.plot(df[anomalies],'ro',label='Anomalies')
plt.legend()
plt.show()


# Detect outliers using mean and standard deviation principle
rolling_windows = df.rolling(window = 12, center = True)
rolling_mean = rolling_windows.mean()
rolling_std = rolling_windows.std()

alpha = 1.96
upper = rolling_mean + alpha*rolling_std
lower = rolling_mean - alpha*rolling_std
anomalies = (df > upper) | (df < lower)

# Plot 
plt.figure(figsize=(20,10))
plt.plot(df,'k',label='Data')
plt.plot(upper, 'b-',label='Bands',alpha=0.3)
plt.plot(lower, 'b-',label='Bands',alpha=0.3)
plt.plot(df[anomalies],'ro',label='Anomalies')
plt.fill_between(upper, lower, facecolor='blue',alpha=0.2)
plt.legend()
plt.show()

# Detect outliers using median and median absolute deviation
rolling_median = rolling_windows.median()
rolling_mad = rolling_windows.apply(robust.mad)

upper = rolling_median + alpha*rolling_mad
lower = rolling_median - alpha*rolling_mad
anomalies = (df > upper) | (df < lower)

# Plot 
plt.figure(figsize=(20,10))
plt.plot(df,'k',label='Data')
plt.plot(upper, 'b-',label='Bands',alpha=0.3)
plt.plot(lower, 'b-',label='Bands',alpha=0.3)
plt.plot(df[anomalies],'ro',label='Anomalies')
plt.fill_between(upper, lower, facecolor='blue',alpha=0.2)
plt.legend()
plt.show()

# Time series segmentation + anomaly detection in each segment
res = rpt.Pelt(model="normal", min_size=5, jump=1).fit(df)
my_bkps = res.predict(pen=2*math.log(len(df)))
rpt.show.display(df, my_bkps, figsize=(10, 6))
# Exercice : detect the potential anomalies in each segment
# Are there any segments where anomaly detection is useless? Why?

##### 
# Anomaly detection using residuals after 
# trend and seasonality fit
# trend is fitted using a moving average of order 12
# seasonality is fitted by supposing the no. of periods = 12

model = seasonal_decompose(df, model="additive", period=12)
model.plot()

plt.figure(figsize=(20,10))
plt.plot(df,'k',label='Data', color="tab:blue")
plt.plot(model.seasonal+model.trend, 'b-',label='Predicted',
         color="tab:orange")
plt.legend()
plt.show()

plt.figure(figsize=(20,10))
plt.plot(model.resid,'k',label='Residuals', color="tab:blue")
plt.show()


#####
# Anomaly detection using the residuals of an AR(12) model 
# fit model
model = ARIMA(df, order=(12,0,0))
model_fit = model.fit()
print(model_fit.summary())

model_fit.plot_predict(dynamic=False)
plt.show()

residuals = pd.DataFrame(model_fit.resid)
residuals.plot()

# Point anomalies
sb.histplot(residuals, stat="density", bins=100)
plt.show()

sb.distplot(df, bins=10, kde=True, rug=True)
plt.show()

# Detect outliers using mean and standard deviation principle
rolling_windows = residuals.rolling(window = 12, center = True)
rolling_mean = rolling_windows.mean()
rolling_std = rolling_windows.std()

alpha = 1.96
upper = rolling_mean + alpha*rolling_std
lower = rolling_mean - alpha*rolling_std
anomalies = (residuals > upper) | (residuals < lower)

# Plot 
plt.figure(figsize=(20,10))
plt.plot(residuals,'k',label='Data')
plt.plot(upper, 'b-',label='Bands',alpha=0.3)
plt.plot(lower, 'b-',label='Bands',alpha=0.3)
plt.plot(residuals[anomalies],'ro',label='Anomalies')
plt.legend()
plt.show()

# Detect outliers using median and median absolute deviation
rolling_median = rolling_windows.median()
rolling_mad = rolling_windows.apply(robust.mad)

alpha = 1.5
upper = rolling_median + alpha*rolling_mad
lower = rolling_median - alpha*rolling_mad
anomalies = (residuals > upper) | (residuals < lower)

# Plot 
plt.figure(figsize=(20,10))
plt.plot(residuals,'k',label='Data')
plt.plot(upper, 'b-',label='Bands',alpha=0.3)
plt.plot(lower, 'b-',label='Bands',alpha=0.3)
plt.plot(residuals[anomalies],'ro',label='Anomalies')
plt.legend()
plt.show()

#################################################
# The issue of detecting anomalous patterns 

# Matrix profiles method

# Discords of length 6
profile = mp.compute(df['ENPLANED'].values, windows = 6)
mp.visualize(profile)
profile = mp.discover.discords(profile, k=5)
figures = mp.visualize(profile)

fig, ax = plt.subplots(1, 1, sharex=True, figsize=(20,7))
ax.plot(np.arange(len(profile['data']['ts'])), profile['data']['ts'])
ax.set_title('Raw Data', size=22)
for discord in profile['discords']:
    x = np.arange(discord, discord + profile['w'])
    y = profile['data']['ts'][discord:discord + profile['w']]
    ax.plot(x, y, c='r')
plt.show()

# Discords of length 12

profile = mp.compute(df['ENPLANED'].values, windows = 12)
mp.visualize(profile)
profile = mp.discover.discords(profile, k=5)
figures = mp.visualize(profile)

fig, ax = plt.subplots(1, 1, sharex=True, figsize=(20,7))
ax.plot(np.arange(len(profile['data']['ts'])), profile['data']['ts'])
ax.set_title('Raw Data', size=22)
for discord in profile['discords']:
    x = np.arange(discord, discord + profile['w'])
    y = profile['data']['ts'][discord:discord + profile['w']]
    ax.plot(x, y, c='r')
plt.show()

## Exercice : use different methods to detect anomalies in the series
## https://fred.stlouisfed.org/series/TOTALSA 


# Compute anomalies for multivariate time series 
cars = pd.read_csv('TOTALSA.csv')
cars['DATE'] = pd.to_datetime(cars['DATE'])
cars = cars.set_index('DATE').sort_index()

df['TOTALSA'] = cars['TOTALSA']
df_scaled = preprocessing.StandardScaler().fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=['Enplaned', 'Cars'])
df_scaled['DATE'] = cars.index.get_level_values(0)
df_scaled = df_scaled.set_index('DATE').sort_index()

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(20,7))
ax[0].plot(df_scaled['Enplaned'], 'tab:blue')
ax[0].set_title("Air Passengers")
ax[1].plot(df_scaled['Cars'], 'tab:green')
ax[1].set_title("Car sales")
plt.show()

plt.plot(cars,'k',label='Data')
plt.plot(cars[anomalies],'ro',label='Anomalies')

# Isolation forests

df_if = ensemble.IsolationForest(max_samples=len(df), random_state=0)
df_if.fit(df_scaled)
df_if.score_samples(df_scaled)
df_if.decision_function(df_scaled)
df_if.predict(df_scaled)

anomalies = df_if.decision_function(df_scaled) < 0
# Plot the outliers
plt.figure(figsize=(20,10))
plt.plot(cars,'k',label='Car sales')
plt.plot(cars[anomalies],'ro',label='Anomalies')
plt.legend()
plt.show()

plt.figure(figsize=(20,10))
plt.plot(df['ENPLANED'],'k',label='Air passengers')
plt.plot(df['ENPLANED'][anomalies],'ro',label='Anomalies')
plt.legend()
plt.show()

# Histogram of the normality score
sb.histplot(df_if.score_samples(df_scaled), stat="density", bins=100)
plt.show()

anomalies = df_if.score_samples(df_scaled) < -0.6

plt.figure(figsize=(20,10))
plt.plot(cars,'k',label='Car sales')
plt.plot(cars[anomalies],'ro',label='Anomalies')
plt.legend()
plt.show()

plt.figure(figsize=(20,10))
plt.plot(df['ENPLANED'],'k',label='Air passengers')
plt.plot(df['ENPLANED'][anomalies],'ro',label='Anomalies')
plt.legend()
plt.show()

# Compute anomalies in a multivariate setting
# Use the electricity profiles

# Import data and compute daily profiles with hourly consumption
data_path =  'household_power_consumption.txt'
cols_to_use = ['Date', 'Time','Global_active_power']
data_cons = pd.read_csv(data_path, sep=';', usecols=cols_to_use)
data_cons['Date'] = data_cons['Date'].apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y').date())
data_cons['Time'] = data_cons['Time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').time())
data_cons['Hour'] = data_cons['Time'].apply(lambda x: x.hour)
data_cons = data_cons.replace('?', np.nan)
data_cons = data_cons.dropna()
data_cons['Global_active_power'] = pd.to_numeric(data_cons['Global_active_power'])
data_cons_hourly = data_cons.groupby(['Date', 'Hour'], as_index=False)['Global_active_power'].sum()
data_cons_hourly = data_cons_hourly.set_index('Date')
data_cons_pivot = data_cons_hourly.pivot(columns='Hour')
data_cons_pivot = data_cons_pivot.dropna()
data_cons_scaled = preprocessing.StandardScaler().fit_transform(data_cons_pivot.T)
data_cons_scaled = pd.DataFrame(data_cons_scaled.T)
data_cons_scaled = data_cons_scaled.set_index(data_cons_pivot.index.get_level_values(0))

data_cons_scaled.T.plot(figsize=(13,8), legend=False, color='blue', alpha=0.02)


dtw_dissim_mat = dtw.distance_matrix_fast(np.matrix(data_cons_scaled))

from sklearn.neighbors import NearestNeighbors
neighbors = NearestNeighbors(n_neighbors=24, metric="precomputed")
neighbors_fit = neighbors.fit(dtw_dissim_mat)
distances, indices = neighbors_fit.kneighbors(dtw_dissim_mat)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)

electric_profiles_dbscan = cluster.DBSCAN(eps=2, metric="precomputed", 
                        min_samples=5).fit(dtw_dissim_mat)

# plot the clusters on t-SNE
# Mapping with dtw distances
tsne_map = manifold.TSNE(n_components=2, metric="precomputed",
                         perplexity=30, angle=0.5)
results_tsne = tsne_map.fit_transform(dtw_dissim_mat)

plt.scatter(results_tsne[:,0], results_tsne[:,1],
    c=electric_profiles_dbscan.labels_,
    cmap="Set2",  alpha=0.6)

plt.close()

pd.value_counts(electric_profiles_dbscan.labels_)
# Noisy samples are having label -1
profiles_outliers = data_cons_pivot.loc[electric_profiles_dbscan.labels_==-1]
profiles_outliers.T.plot(figsize=(13,8), legend=False, 
                        color='blue', alpha=0.1)
plt.close()

# Cross with additional information
pd.to_datetime(profiles_outliers.index.get_level_values(0)).dayofweek.value_counts()
pd.to_datetime(profiles_outliers.index.get_level_values(0)).month.value_counts()

pd.crosstab(pd.to_datetime(profiles_outliers.index.get_level_values(0)).dayofweek,
pd.to_datetime(profiles_outliers.index.get_level_values(0)).month)


