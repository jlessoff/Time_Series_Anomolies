#predict quality variable using other variables
import pandas as pd
#import data
wines=pd.read_csv('~/Documents/GitHub/Data/winequality-red.csv',delimiter=';')
#explanatory variable
wines_X=wines[wines.columns[0:10]]
#target variable
wines_Y=wines['quality']
from sklearn.neighbors import KNeighborsRegressor
#when you create object in python, can specify metaparameters. not best way, but can for first example
knn_algo= KNeighborsRegressor(3)
#fitting with explanatory variables- target valuable
knn_algo.fit(wines_X,wines_Y)
#do prediction on same data set used to learn the model. result is array of predicted values
wines_pred=knn_algo.predict(wines_X)
#different metrics to compare prediction to true value
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
#getting r2 value, 0.3886 which is not that good
#as increase k, the root and square error become larger (ex, change k from 5 to 15), and r2 goes down.  if use 3, get a better value
#3 provides better values for square error and r2
print(r2_score(wines_Y,wines_pred))
print(mean_absolute_error(wines_Y,wines_pred))
#mean squared error: average value of the error
print(mean_squared_error(wines_Y,wines_pred))
print(mean_absolute_error(wines_Y,wines_pred))
import numpy as np

#how to choose k and are the results good?
wines_Y_noisy=wines_Y+0.1*np.random.random_sample(wines_Y.shape[0])-0.05
wines_res=pd.DataFrame({'true_quality':wines_Y_noisy,
                        'predicted_quality':wines_pred})
print(wines_res)
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data=wines_res,x='true_quality',y='predicted_quality',alpha=0.1)
plt.show()