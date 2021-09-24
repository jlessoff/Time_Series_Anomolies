import pandas as pd

#import data
wines=pd.read_csv('~/Documents/GitHub/Data/winequality-red.csv',delimiter=';')

#explanatory variable
wines_X=wines[wines.columns[0:10]]
#target variable
wines_Y=wines['quality']


from sklearn.neighbors import KNeighborsRegressor
knn_algo= KNeighborsRegressor(1)
knn_algo.fit(wines_X,wines_Y)
wines_pred=knn_algo.predict(wines_X)

from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
import numpy as np

wines_Y_noisy=wines_Y+0.1*np.random.random_sample(wines_Y.shape[0])-0.05
wines_res=pd.DataFrame({'true_quality':wines_Y_noisy,
                        'predicted_quality':wines_pred})
print(wines_res)
import seaborn as sns

sns.scatterplot(data=wines_res,x='true_quality',y='predicted_quality',alpha=0.1)


from sklearn.model_selection import train_test_split
w_X_tr, w_X_test, w_Y_tr, w_Y_test=train_test_split(wines_X,wines_Y,
                                                    train_size=0.7, random_state=42)

knn_algo= KNeighborsRegressor(15)
knn_algo.fit(w_X_tr,w_Y_tr)
wines_pred=knn_algo.predict(w_X_test)

wines_res=pd.DataFrame({'true_quality':w_Y_test,
                        'predicted_quality':wines_pred})

sns.scatterplot(data=wines_res,x='true_quality',y='predicted_quality',alpha=0.1)

#scale data to zero mean zero standard deviation for all variables
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(w_X_tr)

knn_algo= KNeighborsRegressor(1)
knn_algo.fit(scaler.transform(w_X_tr), w_Y_tr)
wines_pred=knn_algo.predict(scaler.transform(w_X_test))

print(r2_score(w_Y_test,wines_pred))
print(mean_squared_error(w_Y_test,wines_pred))
print(mean_absolute_error(w_Y_test,wines_pred))

