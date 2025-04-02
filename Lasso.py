#lasso rigid regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV
import pickle
import joblib
import statsmodels.api as sm


dataset1 = pd.read_csv('50_Startups.csv')
dataset1.describe()
dataset1.info()

X= pd.DataFrame(dataset1.iloc[:,0:4])
Y = pd.DataFrame(dataset1.iloc[:,4])


numerical_features= X.select_dtypes(exclude=['object']).columns
categorical_features = X.select_dtypes(include = ['object']).columns

scaler_pipeline= Pipeline([('scaler', MinMaxScaler())])
transformer1= ColumnTransformer([('scaler', scaler_pipeline, numerical_features)])
scale_transform = transformer1.fit(X)
joblib.dump(scale_transform, 'scale_transform')
scaled_transform= pd.DataFrame(scale_transform.transform(X), columns= numerical_features)

Encoding= Pipeline([('encoding', OneHotEncoder())])
transformer2 =  ColumnTransformer([('encoding', Encoding, categorical_features)])
Onehot = transformer2.fit(X)
joblib.dump(Onehot, 'onehot')
Onehot_encoding = pd.DataFrame(Onehot.transform(X))

Onehot_encoding.columns = Onehot.get_feature_names_out(input_features = X.columns)

data = pd.concat([scaled_transform, Onehot_encoding], axis=1)

constant = sm.add_constant(data)
basemodel= sm.OLS(Y,constant).fit()
basemodel.summary()
constant.shape

vif = pd.Series([variance_inflation_factor(constant.values, i) for i in range(constant.shape[1])], index= constant.columns)
vif
sm.graphics.influence_plot(basemodel)

data_new= constant.drop(data.index[[48,49]])
Y_new = Y.drop(Y.index[[48,49]])


basemodel= sm.OLS(Y_new,data_new).fit()
basemodel.summary()

parameters = {'alpha': [1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.13, 0.2, 1, 5, 10, 20,50,1000]}
lasso = Lasso()
lass= GridSearchCV((lasso), parameters, scoring= 'r2', cv=5)
lass.fit(data, Y)
lass.best_params_
lass.best_score_
