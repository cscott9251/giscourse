import xgboost as xgb
#print(xgboost.__version__)
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold

import numpy as np

from datasetup_wsl import CSVFile
import joblib
#from sklearn.model_selection import train_test_split
model_XGB = xgb.XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=1, colsample_bytree=1)

X_train = joblib.load("X_train.sav")
X_test = joblib.load("X_test.sav")
y_train = joblib.load("y_train.sav")
y_test = joblib.load("y_test.sav")

# model_XGB.fit(X_train,y_train)
# print(model_XGB.score(X_train,y_train))

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model_XGB, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv.get_n_splits(X_train,y_train), n_jobs=-1)
scores = np.absolute(scores)

