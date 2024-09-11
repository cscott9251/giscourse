import xgboost as xgb
from xgboost import callback
from sklearn.svm import SVR, SVC
from sklearn.ensemble import GradientBoostingRegressor
#print(xgboost.__version__)
from sklearn.model_selection import train_test_split, cross_val_score, PredefinedSplit, RepeatedKFold, KFold
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score, d2_absolute_error_score, mean_absolute_error, mean_squared_error

from sklearn.model_selection import GridSearchCV

import sys 
import multiprocessing

import numpy as np
import pandas as pd
import neptune
from datasetup_wsl import CSVFile
import joblib

X = joblib.load("X_full.sav")
y = joblib.load("y_full.sav")

# X=X.to_numpy()
# y=y.to_numpy()
print(y.ravel())
print(y.shape)
print(y)

#y=y.ravel()

X_train = joblib.load("X_train.sav")
X_test = joblib.load("X_test.sav")
y_train = joblib.load("y_train.sav")
y_test = joblib.load("y_test.sav")
X_val = joblib.load("X_val.sav")
y_val = joblib.load("y_val.sav")

es = callback.EarlyStopping(
    rounds=3,
    min_delta=1e-3,
    save_best=True,
    maximize=False,
    data_name=X_val,        
    metric_name="rmse",
)

model_XGB = xgb.XGBRegressor(
    
    learning_rate=0.01,
    n_estimators=1000, 
    max_depth=7, 
    eta=0.1, 
    subsample=1, 
    colsample_bytree=1, 
    n_jobs=2,
    tree_method="hist",
    device="cuda", 
    callbacks=[es]
    
    )

parameters = {
    
    'learning_rate': [0.1,0.01,0.05],
    'subsample'    : [0.9, 0.75, 0.5],
    'n_estimators' : [500,1000,1500],
    'max_depth'    : [4,6,8],
    'tree_method'  : ["hist","approx"]
                 
}

# xgbclf = GridSearchCV(
#         model_XGB,
#         param_grid=parameters,
#         verbose=1,
#         n_jobs=2,
#     )

# xgbclf.fit(X_train,y_train)

# print(xgbclf.best_score_)
# print(xgbclf.best_params_)



# model_GBR = GradientBoostingRegressor(learning_rate=0.025)

# grid_GBR = GridSearchCV(estimator=model_GBR, param_grid = parameters, n_jobs=2)
# grid_GBR.fit(X_train, y_train)



# model_XGB.fit(X_train,y_train)
# print(model_XGB.score(X_train,y_train))


# grid_GBR = GridSearchCV(estimator=model_GBR, param_grid = parameters, n_jobs=2)
# grid_GBR.fit(X_train, y_train)

# print(" Results from Grid Search " )
# print("\n The best estimator across ALL searched params:\n",grid_GBR.best_estimator_)
# print("\n The best score across ALL searched params:\n",grid_GBR.best_score_)
# print("\n The best parameters across ALL searched params:\n",grid_GBR.best_params_)

# sys.exit()

NUM_TRIALS = 10
scores = []

for i in range(NUM_TRIALS):
    cv = KFold(n_splits=5, shuffle=True, random_state=i)
    clf = GridSearchCV(estimator=model_XGB, scoring=["r2","neg_mean_squared_error"], param_grid=parameters, cv=cv, refit="r2")
   
    #scores.append(clf.best_score_)
clf.fit(X_train,y_train)  

print(clf.best_params_)
print(clf.best_score_)

scoresMAE = cross_val_score(model_XGB, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv.get_n_splits(X_train,y_train), n_jobs=2)
scoresMAE = np.absolute(scoresMAE)

scoresMSE= cross_val_score(model_XGB, X_train, y_train, scoring='neg_mean_squared_error', cv=cv.get_n_splits(X_train,y_train), n_jobs=2)
scoresMSE = np.absolute(scoresMSE)

  
    #scores.append(clf.best_score_)
#clf.fit(X_train,y_train)  

# print(clf.best_params_)
# print(clf.best_score_)

model_XGB.fit(X_train,y_train)
#y_pred = model_XGB.predict(X_test)

# print(mean_absolute_error(y_test, y_pred))
# print(mean_squared_error(y_test, y_pred))
# print(explained_variance_score(y_test, y_pred))
# print(r2_score(y_test,y_pred))

#sys.exit()


#cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)





print('MAE: %.3f (%.3f)' % (scoresMAE.mean(), scoresMAE.std()))
print('MSE: %.3f (%.3f)' % (scoresMSE.mean(), scoresMSE.std()))


print(model_XGB.score(X_train,y_train))

sys.exit()

scoresMAE = cross_val_score(model_XGB, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv.get_n_splits(X_train,y_train), n_jobs=2)
scoresMAE = np.absolute(scoresMAE)

scoresMSE= cross_val_score(model_XGB, X_train, y_train, scoring='neg_mean_squared_error', cv=cv.get_n_splits(X_train,y_train), n_jobs=2)
scoresMSE = np.absolute(scoresMSE)



print('MAE: %.3f (%.3f)' % (scoresMAE.mean(), scoresMAE.std()))
print('MSE: %.3f (%.3f)' % (scoresMSE.mean(), scoresMSE.std()))


print(model_XGB.score(X_train,y_train))

sys.exit()

r2 = r2_score(y_test,y_pred)
#accuracy = accuracy_score(y_test, y_pred)

print(r2)
#print(accuracy)




