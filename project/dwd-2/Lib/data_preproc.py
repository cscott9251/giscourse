from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
from xgboost import callback

import pandas as pd

from datasetup_wsl import CSVFile
import sys
import joblib

df = pd.read_csv("data.csv")
print(df.head())

X_train = joblib.load("X_train.sav")
X_test = joblib.load("X_test.sav")
y_train = joblib.load("y_train.sav")
y_test = joblib.load("y_test.sav")
X_val = joblib.load("X_val.sav")
y_val = joblib.load("y_val.sav")

D_train = xgb.DMatrix(X_train, y_train)
D_valid = xgb.DMatrix(X_val, y_val)

early_stop = callback.EarlyStopping(rounds=10, metric_name='rmse')

xgbmodel = xgb.XGBRegressor(n_jobs=-1)


preprocessor = Pipeline([
    
    ('scaler', RobustScaler())
])



final_pipe = Pipeline([
    
    ('preproc', preprocessor),
    ('model_XGBoost', xgbmodel)
    
])



print(final_pipe.get_params())

#sys.exit()




params={
    
    'model_XGBoost__objective'    : ["reg:squarederror"],
    'model_XGBoost__device'       : ["cuda"],
    'model_XGBoost__learning_rate': [0.01,0.005],
    'model_XGBoost__subsample'    : [0.75, 0.5],
    'model_XGBoost__n_estimators' : [1000,1500],
    'model_XGBoost__max_depth'    : [6,8],
    'model_XGBoost__tree_method'  : ["hist"],
    'model_XGBoost__eval_metric'  : ["rmse"]
          
}


random_search = RandomizedSearchCV(
    
    final_pipe,
    param_distributions=params,
    cv=5,
    n_iter=50,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    random_state=42,
    verbose=3
    
)

#print(random_search.n_features_in_)

#random_search.fit(X_train,y_train)

#tuned_pipe = random_search.best_estimator_

random_search = joblib.load("random_searchXGB.joblib")

tuned_pipe = random_search.best_estimator_


print(random_search.best_score_)
print(random_search.best_params_)


scaler = RobustScaler()
scaler.fit(X_train,y_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)

X_val_scaled = scaler.transform(X_val)

y_pred = random_search.predict(X_test_scaled)

cvs = cross_val_score(tuned_pipe, X_val_scaled, y_val, scoring="r2", cv=20)

print(cvs)

#joblib.dump(random_search, "random_searchXGB.joblib")


# booster = xgb.train(
#     {'objective': 'reg:squarederror',
#      'eval_metric': ['rmse'],
#      'tree_method': 'hist'}, D_train,
#     evals=[(D_train, 'Train'), (D_valid, 'Valid')],
#     num_boost_round=1000,
#     callbacks=[early_stop],
#     verbose_eval=False)




