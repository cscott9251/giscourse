from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
from xgboost import callback

import pandas as pd

import sys
import joblib

X_train_reduced = joblib.load("Reduced_dataset/X_train_reduced.sav")
X_test_reduced = joblib.load("Reduced_dataset/X_test_reduced.sav")
y_train_reduced = joblib.load("Reduced_dataset/y_train_reduced.sav")
y_test_reduced = joblib.load("Reduced_dataset/y_test_reduced.sav")
X_val_reduced = joblib.load("Reduced_dataset/X_val_reduced.sav")
y_val_reduced = joblib.load("Reduced_dataset/y_val_reduced.sav")




D_train = xgb.DMatrix(X_train_reduced, y_train_reduced)
D_valid = xgb.DMatrix(X_val_reduced, y_val_reduced)

eval_set = [X_val_reduced, y_val_reduced]

early_stop = callback.EarlyStopping(rounds=25, metric_name='rmse',save_best=True)

xgbmodel = xgb.XGBRegressor(verbosity=3)
scaler = RobustScaler()




preprocessor = Pipeline([

    ('scaler', scaler)
])



final_pipe = Pipeline([

    ('preproc', preprocessor),
    ('model_XGBoost', xgbmodel)

])




#sys.exit()




params={

    'model_XGBoost__objective'    : ["reg:gamma"],
    #'model_XGBoost__device'       : ["cuda:0"],
    'model_XGBoost__learning_rate': [0.01,0.02],
    'model_XGBoost__subsample'    : [0.6, 0.5],
    'model_XGBoost__n_estimators' : [900,1000],
    'model_XGBoost__max_depth'    : [5,6],
    'model_XGBoost__tree_method'  : ["hist"],
    'model_XGBoost__eval_metric'  : ["gamma-deviance","rmse"],
    'model_XGBoost__booster'      : ["gbtree","gblinear"],
    #'model_XGBoost__early_stopping_rounds' : [35],
    'model_XGBoost__eval_metric' : ["gamma-deviance","rmse"]
    #'model_XGBoost__eval_set' : [[X_val_reduced, y_val_reduced]]
    #'model_XGBoost__callbacks'    : [early_stop],
    #'model_XGBoost__verbosity'    : [3],


}


fit_params={"early_stopping_rounds":35, 
            "eval_metric" : ["gamma-deviance","rmse"], 
            "eval_set" : [[X_val_reduced, y_val_reduced]]}

random_search = RandomizedSearchCV(

    final_pipe,
    param_distributions=params,
    cv=5,
    n_iter=50,
    #scoring="neg_mean_squared_error",
    n_jobs=-1,
    random_state=42,
    verbose=3

)




grid_search = GridSearchCV(

    final_pipe,
    param_grid=params,
    verbose=3,
    cv=5,
    n_jobs=-1

)

#print(random_search.n_features_in_)



# grid_search.fit(X_train_reduced,y_train_reduced)
# tuned_pipe_grid = grid_search.best_estimator_
# joblib.dump(grid_search, "grid_searchXGB.joblib")

#grid_search = joblib.load("grid_searchXGB.joblib")
#tuned_pipe_grid = grid_search.best_estimator_

#scaler.fit(X_train_reduced,y_train_reduced)


# y_pred = grid_search.predict(X_val_scaled)

# print(clf.best_score_)


def gridsearch_sq_err(X,y):
            
    params={

    'model_XGBoost__objective'    : ["reg:gamma"],
    #'model_XGBoost__device'       : ["cuda:0"],
    'model_XGBoost__learning_rate': [0.03,0.04,0.05],
    'model_XGBoost__subsample'    : [0.6, 0.75],
    'model_XGBoost__n_estimators' : [900,1000],
    'model_XGBoost__max_depth'    : [5,6],
    'model_XGBoost__tree_method'  : ["hist"],
    'model_XGBoost__eval_metric'  : ["gamma-deviance"]
    #'model_XGBoost__booster'      : ["gbtree"],
    #'model_XGBoost__early_stopping_rounds' : [50],
    #'model_XGBoost__eval_metric' : ["gamma-deviance","rmse"]
    #'model_XGBoost__eval_set' : [[X_val_reduced, y_val_reduced]]
    #'model_XGBoost__callbacks'    : [early_stop],
    #'model_XGBoost__verbosity'    : [3],


    }



    grid_search_sq_err = GridSearchCV(

    final_pipe,
    param_grid=params,
    verbose=3,
    cv=5,
    n_jobs=-1

    )
    

    grid_search_sq_err.fit(X,y)
    # tuned_pipe_grid_sq_err = clf.best_estimator_
    
    # X_val_scaled = tuned_pipe_grid_sq_err.transform(X_val_reduced)
    
    # y_pred = tuned_pipe_grid_sq_err.predict(X_val_scaled)
    


#clf_sq = gridsearch_sq_err(X_train_reduced,y_train_reduced)

#joblib.dump(clf_sq,"clf_xgboost_sqv3.joblib")

clf_sq = joblib.load("clf_xgboost_sq.joblib")

clf_sqv2 = joblib.load("clf_xgboost_sqv2.joblib")



y_pred = clf_sqv2.predict(X_test_reduced)


tuned_pipe_sq = clf_sq.best_estimator_


#print(clf_sq.best_estimator_)


# 
#robust = final_pipe.fit(X_train_reduced,y_train_reduced)

# X_train_scaled = clf_sq.transform(X_train_reduced)
# X_test_scaled = tuned_pipe_sq.transform(X_test_reduced)
# X_val_scaled = tuned_pipe_sq.transform(X_val_reduced)

#tuned_pipe_sq.fit(X_val_reduced,y_val_reduced)

#y_pred = tuned_pipe_sq.predict(X_test_reduced)

r2 = r2_score(y_test_reduced, y_pred)
r2_adj = 1-(1-r2_score(y_test_reduced, y_pred))*((len(X_test_reduced)-1)/(len(X_test_reduced)-X_test_reduced.shape[1])-1)
mse = mean_squared_error(y_test_reduced, y_pred)

print(r2)
print(r2_adj)
print(mse)
    
colab_crossval = joblib.load("Colab/crossval_scores.joblib")
#randsearchXGB = joblib.load("Colab/random_searchXGB.joblib")
#randsearchXGBv2 = joblib.load("Colab/random_searchXGBv2.joblib")

print(clf_sq.best_score_)
print(clf_sqv2.best_score_)
print(colab_crossval)
print(type(colab_crossval))
#print(randsearchXGB.best_score_)
#print(randsearchXGBv2.best_score_)


    
# model_GBR = GradientBoostingRegressor(learning_rate=0.025)

# grid_GBR = GridSearchCV(estimator=model_GBR, param_grid = parameters, n_jobs=2)
# grid_GBR.fit(X_train, y_train)



# # model_XGB.fit(X_train,y_train)
# # print(model_XGB.score(X_train,y_train))


# # grid_GBR = GridSearchCV(estimator=model_GBR, param_grid = parameters, n_jobs=2)
# # grid_GBR.fit(X_train, y_train)

# print(" Results from Grid Search " )
# print("\n The best estimator across ALL searched params:\n",grid_GBR.best_estimator_)
# print("\n The best score across ALL searched params:\n",grid_GBR.best_score_)
# print("\n The best parameters across ALL searched params:\n",grid_GBR.best_params_)

# sys.exit()

# model_XGB.fit(X_train,y_train)
# y_pred = model_XGB.predict(X_test)

# print(mean_absolute_error(y_test, y_pred))
# print(mean_squared_error(y_test, y_pred))
# print(explained_variance_score(y_test, y_pred))
# print(r2_score(y_test,y_pred))

# #sys.exit()


# #cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)

# NUM_TRIALS = 10
# scores = []

# for i in range(NUM_TRIALS):
#     cv = KFold(n_splits=5, shuffle=True, random_state=i)
#     clf = GridSearchCV(estimator=model_XGB, scoring=["r2"], param_grid=parameters, cv=cv, refit="r2")
   
#     #scores.append(clf.best_score_)
# clf.fit(X_train,y_train)  
# print(clf.best_params_)
# print(clf.best_score_)



# sys.exit()

# scoresMAE = cross_val_score(model_XGB, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv.get_n_splits(X_train,y_train), n_jobs=2)
# scoresMAE = np.absolute(scoresMAE)

# scoresMSE= cross_val_score(model_XGB, X_train, y_train, scoring='neg_mean_squared_error', cv=cv.get_n_splits(X_train,y_train), n_jobs=2)
# scoresMSE = np.absolute(scoresMSE)



# print('MAE: %.3f (%.3f)' % (scoresMAE.mean(), scoresMAE.std()))
# print('MSE: %.3f (%.3f)' % (scoresMSE.mean(), scoresMSE.std()))


# print(model_XGB.score(X_train,y_train))

# sys.exit()

# r2 = r2_score(y_test,y_pred)
# #accuracy = accuracy_score(y_test, y_pred)

# print(r2)
# #print(accuracy)




