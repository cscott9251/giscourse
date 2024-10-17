from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import joblib

X_train = joblib.load("Reduced_dataset/X_train_reduced.sav")
X_test = joblib.load("Reduced_dataset/X_test_reduced.sav")
y_train = joblib.load("Reduced_dataset/y_train_reduced.sav")
y_test = joblib.load("Reduced_dataset/y_test_reduced.sav")
X_val = joblib.load("Reduced_dataset/X_val_reduced.sav")
y_val = joblib.load("Reduced_dataset/y_val_reduced.sav")

# y_train = y_train.ravel()
# y_test = y_test.ravel()
# y_val = y_val.ravel()

param = {'model_SVR__kernel' : ('rbf', 'sigmoid'),
         'model_SVR__C'      : [1,5,10],
         'model_SVR__degree' : [3,8],
         'model_SVR__coef0'  : [0.01,10,0.5],
         'model_SVR__gamma'  :  [0.01,0.001,1]}

modelsvr = SVR(verbose=True,shrinking=False)


preprocessor = Pipeline([
    
    ('scaler', RobustScaler())
])



final_pipe = Pipeline([
    
    ('preproc', preprocessor),
    ('model_SVR', modelsvr)
    
])

svrgrid = GridSearchCV(final_pipe,param,cv=5,n_jobs=-1,verbose=3)

svrgrid.fit(X_train,y_train.values.ravel())

joblib.dump(svrgrid, "svrgrid.joblib")

tunedsvrpipe = svrgrid.best_estimator_

joblib.dump(tunedsvrpipe, "tunedsvrpipe.joblib")

#print(final_pipe.get_params())