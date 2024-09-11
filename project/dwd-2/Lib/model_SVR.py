from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import joblib

X_train = joblib.load("X_train.sav")
X_test = joblib.load("X_test.sav")
y_train = joblib.load("y_train.sav")
y_test = joblib.load("y_test.sav")
X_val = joblib.load("X_val.sav")
y_val = joblib.load("y_val.sav")

# y_train = y_train.ravel()
# y_test = y_test.ravel()
# y_val = y_val.ravel()

param = {'model_SVR__kernel' : ('rbf', 'sigmoid'),
         'model_SVR__C'      : [1,5,10],
         'model_SVR__degree' : [3,8],
         'model_SVR__coef0'  : [0.01,10,0.5],
         'model_SVR__gamma'  : ('auto','scale')}

modelsvr = SVR(verbose=True)


preprocessor = Pipeline([
    
    ('scaler', RobustScaler())
])



final_pipe = Pipeline([
    
    ('preproc', preprocessor),
    ('model_SVR', modelsvr)
    
])

svrrandgrid = RandomizedSearchCV(final_pipe,param,cv=5,n_jobs=-1,n_iter=50)

svrrandgrid.fit(X_train,y_train.values.ravel())

joblib.dump(svrrandgrid, "svrrandgrid.joblib")

tunedsvrpipe = svrrandgrid.best_estimator_

joblib.dump(tunedsvrpipe, "tunedsvrpipe.joblib")

#print(final_pipe.get_params())