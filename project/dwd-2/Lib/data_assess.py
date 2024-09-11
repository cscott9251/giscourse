import pandas as pd
import numpy as np 

import joblib

X = joblib.load("X_full.sav")
y = joblib.load("y_full.sav")

#print(X.notnull().sum().sort_values(ascending=False))

#X.isnull().sum().sort_values(ascending=False)/len(X)

X_train = joblib.load("X_train.sav")
X_test = joblib.load("X_test.sav")
y_train = joblib.load("y_train.sav")
y_test = joblib.load("y_test.sav")
X_val = joblib.load("X_val.sav")
y_val = joblib.load("y_val.sav")

setlist=[X_train, X_test, y_train, y_test, X_val, y_val]

for set in setlist:
    print(type(set))
    # print(f"{set.columns} %n")
    # print(f"{set.shape()} %n")
    # print(f"{set.head()} %n")