#import arcpy
from pyearth import Earth
import pandas as pd
import numpy as np  
import statistics
from scipy.stats import norm
#import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import  (
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
    minmax_scale
)
    
from datasetup_wsl import CSVFile
# from datasetup import fieldnames
# from datasetup import ws

#from sklearn import linear_model
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.metrics import r2_score

import h5py
import joblib
import sys

df = pd.read_csv(CSVFile)

print(df.shape)
print(df.columns)

df = df.dropna()

X = df.drop(columns=["LST", 'Shape_Length', 'Shape_Area','OBJECTID', 'Shape'])

y = df.filter(["LST"], axis=1)

print(X.shape)
print(X.columns)

print(y.shape)
print(y.columns)



# print(X)
# print(y)

#plt.plot(X["NDVI"], y, marker=".", linestyle="none")
# plt.figure(figsize=(24,200))
# try:
#     for i, col in enumerate(X.columns.to_list()):
#         plt.subplot(10, 3, i + 1)
#         plt.hist(X[col], label=col,color='blue')
#         plt.legend()
#         plt.title(col)
#         plt.tight_layout()
# except Exception as e:
#     print(col,e)
    
# plt.savefig('foo.pdf')
# plt.savefig('foo.png')
#plt.show()
# plt.xlabel("Body Temperature (F)")
# plt.ylabel("Cumulative Distribution Function")

#pd.set_option('display.width', 10000)

pd.set_option('display.max_columns', None)
# print(X.corr())

# sns.heatmap(X.corr(), cmap='coolwarm')
# # sns.heatmap(X.corr().sort_values(by=["LST"], ascending=False), cmap='coolwarm')

# plt.show()

#print(X.corr()["LST"].abs().sort_values(ascending=False))

# LST is most highly correlated with NDVI, impervious, DGM,

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# joblib.dump(X_train, "X_train.sav")
# joblib.dump(X_test, "X_test.sav")
# joblib.dump(y_train, "y_train.sav")
# joblib.dump(y_test, "y_test.sav")

print("------------------------------")

print(X_train.shape)
print(X_train.columns)


print(y_train.shape)
print(y_train.columns)

print(X_train)
print(y_train)

# plt.plot(X,y)

# plt.show()

sys.exit()

model_MARS = Earth(max_degree=2,max_terms=200,verbose=2 )

model_MARS.fit(X_train, y_train)

#Print the model
print(model_MARS.trace())
print(model_MARS.summary())
print(cross_val_score(model_MARS, X_train, y_train, cv=3))
print("----------------")
print(model_MARS.score(X_train,y_train))
print(model_MARS.score(X_test,y_test))
print("----------------")

y_pred = model_MARS.predict(X_test)

filename = 'finalized_model.sav'
joblib.dump(model_MARS, filename)

r2 = r2_score(y_test,y_pred)

print(r2)
print("----------------")
model_LR = LinearRegression().fit(X_train,y_train)
print(model_LR.score(X_train,y_train))
print(model_LR.score(X_test,y_test))
print("----------------")






