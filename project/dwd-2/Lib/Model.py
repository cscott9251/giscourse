#import arcpy
from pyearth import Earth
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
from datasetup_wsl import CSVFile
# from datasetup import fieldnames
# from datasetup import ws

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score

df = pd.read_csv(CSVFile)

df = df.dropna()

X = df.drop(columns=["LST", 'Shape_Length', 'Shape_Area','OBJECTID', 'Shape'])

y = df.filter(["LST"], axis=1)

print(X.shape)
print(X.columns)

print(y.shape)
print(y.columns)


#pd.set_option('display.width', 10000)
pd.set_option('display.max_columns', None)
# print(X.corr())

# sns.heatmap(X.corr(), cmap='coolwarm')
# # sns.heatmap(X.corr().sort_values(by=["LST"], ascending=False), cmap='coolwarm')

# plt.show()

#print(X.corr()["LST"].abs().sort_values(ascending=False))

# LST is most highly correlated with NDVI, impervious, DGM,

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape)
print(X_train.columns)


print(y_train.shape)
print(y_train.columns)


model_MARS = Earth(max_degree=3,max_terms=200,verbose=2 )

model_MARS.fit(X_train, y_train)

#Print the model
print(model_MARS.trace())
print(model_MARS.summary())





