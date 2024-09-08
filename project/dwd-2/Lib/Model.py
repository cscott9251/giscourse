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
#from sklearn import sklearn-contrib-py-earth
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv(CSVFile)

df = df.dropna()

X = df.drop(columns=['Shape_Length', 'Shape_Area','OBJECTID', 'Shape'])

y = df.filter(["LST"], axis=1)

# print(X.shape)
# print(X.columns)
#
# print(y.shape)
# print(y.columns)

# sns.heatmap(X.corr(), cmap='coolwarm')

# plt.show()

pd.set_option('display.width', 10000)
pd.set_option('display.max_columns', None)
# print(X.corr())

# 

# sns.heatmap(X.corr(), cmap='coolwarm')
# # sns.heatmap(X.corr().sort_values(by=["LST"], ascending=False), cmap='coolwarm')

# plt.show()

print(X.corr()["LST"].abs().sort_values(ascending=False))

# LST is most highly correlated with NDVI, impervious, DGM,

# sample = X.sample(10000, random_state=42)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)





