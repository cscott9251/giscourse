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


#df["OBJECTID"] = df["OBJECTID"].shift(1)
#df.index = df.index + 1

print(df.describe())

null_mask = df.isnull().all(axis=1)
null_rows = df[null_mask]

new_df = df.dropna()

print(new_df.shape)
print(new_df)

new_df = new_df.set_index("OBJECTID")

#new_df = new_df.fillna(0)

print(new_df.describe())
print(new_df)

df=new_df.copy(deep=True)
joblib.dump(df,"reduced.sav")
df.to_csv("df_reduced.csv")
#df.drop_duplicates(inplace=True) 

#print(df.describe())

df.boxplot() # Graph boxplot of dataset

df.plot.hist(bins=100) # Graph histograms of dataset

#objectids = df["OBJECTID"]
#objectids.to_csv("objectIDs.csv")
#df=df.drop("OBJECTID", axis=1)

X_orig_reduced = df.drop(columns="LST")
joblib.dump(X_orig_reduced, "X_reduced.sav")
X_orig_reduced.to_csv("X_orig_reduced.csv")

    

y_orig_reduced = df[["LST"]]
y_orig_reduced.to_csv("y_orig_reduced.csv")
joblib.dump(y_orig_reduced, "y_orig.sav")


#forkbestinput(X_orig,y_orig)

    
    

#plt.show()

#df = df[["LST","flaeche","buildup","imprevious","DGM","NDVI"]]

#df.to_csv("data.csv")


#X = df[["flaeche","buildup","imprevious","DGM","NDVI"]]



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

train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15


# train is now 70% of the entire data set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio,shuffle=True,random_state=42)

X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(X_orig_reduced, y_orig_reduced, test_size=1 - train_ratio,shuffle=True,random_state=42)

# test is now 15% of the initial data set
# validation is now 15% of the initial data set
X_val_reduced, X_test_reduced, y_val_reduced, y_test_reduced = train_test_split(X_test_reduced, y_test_reduced, test_size=test_ratio/(test_ratio + validation_ratio))

joblib.dump(X_train_reduced, "X_train_reduced.sav")
joblib.dump(X_test_reduced, "X_test_reduced.sav")
joblib.dump(y_train_reduced, "y_train_reduced.sav")
joblib.dump(y_test_reduced, "y_test_reduced.sav")
joblib.dump(X_val_reduced, "X_val_reduced.sav")
joblib.dump(y_val_reduced, "y_val_reduced.sav")

X_train_reduced.to_csv("X_train_reduced.csv")
X_test_reduced.to_csv("X_test_reduced.csv")
y_train_reduced.to_csv("y_train_reduced.csv")
y_test_reduced.to_csv("y_test_reduced.csv")
X_val_reduced.to_csv("X_val_reduced.csv")
y_val_reduced.to_csv("y_val_reduced.csv")

print("------------------------------")


testtrainvallist = [X_train_reduced,X_test_reduced,y_train_reduced,y_test_reduced,X_val_reduced,y_val_reduced]

for d in testtrainvallist:

    print(d)
    print(d.shape)

# plt.plot(X,y)

# plt.show()