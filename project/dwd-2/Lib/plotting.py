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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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

X = joblib.load("X_full.sav")
y = joblib.load("y_full.sav")

X_train_reduced = joblib.load("Reduced_dataset/X_train_reduced.sav")
X_test_reduced = joblib.load("Reduced_dataset/X_test_reduced.sav")
y_train_reduced = joblib.load("Reduced_dataset/y_train_reduced.sav")
y_test_reduced = joblib.load("Reduced_dataset/y_test_reduced.sav")
X_val_reduced = joblib.load("Reduced_dataset/X_val_reduced.sav")
y_val_reduced = joblib.load("Reduced_dataset/y_val_reduced.sav")



df_complete = joblib.load("df_complete.sav")

num_of_folds = 4
num_of_groups = 100


def fold_visualizer(data, fold_idxs, seed_num):
    fig, axs = plt.subplots(len(fold_idxs)//2, 2, figsize=(15,(len(fold_idxs)//2)*5))
    fig.suptitle("Seed: " + str(seed_num), fontsize=16)
    for fold_id, (train_ids, val_ids) in enumerate(fold_idxs):
        sns.histplot(data=data[train_ids],
                     kde=True,
                     stat="density",
                     alpha=0.15,
                     label="Train Set",
                     bins=30,
                     line_kws={"linewidth":4},
                     ax=axs[fold_id%(len(fold_idxs)//2), fold_id//(len(fold_idxs)//2)])
        sns.histplot(data=data[val_ids],
                     kde=True,
                     stat="density", 
                     color="darkorange",
                     alpha=0.15,
                     label="Validation Set",
                     bins=30,
                     line_kws={"linewidth":4},
                     ax=axs[fold_id%(len(fold_idxs)//2), fold_id//(len(fold_idxs)//2)])
        axs[fold_id%(len(fold_idxs)//2), fold_id//(len(fold_idxs)//2)].legend()
        axs[fold_id%(len(fold_idxs)//2), fold_id//(len(fold_idxs)//2)].set_title("Split " + str(fold_id+1))
    #plt.show()
    plt.savefig(f"StratSeed{seed_num}.png")

def create_cont_folds(df, n_s=8, n_grp=1000, seed=1):
    
    skf = StratifiedKFold(n_splits=n_s, shuffle=True, random_state=seed)
    grp = pd.qcut(df, n_grp, labels=False)
    target = grp
    
    fold_nums = np.zeros(len(df))
    for fold_no, (t, v) in enumerate(skf.split(target, target)):
        fold_nums[v] = fold_no
    
    cv_splits = []

    for i in range(num_of_folds):
        test_indices = np.argwhere(fold_nums==i).flatten()
        train_indices = list(set(range(len(y))) - set(test_indices))
        cv_splits.append((train_indices, test_indices))
        
    return cv_splits

# for i in range(5):
#     cv_splits = create_cont_folds(y, n_s=num_of_folds, n_grp=num_of_groups, seed=i)
#     fold_visualizer(data=y,
#                     fold_idxs=cv_splits,
#                     seed_num=i)


def plotinputs(X):
    

    fig, axis = plt.subplots(1, len(X.axes[1]), figsize=(16,3))
    X.hist(ax=axis, edgecolor='black', grid=False)
    plt.show()
    plt.savefig("InputFeaturesHistPlots.png")
    
    X.boxplot()
    X.plot.hist(bins=100)

    plt.show()
    
    
def plotcorr(X):
    
    sns.heatmap(X.corr(), cmap='coolwarm')
    print(X.corr())
    print(df_complete.corr())
    plt.show()


#plotinputs(X)

plotcorr(X_train_reduced)




# r2 = r2_score(y_test,y_pred)

# print(r2)
# print("----------------")
# model_LR = LinearRegression().fit(X_train,y_train)
# print(model_LR.score(X_train,y_train))
# print(model_LR.score(X_test,y_test))
# print("----------------")






