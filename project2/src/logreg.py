#%% [markdown]
import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report

#%% Functions
def cross_entropy(beta, X, y):
    # Cross-entropy loss for logistic regression:
    a = np.dot(X,beta)
    reg = 0 # Regularization term
    return -np.sum(y*a - np.log(1+np.exp(a))) + reg

def SGD(beta, X,y, lr = 0.01, iter=100): 
    n = len(y)
    costs = np.zeros(iter)
    for i in range(iter):
        cost = 0.0
        for j in range(n):
            idx = np.random.randint(0,n)
            X_ = X[idx,:].reshape(1,X.shape[1])
            y_ = y[idx].reshape(1,1)
            a = np.dot(X_,beta)
            
            beta -= 1/m * lr * X_.T.dot(a-y)
            cost += cross_entropy(beta,X_,y_)
        costs[i] = cost
    return beta, costs

#%%
beta = np.random(-0.5, 0.5, X.shape[1])
