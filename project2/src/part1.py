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

from logreg import LogReg
from initdata import InitData

data = InitData()
XTrain, yTrain, XTest, yTest, Y_train_onehot, Y_test_onehot = data.credit_data()

logreg = LogReg()
beta, costs = logreg.SGD(XTrain,yTrain) # Fit using SGD
y_pred = logreg.predict(XTest)
print(y_pred)