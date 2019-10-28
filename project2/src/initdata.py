#%% [markdown]
import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report

#%% Chosing seeds
seed = 42069
np.random.seed(seed)
random.seed(seed)

#%% Read data as pandas dataframe from excel format
cwd = os.getcwd() # Get current path
cwd = '/Users/svalheim/work/fys-stk4155/project2' # quickfix
filename = cwd + '/default of credit card clients.xls'
nanDict = {} # Empty dictionary for storing nanvalues
df = pd.read_excel(filename, header = 1, skiprows=0, index_col=0, na_values=nanDict)
df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

df = df.drop(df[(df.BILL_AMT1 == 0) &
                (df.BILL_AMT2 == 0) &
                (df.BILL_AMT3 == 0) &
                (df.BILL_AMT4 == 0) &
                (df.BILL_AMT5 == 0) &
                (df.BILL_AMT6 == 0)].index)

df = df.drop(df[(df.PAY_AMT1 == 0) &
                (df.PAY_AMT2 == 0) &
                (df.PAY_AMT3 == 0) &
                (df.PAY_AMT4 == 0) &
                (df.PAY_AMT5 == 0) &
                (df.PAY_AMT6 == 0)].index)

## Features and targets
# Target is last column (defaultpayment 0 or 1), featues is everything else
X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values


## Categorical variables to one-hot's
# Dividing every variable into 2 categories
onehotencoder = OneHotEncoder(categories="auto")

X = ColumnTransformer(
    [("", onehotencoder, [3]),],
    remainder="passthrough"
).fit_transform(X)

#%% Train-test split
trainingShare = 0.5 
XTrain, XTest, yTrain, yTest=train_test_split(X, y, train_size=trainingShare, \
                                              test_size = 1-trainingShare,
                                             random_state=seed)

#%% Input Scaling
sc = StandardScaler() # Scale to zero mean and unit variance
XTrain = sc.fit_transform(XTrain) 
XTest = sc.transform(XTest) 

#%% One-hot's of the target vector
# ??? Is this necessary if we only have 1 type of prediction?
Y_train_onehot, Y_test_onehot = onehotencoder.fit_transform(yTrain), onehotencoder.fit_transform(yTest)


"""
#%% Setting up grid search for optimal parameters of Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

lambdas=np.logspace(-5,7,13)
parameters = [{'C': 1./lambdas, "solver":["lbfgs"]}]#*len(parameters)}]
scoring = ['accuracy', 'roc_auc']
logReg = LogisticRegression()
# ??? Finds best hyperparameters, then does regression.
gridSearch = GridSearchCV(logReg, parameters, cv=5, scoring=scoring, refit='roc_auc') 

#%% Fit stuff
gridSearch.fit(XTrain, yTrain.ravel())
yTrue, yPred = yTest, gridSearch.predict(XTest)
print(classification_report(yTrue,yPred))
rep = pd.DataFrame(classification_report(yTrue,yPred,output_dict=True)).transpose()
display(rep)

logreg_df = pd.DataFrame(gridSearch.cv_results_) # Shows behaviour of CV
display(logreg_df)


logreg_df.columns
logreg_df.plot(x='param_C', y='mean_test_accuracy', yerr='std_test_accuracy', logx=True)
logreg_df.plot(x='param_C', y='mean_test_roc_auc', yerr='std_test_roc_auc', logx=True)
plt.show()
"""
# %%
