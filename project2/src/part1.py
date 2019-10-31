import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
#plt.style.use(u"~/.matplotlib/stylelib/trygveplot_astro.mplstyle")
plt.style.use(u"../trygveplot_astro.mplstyle")

from logreg import LogReg
from initdata import InitData

"""
In this part of the project, we assess the predictive ability of logistic regression on 
determining default based on credit card data. The weights are trained using a gradient
solver and compared with Scikit-Learns Logistic regression method.
"""

## Get data from InitData Class
data = InitData()
XTrain, yTrain, XTest, yTest, Y_train_onehot, Y_test_onehot = data.credit_data(trainingShare=0.5)

## Initialize Logreg Class
logreg = LogReg() # init Logreg class

# Optimize parameters
#lrs = np.logspace(-5,7,13)
lrs = [0.01]
for lr in lrs:
    beta, costs = logreg.SGD_batch(XTrain,yTrain.ravel(),lr=lr,adj_lr=True, n_epoch=100) # Fit using SGD. This can be looped over for best lambda.
    plt.plot(costs)
plt.show()

# Check results statistics
print("---------—--------—--- Our Regression --------—--------—--------—")
yTrue, yPred = yTest, logreg.predict(XTest)     # Predict
print(classification_report(yPred, yTrue))

# Compare with sklearn
if True: # Simple sklearn
    from sklearn.linear_model import LogisticRegression
    logReg = LogisticRegression(solver="lbfgs").fit(XTrain,yTrain.ravel())
    yTrue, yPred = yTest, logReg.predict(XTest)
    print("---------—--------—-- Sklearn Regression --------------—--------—")
    print(classification_report(yPred, yTrue))
else:   # Fancy optimal sklearn
    logreg.sklearn_alternative(XTrain, yTrain, XTest, yTest)
