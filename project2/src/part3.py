import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

plt.style.use(u"~/.matplotlib/stylelib/trygveplot_astro.mplstyle")

from logreg     import LogReg
from initdata   import InitData
from DNN        import NeuralNetwork

## Get data from InitData Class
data = InitData()
XTrain, yTrain, XTest, yTest =  data.franke_data()

logreg = LogReg() # init Logreg class

if False:
    # NN Regress using sklearn
    mlp = MLPRegressor()
    mlp.fit(XTrain, yTrain)
    yTrue, yPred = yTest, mlp.predict(XTest)
    print(mlp.score(XTest,yTrue.ravel()))
else: 
    # Linear regression using Sklearn
    linreg = LinearRegression()
    linreg.fit(XTrain,yTrain)
    yTrue, yPred = yTest, linreg.predict(XTest)
    print(linreg.score(XTest,yTrue.ravel()))

