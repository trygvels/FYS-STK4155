import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
plt.style.use(u"~/.matplotlib/stylelib/trygveplot_astro.mplstyle")

from logreg     import LogReg
from initdata   import InitData
from DNN        import NeuralNetwork
"""
In this part of the project, we assess the predictive ability of a feed-forward Neural 
network on determining default based on credit card data. 
"""
## Get data from InitData Class
data = InitData()
XTrain, yTrain, XTest, yTest, Y_train_onehot, Y_test_onehot = data.credit_data(trainingShare=0.5)
# Running neural network, Needs onehot input
dnn = NeuralNetwork(XTrain,Y_train_onehot.toarray())
dnn.train()

yPred = dnn.predict_probabilities(XTest)
yTrue, yPred = yTest, dnn.predict(XTest)
print(classification_report(yPred, yTrue))