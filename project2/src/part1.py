import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logreg import LogReg
from initdata import InitData

data = InitData()
XTrain, yTrain, XTest, yTest, Y_train_onehot, Y_test_onehot = data.credit_data()


from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression().fit(XTrain,yTrain)
logReg.predict(XTest)

logreg = LogReg() # This does not use SGD

lrs = np.logspace(-5,7,13)
for lr in lrs:
    beta, costs = logreg.GD(XTrain,yTrain,lr=lr) # Fit using SGD. This can be looped over for best lambda.
    plt.plot(costs)
plt.show()
print(np.mean(beta),np.mean(logReg.coef_))
print("---------—--------—--------—--------—--------—--------—")
print(yTest[0],Y_test_onehot[0,0], Y_test_onehot[0,1])
logreg.predict(XTest)     # Predict
logreg.stats(Y_test_onehot)            # Prits stats. Figure out which one is accuracy ;)
sys.exit()
logreg.sklearn_alternative(XTrain, yTrain, XTest, yTest)