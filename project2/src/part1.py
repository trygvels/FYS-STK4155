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

logreg = LogReg()
beta, costs = logreg.SGD(XTrain,yTrain) # Fit using SGD. This can be looped over for best lambda.
logreg.predict(XTest)     # Predict
logreg.stats(Y_test_onehot)            # Prits stats. Figure out which one is accuracy ;)

plt.plot(costs)
plt.show()