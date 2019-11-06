import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

plt.style.use(u"~/.matplotlib/stylelib/trygveplot_astro.mplstyle")

from logreg     import LogReg
from initdata   import InitData
from DNN        import NeuralNetwork
from cost_functions import CostFunctions
## Get data from InitData Class
data = InitData()
noise = 0.00
XTrain, yTrain, XTest, yTest, X, y, b, x_, y_ =  data.franke_data(noise=noise)

logreg = LogReg() # init Logreg class
 
# ----- PARAMETERS ------
sklearn = False
NN = True
lmbd = 0.1
eta = 0.01
act_h = "sigmoid"
act_o = "identity" 
hn = 12
epochs=20
n_categories = 100 # WHy does this work ???

# ------ Run network ------
if sklearn:
    if NN: # NN Regress using sklearn - Somewhat reasonable, needs tuning.
        mlp = MLPRegressor()
        mlp.fit(XTrain, yTrain)

        yTrue, yPred = yTest, mlp.predict(XTest)
        print(mlp.score(XTest,yTrue.ravel()))

        ypred = mlp.predict(X)

    else: # Linear regression using Sklearn  - Initial results prefer this
        linreg = LinearRegression()
        linreg.fit(XTrain,yTrain) 

        yTrue, yPred = yTest, linreg.predict(XTest)
        print(linreg.score(XTest,yTrue.ravel()))
        ypred = linreg.predict(X)
else:
    if NN: # My regression NN - Outputs 50 :)
        dnn = NeuralNetwork(XTrain, yTrain,
                            eta=eta, 
                            lmbd=lmbd, 
                            n_hidden_neurons=hn,
                            act_h=act_h, 
                            act_o=act_o, 
                            epochs=epochs, 
                            nn_type="regression", 
                            n_categories=n_categories)
        dnn.train()
                            
        yTrue, yPred = yTest, dnn.predict(XTest)
        cf = CostFunctions("mse") # init Logreg class
        print(cf.f(yTrue,yPred))
        print(yPred)
        # Find full data prediction
        ypred = dnn.predict(X)
    else:  # Run my regression
        raise NotImplementedError("Insert lasso n stuff")



# ---- PLOT PREDICTION --------

# Plotting parameters
fig = plt.figure()
ax1 = fig.add_subplot(111, projection = '3d')
#ax1.set_zlim3d(-0.2,1.2) # Zlimit
# Remove background
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False
ax1.xaxis.pane.set_edgecolor('w')
ax1.yaxis.pane.set_edgecolor('w')
ax1.zaxis.pane.set_edgecolor('w')

ax1.view_init(5,60) # Optimal viewing angle
#surf = ax1.plot_surface(x_, y_, b, alpha=0.3, cmap=plt.cm.coolwarm,label=r"Franke function $N(0,%.2f)$" %noise)
#surf._facecolors2d=surf._facecolors3d # Bugfix for legend
#surf._edgecolors2d=surf._edgecolors3d

ax1.scatter(x_,y_,ypred.reshape(b.shape),alpha=1, s=1, color="C1")
plt.show()