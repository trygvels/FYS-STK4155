import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report

from cost_functions import CostFunctions
from initdata import InitData

class LogReg: # Logistic regression class
    def __init__(self, # Specify cost function to calculate with
                cost = 'cross_entropy'):

        self.cost = CostFunctions(cost)             # Init cross_entropy cost function
        self.initdata = InitData()                  # Init cross_entropy cost function
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def GD(self, X, y, lr = 1, tol=1e-2):           #Gradient descent method
        print("Doing GD for logreg")
        n = len(y) 
        costs = []                                  # Initializing cost list
        self.beta = np.random.randn(X.shape[1],1)   # Drawing initial random beta values

        i = 0; t = 1
        while t > tol:                              # Do gradient descent while below threshold
            b = X@self.beta                         # Calculate current prediction
            gradient = 1/n*( X.T @ (self.sigmoid(b)-y) ) # Calculate gradient
            self.beta -= lr*gradient                # Calculate perturbation to beta
            costs.append(self.cost(self.beta,X,y))  # Save cost of new beta
            t = np.linalg.norm(gradient)            # Calculate norm of gradient
            i += 1  
            if i > 1e5:                             # If descent takes too long, break.
                print("This takes way too long, %d iterations, with learning rage %e" %(i,lr))
                break

        print("Gradient solver has converged after %d iterations" % i )

        #plt.plot(range(iter), costs)
        #plt.show()
        return self.beta, costs

    def SGD(self, X, y, lr = 0.01, tol=1e-4): # Stochastic gradient descent method
        print("Doing SGD for logreg")
        n = len(y) 
        costs = []                                  # Initializing cost list
        self.beta = np.random.randn(X.shape[1],1)   # Drawing initial random beta values

        i = 0; t = 1
        while t > tol:                              # Do gradient descent while below threshold
            cost = 0
            for j in range(n):
                idx = np.random.randint(0,n)        # Chose random data row
                X_ = X[idx,:].reshape(1,X.shape[1]) # Select random data row
                y_ = y[idx].reshape(1,1)            # select corresponding prediction

                b = X_@self.beta                # Calculate current prediction
                gradient = 1/n*( X_.T @ (self.sigmoid(b)-y_)) # Calculate gradient
                self.beta -= lr*gradient                # Calculate perturbation to beta
                cost += self.cost(self.beta,X_,y_)

            costs.append(cost)                      # Save cost of new beta
            t = np.linalg.norm(gradient)            # Calculate norm of gradient #Fix this for SGD
            i += 1  
            if i > 1e5:                             # If descent takes too long, break.
                print("This takes way too long, %d iterations, with learning rage %e" %(i,lr))
                break

        print("Stochastic gradient solver has converged after %d iterations" % i )
        return self.beta, costs

    def predict(self,X):                           # Calculates probabilities and onehots for y
        print("Predicting y using logreg")
        # Returns probabilities
        self.yprobs = self.sigmoid(X@self.beta)
        self.yPred = (self.yprobs > 0.5).astype(int)
        self.y_pred_onehot = self.initdata.onehotencoder.fit_transform(self.yPred) # Converts to onehot
        return self.yPred

    def sklearn_alternative(self, XTrain, yTrain, XTest, yTest): # Does SKLEARN method
        print("Doing logreg using sklearn")
        #%Setting up grid search for optimal parameters of Logistic regression
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report

        lambdas=np.logspace(-5,7,13)
        parameters = [{'C': 1./lambdas, "solver":["lbfgs"]}]#*len(parameters)}]
        scoring = ['accuracy', 'roc_auc']
        logReg = LogisticRegression()
        # Finds best hyperparameters, then does regression.
        gridSearch = GridSearchCV(logReg, parameters, cv=5, scoring=scoring, refit='roc_auc') 

        # Fit stuff
        gridSearch.fit(XTrain, yTrain.ravel())
        yTrue, yPred = yTest, gridSearch.predict(XTest)
        print(classification_report(yTrue,yPred))
        rep = pd.DataFrame(classification_report(yTrue,yPred,output_dict=True)).transpose()
        display(rep)

        logreg_df = pd.DataFrame(gridSearch.cv_results_) # Shows behaviour of CV
        display(logreg_df[['param_C','mean_test_accuracy', 'rank_test_accuracy','mean_test_roc_auc', 'rank_test_roc_auc']])

        logreg_df.columns
        logreg_df.plot(x='param_C', y='mean_test_accuracy', yerr='std_test_accuracy', logx=True)
        logreg_df.plot(x='param_C', y='mean_test_roc_auc', yerr='std_test_roc_auc', logx=True)
        plt.show()