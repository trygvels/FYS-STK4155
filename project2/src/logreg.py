import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report

from cost_functions import CostFunctions
from initdata import InitData

class LogReg:
    def __init__(self, 
                cost = 'cross_entropy', 
                path=None):

        self.cost = CostFunctions(cost) # Init cross_entropy cost function
        self.initdata = InitData() # Init cross_entropy cost function
        self.path = path
        
    def GD(self, X, y, lr = 1, tol=1e-2):
        print("Doing GD for logreg")

        # Fits beta using stochastic gradient descent
        n = len(y)
        costs = []
        self.beta = np.random.randn(X.shape[1],1)

        i = 0; t = 1
        while t > tol:
            b = np.dot(X,self.beta) # Calculate current prediction
            gradient = 1/n*np.dot(X.T,expit(b)-y) # Calculate gradient
            self.beta -= lr*gradient # Calculate perturbation to beta
            costs.append(self.cost(self.beta,X,y)) # Save cost of new beta
            t = np.linalg.norm(gradient) 
            i += 1
            if i > 1e5:
                print("This takes way too long, %d iterations, with learning rage %e" %(i,lr))
                break

        print("Gradient solver has converged after %d iterations" % i )

        #plt.plot(range(iter), costs)
        #plt.show()
        return self.beta, costs

    def SGD(self, X, y, lr = 0.01, iter=150, tol=1e-4):
        print("Doing SGD for logreg")

        # Fits beta using stochastic gradient descent
        n = len(y)
        costs = np.zeros(iter)
        self.beta = np.random.randn(X.shape[1],1)
        self.beta = np.random.uniform(-0.5,0.5,(X.shape[1], 1))
    
        i = 0
        while 1 > tol and i < iter: # Tol placeholder
            cost = 0.0
            for j in range(n):
                idx = np.random.randint(0,n) # Chose random data row
                X_ = X[idx,:].reshape(1,X.shape[1]) # Select random data row
                y_ = y[idx].reshape(1,1)            # select corresponding prediction

                b = np.dot(X_,self.beta) # Calculate current prediction
                self.beta -= lr/n*np.dot(X_.T,b-y_) #Calculate change to prediction
                cost += self.cost(self.beta,X_,y_)
            costs[i] = cost # Saves cost of beta over iterations
            i+=1
        return self.beta, costs

    def predict(self,X):
        print("Predicting y using logreg")
        # Returns probabilities
        self.probabilities = expit(np.dot(X,self.beta)) # Makes prediction using trained betas
        ints = (self.probabilities > 0.5).astype(int) 
        self.y_pred_onehot = self.initdata.onehotencoder.fit_transform(ints)
        print("---------—--------—--------—--------—--------—--------—")
        print("Checking prections", self.probabilities.shape, self.y_pred_onehot.shape)
        print("Checking prections", self.probabilities[0], self.y_pred_onehot[0,0], self.y_pred_onehot[0,1])
        print("---------—--------—--------—--------—--------—--------—")
        return self.probabilities, self.y_pred_onehot

    def stats(self, y_test_onehot):
        # Make this output a nice dictionary
        stats = classification_report(self.y_pred_onehot,y_test_onehot)
        print(stats)
        return stats

    def sklearn_alternative(self, XTrain, yTrain, XTest, yTest):
        print("Doing logreg using sklearn")
        #%Setting up grid search for optimal parameters of Logistic regression
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report

        lambdas=np.logspace(-5,7,13)
        parameters = [{'C': 1./lambdas, "solver":["lbfgs"]}]#*len(parameters)}]
        scoring = ['accuracy', 'roc_auc']
        logReg = LogisticRegression()
        # ??? Finds best hyperparameters, then does regression.
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