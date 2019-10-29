import numpy as np
from cost_functions import CostFunctions
from scipy.special import expit
from initdata import InitData

class LogReg:
    def __init__(self, 
                cost = 'cross_entropy', 
                path=None):

        self.cost = CostFunctions(cost) # Init cross_entropy cost function
        self.initdata = InitData() # Init cross_entropy cost function
        self.path = path
        
      
    def SGD(self, X, y, lr = 0.01, iter=100, tol=1e-4): 
        # Fits beta using stochastic gradient descent
        n = len(y)
        costs = np.zeros(iter)
        self.beta = np.random.randn(X.shape[1],1)

        i = 0
        while 1 > tol and i < iter: # Tol placeholder
            cost = 0.0
            for j in range(n):
                idx = np.random.randint(0,n)
                X_ = X[idx,:].reshape(1,X.shape[1])
                y_ = y[idx].reshape(1,1)

                b = np.dot(X_,self.beta)
                self.beta -= lr/n*np.dot(X_.T,b-y_)
                cost += self.cost(self.beta,X_,y_)
            costs[i] = cost # Saves cost of beta over iterations
            i+=1
        return self.beta, costs

    def predict(self,X):
        # Returns probabilities
        self.probabilities = expit(np.dot(X,self.beta))
        ints = (self.probabilities > 0.5).astype(int) 
        self.y_pred_onehot = self.initdata.onehotencoder.fit_transform(ints)
        return self.probabilities, self.y_pred_onehot

    def stats(self, y_test_onehot):
        # Make this output a nice dictionary
        s = np.sum(y_test_onehot, axis=0)
        print("number of true defaults / not defaults", s)
        m = np.dot(self.y_pred_onehot.T,y_test_onehot) # Number of correctly characterized defaults
        stats = m/s
        print(stats)
        return stats