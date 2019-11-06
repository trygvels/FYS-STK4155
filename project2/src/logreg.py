import time
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report

from cost_functions import CostFunctions
from initdata       import InitData
from activations    import Activations


class LogReg: # Logistic regression class
    def __init__(self, # Specify cost function to calculate with
                cost = 'cross_entropy'):

        self.cost = CostFunctions(cost)             # Init cross_entropy cost function
        self.initdata = InitData()                  # Init data set
        self.act = Activations("sigmoid")
        
    def GD(self, X, y, lr = 1, tol=1e-2):           #Gradient descent method
        print("Doing GD for logreg")
        n = len(y) 
        costs = []                                  # Initializing cost list
        self.beta = np.random.randn(X.shape[1])   # Drawing initial random beta values

        i = 0; t = 1
        while t > tol:                              # Do gradient descent while below threshold
            b = X@self.beta                         # Calculate current prediction
            gradient = 1/n*( X.T @ (self.act.f(b)-y) ) # Calculate gradient
            self.beta -= lr*gradient                # Calculate perturbation to beta
            tar = X@self.beta
            costs.append(self.cost(tar,y))  # Save cost of new beta
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
                X_ = X[idx,:].reshape(1,X.shapels
                [1]) # Select random data row
                y_ = y[idx].reshape(1,1)            # select corresponding prediction

                b = X_@self.beta                # Calculate current prediction
                gradient = 1/n*( X_.T @ (self.act.f(b)-y_)) # Calculate gradient
                self.beta -= lr*gradient                # Calculate perturbation to beta
                tar = X_@self.beta
                cost += self.cost(tar,y_)  # Save cost of new beta

            costs.append(cost)                      # Save cost of new beta
            t = np.linalg.norm(gradient)            # Calculate norm of gradient #Fix this for SGD
            i += 1  
            if i > 1e5:                             # If descent takes too long, break.
                print("This takes way too long, %d iterations, with learning rage %e" %(i,lr))
                break

        print("Stochastic gradient solver has converged after %d iterations" % i )
        return self.beta, costs

    # Stochastic gradient descent method with batches
    def SGD_batch(self, X, y, lr = 0.01, tol=1e-4, max_iter=1, batch_size=100, n_epoch=100, rnd_seed=False, adj_lr=False, rnd_batch=False, verbosity=0): 
        print("Doing SGD for logreg")
        n = len(y) 
        costs = []                                  # Initializing cost list
        
        if (rnd_seed):
            np.random.seed(int(time.time()))
        self.beta = np.random.randn(X.shape[1],1)   # Drawing initial random beta values
        tar = X@self.beta
        min_cost = self.cost(tar,y)  # Save cost of new beta
        best_beta=self.beta.copy()

        if (adj_lr):
            t0 = n
            lr0=lr

        # We do several SGD searches with new batches for each search, with new searches
        # starting from the previous endpoint
        
        for i in range(max_iter):
            if (verbosity>0):

                print('   search %i of %i'%(i+1,max_iter))
            # Data is (semi) sorted on age after index ~15000,
            # dividing into batches based on index is therefore potentially not random.
            # We therefore have 2 options, (1) draw batch_size random values for each
            # iteration 'j', or (2) split data into m batches before starting
            m=int(n/batch_size)
            if (rnd_batch):
                nbatch=[]
                nbatch.append(batch_size)
                idx=0
            else:
                batch_idx,nbatch=self.split_batch(n,m)
            for k in range(n_epoch):
                if (verbosity>1):
                    print('   epoch %i of %i'%(k+1,n_epoch))
                for j in range(m):
                    #values instead
                    if (rnd_batch):
                        idx_arr = np.random.randint(0,n,batch_size) # Choose n random data rows
                    else:
                        idx=np.random.randint(0,m)
                        idx_arr = batch_idx[idx,:nbatch[idx]]
                    X_ = X[idx_arr,:].reshape(nbatch[idx],X.shape[1]) # Select batch data
                    y_ = y[idx_arr].reshape(nbatch[idx],1)            # select corresponding prediction
                    b = X_@self.beta                # Calculate current prediction
                    gradient = ( X_.T @ (self.act.f(b)-y_)) # Calculate gradient
                    if (adj_lr):
                        lr=(lr0*t0)/(t0+k*n+j*batch_size)
                        #as iterations increase, the step size in beta is reduced
                    self.beta = self.beta - lr*gradient                # Calculate perturbation to beta
                #after each epoch we compute the cost (majority of runtime)
                cost = self.cost(self.beta,X,y) #calculate total cost (This takes a long time!!)
                costs.append(cost)                      # Save cost of new beta
                if (cost < min_cost):
                    min_cost=cost
                    best_beta=self.beta.copy()
        
        self.beta=best_beta.copy() #finally return beta with the lowest total cost

        return best_beta, costs

    def predict(self,X):                           # Calculates probabilities and onehots for y
        print("Predicting y using logreg")
        # Returns probabilities
        self.yprobs = self.act.f(X@self.beta)
        self.yPred = (self.yprobs > 0.5).astype(int)
        self.y_pred_onehot = self.initdata.onehotencoder.fit_transform(self.yPred.reshape(-1,1)) # Converts to onehot
        return self.yPred

    def sklearn_alternative(self, XTrain, yTrain, XTest, yTest): # Does SKLEARN method
        print("Doing logreg using sklearn")
        #%Setting up grid search for optimal parameters of Logistic regression
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import classification_report

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
        pd.display(logreg_df[['param_C','mean_test_accuracy', 'rank_test_accuracy','mean_test_roc_auc', 'rank_test_roc_auc']])

        logreg_df.columns
        logreg_df.plot(x='param_C', y='mean_test_accuracy', yerr='std_test_accuracy', logx=True)
        logreg_df.plot(x='param_C', y='mean_test_roc_auc', yerr='std_test_roc_auc', logx=True)
        plt.show()

    def own_classification_report(self,ytrue,pred,threshold=0.5):
        tp=0
        tn=0
        fp=0
        fn=0
        pred=np.where(pred>threshold,1,0)
        for i in range(len(ytrue)):
            if (pred[i]==1 and ytrue[i]==1):
                tp +=1
            elif (pred[i]==1 and ytrue[i]==0):
                fp +=1
            elif (pred[i]==0 and ytrue[i]==0):
                tn +=1
            elif (pred[i]==0 and ytrue[i]==1):
                fn +=1
        pcp=np.sum(np.where(pred==1,1,0))
        pcn=np.sum(np.where(pred==0,1,0))
        cp=np.sum(np.where(ytrue==1,1,0))
        cn=np.sum(np.where(ytrue==0,1,0))
        ppv=[tn*1.0/pcn, tp*1.0/pcp]
        trp=[tn*1.0/cn, tp*1.0/cp]
        f1=[2.0*ppv[0]*trp[0]/(ppv[0]+trp[0]), 2.0*ppv[1]*trp[1]/(ppv[1]+trp[1])]
        print("              precision     recall     f1-score     true number    predicted number")
        print()
        print("           0      %5.3f      %5.3f        %5.3f        %8i    %16i"%(ppv[0],trp[0],f1[0],cn,pcn))
        print("           1      %5.3f      %5.3f        %5.3f        %8i    %16i"%(ppv[1],trp[1],f1[1],cp,pcp))
        print()
        print("    accuracy                              %5.3f        %8i"%((tp+tn)*1.0/(cp+cn),cp+cn))
        print("   macro avg      %5.3f      %5.3f        %5.3f        %8i"%((ppv[0]+ppv[1])/2.0,(trp[0]+trp[1])/2.0, (f1[0]+f1[1])/2.0,cn+cp))
        print("weighted avg      %5.3f      %5.3f        %5.3f        %8i"%((ppv[0]*cn+ppv[1]*cp)/(cn+cp),(trp[0]*cn+trp[1]*cp)/(cn+cp), (f1[0]*cn+f1[1]*cp)/(cn+cp),cn+cp))
        print()

        return

    def split_batch(self,n,m):

        if (m>n):
            print('m > n in split_batch')
            exit()
            
        #maximum batch size
        if (np.mod(n,m)==0):
            n_m=n//m
        else:
            n_m=n//m+1

        idx_arr=np.zeros(shape=(m,n_m),dtype='int')
        n_ms=np.zeros(shape=(m),dtype='int')
        
        if (m==n): #n_m=1
            idx_arr[:,0]=np.arange(0,m,dtypr='int')
            n_ms += 1
            return idx_arr,n_ms

        arr=np.arange(0,n,dtype='int')
        n_left=n
        for i in range(n):
            m_i = np.mod(i,m) #group number
            nm_i = i//m       #index in group
            n_ms[m_i]=nm_i+1 #update number of values in group
            ind=np.random.randint(n_left) #draw random sample in data that is left
            idx_arr[m_i,nm_i]=arr[ind].copy() #add index of data point to batch m_i 
            arr[ind:n_left-1]=arr[ind+1:n_left].copy() #remove data point from what is left
            n_left-=1

        return idx_arr,n_ms

    def print_beta(self, cols=[]):
        print('------- Beta values -------')
        print('      value   data label' )
        print()
        for i in range(len(self.beta)):
            if i >= len(cols):
                print('  %11.6f'%(self.beta[i,0]))
            else:
                print('  %11.6f   %s'%(self.beta[i,0],cols[i]))
        print()
        return
