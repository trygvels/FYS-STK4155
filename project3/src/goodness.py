import time
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report

from cost_functions import CostFunctions
from initdata       import InitData
from activations    import Activations

class Goodness: # Logistic regression class
    def __init__(self):
        self.print_goodness=True
        
    def classification_report(self,ytrue,pred,print_to_file=False,print_res=True,return_ac=False,return_f1=False,return_f1_weight=False,return_ar=False, return_ar_weight=False):
        """
        Function to calculate and print the goodness of fit parameters for a given 
        prediction.

        Input: numpy arrays of size (N_data, N_classes)
               ytrue: array of true classification {0,1} 
               pred:  array of predicted classification, probabilities [0,1]

        The images (row) is predicted to belong to the class (column) with the highes prob.

        """

        n = pred.shape[0]
        m = pred.shape[1]
        tp = np.zeros(shape=(m),dtype='int')  #number of true positive predictions per class
        tn = np.zeros(shape=(m),dtype='int')  #number of true negative predictions per class
        fp= = np.zeros(shape=(m),dtype='int')  #number of false positive predictions per class
        fn= = np.zeros(shape=(m),dtype='int')  #number of false negative predictions per class
        pred_bin=np.zeros(shape=(n,m),dtype='int') # binary array of predictions

        # set class prediction and count true/flase positives/negatives
        for i in range(n):
            ind=np.argmax(pred[i,:]) # index of higher probability of row i
            pred_bin[i,ind]=1        # set (row i,column ind) to 1 (rest is zero)
            for j in range(m): # this may be vectorized
                if (pred[i,j]==1 and ytrue[i,j]==1):
                    tp[j] +=1
                elif (pred[i,j]==1 and ytrue[i,j]==0):
                    fp[j] +=1
                elif (pred[i,j]==0 and ytrue[i,j]==0):
                    tn[j] +=1
                elif (pred[i,j]==0 and ytrue[i,j]==1):
                    fn[j] +=1

        # count number of predicted targets and true targets per class 
        pcp=np.sum(np.where(pred_bin==1,1,0),axis=0) #predicted number of targets per class
        cp=np.sum(np.where(ytrue==1,1,0),axis=0) #true number of targets per class

        #calculate positive predictive value (precission) and true positive rate (recall) 
        ppv=tp*1.0/pcp
        tpr=tp*1.0/cp

        #calculate accuracy and F1-score (per calss) and weighted F1-score
        ac=np.sum(tp)*1.0/n # accuracy: sum all correct predictions (positive) and divide by total number of images (rows)
        f1=2.0*ppv*tpr/(ppv+tpr) # f1 score per class
        f1_weight = np.sum(f1*cp)/n # weight each f1 score with the relative number of images per class (the true values, not the predicted)

        # Calculate area ratio per class
        """
        code to come
        """

        # print a good and easily viewable classification
        """
        code to come
        """

        # return the desired values
        """
        code to come
        order:
        accuracy, f1, f1_weight, ar, ar_weight 
        """

        return


    def test_classification(self):
        # Write a short code to test the above
        """
        do random values of an 20x4 array
        normalize each row
        create random (ish) ytrue array (40% chance of not recalling highest probability, but a random one)
        """
