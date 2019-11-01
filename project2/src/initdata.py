import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report

# Chosing optimal seed
seed = 42069
np.random.seed(seed)
random.seed(seed)


class InitData: # Class for initializing different data sets
    def __init__(self, path=None): # Chose path of 
        self.path = os.getcwd() # Get current path

        ## Categorical variables to one-hot's
        # Dividing every variable into 2 categories
        self.onehotencoder = OneHotEncoder(categories="auto")


    def credit_data(self, trainingShare): # Function for initializing credit card data
        print("loading Credit card data")
        
        # Read data as pandas dataframe from excel format
        self.filename = self.path + '/../data/default of credit card clients.xls'
        nanDict = {} # Empty dictionary for storing nanvalues
        self.df = pd.read_excel(self.filename, header = 1, skiprows=0, index_col=0, na_values=nanDict) #, nrows=1000) #faster
        self.df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)


        # Drop data points with no bill and payment info
        self.df = self.df.drop(self.df[(self.df.BILL_AMT1 == 0) &
                        (self.df.BILL_AMT2 == 0) &
                        (self.df.BILL_AMT3 == 0) &
                        (self.df.BILL_AMT4 == 0) &
                        (self.df.BILL_AMT5 == 0) &
                        (self.df.BILL_AMT6 == 0)].index)
        self.df = self.df.drop(self.df[(self.df.PAY_AMT1 == 0) &
                        (self.df.PAY_AMT2 == 0) &
                        (self.df.PAY_AMT3 == 0) &
                        (self.df.PAY_AMT4 == 0) &
                        (self.df.PAY_AMT5 == 0) &
                        (self.df.PAY_AMT6 == 0)].index)

        #the following removals are for data where some information is missing
        # Drop data points with marital status 0
        self.df = self.df.drop(self.df[(self.df.MARRIAGE == 0)].index)
        # Drop data points with education status 0, 5, or 6
        self.df = self.df.drop(self.df[(self.df.EDUCATION == 0)].index)
        self.df = self.df.drop(self.df[(self.df.EDUCATION == 5)].index)
        self.df = self.df.drop(self.df[(self.df.EDUCATION == 6)].index)

        # Note: following removals will remove the majority of the data (~26000 samples) 
        if (False): #the majority is here!
            # Drop data points with payment history equal to 0
            self.df = self.df.drop(self.df[(self.df.PAY_0 == 0)].index)
            self.df = self.df.drop(self.df[(self.df.PAY_2 == 0)].index)
            self.df = self.df.drop(self.df[(self.df.PAY_3 == 0)].index)
            self.df = self.df.drop(self.df[(self.df.PAY_4 == 0)].index)
            self.df = self.df.drop(self.df[(self.df.PAY_5 == 0)].index)
            self.df = self.df.drop(self.df[(self.df.PAY_6 == 0)].index)
        if (False):
            # Drop data points with payment history equal to -2
            self.df = self.df.drop(self.df[(self.df.PAY_0 == -2)].index)
            self.df = self.df.drop(self.df[(self.df.PAY_2 == -2)].index)
            self.df = self.df.drop(self.df[(self.df.PAY_3 == -2)].index)
            self.df = self.df.drop(self.df[(self.df.PAY_4 == -2)].index)
            self.df = self.df.drop(self.df[(self.df.PAY_5 == -2)].index)
            self.df = self.df.drop(self.df[(self.df.PAY_6 == -2)].index)

        # Target is last column (defaultpayment 0 or 1), features is everything else
        self.X = self.df.loc[:, self.df.columns != 'defaultPaymentNextMonth'].values
        self.y = self.df.loc[:, self.df.columns == 'defaultPaymentNextMonth'].values

        #Onehotencode column index 1,2 and 3 in data (gender, education and Marriage status)
        self.X = ColumnTransformer(
            [("", self.onehotencoder, [1,2,3]),],
            remainder="passthrough"
        ).fit_transform(self.X)
        
        
        # Train-test split
        self.trainingShare = trainingShare
        self.XTrain, self.XTest, self.yTrain, self.yTest=train_test_split(self.X, self.y, train_size=self.trainingShare, test_size = 1-self.trainingShare, random_state=seed)
        
        
        #%% Input Scaling
        sc = StandardScaler() # Scale to zero mean and unit variance
        self.XTrain = sc.fit_transform(self.XTrain) 
        self.XTest = sc.transform(self.XTest) 


        
        # One-hot's of the target vector
        self.Y_train_onehot, self.Y_test_onehot = self.onehotencoder.fit_transform(self.yTrain), self.onehotencoder.fit_transform(self.yTest)
        
        return self.XTrain, self.yTrain, self.XTest, self.yTest, self.Y_train_onehot, self.Y_test_onehot





