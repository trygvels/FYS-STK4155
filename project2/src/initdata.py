import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
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

    # Function for initializing credit card data
    def credit_data(self, trainingShare, drop_zero=False,drop_neg2=False, per_col=False, exclude_col=['none'],plot_alldata=False, return_cols=False, onehot_encode_col=['SEX','EDUCATION','MARRIAGE']):


        print("loading Credit card data")

        onehot_encode_col=np.array(onehot_encode_col)
        
        # Read data as pandas dataframe from excel format
        self.filename = self.path + '/../data/default of credit card clients.xls'
        nanDict = {} # Empty dictionary for storing nanvalues
        self.df = pd.read_excel(self.filename, header = 1, skiprows=0, index_col=0, na_values=nanDict) #, nrows=1000) #faster
        self.df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

        
        if (plot_alldata):
            #plot data distributions of all the credit card data
            self.plot_credit_data('_alldata')
            
        excl_cols=[]
        # Check if we are to exclude any columns
        if (not exclude_col[0]=='none'):
            for j in range(len(exclude_col)):
                for k in range(len(self.df.columns)):
                    if (exclude_col[j]==self.df.columns[k]):
                        excl_cols.append(k)
            if (len(excl_cols)>0):
                excl_cols=np.sort(excl_cols) #get columns in rising indices
                excl_cols=np.array(excl_cols) #make it a numpy array

        if (return_cols):
            # Make the data string array
            self.data_cols=self.df.columns[:-1].copy()
            if (len(excl_cols)>0):
                for i in np.arange(len(excl_cols)-1,-1,-1,dtype='int'):
                    self.data_cols=np.concatenate((self.data_cols[:excl_cols[i]],self.data_cols[excl_cols[i]+1:]))
            first_cols=[]
            i=0
            if (self.data_cols[0]=='LIMIT_BAL'):
                j=1
            else:
                j=0
            if (np.any(self.data_cols=='SEX')):
                if (np.any(onehot_encode_col=='SEX')):
                    first_cols.append('SEX_male')
                    first_cols.append('SEX_female')
                    i+=1
                    self.data_cols=np.concatenate((self.data_cols[:j],self.data_cols[j+1:]))
                else:
                    j+=1
            if np.any(self.data_cols=='EDUCATION'):
                if (np.any(onehot_encode_col=='EDUCATION')):
                    first_cols.append('EDUCATION_grad_school')
                    first_cols.append('EDUCATION_university')
                    first_cols.append('EDUCATION_high_school')
                    first_cols.append('EDUCATION_other')
                    i+=1
                    self.data_cols=np.concatenate((self.data_cols[:j],self.data_cols[j+1:]))
                else:
                    j+=1
            if np.any(self.data_cols=='MARRIAGE'):
                if (np.any(onehot_encode_col=='MARRIAGE')):
                    first_cols.append('MARRIAGE_married')
                    first_cols.append('MARRIAGE_single')
                    first_cols.append('MARRIAGE_other')
                    i+=1
                    self.data_cols=np.concatenate((self.data_cols[:j],self.data_cols[j+1:]))
            if (i>0):
                self.data_cols=np.concatenate((first_cols,self.data_cols))
            pay_cols=['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
            last_cols=[]
            if (not drop_zero):
                if (per_col):
                    for i in range(6):
                        if (np.any(self.data_cols==pay_cols[i])):
                            last_cols.append(pay_cols[i]+'_flag0')
                else:
                    any_pay=False
                    for i in range(6):
                        if (np.any(self.data_cols==pay_cols[i])):
                            any_pay=True
                    if (any_pay):
                        last_cols.append('PAY_flag0')
            if (not drop_neg2):
                if (per_col):
                    for i in range(6):
                        if (np.any(self.data_cols==pay_cols[i])):
                            last_cols.append(pay_cols[i]+'_flag_neg2')
                else:
                    any_pay=False
                    for i in range(6):
                        if (np.any(self.data_cols==pay_cols[i])):
                            any_pay=True
                    if (any_pay):
                        last_cols.append('PAY_flag_neg2')
            if (len(last_cols)>0):
                self.data_cols=np.concatenate((self.data_cols,last_cols))            

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
        # We must also check if we are to remove any column, i.e. not throw out data based
        # on columns we are not to use
        if (drop_zero): #the majority is here!
            # Drop data points with payment history equal to 0
            # if the column is not to be removed
            if (not np.any(excl_cols==5)):
                self.df = self.df.drop(self.df[(self.df.PAY_0 == 0)].index)
            if (not np.any(excl_cols==6)):
                self.df = self.df.drop(self.df[(self.df.PAY_2 == 0)].index)
            if (not np.any(excl_cols==7)):
                self.df = self.df.drop(self.df[(self.df.PAY_3 == 0)].index)
            if (not np.any(excl_cols==8)):
                self.df = self.df.drop(self.df[(self.df.PAY_4 == 0)].index)
            if (not np.any(excl_cols==9)):
                self.df = self.df.drop(self.df[(self.df.PAY_5 == 0)].index)
            if (not np.any(excl_cols==10)):
                self.df = self.df.drop(self.df[(self.df.PAY_6 == 0)].index)
        if (drop_neg2):
            # Drop data points with payment history equal to -2
            # if the column is not to be removed
            if (not np.any(excl_cols==5)):
                self.df = self.df.drop(self.df[(self.df.PAY_0 == -2)].index)
            if (not np.any(excl_cols==6)):
                self.df = self.df.drop(self.df[(self.df.PAY_2 == -2)].index)
            if (not np.any(excl_cols==7)):
                self.df = self.df.drop(self.df[(self.df.PAY_3 == -2)].index)
            if (not np.any(excl_cols==8)):
                self.df = self.df.drop(self.df[(self.df.PAY_4 == -2)].index)
            if (not np.any(excl_cols==9)):
                self.df = self.df.drop(self.df[(self.df.PAY_5 == -2)].index)
            if (not np.any(excl_cols==10)):
                self.df = self.df.drop(self.df[(self.df.PAY_6 == -2)].index)

        # Target is last column (defaultpayment 0 or 1), features is everything else
        self.X = self.df.loc[:, self.df.columns != 'defaultPaymentNextMonth'].values
        self.y = self.df.loc[:, self.df.columns == 'defaultPaymentNextMonth'].values

        # If we choose not to exclude '0' or '-2' in PAY_i, we try to add extra columns to
        # classify the data containing these flags. Two options: (1) 1 column per flag for 
        # all 'PAY_i' columns combines (i.e. 2 new columns), or (2) 1 column per flag per
        #'PAY_i' column (i.e. up to 12 new columns)

        # we must check if we are to exclude any of the payment history columns
        incl_col=[]
        for i in range(5,11):
            if (not np.any(excl_cols==i)):
                incl_col.append(i)

        if (not drop_zero):
            if len(incl_col)>0:
                if (per_col):
                    #find where X == 0 in the 'PAY_i' columns 
                    Xtemp=np.where(self.X[:,incl_col] == 0, 1, 0)
                else:
                    Xtemp=np.zeros(shape=(self.X.shape[0],1),dtype='int')
                    for i in range(self.X.shape[0]):
                        if (np.any(self.X[i,incl_col] == 0)):
                            Xtemp[i,0]=1
                #merge the arrays (append at the end)
                self.X = np.concatenate((self.X,Xtemp),axis=1)

        if (not drop_neg2):
            if len(incl_col)>0:
                if (per_col):
                    #find where X == 0 in the 'PAY_i' columns 
                    Xtemp=np.where(self.X[:,incl_col] == -2, 1, 0)
                else:
                    Xtemp=np.zeros(shape=(self.X.shape[0],1),dtype='int')
                    for i in range(self.X.shape[0]):
                        if (np.any(self.X[i,incl_col] == -2)):
                            Xtemp[i,0]=1
                #merge the arrays (append at the end)
                self.X = np.concatenate((self.X,Xtemp),axis=1)

        #exclude columns
        if (not exclude_col[0]=='none'):
            if len(excl_cols) > 0:
                for j in np.arange(len(excl_cols)-1,-1,-1): #removing given columns, starting from the last column
                    self.X=np.concatenate((self.X[:,0:excl_cols[j]],self.X[:,excl_cols[j]+1:]),axis=1)

        #We need to find the column numbers of the columns to be onehot encoded 
        onehot_col=[]
        for i in range(len(onehot_encode_col)):
            for j in range(len(self.df.columns)):
                if (self.df.columns[j]==onehot_encode_col[i]):
                    onehot_col.append(j)
        if (len(onehot_col)>0):
            onehot_col=np.sort(onehot_col)
        #check if any of the onehot-encoded columns have been removed.
        # if yes, remove the column and shift the columns of higher indices 1 lower.
        if len(excl_cols) > 0:
            i=0
            j=0
            for k in np.arange(0,len(self.df.columns)):
                if ((i<len(onehot_col)) and (j<len(excl_cols))):
                    if (k==excl_cols[j]):
                        if (k==onehot_col[i]):
                            onehot_col=np.concatenate((onehot_col[:i],onehot_col[i+1:]-1))
                            i += 1
                        else:
                            onehot_col=np.where(onehot_col>excl_cols[j],onehot_col-1,onehot_col)
                            j+=1
                        
        #Onehotencode column index 1,2 and 3 in data (gender, education and Marriage status)
        if len(onehot_col)>0:
            print(onehot_col)
            print(self.data_cols)
            self.X = ColumnTransformer(
                [("", self.onehotencoder, onehot_col),],
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

        if (not return_cols):
            return self.XTrain, self.yTrain, self.XTest, self.yTest, self.Y_train_onehot, self.Y_test_onehot
        else:
            return self.XTrain, self.yTrain, self.XTest, self.yTest, self.Y_train_onehot, self.Y_test_onehot, self.data_cols

    def franke_data(self, N=20, noise=0.1, trainingShare = 0.5, degree=5):
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)
        x_, y_ = np.meshgrid(x,y)

        data = np.c_[(x_.ravel()).T,(y_.ravel()).T]
        data = pd.DataFrame(data)

        # Create and transform franke function data
        b = self.FrankeFunction(x_, y_) + np.random.normal(size=x_.shape)*noise # Franke function with optional gaussian noise
        y = b.ravel()

        # Create design matrix with polynomial features
        poly = PolynomialFeatures(degree=degree) 
        X = poly.fit_transform(data) 

        XTrain, XTest, yTrain, yTest = train_test_split(X,y,test_size=1-trainingShare, shuffle=True)
        return XTrain, yTrain, XTest, yTest
        
    def FrankeFunction(self, x, y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

    def plot_credit_data(self,label):
        #plot sex
        Xtemp = self.df.loc[:, self.df.columns == 'SEX'].values
        ticks=['male','female']
        xval=[1,2]
        plt.figure(1)
        plt.hist(Xtemp,range=(0.5,2.5),bins=2,rwidth=0.8,align='mid', color='b')
        plt.yticks(fontsize=12)
        plt.xticks(xval,labels=ticks,fontsize=12)
        plt.xlabel('Gender',fontsize=14)
        plt.ylabel('Observations count',fontsize=14)
        plt.savefig('plots/gender'+label+'.pdf',bbox_inches='tight',pad_inches=0.02)
        plt.clf()
        
        #plot marriage
        Xtemp = self.df.loc[:, self.df.columns == 'MARRIAGE'].values
        ticks=['missing','married','single','other']
        xval=[0,1,2,3]
        xval_str=['0','1','2','3']
        plt.figure(1)
        plt.hist(Xtemp,range=(-0.5,3.5),bins=4,rwidth=0.8,align='mid', color='b')
        plt.yticks(fontsize=12)
        plt.xticks(xval,labels=ticks,fontsize=12)
        plt.xlabel('Marriage status',fontsize=14)
        plt.ylabel('Observations count',fontsize=14)
        plt.savefig('plots/marriage'+label+'.pdf',bbox_inches='tight',pad_inches=0.02)
        plt.xticks(xval,labels=xval_str,fontsize=12)
        plt.savefig('plots/marriage_val'+label+'.pdf',bbox_inches='tight',pad_inches=0.02)
        plt.clf()

        #plot education
        Xtemp = self.df.loc[:, self.df.columns == 'EDUCATION'].values
        ticks=['flag 0','grad. sch.','university','high sch.','other','flag 5', 'flag 6']
        xval=[0,1,2,3,4,5,6]
        xval_str=['0','1','2','3','4','5','6']
        plt.figure(1)
        plt.hist(Xtemp,range=(-0.5,6.5),bins=7,rwidth=0.8,align='mid', color='b')
        plt.yticks(fontsize=12)
        plt.xticks(xval,labels=ticks,fontsize=12)
        plt.xlabel('Education',fontsize=14)
        plt.ylabel('Observations count',fontsize=14)
        plt.savefig('plots/education'+label+'.pdf',bbox_inches='tight',pad_inches=0.02)
        plt.xticks(xval,labels=xval_str,fontsize=12)
        plt.savefig('plots/education_val'+label+'.pdf',bbox_inches='tight',pad_inches=0.02)
        plt.clf()

        #plot age
        Xtemp = self.df.loc[:, self.df.columns == 'AGE'].values
        xval=[20,40,60,80]
        xval_str=['20','40','60','80']
        plt.figure(1)
        maxage=np.amax(Xtemp[:,0])
        minage=np.amin(Xtemp[:,0])
        plt.hist(Xtemp,range=(minage-0.5,maxage+0.5),bins=(maxage-minage+1),rwidth=1.0,align='mid', color='k')
        plt.yticks(fontsize=12)
        plt.xticks(xval,labels=xval_str,fontsize=12)
        plt.xlabel('Age',fontsize=14)
        plt.ylabel('Observations count',fontsize=14)
        plt.savefig('plots/age'+label+'.pdf',bbox_inches='tight',pad_inches=0.02)
        plt.clf()

        #plot pay history
        for i in ['0','2','3','4','5','6']:
            Xtemp = self.df.loc[:, self.df.columns == 'PAY_'+i].values
            xval=[-2,-1,0,1,2,3,4,5,6,7,8,9]
            xval_str=['-2','-1','0','1','2','3','4','5','6','7','8','9']
            plt.figure(1)
            plt.hist(Xtemp,range=(-2.5,9.5),bins=12,rwidth=0.8,align='mid', color='k')
            plt.yticks(fontsize=12)
            plt.xticks(xval,labels=xval_str,fontsize=12)
            plt.xlabel('Payment history (PAY_'+i+')',fontsize=14)
            plt.ylabel('Observations count',fontsize=14)
            plt.savefig('plots/pay'+i+label+'.pdf',bbox_inches='tight',pad_inches=0.02)
            plt.clf()

