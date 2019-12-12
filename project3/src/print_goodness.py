import numpy as np
import os
from goodness import Goodness 

"""
Script that reads the predictions from file and outputs the goodness of fit parameters
in a Latex table format (for easy table setup), or print the full table in a reader 
friendly format for all files specified.

Added functionality to print a user specified prediction set.
"""

preds=['cifar10_DNN_adam_n10000_D128x128x64_pred_2019_Dec10_1554.dat',
       'cifar10_CNN_adam_n10000_D128x64_C3x3x3_pred_2019_Dec10_1643.dat',
       'cifar10_CNN_adam_n10000_D128x64_C3x3x3_KERAS_pred_2019_Dec11_1617.dat',
       'mnist_DNN_adam_n2000_D128x128x64_pred_2019_Dec10_1649.dat',
       'mnist_CNN_adam_n2000_D128x64_C3x3x3_pred_2019_Dec10_1530.dat',
       'mnist_CNN_adam_n10_D128x64_C3x3x3_KERAS_pred_2019_Dec12_1347.dat',
       'svhn_DNN_adam_n20000_D128x128x64_pred_2019_Dec10_1651.dat',
       'svhn_CNN_adam_n20000_D128x64_C3x3x3_pred_2019_Dec10_1712.dat',
       'svhn_CNN_adam_n20_D128x64_C3x3x3_KERAS_pred_2019_Dec12_1429.dat',
]

trues=['cifar10_DNN_adam_n10000_D128x128x64_true_2019_Dec10_1554.dat',
       'cifar10_CNN_adam_n10000_D128x64_C3x3x3_true_2019_Dec10_1643.dat',
       'cifar10_CNN_adam_n10000_D128x64_C3x3x3_KERAS_true_2019_Dec11_1617.dat',
       'mnist_DNN_adam_n2000_D128x128x64_true_2019_Dec10_1649.dat',
       'mnist_CNN_adam_n2000_D128x64_C3x3x3_true_2019_Dec10_1530.dat',
       'mnist_CNN_adam_n10_D128x64_C3x3x3_KERAS_true_2019_Dec12_1347.dat',
       'svhn_DNN_adam_n20000_D128x128x64_true_2019_Dec10_1651.dat',
       'svhn_CNN_adam_n20000_D128x64_C3x3x3_true_2019_Dec10_1712.dat',
       'svhn_CNN_adam_n20_D128x64_C3x3x3_KERAS_true_2019_Dec12_1429.dat',
]
good=Goodness()


var = input('Print goodness-of-fit to LaTex table for predefined data? [N/y]: ')

if (var =='y' or var=='Y'):
    for i in range(3):
        acc=np.zeros(3)
        f1=np.zeros((3,10))
        f1w=np.zeros(3)
        ar=np.zeros((3,10))
        arw=np.zeros(3)
        cp=np.zeros((3,10))
        for j in range(3):
            ypred=np.loadtxt('../data/'+preds[i*3+j])
            if ypred.shape[0]==10:
                ypred=ypred.T
            ytrue=np.loadtxt('../data/'+trues[i*3+j])
            if ytrue.shape[0]==10:
                ytrue=ytrue.T
            print('    Computing goodness-of-fit for: '+preds[i*3+j])

            acc[j],f1[j,:],f1w[j],ar[j,:],arw[j],cp[j,:]=good.classification_report(ytrue,ypred,return_ac=True,return_f1=True,return_f1_weight=True,return_ar=True,return_ar_weight=True, return_cp=True,print_res=False)

        print()
        # print table values for latex tables to file
        if ('cifar10' in preds[i*3]):
            label='cifar10'
        elif ('mnist' in preds[i*3]):
            label='mnist'
        elif ('svhn' in preds[i*3]):
            label='svhn'
        else:
            label='other'
        label+='_latex'
        filename = "../goodness_parameters/parameters_goodness_" + label + ".txt"
        out = open(filename, "w")
        out.write("class  targets  |       F1-score    |      AR      \n")
        out.write("                |    DNN       CNN  |  DNN      CNN      \n")
        out.write("---------------------------------------------------------------------\n")
        for j in range(10):
            out.write(
                "%3i & %5i  &  %7.4f &  %7.4f & %7.4f &  %7.4f &  %7.4f &  %7.4f \n"
                % (j, int(cp[0,j]), f1[0,j], f1[1,j], f1[2,j], ar[0,j], ar[1,j], ar[2,j]))
        out.write("\n")
        out.write(
            "     CP weighted average (%5i) &  %7.4f & %7.4f &  %7.4f &  %7.4f &  %7.4f &  %7.4f\n" % (int(np.sum(cp[0,:])),f1w[0],f1w[1],f1w[2],arw[0],arw[1],arw[2])
            )
        out.write("---------------------------------------------------------------------\n")
        out.close()


var = input('Print goodness-of-fit to terminal for predefined data? [N/y]: ')
if (var =='y' or var=='Y'):
    for i in range(len(preds)):
        print()
        if ('cifar10' in preds[i]):
            print('CIFAR-10')
        elif ('mnist' in preds[i]):
            print('MNIST')
        elif ('svhn' in preds[i]):
            print('SVHN')
        if ('keras' in preds[i] or 'KERAS' in preds[i]):
            print('KERAS')
        if ('DNN' in preds[i]):
            print('DNN')
        elif ('CNN' in preds[i]):
            print('CNN')
        ypred=np.loadtxt('../data/'+preds[i])
        if ypred.shape[0]==10:
            ypred=ypred.T
        ytrue=np.loadtxt('../data/'+trues[i])
        if ytrue.shape[0]==10:
            ytrue=ytrue.T
        print('--------------------------------------')
        good.classification_report(ytrue,ypred)
        print()

    print()
    print()

var = input('Print goodness-of-fit of other data to terminal? [N/y]: ')
while (var =='y' or var=='Y'):
    print()
    print('Input filenames for the files containing arrays of the class ')
    print('predictions and the one-hot encoded true class array')
    print()
    fpred = input('Predictions data filename: ')
    ftrue = input('True data filename: ')
    if ('cifar10' in fpred):
        print('CIFAR-10')
    elif ('mnist' in fpred):
        print('MNIST')
    elif ('svhn' in fpred):
        print('SVHN')
    if ('keras' in fpred or 'KERAS' in fpred):
        print('KERAS')
    if ('DNN' in fpred):
        print('DNN')
    elif ('CNN' in fpred):
        print('CNN')

    #read predictions data
    if ('../data/' in fpred):
        ypred=np.loadtxt(fpred)
    else:
        ypred=np.loadtxt('../data/'+fpred)
    if ypred.shape[0]==10: #transpose if images are in columns and classes in rows
        ypred=ypred.T

    if ('../data/' in fpred):
        ytrue=np.loadtxt(ftrue)
    else:
        ytrue=np.loadtxt('../data/'+ftrue)
    if ytrue.shape[0]==10: #transpose if images are in columns and classes in rows
        ytrue=ytrue.T
    print('--------------------------------------')
    good.classification_report(ytrue,ypred)
    print()
    print()
    var = input('Print goodness-of-fit of other data to terminal? [N/y]: ')
exit()
