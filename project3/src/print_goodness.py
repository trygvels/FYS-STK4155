import numpy as np
import os
from goodness import Goodness 

"""
Script that reads the predictions from file and outputs the goodness of fit parameters
"""

preds=['cifar10_DNN_adam_n10000_D128x128x64_pred_2019_Dec10_1554.dat',
       'cifar10_CNN_adam_n10000_D128x64_C3x3x3_pred_2019_Dec10_1643.dat',
       'mnist_DNN_adam_n2000_D128x128x64_pred_2019_Dec10_1649.dat',
       'mnist_CNN_adam_n2000_D128x64_C3x3x3_pred_2019_Dec10_1530.dat',
       'svhn_DNN_adam_n20000_D128x128x64_pred_2019_Dec10_1651.dat',
       'svhn_CNN_adam_n20000_D128x64_C3x3x3_pred_2019_Dec10_1712.dat',
       'cifar10_CNN_adam_n10000_D128x64_C3x3x3_KERAS_pred_2019_Dec11_1617.dat'
]

trues=['cifar10_DNN_adam_n10000_D128x128x64_true_2019_Dec10_1554.dat',
       'cifar10_CNN_adam_n10000_D128x64_C3x3x3_true_2019_Dec10_1643.dat',
       'mnist_DNN_adam_n2000_D128x128x64_true_2019_Dec10_1649.dat',
       'mnist_CNN_adam_n2000_D128x64_C3x3x3_true_2019_Dec10_1530.dat',
       'svhn_DNN_adam_n20000_D128x128x64_true_2019_Dec10_1651.dat',
       'svhn_CNN_adam_n20000_D128x64_C3x3x3_true_2019_Dec10_1712.dat',
       'cifar10_CNN_adam_n10000_D128x64_C3x3x3_KERAS_true_2019_Dec11_1617.dat'
]

good=Goodness()
for i in range(3):
    acc=np.zeros(3)
    f1=np.zeros((3,10))
    f1w=np.zeros(3)
    ar=np.zeros((3,10))
    arw=np.zeros(3)
    cp=np.zeros((3,10))
    for j in range(3):
        if (j==2):
            if (not 'cifar10' in preds[i*2]):
                break

        if ('cifar10' in preds[i*2]):
            print('CIFAR-10')
        elif ('mnist' in preds[i*2]):
            print('MNIST')
        elif ('svhn' in preds[i*2]):
            print('SVHN')

        if (j==2):
            print('KERAS')
        elif ('DNN' in preds[i*2+j]):
            print('DNN')
        elif ('CNN' in preds[i*2+j]):
            print('CNN')
        if (j==2):
            ypred=np.loadtxt('../data/'+preds[-1])
            if ypred.shape[0]==10:
                ypred=ypred.T
            ytrue=np.loadtxt('../data/'+trues[-1])
            if ytrue.shape[0]==10:
                ytrue=ytrue.T

        else:
            ypred=np.loadtxt('../data/'+preds[i*2+j])
            if ypred.shape[0]==10:
                ypred=ypred.T
            ytrue=np.loadtxt('../data/'+trues[i*2+j])
            if ytrue.shape[0]==10:
                ytrue=ytrue.T

        acc[j],f1[j,:],f1w[j],ar[j,:],arw[j],cp[j,:]=good.classification_report(ytrue,ypred,return_ac=True,return_f1=True,return_f1_weight=True,return_ar=True,return_ar_weight=True, return_cp=True)
        print()

    # print table values for latex tables to file
    keras=False
    if ('cifar10' in preds[i*2]):
        label='cifar10'
        keras=True
    elif ('mnist' in preds[i*2]):
        label='mnist'
    elif ('svhn' in preds[i*2]):
        label='svhn'
    else:
        label='other'
    label+='_latex'
    filename = "goodness_parameters/parameters_goodness_" + label + ".txt"
    out = open(filename, "w")
    out.write("class  targets  |       F1-score    |      AR      \n")
    out.write("                |    DNN       CNN  |  DNN      CNN      \n")
    out.write("---------------------------------------------------------------------\n")
    for j in range(10):
        if (keras==True):
            out.write(
                "%3i & %5i  &  %7.4f &  %7.4f & %7.4f &  %7.4f &  %7.4f &  %7.4f \n"
                % (j, int(cp[0,j]), f1[0,j], f1[1,j], f1[2,j], ar[0,j], ar[1,j], ar[2,j])
            )
        else:
            out.write(
                "%3i & %5i  &  %7.4f &  %7.4f &  %7.4f &  %7.4f \n"
                % (j, int(cp[0,j]), f1[0,j], f1[1,j], ar[0,j], ar[1,j])
            )
    out.write("\n")
    if (keras==True):
        out.write(
            "     CP weighted average (%5i) &  %7.4f & %7.4f &  %7.4f &  %7.4f &  %7.4f &  %7.4f\n" % (int(np.sum(cp[0,:])),f1w[0],f1w[1],f1w[2],arw[0],arw[1],arw[2])
        )
    else:
        out.write(
            "     CP weighted average (%5i) &  %7.4f & %7.4f &  %7.4f &  %7.4f\n" % (int(np.sum(cp[0,:])),f1w[0],f1w[1],arw[0],arw[1])
        )
        
    out.write("---------------------------------------------------------------------\n")
    out.close()





exit()
