# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 12:21:05 2020

@author: David Wang
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_boston
#boston = load_boston()
#print(boston.DESCR)

#import data set from scikit
from sklearn.datasets import load_boston 
X, t = load_boston(return_X_y=True)
#506 rows (entries) 13 columns (features)->X
#506 rows 1D vector of targets

# split data into trainig and test sets
from sklearn.model_selection import train_test_split
X, X_test, t, t_test = train_test_split(X, t, test_size = 1/4, random_state = 3796)
# print(X_train.shape) # X_train is 2D, but y_train is 1D
#print(t_train.shape)
M = len(X_test) #number rows in test set
N = len(X) #number rows in train set

print(X.shape)
# N=len(X)  # number of rows
# print(N)
# print(t.shape)

SX_Temp=np.ones((N,1)) #generating just dummy vector 503x1
currenterr=9999 #some huge number
##----FUNCTIONS-----

def Wcalculator(Xmatrix,t): #perform (X^t*X)^-1*X^t*t
    global w
    A = np.dot(Xmatrix.T,Xmatrix)
    A1=np.linalg.inv(A)
    t1=np.dot(Xmatrix.T,t)
    w = np.dot(A1,t1)
    # print(w.shape)
    return w

def predictor(Xmatrix): #M(x) =w0+w1x+w2x2+...+wMxM
    global function
    function = np.dot(Xmatrix,w)
    # print(function)
    return function


def error(t,length): #computing average of total error
    global err
    diff = np.subtract(t, function)
    err = np.dot(diff,diff.T)/N
    print(err)


def updater(chosen):#used for most S cases
    global X
    global SX
    if(S==0):
        Extract=X[:,chosen] #extracting selected feature to appen to current feature lis
        #Extract=Extract**2 #basis expansion x^2 activate for all to use
        Extract=Extract**0.5 #basis expansion of root x
        new_col=np.ones(N) # this is for case S=1
        SX_Temp=np.insert(Extract, 0, new_col,)
        SX_Temp=np.reshape(SX_Temp,(-1,N))
        SX=SX_Temp.T
        X=np.delete(X,chosen,1)#deleting selected feature
        
    if(S>0):
        Extract=X[:,chosen] #extracting selected feature to appen to current feature list  
        #Extract=Extract**2 #basis expansion x^2 activate for all to use
        Extract=Extract**0.5 #basis expansion of root x
        SX=np.insert(SX,S+1, Extract,axis=1)
        X=np.delete(X,chosen,1)#deleting selected feature


    
def matrixcreator():
    global S
    global SX_Temp
    if (S==0):
        Extract=X[:,k] #extracting a column from the data
        #Extract=Extract**2 #basis expansion x^2 activate for all to use
        Extract=Extract**0.5 #basis expansion of root x        
        new_col=np.ones(N) # this is for case S=1
        SX_Temp=np.insert(Extract, 0, new_col,)
        SX_Temp=np.reshape(SX_Temp,(-1,N))
        SX_Temp=SX_Temp.T #getting first feature + dummy
    
    if (S>0):
        Extract=X[:,k] #extracting a column from the data
        #Extract=Extract**2 #basis expansion x^2 activate for all to use
        Extract=Extract**0.5 #basis expansion of root x
        SX_Temp=np.insert(SX,S+1, Extract,axis=1)
        
def trainingupdate(chosen):
    global SX_Temp_test
    global X_test
    global SX_test
    if(S==0):
        Extract=X_test[:,chosen] #extracting selected feature to appen to current feature lis
        #Extract=Extract**2 #basis expansion x^2
        Extract=Extract**0.5 #basis expansion of root x
        new_col=np.ones(M) # this is for case S=1
        SX_Temp_test=np.insert(Extract, 0, new_col,)
        SX_Temp_test=np.reshape(SX_Temp_test,(-1,M))
        SX_test=SX_Temp_test.T
        X_test=np.delete(X_test,chosen,1)#deleting selected feature
        
    if(S>0):
        Extract=X_test[:,chosen] #extracting selected feature to appen to current feature list
        #Extract=Extract**2 #basis expansion x^2
        Extract=Extract**0.5 #basis expansion of root x
        SX_test=np.insert(SX_test,S+1, Extract,axis=1)
        X_test=np.delete(X_test,chosen,1)#deleting selected feature  
    
    
#main algo


for S in range(0,13,1):
    print("*********S=",S,"*************")
    plt.figure(S)     #creating 10 figures
    plt.title(S)

    for k in range(0,13-S,1):   
        matrixcreator()
        
        Wcalculator(SX_Temp,t) #trying out features
        predictor(SX_Temp)
        error(t,N)
        plt.scatter(k,err)
        if (err< currenterr):
            currenterr=err
            chosen=k
    print("this is the min error,", currenterr, "at iteration k=,", chosen)  
    updater(chosen) #feature selection
    # print(X.shape)
    # print(SX.shape)
    # print(SX)
    trainingupdate(chosen) #finding training error
    Wcalculator(SX_test,t_test)
    predictor(SX_test)
    error(t_test,M)
    plt.scatter(k+1,err)
   
    currenterr=9999 #some huge number used to reset error
    
                    

                
        
    
        