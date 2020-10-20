# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 12:31:06 2020

@author: David Wang
"""
#given code
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


#features and targets
Xtrain = np.linspace(0.,1.,10) # training set
Xvalid = np.linspace(0.,1.,100) # validation set
np.random.seed(3796)
tvalid = np.sin(4*np.pi*Xvalid) + 0.3 * np.random.randn(100)
ttrain = np.sin(4*np.pi*Xtrain) + 0.3 * np.random.randn(10)

#adding dummy feature to X
N = len(Xtrain) #number rows in train set
H= len(Xvalid) #rows in validation set

def matrixcreator(M,X,length): #this matrix generator will focus on training
    global X3train
    if M==0:
        X3train=np.ones((length,1)) #generating just dummy vector
        return X3train
    
    if M==1: #code to create a 2 dimensional matrix
        new_col=np.ones(length)
        X1train = np.insert(X, 0, new_col,) #insert ones into array of length 
        X2train=np.reshape(X1train,(-1,length))#reshape array in a 2x10matrix
        X3train=X2train.T #transpose to get final result
        return X3train
            
    if M>1: #code to keep adding columns 
        XtrainM=X**M        
        X3train = np.insert(X3train, M, XtrainM, axis=1)
        return X3train
    
def matrixcreator1(M,X,length):#this is a copy focused on validation
    global X4train
    if M==0:
        X4train=np.ones((length,1)) #generating just dummy vector
        return X4train
    
    if M==1: #code to create a 2 dimensional matrix
        new_col=np.ones(length)
        X1train = np.insert(X, 0, new_col,)
        X2train=np.reshape(X1train,(-1,length))
        X4train=X2train.T 
        return X4train
            
    if M>1: #code to keep adding columns 
        XtrainM=X**M        
        X4train = np.insert(X4train, M, XtrainM, axis=1)
        return X4train    

def Wcalculator(Xmatrix,Target): #perform (X^t*X)^-1*X^t*t
    global w
    A = np.dot(Xmatrix.T,Xmatrix)
    A1=np.linalg.inv(A)
    t1=np.dot(Xmatrix.T,Target)
    w = np.dot(A1,t1)
    return w

def predictor(X,type): #M(x) =w0+w1x+w2x2+...+wMxM
    global function
    print(w.shape)
    print(type.shape)  
    function = np.dot(type,w.T)
    plt.plot(X,function) #code used to graph predictor function
    # print(function)
    return function


def error(y, target,number): #computing average of total error
    diff = np.subtract(target, function)
    err = np.dot(diff,diff.T)/number
    print("the" ,y, "error is", err ,"for M as", M)
    
    # #-----activate this part for error graph----
    # if(y=="training"):
    #     plt.scatter(M,err,c='b')
    
    # if(y=="validation"):
    #     plt.scatter(M,err,c='r')

def trueplot():#plot for the true function
    y = np.sin(4*np.pi*Xvalid)
    plt.plot(Xvalid,y)
      
for M in range(0,10,1):
    plt.figure(M+1)     #creating 10 figures, uncomment for plot comparsion
    plt.title(M)  #title for plot comparison graphs
    plt.title("Errors vs M graph")
    #CODE FOR TRAINING
    matrixcreator(M,Xtrain,N)
    Wcalculator(X3train, ttrain)
    predictor(Xtrain, X3train)
    error("training",ttrain,N)
    
    #CODE FOR VALIDATION
    matrixcreator1(M,Xvalid,H)    #BLUE IS TRAINING, GREEN IS TRUE, ORANGE IS VALIDATION
    Wcalculator(X4train, tvalid)
    predictor(Xvalid, X4train)
    error("validation",tvalid,H)
    
    trueplot() #code use to compare to true function
