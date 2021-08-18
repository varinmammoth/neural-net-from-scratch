# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:53:30 2021

@author: Pieumsook
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1+np.exp(-1*z))

def forward_prop3(X, theta1, theta2, theta3):
    #forward propagation
    #X is a 'm by n' matrix
        #m = number of examples
        #n = number of features 
    a1 = np.transpose(X)
    
    z2 = np.matmul(theta1, a1)
    a2 = sigmoid(z2)
    a2 = np.append(np.ones((1,a2.shape[1])),a2,axis=0)
    
    z3 = np.matmul(theta2, a2)
    a3 = sigmoid(z3)
    a3 = np.append(np.ones((1,a3.shape[1])),a3,axis=0)
    
    z4 = np.matmul(theta3, a3)
    a4 = sigmoid(z4)
   
    return a1, a2, a3, a4

def backward_prop3(y_vectors, a1, a2, a3, a4, theta1, theta2, theta3, lamb):
    #backward propagation
    #y_vectors is vectors of results (see notes for clarification)
    #outputs gradiat arrays for theta1 and theta2
    
    m = y_vectors.shape[1]
    
    delta4 = a4 - y_vectors
    
    delta3_matmul_term = np.matmul(np.transpose(theta3),delta4)
    delta3_dot_term = np.multiply(a3, np.ones(a3.shape)-a3)
    delta3 = np.multiply(delta3_matmul_term,delta3_dot_term)
    triangle3 = np.matmul(delta4, np.transpose(a3))
    
    delta2_matmul_term = np.matmul(np.transpose(theta2),delta3)
    delta2_dot_term = np.multiply(a2, np.ones(a2.shape)-a2)
    delta2 = np.multiply(delta2_matmul_term,delta2_dot_term)
    triangle2 = np.matmul(delta3, np.transpose(a2))
    
    triangle1 = np.matmul(delta2[1:,:],np.transpose(a1))
    
    reg3 = np.zeros((theta3.shape[0],1))
    reg3 = np.append(reg3,theta3[:,1:],axis=1)
    grad3 = (1/m)*triangle3 + lamb*reg3
    
    reg2 = np.zeros((theta2.shape[0],1))
    reg2 = np.append(reg2,theta2[:,1:],axis=1)
    grad2 = (1/m)*triangle2 + lamb*reg2
    
    reg1 = np.zeros((theta1.shape[0],1))
    reg1 = np.append(reg1,theta1[:,1:],axis=1)
    grad1 = (1/m)*triangle1 + lamb*reg1
    
    return grad1, grad2, grad3

def grad_des3(theta1, theta2, theta3, grad1, grad2, grad3, alpha):
    #gradiat descent
    theta1 = theta1 - alpha*grad1
    theta2 = theta2 - alpha*grad2
    theta3 = theta3 - alpha*grad3
    return theta1, theta2, theta3



