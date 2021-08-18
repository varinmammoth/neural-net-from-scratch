#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 14:21:31 2021

@author: mammonth
"""

import numpy as np
import matplotlib.pyplot as plt

#the 2 in function names denotes 2 layers (not including dataset features)

#See the link:
    #https://medium.com/secure-and-private-ai-math-blogging-competition/https-medium-com-fadymorris-understanding-vectorized-implementation-of-neural-networks-dae4115ca185

def sigmoid(z):
    return 1/(1+np.exp(-1*z))

def y_vector_single(y_i, num=10):
    #y_i is the outcome
    #num is the total number of outcomes, default is 10
    y = np.zeros(num)
    if y_i == 10:
        #10 used to signify 0 for this particuliar dataset
        y[0] = 1
    else:
        y[int(y_i)] = 1
    return y

def y_vector(y, num=10):
    #y is list of results
    #num is the total number of outcomes, default is 10
    y_out = np.zeros((num,1))
    for i in y:
        y_out = np.append(y_out, y_vector_single(i, num).reshape(num, 1), axis=1)
    return y_out[:,1:]  

def y_vector_result(y):
    for i in range (0, len(y)):
        if y[i] == 1:
            result = i
    return result

def y_vector_to_list(y_vector):
    #converts the final output of neural network back to list of results
    m = y_vector.shape[1]
    y_list = []
    for i in range (0, m):
        max_index = np.argmax(y_vector[:,i])
        y_list.append(max_index)
        
    result = np.asarray(y_list)
    result = result.reshape(len(y_list),1)
    
    return result

def get_accuracy(y_real, y_predict):
    #takes in two (m,1) np arrays
    #y_real is actual results
    #y_prediction is final output of neural network
    #note: this particuliar function is for predicting digits only
        #need to rewrite this function for other application
    m = y_real.shape[0]
    array = np.abs(y_real - y_predict)
    score = 0
    for i in array:
        if i == 0:
            score = score + 1
        elif i == 10:
            score = score + 1
    return score/m

def forward_prop2(X, theta1, theta2):
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
   
    return a1, a2, a3

def backward_prop2(y_vectors, a1, a2, a3, theta1, theta2, lamb):
    #backward propagation
    #y_vectors is vectors of results (see notes for clarification)
    #outputs gradiat arrays for theta1 and theta2
    
    m = y_vectors.shape[1]
    
    delta3 = a3 - y_vectors
    
    delta2_matmul_term = np.matmul(np.transpose(theta2),delta3)
    delta2_dot_term = np.multiply(a2, np.ones(a2.shape)-a2)
    delta2 = np.multiply(delta2_matmul_term,delta2_dot_term)
    
    triangle2 = np.matmul(delta3, np.transpose(a2))
    
    triangle1 = np.matmul(delta2[1:,:],np.transpose(a1))
    
    reg2 = np.zeros((theta2.shape[0],1))
    reg2 = np.append(reg2,theta2[:,1:],axis=1)
    
    grad2 = (1/m)*triangle2 + lamb*reg2
    
    reg1 = np.zeros((theta1.shape[0],1))
    reg1 = np.append(reg1,theta1[:,1:],axis=1)
    
    grad1 = (1/m)*triangle1 + lamb*reg1
    
    return grad1, grad2

def grad_des(theta1, theta2, grad1, grad2, alpha):
    #gradiat descent
    theta1 = theta1 - alpha*grad1
    theta2 = theta2 - alpha*grad2
    return theta1, theta2
    
def J2(y_vectors, a3, theta1, theta2, lamb):
    #cost function J
    
    m = y_vectors.shape[1]
    
    cost = np.multiply(y_vectors, np.log(a3)) + np.multiply((1-y_vectors),np.log(1-a3))
    cost = np.sum(cost)
    cost = (-1/m)*cost
    
    reg = np.sum(np.square(theta1)) + np.sum(np.square(theta2))
    reg = (lamb/2*m)*reg
    
    J = cost + reg
    return J
    
def test_train_split(X, y):
    #takes X array as input (with 1s added)
    #takes y column vector as input
    #splits into training and testing set 
    #50-50 split
    
    X_train = np.zeros((1, X.shape[1]))
    X_test = np.zeros((1, X.shape[1]))
    y_train = np.zeros((1,1))
    y_test = np.zeros((1,1))
    
    for i in range (0,X.shape[0]):
        if i%2 == 0:
            X_append = X[i,:].reshape(1, X[i,:].shape[0])
            X_train = np.append(X_train, X_append ,axis=0)
            y_train = np.append(y_train, [y[i]] ,axis=0)
        else:
            X_append = X[i,:].reshape(1, X[i,:].shape[0])
            X_test = np.append(X_test, X_append ,axis=0)
            y_test = np.append(y_test, [y[i]] ,axis=0)

    X_train = X_train[1:,:]
    X_test = X_test[1:,:]
    y_train = y_train[1:]
    y_test = y_test[1:]
    
    return X_train, X_test, y_train, y_test

def print_check(example):
    plt.imshow(X_test[example,1:].reshape(20,20,order='F'))
    print('Prediction: ', np.argmax(a3_test[:,example]))
    print('Likelihood: ', np.max(a3_test[:,example]))
    return

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

# def backward_prop3(y_vectors, a1, a2, a3, a4, theta1, theta2, theta3, lamb):
#     #backward propagation
#     #y_vectors is vectors of results (see notes for clarification)
#     #outputs gradiat arrays for theta1 and theta2
    
#     m = y_vectors.shape[1]
    
#     delta4 = a4 - y_vectors
    
#     delta3_matmul_term = np.matmul(np.transpose(theta3),delta4)
#     delta3_dot_term = np.multiply(a3, np.ones(a3.shape)-a3)
#     delta3 = np.multiply(delta3_matmul_term,delta3_dot_term)
#     triangle3 = np.matmul(delta4, np.transpose(a3))
    
#     delta2_matmul_term = np.matmul(np.transpose(theta2),delta3)
#     delta2_dot_term = np.multiply(a2, np.ones(a2.shape)-a2)
#     delta2 = np.multiply(delta2_matmul_term,delta2_dot_term)
#     triangle2 = np.matmul(delta3, np.transpose(a2))
    
#     triangle1 = np.matmul(delta2[1:,:],np.transpose(a1))
    
#     reg3 = np.zeros((theta3.shape[0],1))
#     reg3 = np.append(reg3,theta3[:,1:],axis=1)
#     grad3 = (1/m)*triangle3 + lamb*reg3
    
#     reg2 = np.zeros((theta2.shape[0],1))
#     reg2 = np.append(reg2,theta2[:,1:],axis=1)
#     grad2 = (1/m)*triangle2 + lamb*reg2
    
#     reg1 = np.zeros((theta1.shape[0],1))
#     reg1 = np.append(reg1,theta1[:,1:],axis=1)
#     grad1 = (1/m)*triangle1 + lamb*reg1
    
#     return grad1, grad2, grad3

def grad_des3(theta1, theta2, theta3, grad1, grad2, grad3, alpha):
    #gradiat descent
    theta1 = theta1 - alpha*grad1
    theta2 = theta2 - alpha*grad2
    theta3 = theta3 - alpha*grad3
    return theta1, theta2, theta3

def J3(y_vectors, a3, theta1, theta2, theta3, lamb):
    #cost function J
    
    m = y_vectors.shape[1]
    
    cost = np.multiply(y_vectors, np.log(a3)) + np.multiply((1-y_vectors),np.log(1-a3))
    cost = np.sum(cost)
    cost = (-1/m)*cost
    
    reg = np.sum(np.square(theta1)) + np.sum(np.square(theta2)) + np.sum(np.square(theta3))
    reg = (lamb/2*m)*reg
    
    J = cost + reg
    return J

def get_delta(nodes_current, theta_current, delta_previous):
    derivative = np.multiply(nodes_current, 1-nodes_current)
    matmul_term = np.matmul(np.transpose(theta_current), delta_previous)
    delta_current = np.multiply(matmul_term, derivative)
    return delta_current

def get_delta_mid(nodes_current, theta_current, delta_previous):
    derivative = np.multiply(nodes_current, 1-nodes_current)
    matmul_term = np.matmul(np.transpose(theta_current), delta_previous[1:,:])
    delta_current = np.multiply(matmul_term, derivative)
    return delta_current

def backward_prop3(y_vectors, a1, a2, a3, a4, theta1, theta2, theta3, lamb):
    m = y_vectors.shape[1]
     
    delta4 = a4 - y_vectors
    
    delta3 = get_delta(a3, theta3, delta4)
    triangle3 = np.matmul(delta4, np.transpose(a3))
    
    delta2 = get_delta_mid(a2, theta2, delta3)
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