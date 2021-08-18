#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 14:11:08 2021

@author: mammonth
"""
#instructions:

#X is a 'm by n' matrix
#m = number of examples
#n = number of features 

#y is a 'm by 1' list of the results

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import neuralfunc as nn

#loads data
data=loadmat("ex3data1.mat")
X=data["X"]
y=data["y"]

#adds 1 as the first element of each example
X = np.append(np.ones((X.shape[0],1)),X,axis=1)

#splits X and y into training and test set, half and half
X_train, X_test, y_train, y_test = nn.test_train_split(X, y)

#converts y list into array of vectors
y_vectors_train = nn.y_vector(y_train, num=10)
y_vectors_test = nn.y_vector(y_test, num=10)

print('Data arrays ready. Training...')

#creates theta arrays
#layers: 3 (including data)
theta1_rand = np.random.uniform(low=0.00001, high=0.0001, size=(40,401))
theta2_rand = np.random.uniform(low=0.00001, high=0.0001, size=(10,41))

theta1 = theta1_rand
theta2 = theta2_rand

theta1_3 = np.random.uniform(low=0.00001, high=0.0001, size=(40,401))
theta2_3 = np.random.uniform(low=0.00001, high=0.0001, size=(50,41))
theta3_3 = np.random.uniform(low=0.00001, high=0.0001, size=(10,51))

#training
#forward prop.
#backward prop.
#update theta using gradient descent
#compute cost function 

# iterations = 5000
# alpha = 1
# lamb = 0
# J2 = []
# for i in range (0, iterations):
#     a1, a2, a3 = nn.forward_prop2(X_train, theta1, theta2)
#     grad1, grad2 = nn.backward_prop2(y_vectors_train, a1, a2, a3, theta1, theta2, lamb)
#     theta1 ,theta2 = nn.grad_des(theta1, theta2, grad1, grad2, alpha)
#     J2.append(nn.J2(y_vectors_train, a3, theta1, theta2, lamb))
    
# plt.plot(J2)
# plt.xlabel('Iterations')
# plt.ylabel('J')
# plt.show()

# print(min(J2))
# print('Training finished.')

# #testing

# #a3_test is the prediction array of vectors
# print('Testing...')
# a1_test, a2_test, a3_test = nn.forward_prop2(X_test, theta1, theta2)

# #converts a3_test array of vectors back to list of results (0 to 10)
# y_predict = nn.y_vector_to_list(a3_test)

# #calculates accuracy
# accuracy = nn.get_accuracy(y_test, y_predict)
# print('Testing done.')
# print('Accuracy: ', accuracy)

#use this code to manually check
    #prints the image from testing set
    #prints the y_prediction vector for that testing set
#plt.imshow(X_test[1778,1:].reshape(20,20,order='F'))
#np.argmax(a3_test[:,1778])
    #also can use the nn.print_check(test example number) command

print('Training 3 layers...')
iterations = 5000
alpha = 1
lamb = 0
J3 = []
for i in range (0, iterations):
    a1, a2, a3, a4 = nn.forward_prop3(X_train, theta1_3, theta2_3, theta3_3)
    grad1, grad2, grad3 = nn.backward_prop3(y_vectors_train, a1, a2, a3, a4, theta1_3, theta2_3, theta3_3, lamb)
    theta1_3 ,theta2_3, theta3_3 = nn.grad_des3(theta1_3, theta2_3, theta3_3, grad1, grad2, grad3, alpha)
    J3.append(nn.J3(y_vectors_train, a3, theta1_3, theta2_3, theta3_3, lamb))
    
plt.plot(J3)
plt.xlabel('Iterations')
plt.ylabel('J')
plt.show()

print(min(J3))
print('Training finished.')

print('Testing 3 layers...')
a1_test3, a2_test3, a3_test3, a4_test3 = nn.forward_prop3(X_test, theta1_3, theta2_3, theta3_3)

y_predict3 = nn.y_vector_to_list(a4_test3)
accuracy3 = nn.get_accuracy(y_test, y_predict3)
print('Testing done.')
print('Accuracy: ', accuracy3)






    