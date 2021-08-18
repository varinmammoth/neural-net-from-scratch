# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 12:07:07 2021

@author: Pieumsook
"""

def get_delta(nodes_current, theta_current, delta_previous):
    derivative = np.multiply(nodes_current, 1-nodes_current)
    matmul_term = np.matmul(np.transpose(theta_current), delta_previous)
    delta_current = np.multiply(matmul_term, derivative)
    return delta_current

def backward_prop3(y_vectors, a1, a2, a3, a4, theta1, theta2, theta3, lamb):
    delta4 = a4 - y_vectors
    
    delta3 = get_delta(a3, theta3, delta4)
    triangle3 = np.matmul(delta4, np.transpose(a3))
    
    delta2 = get_delta(a2, theta2, delta2)
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