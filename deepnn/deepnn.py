# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 13:14:18 2021

@author: Pieumsook
"""
"""
Notation:
    A_previous means A_l-1
    A_current means A_l
    Unless stated otherwise, cache means cache_current
    
Cache:
    Cache is the tuple (A_current, Z_current, W_current, b_current, A_previous, activation_current)
    
Instructions:
    1) Create the weights and biases.
    2) Create caches using init_cache()
    3) Update caches using forward_prop(cache)
    4) Start backward propogation using back_prop(Y, cache_L)
    5) Obtain dA_previous and grad_current using back_prop(dA_current, cache)
"""


import numpy as np

def sigmoid(z):
    """
    sigmoid activation
    """
    return 1/(1+np.exp(-z))

def d_sigmoid(z):
    """
    sigmoid derivative
    """
    return sigmoid(z) * (1- sigmoid(z))

def relu(z):
    """
    relu activation
    """
    return max(0.0, z)
    
def d_relu(z):
    """
    relu derivative
    """
    if z > 0:
        return 1
    else:
        return 0

def init_W_b(n_previous, n_current, low=0.0000001, high=0.000001):
    """
    Parameters
    ----------
    n_previous : INT, number of nodes in previous layer.
    n_current : INT, number of nodes in current layer.
    low : FLOAT, Min value of initial weight/bias. The default is 0.00001.
    high : FLOAT, max value of initial weight/bias. The default is 0.0001.

    Returns
    -------
    (W,b) : tuple with weight W and bias b
    """
    W = np.random.uniform(low, high, size=(n_current,n_previous))
    b = np.random.uniform(low, high, size=(n_current,1))
    return (W, b)

def init_cache(A_current=[], W_current=[], b_current=[], activation_current=[], dZ_current=[]):
    """
    initializes cache.
    Always leave A_current empty except for the first initalization ie. cache0, in this case
    put A_current=X. Leave W_current and b_current empty for cache0.
    """
    Z_current = []
    cache = (A_current, Z_current, W_current, b_current, activation_current, dZ_current)
    return cache

def forward_prop(cache_previous, cache_current):
    #W_current is the weights to get the current node A_current
    #b_current is the biases to get the current node A_current
    #A_previous is the nodes from previous layer
    #activation is the activation function
        #can choose 'sigmoid', 'relu'
    #returns the nodes for the current layer A_current
    
    A_current, Z_current, W_current, b_current, activation_current, dZ_current = cache_current
    A_previous, Z_previous, W_previous, b_previous, activation_previous, dZ_previous = cache_previous
    
    Z_current = np.matmul(W_current, A_previous) + b_current
    
    if activation_current == 'sigmoid':
        A_current = sigmoid(Z_current)
    elif activation_current == 'relu':
        A_current = relu(Z_current)
    
    cache_current = (A_current, Z_current, W_current, b_current, activation_current, dZ_current)
    
    return cache_current

def back_prop(cache_previous, cache_current, cache_next):
    #takes input dA_current and
    #cache is the tuple (Z_current, W_current, b_current, A_previous, activation_current)
    #returns dA_previous and tuple (dW_current, db_current)
    
    A_current, Z_current, W_current, b_current, activation_current, dZ_current = cache_current
    A_previous, Z_previous, W_previous, b_previous, activation_previous, dZ_previous = cache_previous
    A_next, Z_next, W_next, b_next, activation_next, dZ_next = cache_next
    
    m = Z_current.shape[1]
    
    dA_current = np.matmul(np.transpose(W_next), dZ_next)
    
    if activation_current == 'sigmoid':
        dZ_current = np.multiply(dA_current, d_sigmoid(Z_current))
    elif activation_current == 'relu':
        dZ_current = np.multiply(dA_current, d_relu(Z_current))
        
    dW_current = (1/m)*np.matmul(dZ_current, np.transpose(A_previous))
    db_current = (1/m)*np.sum(dZ_current, axis=1, keepdims=True)
    
    cache_current = (A_current, Z_current, W_current, b_current, activation_current, dZ_current)
    gradients_current = (dW_current, db_current)
    
    return cache_current, gradients_current

def start_back_prop(Y, cache_previous, cache_L):
    #Y_predict is the result from forward propagation
    #Y is targets from training examples
    
    Y_predict, Z_current, W_L, b_current, activation_current, dZ_L = cache_L
    A_previous, Z_previous, W_previous, b_previous, activation_previous, dZ_previous = cache_previous
    
    m = Z_current.shape[1]
    
    dZ_L = Y_predict - Y
    dW_L = (1/m)*np.matmul(dZ_L, np.transpose(A_previous))
    db_L = (1/m)*np.sum(dZ_L, axis=1, keepdims=True)
    
    
    cache_L = (Y_predict, Z_current, W_L, b_current, activation_current, dZ_L) 
    gradient_L = (dW_L, db_L)
    
    return cache_L, gradient_L

def grad_des(cache_current, grad_current, alpha):
    
    A_current, Z_current, W_current, b_current, activation_current, dZ_current = cache_current
    
    dW_current, db_current = grad_current
    W_current = W_current - alpha*dW_current
    b_current = b_current - alpha*db_current
    
    cache_current = (A_current, Z_current, W_current, b_current, activation_current, dZ_current)
    
    return cache_current

def compute_cost(Y, cache_L, cost_list):
    """
    Y is targets from training examples
    cache_L is cache of last layer obtained from forward_prop function
    cost_list is a list containing cost at each iteration
    """
    
    Y_predict, Z_current, W_L, b_current, activation_current, dZ_current = cache_L
    
    m = Y.shape[1]
    cost = np.multiply(Y, np.log(Y_predict)) + np.multiply((1-Y),np.log(1-Y_predict))
    cost = np.sum(cost)
    cost = (-1/m)*cost
    
    cost_list.append(cost)
    
    return cost_list





    