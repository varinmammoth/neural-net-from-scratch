# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 16:32:26 2021

@author: Pieumsook
"""

import numpy as np
import matplotlib.pyplot as plt
import deepnn as nn
import neuralfunc as nf
from scipy.io import loadmat

#loads data
data=loadmat("ex3data1.mat")
X=data["X"]
X=np.transpose(X)
y=data["y"]
Y=nf.y_vector(y, num=10)

Wb1 = nn.init_W_b(400,40)
Wb2 = nn.init_W_b(40,10)


cache0 = nn.init_cache(X)
cache1 = nn.init_cache([], *Wb1, 'sigmoid')
cache2 = nn.init_cache([], *Wb2, 'sigmoid')



iterations = 1000
alpha = 0.01
cost = []
for i in range (0, iterations):
    print(i)
    
    cache1 = nn.forward_prop(cache0, cache1)
    cache2 = nn.forward_prop(cache1, cache2)
    
    cache2, grad2 = nn.start_back_prop(Y, cache1, cache2)
    cache1, grad1 = nn.back_prop(cache0, cache1, cache2)
    
    cost = nn.compute_cost(Y, cache2, cost)
    
    cache1 = nn.grad_des(cache1, grad1, alpha)
    cache2 = nn.grad_des(cache2, grad2, alpha)
    
    
    



