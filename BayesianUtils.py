import numpy as np
import scipy.stats as stats
import math

def softmax(x):
    return np.exp(x) / (1 + np.exp(x))

def V_logistic(theta, Y, X):
    first = np.log(1 + np.exp(np.dot(X.reshape(-1,1), theta.reshape(1,-1)))).sum(axis=0)
    second = np.dot(Y, np.dot(X.reshape(-1,1),  theta.reshape(1,-1)))
    return first - second

def gradV_logistic(theta, Y, X):
    Y_reshaped = Y.reshape(-1, 1)
    grad = -np.dot(X.T, Y_reshaped - softmax(np.dot(X, theta.T)))
    return grad.T
