import numpy as np
import scipy.stats as stats
import math

def rho_gaussian_samples(num_samples, dim=1):
    return np.random.normal(size=(dim, num_samples))

def gradV_Gaussian_highd(x, mean, invcov):
    return np.dot(invcov, (x - mean).T).T

def V_Gaussian_highd(x, mean, invcov):
    x = np.atleast_2d(x)
    d = invcov.shape[0]
    x_minus_mean = x - mean
    quadratic_form = np.einsum('ij,jk,ik->i', x_minus_mean, invcov, x_minus_mean)
    constant_term = (d / 2) * np.log(2 * math.pi) - (1 / 2) * np.log(np.linalg.det(invcov))
    result = quadratic_form / 2 + constant_term
    return result if x.shape[0] > 1 else result[0]

####2D Gaussian 2-mixture target
def gradV_mixture2D(x,m,w,s):
    m11,m12,m21,m22 = m
    w11,w12,w21,w22 = w 
    grad_term = np.zeros_like(x)
    grad_term[:,0] = gradV_mixture(x[:,0],m11,m12,w11,w12,s)
    grad_term[:,1] = gradV_mixture(x[:,1],m21,m22,w21,w22,s)
    return grad_term

def V_mixture2D(x,m,w,s):
    m11,m12,m21,m22 = m
    w11,w12,w21,w22 = w   
    return V_mixture(x[:,0],m11,m12,w11,w12,s) + V_mixture(x[:,1],m21,m22,w21,w22,s)

####Univariate Gaussian 2-mixture target
def gradV_mixture(x, m1,m2,w1,w2,s):
    pdf1 = lambda x: stats.norm.pdf(x, loc=m1, scale=s)
    pdf2 = lambda x: stats.norm.pdf(x, loc=m2, scale=s)
    return ( w1*(x-m1)/s**2*pdf1(x) + w2*(x-m2)/s**2*pdf2(x) )/ ( w1*pdf1(x) + w2*pdf2(x) )

def V_mixture(x,m1,m2,w1,w2,s):
    pdf1 = lambda x: stats.norm.pdf(x, loc=m1, scale=s)
    pdf2 = lambda x: stats.norm.pdf(x, loc=m2, scale=s)
    return -np.log((w1*pdf1(x) + w2*pdf2(x))/2)
