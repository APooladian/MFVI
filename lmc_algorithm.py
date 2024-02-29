import numpy as np
from numpy.linalg import eigvalsh
from numpy import sqrt

def langevin_monte_carlo(gradV, N_samples, dim, h, iters=None, tol=None):
    if tol == None:
        tol = 1e-2
    if iters == None:
        iters = 5000
        
    #initialize particle dynamics at the origin
    Xold = np.zeros((N_samples, dim)) 
    for k in range(iters):
        Xold += (-h * gradV(Xold) + sqrt(2*h)*np.random.randn(N_samples, dim))
    
    return Xold