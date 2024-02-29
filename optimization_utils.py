import numpy as np
import scipy.stats as stats
from misc_utils import compute_T, gaussian_int
from scipy.linalg import sqrtm
from scipy.optimize import minimize
from math import sqrt

############################################
#### Compute \nabla \cV(\mu_\lambda) #######
############################################

def compute_potential_gradient_mc_highd(grad_V, M_1d, dim, α, λ, v, x_samples, compute_grad_v = True):
    d, n = x_samples.shape
    J = len(M_1d)
    grad = np.zeros(shape=(d, J))
    T = compute_T(M_1d, α, λ, v)
    grad_V_pushed = grad_V(T(x_samples.T))
    for i in range(d):
        for j in range(J):
            grad[i, j] = np.average(M_1d[j](x_samples[i,:].T) * grad_V_pushed[:, i])
    
    if compute_grad_v:
        v_grad = np.average(grad_V_pushed, axis=0)
    else:
        v_grad = np.zeros(d)
    
    return grad, v_grad

############################################
#### Compute \nabla \cH(\mu_\lambda) #######
############################################

def compute_entropy_gradient_highd(alpha, lamb, mesh, gradent_numerator):
    return gradent_numerator/(lamb + alpha*mesh)

############################################
####### Compute KL(\mu_\lambda\|\pi) #######
############################################

def compute_KL_mc(neg_log_pi, M_1d, mesh, trunc, α, λ, v, ρ_samples):
    d, n = ρ_samples.shape
    J = len(M_1d)
    λ = λ.reshape(d,J) 
    T_map = compute_T(M_1d, α, λ, v)

    last_k = int(2 * trunc / mesh)
    if J != last_k:
        raise IndexError("Size of dictionary not consistent with truncation and mesh!")
    log_det_term = 0
    for i in range(d):
        for k in range(last_k):
            a = -trunc + k * mesh
            log_det_term -= np.log(α + λ[i][k]/mesh) * gaussian_int(a, a + mesh)
    log_det_term -= d * np.log(α) * (1 - gaussian_int(-trunc, -trunc + last_k * mesh))

    μ_samples = T_map(ρ_samples.T)
    return np.average(neg_log_pi(μ_samples)) - (d/2)*(np.log(2*np.pi)+1) + log_det_term

#####################################################
#### Projection w.r.t. \|\cdot\|_Q via L-BFGS-B #####
#####################################################

def Qproj(lamb, prev, Q):
    J = len(Q)
    bounds_J =  [(0, None)] * J
    optQ =  minimize(lambda l: np.dot(l - lamb, Q @ (l - lamb)), x0 = prev, jac = lambda l: Q @ (l - lamb),  method='L-BFGS-B', bounds = bounds_J)
    return optQ['x']


#######################################################
### Stochastic projected gradient descent algorithm ###
#######################################################
def spgd(M_1d, dim, step_size, step_size_v, alpha, Q, Qinv, gradent_num, lamb0, mesh, trunc, grad_V, neg_log_PI, num_iter, stochastic_samples=1, 
         compute_W2=False, ground_cov=None, compute_KL=False, W_2_tol=1e-3, KL_tol=1e-3, stopping_cond=0, save_vals=False):
    
    v_init = np.zeros(dim)
    v = v_init
    if step_size_v > 0:
        compute_grad_v = True
    else:
        compute_grad_v = False
    lamb_new = lamb0.copy()
    J = len(M_1d)
    
    if save_vals == True:
        lam_vals = []
        v_vals=[]

    KL_vals = []
    W2_vals = []

    for itr in range(num_iter):
        lamb_cur = lamb_new.copy()
        x_samples = np.random.normal(size=(dim, stochastic_samples))
        gradcV, gradv = compute_potential_gradient_mc_highd(grad_V, M_1d, dim, alpha, lamb_cur, v, x_samples,compute_grad_v=compute_grad_v)
        gradcH = compute_entropy_gradient_highd(alpha, lamb_cur, mesh, gradent_num)
        gradKL = gradcV + gradcH
        
        step = (Qinv @ gradKL.T).T
        lamb_plus = lamb_cur - step_size * step
        lamb_new = lamb_plus.copy()
        
        for i in range(dim):
            if (lamb_plus[i]<0).any():
                lamb_new[i] = Qproj(lamb_plus[i], lamb_cur[i], Q) 

        v -= step_size_v * gradv

        if compute_W2 == True or compute_KL == True:
            ρ_samples = np.random.normal(size=(dim, 10000))

        if compute_W2 == True:
            assert ground_cov is not None
            T_conv = compute_T(M_1d, alpha, lamb_new, v)
            pushforward_samples = T_conv(ρ_samples.T).T
            sample_cov = np.cov(pushforward_samples)
            W2_vals.append( np.trace( ground_cov + sample_cov - 2*sqrtm(ground_cov**(1/2)@sample_cov@ground_cov**(1/2) )) )

        if compute_KL:
            KL_val = compute_KL_mc(neg_log_PI, M_1d, mesh, trunc, alpha, lamb_new, v, ρ_samples)
            KL_vals.append(KL_val)
            print("KL:", KL_val)
            KL_running_diff = np.average(KL_vals[-10:-1]) - np.average(KL_vals[-20:-10])
            print("KL diff", KL_running_diff)
            if stopping_cond == 1:
                if KL_val < KL_tol:
                    break
            elif stopping_cond == 0:
                if abs(KL_running_diff) < KL_tol:
                    break
        else:
            KL_val = 0
        if itr % 50 == 0 and itr > 0:
            print('Iter:', itr)

        if save_vals == True and itr % 100 == 0:
            lam_vals.append(lamb_new)
            v_vals.append(v)

    if save_vals == True:
        return lamb_new, v, KL_vals, W2_vals, lam_vals, v_vals
    else:
        return lamb_new, v, KL_vals, W2_vals
