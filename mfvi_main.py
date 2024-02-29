import numpy as np

from optimization_utils import *
from misc_utils import *
from gaussian_utils import rho_gaussian_samples

class MFVI:
    def __init__(self, V, grad_V, mesh, trunc,dim):
        self.V = V
        self.grad_V = grad_V
        self.mesh = mesh
        self.trunc = trunc
        self.dim = dim
        self.J = int(2 * self.trunc / self.mesh)

        self.lamb_opt, self.v_opt, self.T_opt = None, None, None

        self.KL_vals, self.W2_vals = None, None

        self.M_1d, _, self.Q, self.Qinv, means , self.gradent_num = build_M_FAST(dim=self.dim, mesh=self.mesh, truncation=self.trunc)
    def SPGD(self,alpha, h, h_v, lamb0, batch_size=1, num_iters=1000, tol=1e-3, compute_KL=False, compute_W2=False,ground_cov=None,stopping_cond=0,save_vals=False):
        self.alpha = alpha
        if save_vals:
            self.lamb_opt, self.v_opt, self.KL_vals, self.W2_vals, self.lamb_vals, self.v_vals = spgd(self.M_1d, self.dim, h, h_v, alpha, self.Q, self.Qinv, self.gradent_num , lamb0, self.mesh, self.trunc, 
                                                                     self.grad_V, self.V, num_iter=num_iters, stochastic_samples = batch_size,
                                                                     compute_KL=compute_KL, compute_W2=compute_W2, ground_cov = ground_cov,KL_tol=tol, stopping_cond=stopping_cond, save_vals=True)
        else:
            self.lamb_opt, self.v_opt, self.KL_vals, self.W2_vals = spgd(self.M_1d, self.dim, h, h_v, alpha, self.Q, self.Qinv, self.gradent_num , lamb0, self.mesh, self.trunc, 
                                                                     self.grad_V, self.V, num_iter=num_iters, stochastic_samples = batch_size,
                                                                     compute_KL=compute_KL, compute_W2=compute_W2, ground_cov = ground_cov,KL_tol=tol, stopping_cond=stopping_cond, save_vals=False)


    def get_Topt(self):
        self.T_opt = compute_T(self.M_1d, self.alpha, self.lamb_opt, self.v_opt)
        return self.T_opt
     
    def gen_mfapprox(self,n_samples):
        rho_samples = rho_gaussian_samples(n_samples,self.dim).T
        if self.T_opt == None:
            self.T_opt = self.get_Topt()
        return self.T_opt(rho_samples)
    