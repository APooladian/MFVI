import numpy as np
import scipy.stats as stats
import math

##########################################
##### Build maps and Gram matrix "Q" #####
##########################################

def build_M_FAST(dim=1, mesh=0.5, truncation=2):

  def step_fn(k):
    return lambda x: np.minimum(1, np.maximum(0, (x - (-truncation + k * mesh)) / mesh))

  # compute the centered 1D maps and their derivatives
  means_M_1d = []
  for k in range(int(2 * truncation / mesh)):
    a = -truncation + k * mesh
    means_M_1d.append((gaussian_x_int(a, a + mesh) - a * gaussian_int(a, a + mesh)) / mesh + gaussian_int(a + mesh, np.inf))
  M_1d = [lambda x, k=k: step_fn(k)(x) - means_M_1d[k] for k in range(int(2 * truncation / mesh))]
    
  # build the high-dim dictionary
  def basis_vec(i):
    vec = np.zeros(dim)
    vec[i] = 1.0
    return vec

  M = [[lambda x, i=i, T=T: T(x[i]) * basis_vec(i) for T in M_1d] for i in range(dim)]

  if dim == 1:
    M = M[0]
    
  # compute Q and its inverse
  J = len(M_1d)
  Q = np.zeros((J, J))
  for j in range(J):
    for i in range(j + 1):
      a0 = -truncation + i * mesh
      a1 = -truncation + j * mesh
      I = 0
      if a1 <= a0 + mesh:
        I += (gaussian_xsq_int(a1, a0 + mesh) - (a0 + a1) * gaussian_x_int(a1, a0 + mesh) + (a0 * a1) * gaussian_int(a1, a0 + mesh)) / mesh**2
        I += (gaussian_x_int(a0 + mesh, a1 + mesh) - a1 * gaussian_int(a0 + mesh, a1 + mesh)) / mesh
      else:
        I += (gaussian_x_int(a1, a1 + mesh) - a1 * gaussian_int(a1, a1 + mesh)) / mesh
      I += gaussian_int(a1 + mesh, np.inf)
      I -= means_M_1d[i] * means_M_1d[j]
      Q[i][j] = I
      Q[j][i] = I
  Qinv = np.linalg.pinv(Q)

  gradent_num = np.zeros((1,int(2 * truncation / mesh)))
  for k in range(int(2 * truncation / mesh)):
    a = -truncation + k * mesh
    gradent_num[0,k] = -(stats.norm.cdf(a + mesh) - stats.norm.cdf(a))

  return M_1d, M, Q, Qinv, np.array(means_M_1d), gradent_num

### Relevant Gaussian integrals in "closed" form
def gaussian_int(a, b):
  return stats.norm.cdf(b) - stats.norm.cdf(a)

def gaussian_x_int(a, b):
  return stats.norm.pdf(a) - stats.norm.pdf(b)

def gaussian_xsq_int(a, b):
  return a * stats.norm.pdf(a) - b * stats.norm.pdf(b) + gaussian_int(a, b)

### create map T based on prescribed parameters (α, λ, v) corresponding to a family M_1d
def compute_T(M_1d, α, λ, v):
    d, J = λ.shape
    return lambda x: α*x + sum([λ[:,j] * M_1d[j](x) for j in range(J)]) + v


