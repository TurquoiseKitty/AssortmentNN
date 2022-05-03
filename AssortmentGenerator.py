import numpy as np
import random

# for given 0 < lam < 1, given N candidate products, generated assortment will contain averagely lam * N products
def GenAssortment_Even(Vec_Len = 10, lam=1/2):
    potential_vec = np.random.uniform(low=0., high=1, size=Vec_Len)
    assortment_vec = np.zeros(Vec_Len)
    assortment_vec[potential_vec > lam] = 0
    assortment_vec[potential_vec <= lam] = 1
    return assortment_vec

# for function psi, given N candidate products, the generated assortment will contain averagely psi(N) products
def GenAssortment_Sparse(Vec_Len = 10, sparse_fun = lambda x : np.sqrt(x)):
    potential_vec = np.random.uniform(low=0., high=1, size=Vec_Len)
    assortment_vec = np.zeros(Vec_Len)
    assortment_vec[potential_vec > (sparse_fun(Vec_Len) / Vec_Len)] = 0
    assortment_vec[potential_vec <= (sparse_fun(Vec_Len) / Vec_Len)] = 1
    return assortment_vec

# generate assortment containing fixed number of products
def GenAssortment_Fixed(Vec_Len = 10, fixed_num = 6):
    positions = random.sample(list(range(Vec_Len)),k=fixed_num)
    print(positions)
    assortment_vec = np.zeros(Vec_Len)
    assortment_vec[positions] = 1
    return assortment_vec

# in some cases we want product 0 in assortment, so use the following function
def Product_0(vec):
    return np.insert(vec, 0, 0)