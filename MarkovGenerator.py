import numpy as np

# one thing to point out is that, for all possible transition matrix,
# 0 will always be an absorbing state
# and for every state i, the probability of transiting to 0, M[i,0] must be positive


# start with generating the transition probability of single state
# remember that p[0] > 0
# in fact, a lower bound for p[0] will roughly induce an uppoer bound for iterations

def GenTransprob(product_num, lb_0 = 0.1, ub_0 = 0.2):
    potential_vec = np.random.uniform(low=0., high=1, size=product_num)
    prob_0 = np.random.uniform(low=lb_0, high=ub_0)

    prob_vec = potential_vec / max(sum(potential_vec), 1e-6) * (1-prob_0)
    prob_vec = np.insert(prob_vec, 0, 1- sum(prob_vec))
    return prob_vec

# zero out some elements
def GenTransprob_Even(product_num, lb_0 = 0.1, ub_0 = 0.2, lam = 1/2):
    potential_vec = np.random.uniform(low=0., high=1, size=product_num)
    potential_vec[potential_vec > lam] =0
    prob_0 = np.random.uniform(low=lb_0, high=ub_0)
    prob_vec = potential_vec / max(sum(potential_vec), 1e-6) * (1-prob_0)
    prob_vec = np.insert(prob_vec, 0, 1- sum(prob_vec))
    return prob_vec


# a sparse structure is less computational heavy, so further zero out some elements
def GenTransprob_Sparse(product_num, lb_0 = 0.1, ub_0 = 0.2, sparse_fun = lambda x : np.sqrt(x)):
    potential_vec = np.random.uniform(low=0., high=1, size=product_num)
    potential_vec[potential_vec > (sparse_fun(product_num) / product_num)] =0
    prob_0 = np.random.uniform(low=lb_0, high=ub_0)
    prob_vec = potential_vec / max(sum(potential_vec), 1e-6) * (1-prob_0)
    prob_vec = np.insert(prob_vec, 0, 1- sum(prob_vec))
    return prob_vec

# generate transition matrix. Remember that 0 is absorbing and diagonal must be 0
def GenMarkovM(product_num, gen_func = lambda x : GenTransprob(x)):
    starting = np.zeros(product_num + 1)
    starting[0] = 1.0

    M = np.expand_dims(starting, axis = 0)

    for i in range(product_num):
        trans_vec = gen_func(product_num)

        M = np.concatenate((M,np.expand_dims(trans_vec,axis=0)), axis=0)

    for i in range(1,product_num+1):
        M[i,i]=0
        
    M = M / M.sum(axis=1, keepdims = True)
    return M

# probability vec to instance choice
def Pvec_to_Choice(p_vec):
    if not sum(p_vec) > 1.0 - 1e-6 and sum(p_vec) < 1.0 + 1e-6:
        print("WRONG PROBABILITY!")
        return
    index = np.random.choice(len(p_vec), 1, p=p_vec)[0]
    indicate = np.zeros(len(p_vec))
    indicate[index] = 1
    return indicate

# Note that, the arriving frequency Lams corresponds to each product
# but not product 0
# same is the case with Assort
# which does not contain product 0
def Absorbing_Calculator(Lams, TransP, Assorts):
    Lams = np.insert(Lams, 0, 0)
    Assorts = np.insert(Assorts, 0, 1)
    S_plus = np.squeeze(np.argwhere(Assorts == 1),axis=1)
    S_bar = np.squeeze(np.argwhere(Assorts == 0),axis=1)
    B = TransP[np.expand_dims(S_bar, axis=1), S_plus]
    C = TransP[np.expand_dims(S_bar, axis=1), S_bar]
    
    distri = np.zeros(len(Lams))
    
    addi = np.matmul(np.matmul(np.expand_dims(Lams[S_bar], axis=0), np.linalg.inv(np.identity(len(C)) - C)), B)
    
    count = 0
    for i in S_plus:
        distri[i] = Lams[i] + addi[0,count]
        count += 1
    
    return distri
