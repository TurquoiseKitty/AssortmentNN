from ctypes import util
import numpy as np
import AssortmentGenerator as AG

# functions regarding the mixture of multinomial logit model
class MMNL_generator:

    # N_prod denotes the number of products
    # N_mix=K means the model is a mixture of K MNL models
    def __init__(self, N_prod, N_mix):
        # sum(mixture_para) = 1, sum(choose_para[i,:]) = 1
        self.N_mix = N_mix
        self.N_prod = N_prod
        self.mixture_para = np.zeros(N_mix)
        self.choose_para = np.zeros((N_mix, N_prod + 1))

        # by default, we automatically generate the parameters for problem instance
        self.self_gen_instance()


    def self_gen_instance(self):
        probs = np.random.uniform(low=0., high=1, size=self.N_mix)
        self.mixture_para = probs / sum(probs)

        for i in range(self.N_mix):
            probs = np.random.uniform(low=0., high=1, size=self.N_prod + 1)
            self.choose_para[i] = probs / sum(probs)

    # assume that assortment is already one-hot encoded
    # and does not include product 0
    def prob_for_assortment(self, assortment):
        assortment = AG.Product_1(assortment)

        probs = np.zeros(self.N_prod + 1)
        for mix in range(self.N_mix):
            mix_para = self.mixture_para[mix]

            util_paras = self.choose_para[mix] * assortment
            util_paras = util_paras / sum(util_paras)

            probs += mix_para * util_paras

        return probs

    # the output is also one_hot encoded
    def gen_final_choice(self, assortment):
        assortment = AG.Product_1(assortment)
        cati = np.random.choice(self.N_mix, 1, p=self.mixture_para)[0]

        probs = self.choose_para[cati] * assortment
        probs = probs / sum(probs)

        fin_choice = np.random.choice(self.N_prod+1, 1, p=probs)[0]

        ret = np.zeros(self.N_prod + 1)
        ret[fin_choice] = 1
        return ret

# functions regarding the mixture of permutation model
class MP_generator:
    def __init__(self,N_prod,N_mix):
        # sum(mixture_para) = 1, choose_para[i,:] is a permutaion
        self.N_mix = N_mix
        self.N_prod = N_prod
        self.mixture_para = np.zeros(N_mix)
        self.permutate_para = np.zeros((N_mix, N_prod + 1))

        # by default, we automatically generate the parameters for problem instance
        self.self_gen_instance()

    def self_gen_instance(self):
        probs = np.random.uniform(low=0., high=1, size=self.N_mix)
        self.mixture_para = probs / sum(probs)

        for i in range(self.N_mix):
            self.permutate_para[i] = np.random.permutation(self.N_prod + 1)

    # assume that assortment is already one-hot encoded
    # and does not include product 0
    def prob_for_assortment(self, assortment):
        assortment = AG.Product_1(assortment)
        bundle = np.array([i for i in range(self.N_prod+1) if assortment[i] == 1 ])
        
        probs = np.zeros(self.N_prod + 1)
        for mix in range(self.N_mix):
            mix_para = self.mixture_para[mix]
            util_paras = self.permutate_para[mix]

            for idx in range(self.N_prod + 1):
                if util_paras[idx] in bundle:
                    
                    probs[int(util_paras[idx])] += mix_para
                    break

        return probs

    # the output is also one_hot encoded
    def gen_final_choice(self, assortment):
        assortment = AG.Product_1(assortment)
        cati = np.random.choice(self.N_mix, 1, p=self.mixture_para)[0]

        permu = self.permutate_para[cati]
        
        ret = np.zeros(self.N_prod + 1)

        for idx in range(self.N_prod + 1):
            if assortment[int(permu[idx])] == 1:
                ret[int(permu[idx])] = 1
                break

        return ret


# a little check
if __name__ == "__main__":
    mm_model = MMNL_generator(N_prod = 5,N_mix = 3)
    mp_model = MP_generator(N_prod=5, N_mix = 4)
    assortment = np.array([1,0,0,1,0])

    # print(mm_model.prob_for_assortment(assortment))
    # print(mp_model.permutate_para)
    print(mp_model.prob_for_assortment(assortment))

    histo = np.zeros(6)
    for sample in range(100000):
        histo += mp_model.gen_final_choice(assortment)

    print(histo/sum(histo))





        