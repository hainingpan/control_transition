from .utils import dec2bin, bin_pad

from fractions import Fraction
# import numpy as np
import random

class APT_classical:
    def __init__(self,L,seed=None,seed_vec=None,seed_C=None,x0=None):
        self.L=L
        # self.rng=np.random.default_rng(seed)
        # self.rng_vec=np.random.default_rng(seed_vec) if seed_vec is not None else self.rng
        # self.rng_C=np.random.default_rng(seed_C) if seed_C is not None else self.rng
        self.rng=random.Random(seed)
        self.rng_vec=random.Random(seed_vec) if seed_vec is not None else self.rng
        self.rng_C=random.Random(seed_C) if seed_C is not None else self.rng
        self.x0=x0
        self.vec=self._initialize_vector()
        self.single_mask,self.double_mask=self._initialize_binary()
        self.right_idx={True:list(range(0,L,2)),False:list(range(1,L,2))}    # here, uses an index labeling of L, L-1, ..., 1, 0
        
    def _initialize_vector(self):
        '''save using an array of L'''
        if self.x0 is None:
            # vec=self.rng_vec.integers(low=0,high=1<<self.L)
            vec=self.rng_vec.randint(0,1<<self.L-1)
        else:
            # vec=int(self.x0*(1<<self.L))
            vec=dec2bin(self.x0,self.L)
        return vec

    def _initialize_binary(self):
        single_mask=[1<<i for i in range(self.L)]
        double_mask=[(1<<i)+(1<<(i+1)%self.L) for i in range(self.L)] # the index is for the right index of the two bits
        return single_mask,double_mask

    def unitary_layer(self,even=True):
        for i in self.right_idx[even]:
            self.unitary(i)
            

    def unitary(self,i):
        bits=self.double_mask[i]&self.vec
        if i<self.L-1:
            if bits>>i != 0:
                new_bits=(generate_U3(self.rng_C)<<i)
                self.vec-=bits
                self.vec+=new_bits
        elif i==self.L-1:
            if (bits&1)+(bits>>i-1) != 0:
                new_bits=(generate_U3(self.rng_C))
                self.vec-=(bits)
                self.vec+=(new_bits>>1) # the first bit should be assigned to the last bit
                self.vec+=((new_bits&1)<<i) # the last bit should be assigned to the first bit
    
    def reset(self,i):
        self.vec&=~(1<<i)

    def random_circuit(self,p,even):
        self.unitary_layer(even)
        for i in range(self.L):
            bit = self.single_mask[i]&self.vec
            if (bit >> i) !=0:
                if self.rng.random() < p: # flip from 0 to 1
                    self.reset(i)
    def order_parameter(self):
        return bin(self.vec).count('1')/self.L
        
                



def generate_U3(rng):
    # x=rng.integers(low=1,high=4)
    # print(x)
    # return x
    # return rng.integers(low=1,high=4)
    return rng.randint(1,3)

            

        