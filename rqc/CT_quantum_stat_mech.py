from fractions import Fraction
import random
from .utils import dec2bin, bin_pad
import numpy as np

class CT_quantum_stat_mech:
    def __init__(self,L,seed=None,seed_vec=None,seed_C=None,x0=None):
        self.L=L
        self.rng = np.random.default_rng(seed)
        # self.rng_vec = np.random.default_rng(seed_vec) if seed_vec is not None else self.rng
        self.rng_C = np.random.default_rng(seed_C) if seed_C is not None else self.rng
        self.x0 = x0
        self.vec = self._initialize_vector()
    

    def _initialize_vector(self):
        '''"0" is for maximal mixed state as <Z>=0
        <1> is for fixed state as <Z>=1'''
        vec = np.zeros((self.L,))
        # vec[(0,)*(self.L-2)+(1,0)]=1
        # vec = np.ones((2,)*self.L)/2**self.L
        return vec
    
    def Bernouli(self):
        self.vec = np.roll(self.vec,-1)
        self.vec[-2:]=0

    def control(self):
        self.vec[-1]=1
        self.vec = np.roll(self.vec,1)
    
    def variance(self):
        return 1/self.L - np.sum(self.vec**2)/self.L**2

    def random_circuit(self,p):
        prob=self.rng_C.random()
        if prob>=p:
            self.Bernouli()
        else:
            self.control()

        


