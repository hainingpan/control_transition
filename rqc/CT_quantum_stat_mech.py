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
        vec = np.ones((self.L,))
        # vec = np.zeros((self.L,))
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
    
    def variance_FDW(self):
        positions_from_right = np.arange(self.L, 0, -1)
        zero_mask = (self.vec == 0)
        q_values = np.cumsum(zero_mask)
        weights = 0.5**q_values * zero_mask
        mean = np.sum(weights * positions_from_right)
        mean_sq = np.sum(weights * positions_from_right**2)
        return mean_sq - mean**2 

    def random_circuit(self,p):
        prob=self.rng_C.random()
        if prob>=p:
            self.Bernouli()
        else:
            self.control()

    def mean_Mz_moment(self, n=1):
        """Compute the n-th moment <Mz^n> where Mz = (1/L) * sum_j Z_j

        For the quantum state rho at each site:
        - vec[j]=1 (U state): <Z_j>=1, <Z_j^2>=1
        - vec[j]=0 (M state): <Z_j>=0, <Z_j^2>=1

        Returns:
        --------
        <Mz^n> : float
            Quantum expectation value of Mz^n
        """
        N_U = np.sum(self.vec)  # Number of U sites

        if n == 1:
            # <Mz> = (1/L) * sum_j <Z_j> = N_U / L
            return N_U / self.L

        elif n == 2:
            # <Mz^2> = <(1/L^2) sum_{i,j} Z_i Z_j>
            # = (1/L^2) [sum_i <Z_i^2> + sum_{i!=j} <Z_i><Z_j>]
            # = (1/L^2) [L + (N_U^2 - N_U)]
            # = 1/L + (N_U^2 - N_U)/L^2
            return 1.0/self.L + (N_U**2 - N_U) / self.L**2

        else:
            raise NotImplementedError(f"Moment n={n} not implemented for Mz. Only n=1,2 supported.")

    def mean_FDW_moment(self, n=1):
        """Compute the n-th moment of First Domain Wall position from the right"""
        positions_from_right = np.arange(self.L, 0, -1)
        zero_mask = (self.vec == 0)
        q_values = np.cumsum(zero_mask)
        weights = 0.5**q_values * zero_mask
        mean_n = np.sum(weights * positions_from_right**n)
        return mean_n




