from .utils import U
import numpy as np
from opt_einsum import contract
from functools import partial
class CT_classical_markov:
    def __init__(self,L,seed=None,seed_vec=None,seed_C=None,x0=None):
        self.L=L
        self.rng = np.random.default_rng(seed)
        # self.rng_vec = np.random.default_rng(seed_vec) if seed_vec is not None else self.rng
        self.rng_C = np.random.default_rng(seed_C) if seed_C is not None else self.rng
        self.x0 = x0
        self.vec = self._initialize_vector()
        self.op_list=self._initialize_op()
        self.fdw_vec=self.generate_FDW_vec()
    def _initialize_op(self):
        """Initialize the operators in the circuit, including the control, projection, and Bernoulli map. `C` is the control map, `P` is the projection, `B` is the Bernoulli map, `I` is the identity map. The second element in the tuple is the outcome. The number after "P" is the position of projection (0-index).

        Returns
        -------
        dict of operators
            possible operators in the circuit
        """ 
        return {("C",0):partial(self.control_map,pos=[self.L-1],n=[0]),
                ("C",1):partial(self.control_map,pos=[self.L-1],n=[1]),
                ("B",):self.Bernoulli_map,}
    def _initialize_vector(self):
        '''This is a classical probablity vector of 2^L, with L1 norm'''
        vec = np.zeros((2,)*self.L,dtype=float)
        vec[(0,)*(self.L)]=1
        # vec = np.ones((2,)*self.L)/2**self.L
        return vec

    def T_tensor(self,vec,left=True):
        """Left shift (times 2) and right shift (divided by 2). Directly transpose the index of tensor.

        Parameters
        ----------
        vec : numpy.array, shape=(2**L_T,) or (2,)*L_T
            state vector
        left : bool, optional
            if True, left shift, else, right shift, by default True

        Returns
        -------
        numpy.array, shape=(2,)*L_T
            state vector after shift
        """
        # if vec.ndim!=self.L_T:
        #     vec=vec.reshape((2,)*self.L_T)
        if left:
            idx_list_2=list(range(1,self.L))+[0]
        else:
            idx_list_2=[self.L-1]+list(range(self.L-1))
        # if self.ancilla:
        #     idx_list_2.append(self.L)
        return vec.transpose(idx_list_2)

    def S_tensor(self,vec,rng):
        """Apply scrambler gate to the last two qubits. Directly convert to tensor and apply to the last two indices.

        Parameters
        ----------
        vec : numpy.array, shape=(2**L_T,) or (2,)*L_T
            state vector
        rng : numpy.random.Generator
            random number generator

        Returns
        -------
        numpy.array, 
            state vector after applying scrambler gate
        """
        U_4=U(4,rng)
        U_4=(U_4 * U_4.conj()).real.reshape((2,2,2,2))
        return contract(vec, range(self.L), U_4, range(self.L-2,self.L+2), list(range(self.L-2)) + [self.L,self.L+1] )

    def Bernoulli_map(self,vec):
            """Apply Bernoulli map to the state vector `vec`. The Bernoulli map contains a left shift and a scrambler gate (U(4) Haar random unitary matrix) to the last two qubits.

            Parameters
            ----------
            vec : np.array, shape=(2**L_T,) or (2,)*L_T
                state vector

            Returns
            -------
            np.array, shape=(2**L_T,)
                state vector after Bernoulli map
            """
            vec=self.T_tensor(vec,left=True)
            vec=self.S_tensor(vec,rng=self.rng_C)
            return vec


    def control_map(self,vec,n,pos):
        """Apply control map to the state vector `vec`, usually input vector is in the shape where (ensemble, ensemble_m) are flatten.

        Parameters
        ----------
        vec : torch.tensor
            state vector
        n : int, {0,1}
            the outcome of the measurement of the last qubit, 0 or 1

        Returns
        -------
        torch.tensor
            state vector after control map
        """
        # projection on the last bits
        vec=self.R_tensor(vec,n,pos)
        # right shift 
        vec=self.T_tensor(vec,left=False)

        return vec
    
    def R_tensor(self,vec,n,pos):
        vec=self.P_tensor(vec,n,pos)
        # if self.xj in [frozenset([Fraction(1,3),Fraction(2,3)]),frozenset([0]),frozenset([1]),frozenset([-1])]:
        if len(n)==1:
            # projection on the last bits
            if n[0]==1:
                vec=self.XL_tensor(vec)
        vec=self.normalize(vec)
        return vec
    
    def P_tensor(self,vec,n,pos):
        for n_i,pos_i in zip(n,pos):
            idx_list=[slice(None)]*self.L
            idx_list[pos_i]=1-n_i
            vec[tuple(idx_list)]=0
        return vec

    def XL_tensor(self,vec):
        """Flip the last qubit of the state vector `vec` (excluding the ancilla qubit). Directly swap 0 and 1 index.

        Parameters
        ----------
        vec : numpy.array, shape=(2**L_T,) or (2,)*L_T
            state vector

        Returns
        -------
        numpy.array, shape=(2,)*L_T
            state vector after applying sigma_x to the last qubit
        """
        if vec.ndim!=self.L:
            vec=vec.reshape((2,)*self.L)
        vec=np.roll(vec,1,axis=self.L-1)
        return vec

    def normalize(self,vec):
        """Normalize the state vector `vec`. If the norm is zero, leave it unchanged.

        Parameters
        ----------
        vec : numpy.array, shape=(2**L_T,) or (2,)*L_T
            state vector

        Returns
        -------
        numpy.array, shape=(2**L_T,) or (2,)*L_T
            normalized state vector
        """
        norm=vec.sum()
        if norm>0:
           return vec/(norm)

    def inner_prob(self,vec,pos,n_list):
        """Calculate the probability of measuring 0 at position `pos` for the state vector `vec`. First, convert the vector to tensor (2,2,..), take about the specific `pos`-th index, and flatten to calculate the inner product.

        Parameters
        ----------
        vec : numpy.array, shape=(2**L_T,) or (2,)*L_T
            state vector
        pos : int, {0,1,...,L-1}
            position to apply to calculate the probability of measuring 0

        Returns
        -------
        float, 0<=inner_prod<=1
            probability of measuring 0 at position `pos`
        """
        if vec.ndim != (2,)*self.L:
            vec=vec.reshape((2,)*self.L)
        idx_list=np.array([slice(None)]*self.L)
        idx_list[pos]=n_list
        prob=vec[tuple(idx_list)].sum()

        return prob

    def random_control(self,vec,p_ctrl):
            """The competition between chaotic random unitary, control map and projection, where the projection can only be applied after the unitary. The probability of control is `p_ctrl`, and the probability of projection is `p_proj`.

            Parameters
            ----------
            p_ctrl : float, 0<=p_ctrl<=1
                probability of control
            p_proj : float, 0<=p_proj<=1
                probability of projection
            p_global : float, 0<=p_global<=1
                probability of using global control
            """ 
            
            if self.rng_C.random()<=p_ctrl:
                # control map
                p_0=self.inner_prob(vec, pos=[self.L-1],n_list=[0])
                op=('C',0) if self.rng.random()<=p_0 else ('C',1)
            else:
                # chaotic map
                op=('B',)
            vec=self.op_list[op](vec)
            return vec
    # def FDW(self,vec):
    def generate_FDW_vec(self):
        FDW_v = np.zeros((2,)*self.L,dtype=float)
        idx = np.array([slice(None)]*self.L)
        for i in range(self.L):
            idx[i] = 1
            if i-1>=0:
                idx[i-1] = 0
            FDW_v[tuple(idx)] = self.L-i
        return FDW_v
    
    def FDW(self,vec):
        fdw = contract(self.fdw_vec , range(self.L),vec,range(self.L)).item()
        fdw2 = contract(self.fdw_vec**2 , range(self.L),vec,range(self.L)).item()
        return fdw, fdw2

    def Z_tensor(self,vec):
        """Calculate the order parameter for Ferromagnetic state. The order parameter is defined as \sum_{i=0..L-1} <Z_i>, where Z_i is the Pauli Z matrix at site i.

        Parameters
        ----------
        vec : numpy.array, shape=(2**L_T,) or (2,)*L_T
            state vector

        Returns
        -------
        float
            order parameter for ferromagnetic state
        """
        rs=0
        for i in range(self.L):
            P0=self.inner_prob(vec,pos=[i],n_list=[0])
            rs+=P0*1+(1-P0)*(-1)
        return rs/self.L
