import numpy as np
from functools import reduce
import scipy.sparse as sp
import scipy
from fractions import Fraction
from functools import partial, lru_cache
import torch

# needs optimizations, docstrings.
class CT_classical:
    def __init__(self,L,history=False,seed=None,x0=None):
        '''
        
        if classical is False: save using an array of 2^L
        '''
        self.L=L
        self.history=history
        self.rng=np.random.default_rng(seed)
        # self.x0=self.rng.random() if x0 is None else x0
        self.x0=x0
        self.op_history=[]  # control: true, Bernoulli: false
        self.binary=self._initialize_binary([Fraction(1,6),Fraction(1,3)])
        self.binary[Fraction(1,6)][-1]=1

        self.vec=self._initialize_vector()
        self.vec_history=[self.vec]
    def _initialize_vector(self):
        '''save using an array of L'''
        if self.x0 is None:
            vec=self.rng.integers(low=0,high=2,size=(self.L,))
        else:
            vec=dec2bin(self.x0,self.L)
        return vec
    def _initialize_binary(self,x_list):
        return {x:dec2bin(x, self.L) for x in x_list}

    def Bernoulli_map(self,vec):
        vec=np.roll(vec,-1)
        vec[-3:]=self.rng.permutation(vec[-3:])        
        return vec
    
    def control_map(self,vec):
        vec[-1]=0
        vec=np.roll(vec,1)
        if vec[1]==0:
            # attract to 1/3
            # here is a slight difference for the last digit, but this does not matter in the thermaldynamic limit
            vec=add_binary(vec,self.binary[Fraction(1,6)])
        else:
            # attract to 2/3
            vec=add_binary(vec,self.binary[Fraction(1,3)])
        return vec

    def random_control(self,p):
        '''
        p: the control probability
        '''
        p0=self.rng.random()
        vec=self.vec_history[-1].copy()
        if p0<p:
            vec=self.control_map(vec)
        else:
            vec=self.Bernoulli_map(vec)
        if self.history:
            self.vec_history.append(vec)
            self.op_history.append((p0<p))
        else:
            self.vec_history=[vec]
            self.op_history=[(p0<p)]

    def order_parameter(self):
        vec=self.vec_history[-1].copy()
        vec_Z=2*vec-1
        vec_Z_shift=np.roll(vec_Z,1)
        return -vec_Z@vec_Z_shift/self.L

def add_binary(vec1,vec2):
    ''' adder for two `vec1` and `vec2`
    both are np.array
    return vec1
    '''
    assert vec1.shape[0]==vec2.shape[0], f'len of {vec1.shape[0]} is not same as len of {vec2.shape[0]}'
    vec_bin1=int(''.join(map(str,vec1)),2)
    vec_bin2=int(''.join(map(str,vec2)),2)
    vec_bin1=vec_bin1+vec_bin2
    vec1_sum=bin(vec_bin1)[2:]
    vec1_sum=vec1_sum.rjust(vec1.shape[0],'0')
    
    vec1_sum=list(map(int,list(vec1_sum)))
    if len(vec1_sum)>(vec1.shape[0]):
        vec1_sum = np.array(vec1_sum[1:]) #drop carry
    else:
        vec1_sum = np.array(vec1_sum)

    if np.array_equal(vec1_sum[[0,-3,-1]],np.array([1,0,1])) or np.array_equal(vec1_sum[[0,-3,-1]],np.array([0,1,0])):
        vec1_sum[-2]=1-vec1_sum[-2]

    return (vec1_sum)

class CT_quantum:
    def __init__(self,L,store_vec=False,store_op=False,store_prob=False,seed=None,seed_vec=None,seed_C=None,x0=None,xj=set([Fraction(1,3),Fraction(2,3)]),_eps=1e-10, ancilla=False,normalization=True,debug=False):
        """Initialize the quantum circuit for the control transition (CT)

        Parameters
        ----------
        L : int
            the length of the physical system, excluding ancilla qubit
        store_vec : bool, optional
            store the history of state vector , by default False
        store_op : bool, optional
            store the history of operators applied to the circuit, by default False
        store_prob : bool, optional
            store the history of each probability at projective measurement, by default False
        seed : int, optional
            (1) the random seed in the measurement outcome; (2) if `seed_vec` and `seed_C` is None, this random seed also applied to initial state vector and circuit, by default None
        seed_vec : int, optional
            the random seed in the state vector, by default None
        seed_C : int, optional
            the random seed in the circuit, by default None
        x0 : float|Fraction(a,b), optional
            the initial state represented by a float number within [0,1), by default None, if None, the initial state is Haar random state
        xj : set of Fractions, optional
            the set of attractors using Fractions, by default set([Fraction(1,3),Fraction(2,3)])
        _eps : float, optional
            error threshold, by default 1e-10
        ancilla : bool, optional
            if true, an ancilla qubit is maximally entangled to the system , by default False
        normalization : bool, optional
            if false, the state vector will not be normalized after the projective measurement, by default True
        debug : bool, optional
            if trure, all assertions will be checked, by default False
        """        
        self.L=L 
        self.L_T=L+1 if ancilla else L # the length combining the physical system and the ancilla qubit
        self.store_vec=store_vec
        self.store_op=store_op
        self.store_prob=store_prob
        self.rng=np.random.default_rng(seed)
        self.rng_vec=np.random.default_rng(seed_vec) if seed_vec is not None else self.rng
        self.rng_C=np.random.default_rng(seed_C) if seed_C is not None else self.rng
        self.x0=x0
        self.op_history=[] # store the history of operators applied to the circuit
        self.prob_history=[]  # store the history of each probability at projective measurement
        self.ancilla=ancilla
        self.vec=self._initialize_vector() # initialize the state vector
        self.vec_history=[self.vec] # store the history of state vector
        self._eps=_eps
        self.xj=set(xj)
        self.op_list=self._initialize_op() # initialize operators in the circuit
        self.normalization=normalization
        self.debug=debug    
        
    def _initialize_vector(self):
        """Save the state vector using an array of 2**L_T, if ancilla qubit, the last qubit is the ancilla qubit.
        If ancilla is False, the initial state is either the Haar random state or the state represented by `x0` (if specified)
        If ancilla is True, the initial state is the maximally entangled state between the system and the ancilla qubit, where the system is Haar random state. Namely, |psi>=1/sqrt(2)(|psi_0>|0>+|psi_1>|1>), where |psi_0> and |psi_1> are two orthogonal Haar random states. 

        Returns
        -------
        np.array, shape=(2**L_T,) or (2,)*L_T
            the initial state vector
        """
        if not self.ancilla:
            if self.x0 is not None:
                vec_int=int(''.join(map(str,dec2bin(self.x0,self.L))),2)
                vec=np.zeros((2**self.L,),dtype=complex)
                vec[vec_int]=1
            else:
                vec=Haar_state(self.L, 1,rng=self.rng_vec,k=1).flatten()
        else:
            vec=Haar_state(self.L, 1,rng=self.rng_vec,k=2).flatten()/np.sqrt(2)
        return vec.reshape((2,)*self.L_T)
    
    def _initialize_op(self):
        """Initialize the operators in the circuit, including the control, projection, and Bernoulli map. `C` is the control map, `P` is the projection, `B` is the Bernoulli map, `I` is the identity map. The second element in the tuple is the outcome. The number after "P" is the position of projection (0-index).

        Returns
        -------
        dict of operators
            possible operators in the circuit
        """ 
        return {("C",0):partial(self.control_map,n=0),
                ("C",1):partial(self.control_map,n=1),
                (f"P{self.L-1}",0):partial(self.projection_map,pos=self.L-1,n=0),
                (f"P{self.L-1}",1):partial(self.projection_map,pos=self.L-1,n=1),
                (f"P{self.L-2}",0):partial(self.projection_map,pos=self.L-2,n=0),
                (f"P{self.L-2}",1):partial(self.projection_map,pos=self.L-2,n=1),
                ("B",):self.Bernoulli_map,
                ("I",):lambda x:x
                }

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
    
    def control_map(self,vec,n):
        """Apply control map the state vector `vec`. The control map is a combination of projection, right shift and an adder. The projection is applied to the last qubit (if the outcome is 1, sigma_x is applied to flip the last qubit); the right shift is applied to all qubits in the system (excluding the ancilla qubit); the adder is the shuffle of the state basis. 

        Parameters
        ----------
        vec : np.array, shape=(2**L_T,) or (2,)*L_T
            state vector
        n : int, {0,1}
            the outcome of the measurement of the last bit, 0 or 1. 

        Returns
        -------
        np.array, shape=(2**L_T,)
            state vector after the control map
        """
        # projection on the last bits
        vec=self.P_tensor(vec,n)
        if n==1:
            vec=self.XL_tensor(vec)
        if self.normalization:
            vec=self.normalize(vec)

        # right shift 
        vec=self.T_tensor(vec,left=False)
        if self.debug:
            assert np.abs(vec[vec.shape[0]//2:]).sum() == 0, f'first qubit is not zero ({np.abs(vec[vec.shape[0]//2:]).sum()}) after right shift '

        # Adder
        if vec.ndim >1 :
            vec=vec.flatten()
        if not self.ancilla:
            vec=self.adder()@vec
        else:
            vec=(self.adder()@vec.reshape((2**self.L,2))).flatten()
        return vec

    def projection_map(self,vec,pos,n):
        """Projective measurement to `pos`-th qubit with outcome of `n` (0-index). `pos=L-1` is the last bit.

        Parameters
        ----------
        vec : np.array, shape=(2**L_T,) or (2,)*L_T
            state vector
        pos : int, {0,1,...,L-1}
            position to apply the projection
        n : int, {0,1}
            the outcome of the measurement, 0 or 1

        Returns
        -------
        np.array, shape=(2**L_T,)
            state vector after projection
        """
        vec=self.P_tensor(vec,n,pos)
        if self.normalization:
            vec=self.normalize(vec)
        return vec
    def encoding(self):
        """Encoding process: Randomly apply Bernoulli map
        """
        vec=self.vec_history[-1].copy()
        vec=self.op_list[('B',)](vec)
        self.update_history(vec,('B',),None)


    def random_control(self,p_ctrl,p_proj):
        """The competition between chaotic random unitary, control map and projection, where the projection can only be applied after the unitary. The probability of control is `p_ctrl`, and the probability of projection is `p_proj`.
        This is the same protocol as `random_control_2`, the only difference is the way to generate rng, and document operators.

        Parameters
        ----------
        p_ctrl : float, 0<=p_ctrl<=1
            probability of control
        p_proj : float, 0<=p_proj<=1
            probability of projection
        """ 
        vec=self.vec_history[-1].copy()
        

        op_l=[]
        if self.rng_C.random()<=p_ctrl:
            # control map
            p_0=self.inner_prob(vec, pos=self.L-1,)
            op=('C',0) if self.rng.random()<=p_0 else ('C',1)
        else:
            # chaotic map
            op=('B',)
        op_l.append(op)
        vec=self.op_list[op](vec)

        if "B" in op:
            for idx,pos in enumerate([self.L-1,self.L-2]):
                if self.rng_C.random()<p_proj:
                    # projection map
                    p_2=(self.inner_prob(vec, pos=pos,))
                    op_2=(f"P{pos}",0) if self.rng.random()<p_2 else (f"P{pos}",1)
                    vec=self.op_list[op_2](vec)
                    op_l.append(op_2)
        self.update_history(vec,op_l,None)

    def reference_control(self,op_history):
        """The reference protocol, where the operators are specified by `op_history`. The operators are applied sequentially.

        Parameters
        ----------
        op_history : list of list of str
            The history of operators applied to the circuit. See `__init__` for the definition.
        """
        vec=self.vec_history[-1].copy()
        for op_l in op_history:
            self.rng_C.random() # dummy random to keep same random sequence, same as p_ctrl
            for idx,op in enumerate(op_l):
                vec=self.op_list[op](vec)
                if 'C' in op:
                    self.rng.random() # dummy random for C0/C1
                if 'B' in op:
                    self.rng_C.random() # dummy in projection at L-1, L-2
                    self.rng_C.random() # dummy in projection at L-1, L-2
                if 'P' in op:
                    self.rng.random() # dummy in random pro Pi0/Pi1
            self.update_history(vec,op_l,None)
    
    def order_parameter(self,vec=None):
        """Calculate the order parameter. For `xj={1/3,2/3}`, it is \sum Z.Z, for `xj={0}`, it is \sum Z.

        Parameters
        ----------
        vec : np.array, shape=(2**L_T,) or (2,)*L_T, optional
            state vector, by default None

        Returns
        -------
        float
            the order parameter. 
        """
        if vec is None:
            vec=self.vec_history[-1].copy()
        if self.xj== set([Fraction(1,3),Fraction(2,3)]):
            O=self.ZZ_tensor(vec)
        elif self.xj == set([0]):
            O=self.Z_tensor(vec)
        if self.debug:
            assert np.abs(O.imag)<self._eps, f'<O> is not real ({val}) '
        return O.real
    
    def von_Neumann_entropy_pure(self,subregion,vec=None):
        """Calculate the von Neumann entropy of a pure state, where the state vector is `vec` and the subregion is `subregion`. Using the Schmidt decomposition, the von Neumann entropy is -\sum_i \lambda_i^2 \log \lambda_i^2, where \lambda_i is the singular value of the reshaped state vector `vec`.

        Parameters
        ----------
        subregion : list of int or np.array
            The spatial subregion to calculate the von Neumann entropy
        vec : np.array, shape=(2**L_T,) or (2,)*L_T, optional
            state vector, by default None

        Returns
        -------
        float
            Von Neumann entropy of the subregion
        """
        if vec is None:
            vec=self.vec_history[-1].copy()
        vec_tensor=vec.reshape((2,)*(self.L_T))
        subregion=list(subregion)
        not_subregion=[i for i in range(self.L_T) if i not in subregion]
        vec_tensor_T=vec_tensor.transpose(np.hstack([subregion , not_subregion]))
        S=np.linalg.svd(vec_tensor_T.reshape((2**len(subregion),2**len(not_subregion))),compute_uv=False)
        S_pos2=np.clip(S,1e-18,None)**2
        return -np.sum(np.log(S_pos2)*S_pos2)

    def half_system_entanglement_entropy(self,vec=None,selfaverage=False):
        """Calculate the half-system entanglement entropy, where the state vector is `vec`. The half-system entanglement entropy is defined as \sum_{i=0..L/2-1}S_([i,i+L/2)) / (L/2), where S_([i,i+L/2)) is the von Neumann entropy of the subregion [i,i+L/2).

        Parameters
        ----------
        vec : np.array, shape=(2**L_T,) or (2,)*L_T, optional
            state vector, by default None
        selfaverage : bool, optional
            if true, average over all possible halves, namely, \sum_{i=0..L/2-1}S_([i,i+L/2)) / (L/2), by default False

        Returns
        -------
        float
            Half-system entanglement entropy
        """
        if vec is None:
            vec=self.vec_history[-1].copy()
        if selfaverage:
            return np.mean([self.von_Neumann_entropy_pure(np.arange(i,i+self.L//2),vec) for i in range(self.L//2)])
        else:
            return self.von_Neumann_entropy_pure(np.arange(self.L//2),vec)

    def tripartite_mutual_information(self,subregion_A,subregion_B, subregion_C,selfaverage=False,vec=None):
        """Calculate tripartite entanglement entropy. The tripartite entanglement entropy is defined as S_A+S_B+S_C-S_AB-S_AC-S_BC+S_ABC, where S_A is the von Neumann entropy of subregion A, S_AB is the von Neumann entropy of subregion A and B, etc. The system size `L` should be a divided by 4 such that the subregion A, B and C are of the same size.

        Parameters
        ----------
        subregion_A : list of int or np.array
            subregion A
        subregion_B : list of int or np.array
            subregion B
        subregion_C : list of int or np.array
            subregion C
        selfaverage : bool, optional
            if true, average over all possible partitions, by default False
        vec : np.array, shape=(2**L_T,) or (2,)*L_T, optional
            state vector, by default None

        Returns
        -------
        float
            Tripartite entanglement entropy
        """
        if self.debug:
            assert np.intersect1d(subregion_A,subregion_B).size==0 , "Subregion A and B overlap"
            assert np.intersect1d(subregion_A,subregion_C).size==0 , "Subregion A and C overlap"
            assert np.intersect1d(subregion_B,subregion_C).size==0 , "Subregion B and C overlap"
        if vec is None:
            vec=self.vec_history[-1].copy()
        if selfaverage:
            return np.mean([self.tripartite_mutual_information((subregion_A+shift)%self.L,(subregion_B+shift)%self.L,(subregion_C+shift)%self.L,selfaverage=False) for shift in range(len(subregion_A))])
        else:
            S_A=self.von_Neumann_entropy_pure(subregion_A,vec=vec)
            S_B=self.von_Neumann_entropy_pure(subregion_B,vec=vec)
            S_C=self.von_Neumann_entropy_pure(subregion_C,vec=vec)
            S_AB=self.von_Neumann_entropy_pure(np.concatenate([subregion_A,subregion_B]),vec=vec)
            S_AC=self.von_Neumann_entropy_pure(np.concatenate([subregion_A,subregion_C]),vec=vec)
            S_BC=self.von_Neumann_entropy_pure(np.concatenate([subregion_B,subregion_C]),vec=vec)
            S_ABC=self.von_Neumann_entropy_pure(np.concatenate([subregion_A,subregion_B,subregion_C]),vec=vec)
            return S_A+ S_B + S_C-S_AB-S_AC-S_BC+S_ABC

    def update_history(self,vec=None,op=None,p=None):
        """Update the history of state vector, operators and Born probability. If `store_vec` is True, the state vector is appended to the history. If `store_op` is True, the operators are appended to the history. If `store_prob` is True, the Born probability is appended to the history.

        Parameters
        ----------
        vec : np.array, shape=(2**L_T,) or (2,)*L_T, optional
            state vector, if None, do not save it, by default None
        op : tuple, optional
            operations, if None, do not save it, by default None
        p : float, optional
            Born probability, if None, do not save it, by default None
        """
        if vec is not None:
            vec=vec.copy().flatten()
            if self.store_vec:
                self.vec_history.append(vec)
            else:
                self.vec_history=[vec]

        if op is not None:
            if self.store_op:
                self.op_history.append(op)
            else:
                self.op_history=[op]
        
        if p is not None:
            if self.store_prob:
                self.prob_history.append(p)
            else:
                self.prob_history=[p]

    def inner_prob(self,vec,pos):
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
        if vec.ndim != (2,)*self.L_T:
            vec=vec.reshape((2,)*self.L_T)
        idx_list=[slice(None)]*self.L_T
        idx_list[pos]=0
        vec_0=vec[tuple(idx_list)].flatten()
        inner_prod=vec_0.conj()@vec_0
        if self.debug:
            assert np.abs(inner_prod.imag)<self._eps, f'probability for outcome 0 is not real {inner_prod}'
            assert inner_prod>-self._eps, f'probability for outcome 0 is not positive {inner_prod}'
            assert inner_prod<1+self._eps, f'probability for outcome 1 is not smaller than 1 {inner_prod}'
        inner_prod=np.clip(inner_prod.real,0,1)
        return inner_prod

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
        if vec.ndim!=1:
            vec=vec.flatten()
        norm2=(vec.conj().T@vec).real
        self.update_history(None,None,norm2)
        if norm2 > 0:
            return vec/np.sqrt(norm2)
        else:
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
        if vec.ndim!=self.L_T:
            vec=vec.reshape((2,)*self.L_T)
        if not self.ancilla:
            vec=vec[...,[1,0]]
        else:
            vec=vec[...,[1,0],:]
        return vec

    def P_tensor(self,vec,n,pos=None):
        """Directly set zero at tensor[...,0] =0 for n==1 and tensor[...,1] =0 for n==0

        Parameters
        ----------
        vec : numpy.array, shape=(2**L_T,) or (2,)*L_T
            state vector
        n : int, {0,1}
            outcome of projection
        pos : int, optional
            position of projection, if None, apply to the last qubit excluding ancilla qubit, by default None

        Returns
        -------
        numpy.array, shape=(2,)*L_T
            state vector after projection
        """
        if vec.ndim!=self.L_T:
            vec=vec.reshape((2,)*self.L_T)
        vec=vec.reshape((2,)*self.L_T)
        if pos is None or pos==self.L-1:
            # project the last site
            if not self.ancilla:
                vec[...,1-n]=0
            else:
                vec[...,1-n,:]=0
        if pos == self.L-2:
            if not self.ancilla:
                vec[...,1-n,:]=0
            else:
                vec[...,1-n,:,:]=0
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
        if vec.ndim!=self.L_T:
            vec=vec.reshape((2,)*self.L_T)
        if left:
            idx_list_2=list(range(1,self.L))+[0]
        else:
            idx_list_2=[self.L-1]+list(range(self.L-1))
        if self.ancilla:
            idx_list_2.append(self.L)
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
        if not self.ancilla:
            vec=vec.reshape((2**(self.L-2),2**2)).T
            return (U_4@vec).T
        else:
            vec=vec.reshape((2**(self.L-2),2**2,2)).transpose((1,0,2)).reshape((2**2,2**(self.L-1)))
            return (U_4@vec).reshape((2**2,2**(self.L-2),2)).transpose((1,0,2))

    def ZZ_tensor(self,vec):
        """Calculate the order parameter for Neel state. The order parameter is defined as \sum_{i=0..L-1} <Z_iZ_{i+1}>, where Z_i is the Pauli Z matrix at site i.

        Parameters
        ----------
        vec : numpy.array, shape=(2**L_T,) or (2,)*L_T
            state vector

        Returns
        -------
        float
            order parameter for Neel state
        """
        if vec.ndim != (2,)*self.L_T:
            vec=vec.reshape((2,)*self.L_T)
        rs=0
        for i in range(self.L):
            for zi in range(2):
                for zj in range(2):
                    idx_list=[slice(None)]*self.L_T
                    idx_list[i],idx_list[(i+1)%self.L]=zi,zj
                    exp=1-2*(zi^zj) # expectation-- zi^zj is xor of two bits which is only one when zi!=zj
                    vec_i=vec[tuple(idx_list)].flatten()
                    rs+=vec_i.conj()@vec_i*exp
        return -rs/self.L
    
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
        if vec_tensor.ndim != (2,)*self.L_T:
            vec_tensor=vec.reshape((2,)*self.L_T)
        rs=0
        for i in range(self.L):
            P0=self.inner_prob(vec_tensor,i)
            rs+=P0*1+(1-P0)*(-1)
        return rs/self.L
        
    @lru_cache(maxsize=None)
    def adder(self):
        """Calculate the adder matrix, which is the shuffle of the state basis. Note that this is not a full adder, which assumes the leading digit in the input bitstring is zero (because of the T^{-1}R_L, the leading bit should always be zero).

        Returns
        -------
        numpy.array, shape=(2**L,2**L)
            adder matrix
        """
        if self.xj==set([Fraction(1,3),Fraction(2,3)]):
            bin_1_6=dec2bin(Fraction(1,6), self.L)
            bin_1_6[-1]=1
            bin_1_3=dec2bin(Fraction(1,3), self.L)
            int_1_6=int(''.join(map(str,bin_1_6)),2)
            int_1_3=int(''.join(map(str,bin_1_3)),2)
            old_idx=np.arange(2**(self.L-1))
            adder_idx=np.array([int_1_6]*2**(self.L-2)+[int_1_3]*2**(self.L-2))    
            new_idx=old_idx+adder_idx
            # handle the extra attractors, if 1..0x1, then 1..0(1-x)1, if 0..1x0, then 0..1(1-x)0 [shouldn't enter this branch..]
            mask_1=(new_idx&(1<<self.L-1) == (1<<self.L-1)) & (new_idx&(1<<2) == (0)) & (new_idx&(1) == (1))
            mask_2=(new_idx&(1<<self.L-1) == (0)) & (new_idx&(1<<2) == (1<<2)) & (new_idx&(1) == (0))
            new_idx[mask_1+mask_2]=new_idx[mask_1+mask_2]^(0b10)
            ones=np.ones(2**(self.L-1))
            return sp.coo_matrix((ones,(new_idx,old_idx)),shape=(2**self.L,2**self.L))
        if self.xj==set([0]):
            return sp.eye(2**self.L)  

def dec2bin(x,L):
    """convert a float number x in [0,1) to the binary form with maximal length of L, where the leading 0 as integer part is truncated. Example, 1/3 is 010101...

    Parameters
    ----------
    x : float, 0<=x<1
        float number to be converted
    L : int
        length of the binary form

    Returns
    -------
    numpy.array, shape=(L,)
        array of binary form
    """
    assert 0<=x<1, f'{x} is not in [0,1)'
    bits=[]
    for _ in range(L):
        x*=2
        bits.append(int(x))
        x-=int(x)
    return np.array(bits,dtype=int)

def U(n,rng=None,size=1):
    """Calculate Haar random unitary matrix of dimension `n`. The method is based on QR decomposition of a random matrix with Gaussian entries.

    Parameters
    ----------
    n : int
        dimension of the unitary matrix
    rng : numpy.random.Generator, optional
        Random generator, by default None
    size : int, optional
        Number of unitary matrix, by default 1

    Returns
    -------
    numpy.array, shape=(size,n,n)
        Haar random unitary matrix
    """
    if rng is None:
        rng=np.random.default_rng(None)
    return scipy.stats.unitary_group.rvs(n,random_state=rng,size=size)

def Haar_state(L,ensemble,rng=None,k=1):
    """Generate `k` orthogonal Haar random states, using the method in https://quantumcomputing.stackexchange.com/questions/15754/confusion-about-the-output-distribution-of-haar-random-quantum-states

    Parameters
    ----------
    L : int
        Length of the system, gives 2**L dimension of the Hilbert space
    ensemble : int
        Ensemble size
    rng : np.random.Generator, optional
        Random generator, by default None
    k : int, optional, {0,1}
        Number of orthrogonal Haar random state, by default 1

    Returns
    -------
    np.array, shape=(2**L,k,ensemble)
        The orthogonal `k` Haar random states
    """
    if rng is None:
        rng=np.random.default_rng(None)
    assert k in [1,2], f'k ({k}) is not 1 or 2'
    state=rng.normal(size=(2**L,2,k,ensemble)) # (wf, re/im, k, ensemble)
    z=state[:,0,:,:]+1j*state[:,1,:,:] # (wf, k, ensemble)
    norm=np.sqrt((np.abs(z[:,0,:])**2).sum(axis=0)) # (ensemble,)
    z[:,0,:]/=norm
    if k==2:
        z[:,1,:]-=(z[:,0,:].conj()*z[:,1,:]).sum(axis=0)*z[:,0,:]
        norm=np.sqrt((np.abs(z[:,1,:])**2).sum(axis=0))
        z[:,1,:]/=norm
    return z

@lru_cache(maxsize=None)
def bin_pad(x,L):
    """Convert an integer `x` to binary form with length `L`, with 0 padding to the left.

    Parameters
    ----------
    x : int
        integer in decimal form
    L : int
        length of the binary form

    Returns
    -------
    str
        bitstring of length `L` in binary form
    """
    return (bin(x)[2:]).rjust(L,'0')

class CT_tensor:
    def __init__(self,L,store_vec=False,store_op=False,store_prob=False,seed=None,seed_vec=None,seed_C=None,xj=set([Fraction(1,3),Fraction(2,3)]),_eps=1e-10, ancilla=False,gpu=False,complex128=True,ensemble=None,ensemble_m=1,debug=False):
        """Initialize the quantum circuit for control transition and measurement induced transition using PyTorch. The tensor is saved as (0,1,...L-1, ancilla, ensemble, ensemble_m), where `ensemble` is for the ensemble size of the circuit (position of projection and control) and `ensemble_m` is for the ensemble size of outcome within a same circuit. 
        Numpy rng mode: If `seed` is a list, this corresponds to the single state vector simulation defined in `CT_quantum`. Numpy rng generator are used, the ensemble size is the same as `len(seed)`, `ensemble` should be None.
        PyTorch rng mode: when `seed_vec` and `seed_C` are not `None`, and `seed` is a number. `seed` will apply to initial state vec, and circuit. In this case, ensemble size is simply the total number of random samples with different circuit, and `ensemble_m` is 1 (by default). When `seed_vec` and `seed_C` and `seed` are all not None. `ensemble` and `ensemble_m` controls the number of different circuit and number of outcome in each circuit.

        Parameters
        ----------
        L : int
            the length of the physical system, excluding ancilla qubit
        store_vec : bool, optional
            store the history of state vector , by default False
        store_op : bool, optional
            store the history of operators applied to the circuit, by default False
        store_prob : bool, optional
            store the history of each probability at projective measurement, by default False
        seed : list of numpy.rng or int, optional
            (1) the random seed in the measurement outcome; (2) if `seed_vec` and `seed_C` is None, this random seed also applied to initial state vector and circuit, by default None
        seed_vec : list of numpy.rng or int, optional
            the random seed in the state vector, by default None, by default None
        seed_C : list of numpy.rng or int, optional
            the random seed in the circuit, by default None, by default None
        xj : set of Fractions, optional
            the set of attractors using Fractions, by default set([Fraction(1,3),Fraction(2,3)])
        _eps : float, optional
            error threshold, by default 1e-10
        ancilla : bool, optional
            if true, an ancilla qubit is maximally entangled to the system , by default False
        gpu : bool, optional
            if true, attempt to use GPU, by default False
        complex128 : bool, optional
            if true, save `vec` using complex128, otherwise, use `complex64`, by default True
        ensemble : int, optional
            number of different circuits. If None, len(numpy.rng) are used to determine the ensemble size, by default None
        ensemble_m : int, optional
            number of different outcome within a single circuit, by default 1
        debug : bool, optional
            if trure, all assertions will be checked, by default False, by default False
        """
        self.L=L # physical L, excluding ancilla
        self.L_T=L+1 if ancilla else L # tensor L, ancilla
        self.store_vec=store_vec
        self.store_op=store_op
        self.store_prob=store_prob
        self.gpu=gpu
        self.device=self._initialize_device()
        self.ensemble=ensemble
        self.ensemble_m=ensemble_m
        self.rng=self._initialize_random_seed(seed)
        self.rng_vec=self._initialize_random_seed(seed_vec) if seed_vec is not None else self.rng
        self.rng_C=self._initialize_random_seed(seed_C) if seed_C is not None else self.rng

        self.op_history=[]  
        self.prob_history=[]  
        self.ancilla=ancilla
        self.dtype={'numpy':np.complex128,'torch':torch.complex128} if complex128 else {'numpy':np.complex64,'torch':torch.complex64}
        self.vec=self._initialize_vector()
        self.vec_history=[self.vec]
        self._eps=_eps
        self.xj=set(xj)
        self.op_list=self._initialize_op()
        self.new_idx,self.old_idx, self.not_new_idx=self.adder_gpu()
        self.debug=debug

    def _initialize_device(self):
        """Initialize the device, if `gpu` is True, use GPU, otherwise, use CPU.

        Returns
        -------
        cuda instance
            the name of GPU device

        Raises
        ------
        ValueError
            If GPU is not available, raise error.
        """
        if self.gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print('Using',device)
                return device
            else:
                raise ValueError('CUDA is not available')
        else:
            print('Using cpu')

    
    def _initialize_random_seed(self,seed):
        """Initialize random seed

        Parameters
        ----------
        seed : int or list of numpy.rng
            If int, it's the random seed for all states. If list of numpy.rng, it uses numpy random generator for each state.

        Returns
        -------
        rng : torch.Generator or list of numpy.random.Generator
            If seed is int, return torch.Generator, otherwise, return list of numpy.random.Generator
        """
        if self.ensemble is None:
            return np.array([np.random.default_rng(s) for s in seed])
        else:
            rng=torch.Generator(device=self.device)
            rng.manual_seed(seed)
            return rng

    def _initialize_vector(self):
        """Initalize the state vector.

        Returns
        -------
        torch.tensor, shape=(2,)*L+[(anc,)]+(C,M)
            [...] exists if `ancilla` is on. 
        """
        k=1 if not self.ancilla else 2
        factor=np.sqrt(1/k)
        if self.ensemble is None:
            vec=factor*torch.from_numpy(np.array([Haar_state(self.L,ensemble=1,rng=rng,k=k).astype(self.dtype['numpy']).reshape((2,)*(self.L_T)) for rng in self.rng_vec])).permute(list(range(1,self.L_T+1))+[0]).unsqueeze(-1)
            if self.gpu:
                vec=vec.cuda()
        else:
            # Though verbose, but optimize for GPU RAM usage, squeeze() tends to reserve huge memory
            vec=factor*self.Haar_state(k=k,ensemble=self.ensemble,rng=self.rng_vec)
            if k == 1:
                vec=vec[...,0,:]
            vec=vec.unsqueeze(-1).repeat(*(1,)*self.L_T+(1,self.ensemble_m,))
        return vec
    
    def _initialize_op(self):
        """Initialize the operators inthe circuit. `C` is the control map, `P` is the projection, `B` is the Bernoulli map, `I` is the identity map. The second element in the tuple is the outcome. The number after "P" is the position of projection (0-index).

        Returns
        -------
        dict of operatiors
            possible operators in the circuit
        """
        return {("C",0):partial(self.control_map,n=0),
                ("C",1):partial(self.control_map,n=1),
                (f"P{self.L-1}",0):partial(self.projection_map,pos=self.L-1,n=0),
                (f"P{self.L-1}",1):partial(self.projection_map,pos=self.L-1,n=1),
                (f"P{self.L-2}",0):partial(self.projection_map,pos=self.L-2,n=0),
                (f"P{self.L-2}",1):partial(self.projection_map,pos=self.L-2,n=1),
                ("B",):self.Bernoulli_map,
                ("I",):lambda x:x
                }

    def Bernoulli_map(self,vec,rng=None,size=None):
        """Apply Bernoulli map to the state vector `vec`. Note that this is always applied to the whole ensemble_m axis.

        Parameters
        ----------
        vec : torch.tensor
            state vector
        rng : list of np.rng or torch.Generator, optional
            Random generator to use, by default None, if None, use self.rng_C, this is used when `ensemble` is not None, namely using a single torch.Generator. Otherwise `rng` is a list of np.rng, and `ensebme` is ignored. The actual size is inferred from the length of `rng`
        size : int, optional
            If rng is scalar, `size` control the ensemble size, by default None

        Returns
        -------
        torch.tensor
            state vector after Bernoulli map
        """
        vec=self.T_tensor(vec,left=True)
        rng=self.rng_C if rng is None else rng
        vec=self.S_tensor(vec,rng=rng,size=size)
        return vec

    def control_map(self,vec,n):
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
        self.P_tensor_(vec,n,self.L-1)
        if n==1:
            vec=self.XL_tensor(vec)
        self.normalize_(vec)
        # right shift 
        vec=self.T_tensor(vec,left=False)

        # Adder
        if not vec.is_contiguous():
            vec=vec.contiguous()
        self.adder_tensor_(vec)
        
        return vec
    
    def projection_map(self,vec,pos,n):
        """Projective measurement on the state vector `vec` at position `pos` with outcome `n`, usually input vector is in the shape where (ensemble, ensemble_m) are flatten.

        Parameters
        ----------
        vec : torch.tensor
            state vector
        pos : int, {0,1,...,L-1}
            position to apply the projection
        n : int , {0,1}
            the outcome of the measurement, 0 or 1

        Returns
        -------
        torch.tensor
            state vector after projection
        """
        '''projection to `pos` with outcome of `n`
        note that here is 0-index, and pos=L-1 is the last bit'''
        self.P_tensor_(vec,n,pos)
        self.normalize_(vec)
        return vec

    def encoding(self,):
        """Encoding process: Randomly apply Bernoulli map and control map to the state vector `vec`. Note that this is always applied to the whole ensemble_m axis.
        """
        vec=self.vec
        vec=self.op_list[("B",)](vec,size=self.ensemble)
        self.update_history(vec)
        
    def random_control(self,p_ctrl,p_proj,vec=None):
        """Ramdonly apply `bernoulli_map`, `control_map` (with prob of `p_ctrl`) and `projection_map` (with prob of `p_proj`) to the state vector `vec`. 

        Parameters
        ----------
        p_ctrl : float, 0<=p_ctrl<=1
            probability of applying control map
        p_proj : float, 0<=p_proj<=1
            probability of applying projection map
        vec : torch.tensor, optional
            state vector to apply, by default None
        """
        '''the competition between chaotic and random, where the projection can only be applied after the unitary
        Notation: L-1 is the last digits'''
        if vec is None:
            vec=self.vec

        ctrl_idx_dict=self.generate_binary([torch.arange(self.rng.shape[0] if self.ensemble is None else self.ensemble,device=self.device),], p_ctrl,rng=self.rng_C)
        ctrl_0_idx_dict={} # whether projected to zero in control map
        p_0=None
        if (ctrl_idx_dict[True]).numel()>0:
            # apply control map
            vec_ctrl=vec[...,ctrl_idx_dict[True][:,0],:]
            p_0= self.inner_prob(vec=vec_ctrl,pos=[self.L-1],n_list=[0]) # prob for 0
            ctrl_idx=[ctrl_idx_dict[True][:,0], torch.arange(self.ensemble_m,device=self.device)]
            ctrl_0_idx_dict=self.generate_binary(ctrl_idx, p_0,rng=self.rng)
            for key,idx in ctrl_0_idx_dict.items():
                if len(idx)>0:
                    vec[...,idx[:,0],idx[:,1]]=self.op_list[("C",1-key)](vec[...,idx[:,0],idx[:,1]])

        proj_idx_dict={} # {pos: {True: .., False:..}} whether pos is projected site
        proj_0_idx_dict={} # {pos: {True:.., False: ..}} if projected, whether it is projected to 0 
        p_2_dict={}
        if (ctrl_idx_dict[False]).numel()>0:
            # apply Bernoulli map
            rng_ctrl=self.rng_C[ctrl_idx_dict[False][:,0].cpu().numpy()] if self.ensemble is None else None
            size=ctrl_idx_dict[False].shape[0] if self.ensemble is not None else None
            vec[...,ctrl_idx_dict[False][:,0],:]=self.op_list[("B",)](vec[...,ctrl_idx_dict[False][:,0],:],size=size,rng=rng_ctrl)
            for pos in [self.L-1,self.L-2]:
                proj_idx_dict[pos]=self.generate_binary([ctrl_idx_dict[False][:,0],], p_proj,rng=self.rng_C)
                if (proj_idx_dict[pos][True]).numel()>0:
                    vec_p=vec[...,proj_idx_dict[pos][True][:,0],:]
                    p_2 = self.inner_prob(vec=vec_p,pos=[pos], n_list=[0])
                    proj_idx=[proj_idx_dict[pos][True][:,0], torch.arange(self.ensemble_m,device=self.device)]
                    p_2_dict[pos]=p_2
                    proj_0_idx_dict[pos]=self.generate_binary(proj_idx, p_2,rng=self.rng)
                    for key,idx in proj_0_idx_dict[pos].items():
                        if len(idx)>0:
                            vec[...,idx[:,0],idx[:,1]]=self.op_list[(f'P{pos}',(1-key))](vec[...,idx[:,0],idx[:,1]])
                            
        self.update_history(vec,(ctrl_idx_dict,ctrl_0_idx_dict,proj_idx_dict,proj_0_idx_dict),(p_0,p_2_dict))

    def reference_control(self,op,vec=None):
        """Instead of randomly apply `bernoulli_map`, `control_map` and `projection_map`, apply the same operators following force quantum trajectory by applying `op`. This is used in cross entropy benchmark.

        Parameters
        ----------
        op : tuple of dict
            The operator to apply
        vec : torch.tensor, optional
            state vector, by default None
        """
        if vec is None:
            vec=self.vec
        ctrl_idx_dict,ctrl_0_idx_dict,proj_idx_dict,proj_0_idx_dict=op
        self.generate_binary([torch.arange(self.rng.shape[0] if self.ensemble is None else self.ensemble,device=self.device),], p=None,rng=self.rng_C,dummy=True) # dumm run for controvl vs Bernoulli
        p_0=None
        if (ctrl_idx_dict[True]).numel()>0:
            vec_ctrl=vec[...,ctrl_idx_dict[True][:,0],:]
            p_0= self.inner_prob(vec=vec_ctrl,pos=[self.L-1],n_list=[0])
            ctrl_idx=[ctrl_idx_dict[True][:,0], torch.arange(self.ensemble_m,device=self.device)]
            self.generate_binary(ctrl_idx, p=None,rng=self.rng,dummy=True)
            for key, idx in ctrl_0_idx_dict.items():
                if len(idx)>0:
                    vec[...,idx[:,0],idx[:,1]]=self.op_list[("C",1-key)](vec[...,idx[:,0],idx[:,1]])
        p_2_dict={}
        if (ctrl_idx_dict[False]).numel()>0:
            rng_ctrl=self.rng_C[ctrl_idx_dict[False][:,0].cpu().numpy()] if self.ensemble is None else None
            size=ctrl_idx_dict[False].shape[0] if self.ensemble is not None else None
            vec[...,ctrl_idx_dict[False][:,0],:]=self.op_list[("B",)](vec[...,ctrl_idx_dict[False][:,0],:],size=size,rng=rng_ctrl)
            for pos in [self.L-1,self.L-2]:
                vec_p=vec[...,proj_idx_dict[pos][True][:,0],:]
                p_2 = self.inner_prob(vec=vec_p,pos=[pos], n_list=[0])
                proj_idx=[proj_idx_dict[pos][True][:,0], torch.arange(self.ensemble_m,device=self.device)]
                p_2_dict[pos]=p_2
                self.generate_binary([ctrl_idx_dict[False][:,0],], p=None,rng=self.rng_C,dummy=True)
                if (proj_idx_dict[pos][True]).numel()>0:
                    proj_idx=[proj_idx_dict[pos][True][:,0], torch.arange(self.ensemble_m,device=self.device)]
                    self.generate_binary(proj_idx, p=None,rng=self.rng,dummy=True)
                    for key,idx in proj_0_idx_dict[pos].items():
                        if len(idx)>0:
                            vec[...,idx[:,0],idx[:,1]]=self.op_list[(f'P{pos}',1*(1-key))](vec[...,idx[:,0],idx[:,1]])
        self.update_history(vec,(ctrl_idx_dict,ctrl_0_idx_dict,proj_idx_dict,proj_0_idx_dict),(p_0,p_2_dict))

    def order_parameter(self,vec=None):
        """Calculate the order parameter. For `xj={1/3,2/3}`, it is \sum Z.Z, for `xj={0}`, it is \sum Z.

        Parameters
        ----------
        vec : torch.tensor, optional
            state vector, by default None

        Returns
        -------
        torch.tensor
            order parameter
        """
        if vec is None:
            vec=self.vec
        if self.xj== set([Fraction(1,3),Fraction(2,3)]):
            O=self.ZZ_tensor(vec)
        elif self.xj == set([0]):
            O=self.Z_tensor(vec)
        return O  

    def von_Neumann_entropy_pure(self,subregion,vec=None,driver='gesvd'):
        """Calculate the von Neumann entropy of a pure state, where the state vector is `vec` and the subregion is `subregion`. Using the Schmidt decomposition, the von Neumann entropy is -\sum_i \lambda_i^2 \log \lambda_i^2, where \lambda_i is the singular value of the reshaped state vector `vec`.

        Parameters
        ----------
        subregion : list of int or torch.tensor
            
        vec : _type_, optional
            The spatial subregion to calculate the von Neumann entropy, by default None
        driver : str, optional
            The driver of svd, by default 'gesvd'. Numerically found to be optimal here on GPU. If not on GPU, use default driver.

        Returns
        -------
        torch.tensor
            von Neumann entropy
        """
        if vec is None:
            vec=self.vec
        if not self.gpu:
            driver=None
        subregion=list(subregion)
        not_subregion=[i for i in range(self.L_T) if i not in subregion]
        vec=vec.permute([self.L_T,self.L_T+1]+subregion+not_subregion)
        vec_=vec.contiguous().view((self.rng.shape[0] if self.ensemble is None else self.ensemble,self.ensemble_m,2**len(subregion),2**len(not_subregion)))

        S=torch.linalg.svdvals(vec_,driver=driver)
        S_pos=torch.clamp(S,min=1e-18)
        return torch.sum(-torch.log(S_pos**2)*S_pos**2,axis=-1)

    def half_system_entanglement_entropy(self,vec=None,selfaverage=False):
        """Calculate the half-system entanglement entropy, where the state vector is `vec`. The half-system entanglement entropy is defined as \sum_{i=0..L/2-1}S_([i,i+L/2)) / (L/2), where S_([i,i+L/2)) is the von Neumann entropy of the subregion [i,i+L/2).

        Parameters
        ----------
        vec : torch.tensor, optional
            state vector, by default None
        selfaverage : bool, optional
            if true, average over all possible halves, namely, \sum_{i=0..L/2-1}S_([i,i+L/2)) / (L/2), by default False

        Returns
        -------
        torch.tensor
            Half-system entanglement entropy
        """
        '''\sum_{i=0..L/2-1}S_([i,i+L/2)) / (L/2)'''
        if vec is None:
            vec=self.vec
        if selfaverage:
            S_A=torch.mean([self.von_Neumann_entropy_pure(np.arange(i,i+self.L//2),vec) for i in range(self.L//2)])
        else:
            S_A=self.von_Neumann_entropy_pure(np.arange(self.L//2),vec)
        return S_A

    def tripartite_mutual_information(self,subregion_A,subregion_B, subregion_C,selfaverage=False,vec=None):
        """Calculate tripartite entanglement entropy. The tripartite entanglement entropy is defined as S_A+S_B+S_C-S_AB-S_AC-S_BC+S_ABC, where S_A is the von Neumann entropy of subregion A, S_AB is the von Neumann entropy of subregion A and B, etc. The system size `L` should be a divided by 4 such that the subregion A, B and C are of the same size.

        Parameters
        ----------
        subregion_A : list of int or torch.tensor
            subregion A
        subregion_B : list of int or torch.tensor
            subregion B
        subregion_C : list of int or torch.tensor
            subregion C
        selfaverage : bool, optional
            if true, average over all possible partitions, by default False, by default False
        vec : torch.tensor, optional
            state vector, by default None

        Returns
        -------
        torch.tensor
            Tripartite entanglement entropy
        """
        if self.debug:
            assert np.intersect1d(subregion_A,subregion_B).size==0 , "Subregion A and B overlap"
            assert np.intersect1d(subregion_A,subregion_C).size==0 , "Subregion A and C overlap"
            assert np.intersect1d(subregion_B,subregion_C).size==0 , "Subregion B and C overlap"
        if vec is None:
            vec=self.vec
        if selfaverage:
            return torch.mean([self.tripartite_mutual_information((subregion_A+shift)%self.L,(subregion_B+shift)%self.L,(subregion_C+shift)%self.L,selfaverage=False) for shift in range(len(subregion_A))])
        else:
            S_A=self.von_Neumann_entropy_pure(subregion_A,vec=vec)
            S_B=self.von_Neumann_entropy_pure(subregion_B,vec=vec)
            S_C=self.von_Neumann_entropy_pure(subregion_C,vec=vec)
            S_AB=self.von_Neumann_entropy_pure(np.concatenate([subregion_A,subregion_B]),vec=vec)
            S_AC=self.von_Neumann_entropy_pure(np.concatenate([subregion_A,subregion_C]),vec=vec)
            S_BC=self.von_Neumann_entropy_pure(np.concatenate([subregion_B,subregion_C]),vec=vec)
            S_ABC=self.von_Neumann_entropy_pure(np.concatenate([subregion_A,subregion_B,subregion_C]),vec=vec)
            return S_A+ S_B + S_C-S_AB-S_AC-S_BC+S_ABC
    
    def update_history(self,vec=None,op=None,p=None):
        """Update history of state vector (`vec`), operators (`op`) and probability (`p`).
        For state vector, simply append it to the list
        For operators, append the four dictionaries to the list, this is different from `CT_quantum.update_history`. Not easy to visualize but useful in `reference_control`.
        For probability, assembly the probability map first and append it to the list, such that the `prob_history` is in the format of [torch.tensor,torch.tensor,...]


        Parameters
        ----------
        vec : torch.tensor, optional
            state vector, by default None
        op : tuple of dict, optional
            operators, by default None
        p : tuple of prob, optional
            probabilities, by default None
        """
        '''Maybe I should abandon the previous way of storing history, why not just use the four *_idx_dict to store the history directly? This will also create comvenience when call it'''
        if vec is not None:
            if self.store_vec:
                self.vec_history.append(vec.cpu().clone())
            else:
                # Reserved for later use
                pass
        if op is not None:
            if self.store_op:
                # 
                self.op_history.append(op)
            else:
                pass
        if p is not None:
            if self.store_prob:
                p_0,p_2_dict=p
                dtype=torch.float64 if self.dtype['torch']==torch.complex128 else torch.float32
                ctrl_idx_dict,ctrl_0_idx_dict,proj_idx_dict,proj_0_idx_dict=op
                p_map=torch.ones((3,self.rng.shape[0] if self.ensemble is None else self.ensemble,self.ensemble_m),device=self.device,dtype=dtype)
                flip_prob=torch.ones((3,self.rng.shape[0] if self.ensemble is None else self.ensemble,self.ensemble_m),device=self.device,dtype=int)
                if ctrl_idx_dict[True].numel()>0:
                    p_map[0,ctrl_idx_dict[True][:,0]]=p_0
                    if ctrl_0_idx_dict[False].numel()>0:
                        flip_prob[0][tuple((ctrl_0_idx_dict[False]).T)]=-1
                        p_map[0]*=flip_prob[0]
                        p_map[0]+=((1-flip_prob[0])/2)
                for idx,key in enumerate(p_2_dict.keys()):
                    p_map[idx+1,proj_idx_dict[key][True][:,0]]=p_2_dict[key]
                    if key in proj_0_idx_dict:
                        if (proj_0_idx_dict[key][False]).numel()>0:
                            flip_prob[idx+1][tuple((proj_0_idx_dict[key][False]).T)]=-1
                            p_map[idx+1]*=flip_prob[idx+1]
                            p_map[idx+1]+=((1-flip_prob[idx+1])/2)
                self.prob_history.append(p_map.prod(dim=0))
            else:
                pass
        
    def normalize_(self,vec):
        """Normalize the state vector `vec` in-place. This assumes that the state vector has 1 ensemble index, because it only happens after projection which involves measurment, where the two `ensemble` and `ensemble_m` indices are flatten.

        Parameters
        ----------
        vec : torch.tensor
            state vector after normalization
        """
        norm=torch.sqrt(torch.einsum(vec.conj(),[...,0],vec,[...,0],[0]))
        if self.debug:
            assert torch.all(norm != 0) , f'Cannot normalize: norm is zero {norm}'
        vec/=norm
    
    def inner_prob(self,vec,pos,n_list):
        """Calculate the probability of measuring 0 at position `pos` for the state vector `vec`. This assumes `vec` has two ensemble index.

        Parameters
        ----------
        vec : torch.tensor
            state vector
        pos : int or list of int
            position(s) to measure
        n_list : int or list of int
            measuring outcome(s)

        Returns
        -------
        torch.tensor
            probability of measuring 0 at position `pos`
        """
        '''probability of `vec` of measuring `n_list` at `pos`
        convert the vector to tensor (2,2,..), take about the specific pos-th index, and flatten to calculate the inner product'''
        idx_list=np.array([slice(None)]*self.L_T)
        idx_list[pos]=n_list
        vec_0=vec[tuple(idx_list)]
        inner_prod=torch.einsum(vec_0.conj(),[...,0,1],vec_0,[...,0,1],[0,1])

        if self.debug:
            assert torch.all(torch.abs(inner_prod.imag)<self._eps), f'probability for outcome 0 is not real {inner_prod}'
        inner_prod=inner_prod.real
        if self.debug:
            assert torch.all(inner_prod>-self._eps), f'probability for outcome 0 is not positive {inner_prod}'
            assert torch.all(inner_prod<1+self._eps), f'probability for outcome 1 is not smaller than 1 {inner_prod}'
        inner_prod=torch.clamp_(inner_prod,min=0,max=1)
        return inner_prod

    def XL_tensor(self,vec):
        """Flip the last qubit of the state vector `vec` in-place (excluding the ancilla qubit). `roll` seems much faster than the in-place swap as done in `CT_quantum.XL_tensor`. This works both one and two ensemble indices.

        Parameters
        ----------
        vec : torch.tensor
            state vector

        Returns
        -------
        torch.tensor
            state vector after flipping the last qubit
        """
        vec=torch.roll(vec,1,dims=self.L-1)
        return vec

    def P_tensor_(self,vec,n,pos):
        """Directly set zero at tensor[...,0,:] =0 for n==1 and tensor[...,1,:] =0 for n==0. This works both one and two ensemble indices. 

        Parameters
        ----------
        vec : torch.tensor
            state vector
        n : int, {0,1}
            out come of the projection
        pos : int, {0,1,...,L-1}
            position to apply the projection.
        """
        '''directly set zero at tensor[...,0] =0 for n==1 and tensor[...,1] =0 for n==0'
        This is an in-placed operation
        '''
        idx_list=[slice(None)]*vec.dim()
        idx_list[pos]=1-n
        vec[idx_list]=0

    def T_tensor(self,vec,left=True):
        """Left or right shift the state vector `vec` in-place. This works both one and two ensemble indices.

        Parameters
        ----------
        vec : torch.tensor
            state vector
        left : bool, optional
            if true, left shift, else, right shift, by default True

        Returns
        -------
        torch.tensor
            state vector after shift
        """
        if left:
            idx_list=list(range(1,self.L))+[0]
        else:
            idx_list=[self.L-1]+list(range(self.L-1))
        
        idx_list=idx_list+list(range(self.L,vec.dim()))
        return vec.permute(idx_list)

    def S_tensor(self,vec,rng,size=None):
        """Apply scrambler to the state vector `vec` in-place. This works only for two ensemble indices.

        Parameters
        ----------
        vec : torch.tensor
            state vector
        rng : list of np.rng or torch.Generator
            Random generators to use. If rng is a list, `size` is ignored. 
        size : int, optional
            size of ensemble, by default None. If none, the size is inferred from the length of `rng`

        Returns
        -------
        torch.tensor
            state vector
        """
        if self.ensemble is None:
            U_4=torch.from_numpy(np.array([U(4,rng).astype(self.dtype['numpy']).reshape((2,)*4) for rng in rng])).unsqueeze(1)
            if self.gpu:
                U_4=U_4.cuda()
        else:
            U_4=self.U(4,rng=rng,size=(size,1)).reshape((size,1,2,2,2,2)).expand(-1,self.ensemble_m,-1,-1,-1,-1)


        if not self.ancilla:
            vec=torch.einsum(vec,[...,0,1,2,3],U_4,[2,3,4,5,0,1],[...,4,5,2,3]) # ... (L-2,L-1)(C,m) * (c,m)(L-2,L-1)'(L-2,L-1) -> ... (L-2,L-1)'(C,m)
        else:
            vec=torch.einsum(vec,[...,0,1,2,3,4],U_4,[3,4,5,6,0,1],[...,5,6,2,3,4]) # ... (L-2,L-1)(anc)(C,m) * (c,m)(L-2,L-1)'(L-2,L-1) -> ... (L-2,L-1)'(anc)(C,m)
        return vec

    def ZZ_tensor(self,vec):
        """Calculate the order parameter for Neel state. The order parameter is defined as \sum_{i=0..L-1} <Z_iZ_{i+1}>, where Z_i is the Pauli Z matrix at site i.

        Parameters
        ----------
        vec : torch.tensor
            state vector

        Returns
        -------
        torch.tensor
            order parameter for Neel state
        """
        rs=0
        for i in range(self.L):
            for zi in range(2):
                for zj in range(2):
                    inner_prod=self.inner_prob(vec, [i,(i+1)%self.L],[zi,zj])
                    exp=1-2*(zi^zj) # expectation-- zi^zj is xor of two bits which is only one when zi!=zj
                    rs+=inner_prod*exp
        return -rs/self.L
    
    def Z_tensor(self,vec):
        """Calculate the order parameter for Ferromagnetic state. The order parameter is defined as \sum_{i=0..L-1} <Z_i>, where Z_i is the Pauli Z matrix at site i.

        Parameters
        ----------
        vec : torch.tensor
            state vector

        Returns
        -------
        torch.tensor
            order parameter for ferromagnetic state
        """
        rs=0
        for i in range(self.L):
            P0=self.inner_prob(vec,[i],[0])
            rs+=P0*1+(1-P0)*(-1)
        return rs/self.L

    def adder_gpu(self):
        """Calculate the shift adder index, namely, `old_index` -> `new_index`, and `not_new_index` should be all zero. This is not a full adder, which assume the leading digit in the input bitstring is zero (because of the T^{-1}R_L, the leading bit should always be zero). 

        Returns
        -------
        tuple of torch.tensor
            new_idx, old_idx, not_new_idx
        """
        if self.xj==set([Fraction(1,3),Fraction(2,3)]):
            int_1_6=(int(Fraction(1,6)*2**self.L)|1)
            int_1_3=(int(Fraction(1,3)*2**self.L))
                
            old_idx=torch.arange(2**(self.L-1),device=self.device).view((2,-1))
            adder_idx=torch.tensor([[int_1_6],[int_1_3]],device=self.device)
            new_idx=(old_idx+adder_idx)
            # handle the extra attractors, if 1..0x1, then 1..0(1-x)1, if 0..1x0, then 0..1(1-x)0 [shouldn't enter this branch..]
            mask_1=(new_idx&(1<<self.L-1) == (1<<self.L-1)) & (new_idx&(1<<2) == (0)) & (new_idx&(1) == (1))
            mask_2=(new_idx&(1<<self.L-1) == (0)) & (new_idx&(1<<2) == (1<<2)) & (new_idx&(1) == (0))

            new_idx[mask_1+mask_2]=new_idx[mask_1+mask_2]^(0b10)

            if self.ancilla:
                new_idx=torch.hstack((new_idx<<1,(new_idx<<1)+1))
                old_idx=torch.hstack((old_idx<<1,(old_idx<<1)+1))

            not_new_idx=torch.ones(2**(self.L_T),dtype=bool,device=self.device)
            not_new_idx[new_idx]=False

            return new_idx, old_idx, not_new_idx
        if self.xj==set([0]):
            return torch.tensor([]), torch.tensor([]),torch.tensor([])

    def adder_tensor_(self,vec):
        """Apply the index shuffling of `vec` using the `old_idx` and `new_idex` generated from `adder_gpu`. 

        Parameters
        ----------
        vec : torch.tensor
            state vector 
        """
        new_idx=self.new_idx.flatten()
        old_idx=self.old_idx.flatten()
        not_new_idx=self.not_new_idx.flatten()
        if (new_idx).shape[0]>0 and (old_idx).shape[0]>0:
            vec_flatten=vec.view((2**self.L_T,-1))    
            vec_flatten[new_idx]=vec_flatten[old_idx]
            vec_flatten[not_new_idx]=0

    def generate_binary(self,idx_list,p,rng,dummy=False):
        """Generate boolean list, given probability `p` and seed `rng`.

        Parameters
        ----------
        idx_list : list of list
            [[ensemble...],[ensemble_m...]], the second list could be None if no slice on `ensemble_m` axis needed.
        p : float or torch.tensor
            if float, this is used in np.rng, if torch.tensor, this is used in torch.rng, the shape should be (idx_list[0].shape[0], ensemble_m)
        rng : list of np.rng or torch.Generator
            Random generators to use. 
        dummy : bool, optional
            If false, no dictionary will be generated, and exists for reproducibility, by default False

        Returns
        -------
        dict
            A dictionary to store the index of operators.
        """
        if self.ensemble is None:
            true_list=[]
            false_list=[]
            if isinstance(p, float) or isinstance(p, int):
                # for randomness at circuit leverl
                for idx in idx_list[0]:
                    random=rng[idx].random()
                    if not dummy:
                        if random<=p:
                            true_list.append([idx,0])
                        else:
                            false_list.append([idx,0])
            else:
                # for randomness at outcome level
                if not dummy and self.debug:
                    assert len(idx_list[0]) == len(p), f'len of idx_list {len(idx_list)} is not same as len of p {len(p)}'
                # Assume that ensemble_m=1, because this is for np.rng, therefore it always append [idx,0]
                for idx,p in zip(idx_list[0],p):
                    random=rng[idx].random() # This assume idx_list is always in the shape of (-1,2), which means, it is for the outcome randomness
                    if not dummy:
                        if random<=p:
                            true_list.append([idx,0])
                        else:
                            false_list.append([idx,0])
                        
            if not dummy:
                idx_dict={True:torch.tensor(true_list,dtype=int,device=self.device),False:torch.tensor(false_list,dtype=int,device=self.device)}
                return idx_dict
        else:
            if isinstance(p, float) or isinstance(p, int):
                # for randomness at circuit leverl
                p=p*torch.ones((idx_list[0].shape[0],),device=self.device)

            p_rand=torch.rand(*(idx_list[i].shape[0] for i in range(len(idx_list))),device=self.device,generator=rng)
            if not dummy:
                p_rand=(p_rand<p)
                true_idx=torch.nonzero(p_rand,)
                false_idx=torch.nonzero(~p_rand)
                
                true_tensor=torch.vstack([idx_list[i][true_idx[:,i]] for i in range(true_idx.shape[-1])]).T
                false_tensor=torch.vstack([idx_list[i][false_idx[:,i]] for i in range(false_idx.shape[-1])]).T
                idx_dict={True:true_tensor,False:false_tensor}
                return idx_dict


    def U(self,n,rng,size):
        """Generate random unitary matrix of size `n` using Haar measure. This works for two ensemble indices.

        Parameters
        ----------
        n : int
            dimension of unitary matrix
        rng : list of np.rng or torch.Generator
            if rng is list, `size` is ignored, and the size is inferred from the length of `rng`. Othewise, generate `size` copies. random unitary matrices.
        size : tuple
            (ensemble, ensemble_m) copies of random unitary matrices. Usually ensemble_m=1 because the state within same circuit index should evolve under the same unitary matrix..

        Returns
        -------
        torch.tensor, (ensemble, ensemble_m, n, n)
            stacked unitary matrix
        """
        dtype=torch.float64 if self.dtype['torch']==torch.complex128 else torch.float32
        im = torch.randn((*size,n, n), device=self.device,dtype=dtype,generator=rng)
        re = torch.randn((*size,n, n), device=self.device,dtype=dtype,generator=rng)
        z=torch.complex(re,im)
        Q,R=torch.linalg.qr(z)
        r_diag=torch.diagonal(R,dim1=-2,dim2=-1)
        Lambda=torch.diag_embed(r_diag/torch.abs(r_diag))
        Q=torch.einsum(Q,[...,0,1],Lambda,[...,1,2],[...,0,2])
        return Q

    def Haar_state(self,ensemble,rng,k=1):
        """Generate `k` Haar random state. This works for two ensemble indices.

        Parameters
        ----------
        ensemble : int
            `ensemble` copies of Haar random state.
        rng : list of np.rng or torch.Generator
            random generators to use. 
        k : int, {0,1}, optional
            1 or 2 Haar random state, if k=2, the two states are orthogonal, by default 1

        Returns
        -------
        torch.tensor, (2,2,...,k,ensemble)
            Haar random state
        """
        # Generate k orthorgonal Haar random state
        dtype=torch.float64 if self.dtype['torch']==torch.complex128 else torch.float32
        state=torch.randn((2,)*(self.L+1)+(k,)+(ensemble,),device=self.device,dtype=dtype,generator=rng) # wf, re/im, k,ensemble
        state=torch.complex(state[:,0,:,:],state[:,1,:,:]) # wf, k, ensemble
        norm=(torch.einsum(state[...,0,:].conj(),[...,0],state[...,0,:],[...,0],[0])) # ensemble
        state[...,0,:]/=torch.sqrt(norm)
        if k==2:
            overlap=torch.einsum(state[...,0,:].conj(),[...,0],state[...,1,:],[...,0],[0]) #ensemble
            state[...,1,:]-=overlap*state[...,0,:]
            norm=(torch.einsum(state[...,1,:].conj(),[...,0],state[...,1,:],[...,0],[0])) 
            state[...,1,:]/=torch.sqrt(norm)
        return state
