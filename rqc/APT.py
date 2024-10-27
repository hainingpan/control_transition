import numpy as np
# import jax.numpy as np
from .utils import U, Haar_state, dec2bin
from opt_einsum import contract, contract_expression
class APT:
# Absorbing phase transition, may need a better name later
    def __init__(self,L,seed=None,seed_vec=None,seed_C=None,x0=None,store_op=False):
        """
        seed : int, optional
            (1) the random seed in the measurement outcome; (2) if `seed_vec` and `seed_C` is None, this random seed also applied to initial state vector and circuit, by default None
        seed_vec : int, optional
            the random seed in the state vector, by default None
        seed_C : int, optional
            the random seed in the circuit, by default None
        """
        self.L=L
        self.x0=x0
        self.store_op=store_op
        self.rng=np.random.default_rng(seed)
        self.rng_vec=np.random.default_rng(seed_vec) if seed_vec is not None else self.rng
        self.rng_C=np.random.default_rng(seed_C) if seed_C is not None else self.rng
        self.rng=np.random.default_rng(seed)
        self.vec=self._initialize_vector()
        self.U3=np.zeros((2,2,2,2),dtype=complex)
        self.U3[0,0,0,0]=1
        self.sites=np.arange(L)
        self.delta={0:np.array([1,0]),1:np.array([0,1])}  
        self.op_history=[]  
        self.left_idx={True:np.arange(0,L,2),False:np.arange(1,L,2)}


    def _initialize_vector(self):
        """Save the state vector using an array of 2**L_T, if ancilla qubit, the last qubit is the ancilla qubit.
        If ancilla is False, the initial state is either the Haar random state or the state represented by `x0` (if specified)
        If ancilla is True, the initial state is the maximally entangled state between the system and the ancilla qubit, where the system is Haar random state. Namely, |psi>=1/sqrt(2)(|psi_0>|0>+|psi_1>|1>), where |psi_0> and |psi_1> are two orthogonal Haar random states. 

        Returns
        -------
        np.array, shape=(2**L_T,) or (2,)*L_T
            the initial state vector
        """
        if self.x0 is not None:
            vec_int=dec2bin(self.x0,self.L)
            vec=np.zeros((2**self.L,),dtype=complex)
            vec[vec_int]=1
        else:
            vec=Haar_state(self.L, 1,rng=self.rng_vec,k=1).flatten()
        return vec.reshape((2,)*self.L)
            
    def unitary_layer(self,even=True):
        for i in self.left_idx[even]:
            self.unitary(i)
            if self.store_op:
                self.op_history.append([i,self.U3.copy()])

    def unitary(self,i):
        """ applies to (i,i+1)"""
        self.generate_U3(self.rng_C)
        l,r=i,(i+1)%self.L
        new_sites=np.arange(self.L)
        new_sites[[l,r]]=self.L,self.L+1
        self.vec=contract(self.vec, self.sites, self.U3, [self.L,self.L+1,l,r], new_sites)

    def generate_U3(self,rng):
        U3=U(3,rng)
        # U3=np.arange(1,10).reshape((3,3))
        self.U3[1,:,1,:]=U3[1:,1:]
        self.U3[0,1,0,1]=U3[0,0]
        self.U3[0,1,1,:]=U3[0,1:]
        self.U3[1,:,0,1]=U3[1:,0]
        return U3
    
    def P(self,n,pos):
        """pos: position of measurement, n: measurement outcome"""
        idx_list=[slice(None)]*self.L
        idx_list[pos]=1-n
        self.vec[tuple(idx_list)]=0
        # if self.store_op:
        #     self.op_history.append([[pos],n])

    def X(self,pos):
        self.vec=np.roll(self.vec,1,axis=pos)
        # if self.store_op:
        #     self.op_history.append([[pos],'X'])
    
    def inner_prob(self,pos,n):
        """ pos: position of measurement, count from left"""
        return contract(self.vec.conj(), np.arange(self.L), self.vec, np.arange(self.L), self.delta[n],[pos]).real

    def normalize(self):
        norm=np.linalg.norm(self.vec)
        self.vec/=norm

    def random_cicuit(self,p_m,p_f,even):
        """p_m: measurement probability, p_f: feedback probability"""
        self.unitary_layer(even=even)
        measure_idx=np.arange(self.L)[(self.rng_C.random(self.L)<p_m)]
        for idx in measure_idx:
            n0=int(self.rng.random() >= self.inner_prob(idx,0))
            self.P(n0,pos=idx)
            if self.store_op:
                self.op_history.append([idx,n0])
            self.normalize()
            if self.rng_C.random() < p_f and n0:
                self.X(idx)
                if self.store_op:
                    self.op_history[-1].append('X')

    def order_parameter(self):
        return np.sum([self.inner_prob(pos=i,n=1) for i in range(self.L)])/self.L
            
    def half_system_entanglement_entropy(self,vec=None,selfaverage=False,n=1,threshold=1e-10):
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
        # if vec is None:
        #     vec=self.vec.copy()
        if selfaverage:
            return np.mean([self.von_Neumann_entropy_pure(np.arange(i,i+self.L//2),vec) for i in range(self.L//2)])
        else:
            return self.von_Neumann_entropy_pure(np.arange(self.L//2),vec,n=n,threshold=threshold)
    
    def von_Neumann_entropy_pure(self,subregion,vec=None,n=1,threshold=1e-10):
        """Calculate the von Neumann entropy of a pure state, where the state vector is `vec` and the subregion is `subregion`. Using the Schmidt decomposition, the von Neumann entropy is -\sum_i \lambda_i^2 \log \lambda_i^2, where \lambda_i is the singular value of the reshaped state vector `vec`.

        Parameters
        ----------
        subregion : list of int or np.array
            The spatial subregion to calculate the von Neumann entropy
        vec : np.array, shape=(2**L_T,) or (2,)*L_T, optional
            state vector, by default None
        n: int, optional,
            n-th Renyi entropy
        threshold: float, optional
            threshold to clip the singular value, by default 1e-10. For 0-th Reny entropy, threshold is 1e-8, by empirical observation.

        Returns
        -------
        float
            Von Neumann entropy of the subregion
        """
        if vec is None:
            vec=self.vec.copy()
        vec_tensor=vec.reshape((2,)*(self.L))
        subregion=list(subregion)
        not_subregion=[i for i in range(self.L) if i not in subregion]
        vec_tensor_T=vec_tensor.transpose(np.hstack([subregion , not_subregion]))
        S=np.linalg.svd(vec_tensor_T.reshape((2**len(subregion),2**len(not_subregion))),compute_uv=False)
        S_pos=np.clip(S,1e-18,None)
        S_pos2=S_pos**2
        if n==1:
            return -np.sum(np.log(S_pos2)*S_pos2)
        elif n==0:
            return np.log((S_pos2>threshold**2).sum())
        elif n==np.inf:
            return -np.log(np.max(S_pos2))
        else:
            return np.log(np.sum(S_pos2**n))/(1-n)

    def tripartite_mutual_information(self,subregion_A,subregion_B, subregion_C,selfaverage=False,vec=None,n=1,threshold=1e-10):
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
        if selfaverage:
            return np.mean([self.tripartite_mutual_information((subregion_A+shift)%self.L,(subregion_B+shift)%self.L,(subregion_C+shift)%self.L,selfaverage=False) for shift in range(len(subregion_A))])
        else:
            S_A=self.von_Neumann_entropy_pure(subregion_A,vec=vec,n=n,threshold=threshold)
            S_B=self.von_Neumann_entropy_pure(subregion_B,vec=vec,n=n,threshold=threshold)
            S_C=self.von_Neumann_entropy_pure(subregion_C,vec=vec,n=n,threshold=threshold)
            S_AB=self.von_Neumann_entropy_pure(np.concatenate([subregion_A,subregion_B]),vec=vec,n=n,threshold=threshold)
            S_AC=self.von_Neumann_entropy_pure(np.concatenate([subregion_A,subregion_C]),vec=vec,n=n,threshold=threshold)
            S_BC=self.von_Neumann_entropy_pure(np.concatenate([subregion_B,subregion_C]),vec=vec,n=n,threshold=threshold)
            S_ABC=self.von_Neumann_entropy_pure(np.concatenate([subregion_A,subregion_B,subregion_C]),vec=vec,n=n,threshold=threshold)
            return S_A+ S_B + S_C-S_AB-S_AC-S_BC+S_ABC





