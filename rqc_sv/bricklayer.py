import scipy.sparse as sp
import numpy as np
from .utils import U, Haar_state, dec2bin
from functools import lru_cache
from fractions import Fraction

class bricklayer:
    def __init__(self,L,store_vec=False,store_op=False,store_prob=False,seed=None,seed_vec=None,seed_C=None,x0=None,_eps=1e-10,add_x=0,debug=False,feedback=False,xj=None):
        self.L=L
        self.store_vec=store_vec
        self.store_op=store_op
        self.store_prob=store_prob
        self.rng=np.random.default_rng(seed)
        self.rng_vec=np.random.default_rng(seed_vec) if seed_vec is not None else self.rng
        self.rng_C=np.random.default_rng(seed_C) if seed_C is not None else self.rng
        self.x0=x0
        self.op_history=[] # store the history of operators applied to the circuit
        self.prob_history=[]  # store the history of each probability at projective measurement
        self.vec=self._initialize_vector() # initialize the state vector
        self.vec_history=[self.vec] # store the history of state vector
        self._eps=_eps
        self.add_x=add_x
        self.debug=debug
        self.feedback=feedback
        self.xj=xj

    def _initialize_vector(self):
        if self.x0 is not None:
            vec_int=dec2bin(self.x0,self.L)
            vec=np.zeros((2**self.L,),dtype=complex)
            vec[vec_int]=1
        else:
            vec=Haar_state(self.L, 1,rng=self.rng_vec,k=1).flatten()
        return vec.reshape((2,)*self.L)

    def U_tensor(self,vec,i,rng):
        '''Haar random unitary, applies to (i,(i+1)%L)'''
        U_4=U(4,rng).reshape((2,)*4)
        print(U_4)
        if vec.ndim != (2,)*self.L:
            vec=vec.reshape((2,)*self.L)
        sites_list=np.arange(self.L)
        sites_list[[i,(i+1)%self.L]] = [self.L,self.L+1]
        vec = np.einsum(vec,np.arange(self.L),U_4,[self.L,self.L+1,(i)%self.L,(i+1)%self.L], sites_list) # .. (i,i+1%L) ..  * (i,i+1%L)' (i,i+1%L) -> ... (i,i+1%L)' ...
        return vec

    def P_tensor(self,vec,n,i):
        '''n: outcome 0 or 1, i: position'''
        if vec.ndim!=self.L:
            vec=vec.reshape((2,)*self.L)
        idx_list=[slice(None)]*self.L
        idx_list[i]=1-n
        vec[tuple(idx_list)]=0
        return vec

    def adder_cpu(self,vec,i=0):
        if vec.ndim >1 :
            vec=vec.flatten()
        vec=self.adder(i)@vec
        return vec

    def _right_shift(self,d,i):
        ''' make i-th bit in the output the leading bit'''
        d_bin=bin(d)[2:].zfill(self.L)
        return int(d_bin[-i:]+d_bin[:-i],2)

    def _left_shift(self,d,i):
        ''' make i-th bit in the input the leading bit'''
        d_bin=bin(d)[2:].zfill(self.L)
        return int(d_bin[i:]+d_bin[:i],2)


    @lru_cache(maxsize=None)
    def adder(self,i):
        ''' if xj is None, then use add_x as a universal adder, 
        otherwise, use xj as the adder, which is a outcome dependent adder, currently support only xj={Fraction(1,3),Fraction(2,3)}'''
        old_idx=np.arange(2**(self.L))
        if self.xj is None:
            add_x=self._right_shift(self.add_x,i)
            # print(add_x)
            new_idx=(old_idx+add_x)%2**self.L
        else:
            if self.xj==set([Fraction(1,3),Fraction(2,3)]):
                int_1_6=dec2bin(Fraction(1,6), self.L)|1
                int_1_3=dec2bin(Fraction(1,3), self.L)
                # add_x_1_6=self._right_shift(int_1_6,i)
                # add_x_1_3=self._right_shift(int_1_3,i)
                add_x=np.array([int_1_6]*2**(self.L-2)+[int_1_3]*2**(self.L-2)+[int_1_6]*2**(self.L-2)+[int_1_3]*2**(self.L-2))
                new_idx=(old_idx+add_x)%2**self.L
                mask_1=(new_idx&(1<<self.L-1) == (1<<self.L-1)) & (new_idx&(1<<2) == (0)) & (new_idx&(1) == (1))
                mask_2=(new_idx&(1<<self.L-1) == (0)) & (new_idx&(1<<2) == (1<<2)) & (new_idx&(1) == (0))
                new_idx[mask_1+mask_2]=new_idx[mask_1+mask_2]^(0b10)

                # Recover to real basis
                old_idx=[self._right_shift(d=idx,i=i) for idx in old_idx]
                new_idx=[self._right_shift(d=idx,i=i) for idx in new_idx]

                # print(np.vstack([old_idx,new_idx]))
                # print(old_idx_2)
                # print(new_idx_2)
                # print(new_idx)
        
        ones=np.ones(2**(self.L))
        return sp.coo_matrix((ones,(new_idx,old_idx)),shape=(2**self.L,2**self.L))
    def inner_prob(self,vec,n,i):
        '''n: outcome 0 or 1, i: position'''
        if vec.ndim != (2,)*self.L:
            vec=vec.reshape((2,)*self.L)
        idx_list=np.array([slice(None)]*self.L)
        idx_list[i]=n
        vec_0=vec[tuple(idx_list)].flatten()
        inner_prod=vec_0.conj()@vec_0
        if self.debug:
            assert np.abs(inner_prod.imag)<self._eps, f'probability for outcome 0 is not real {inner_prod}'
            assert inner_prod>-self._eps, f'probability for outcome 0 is not positive {inner_prod}'
            assert inner_prod<1+self._eps, f'probability for outcome 1 is not smaller than 1 {inner_prod}'
        inner_prod=np.clip(inner_prod.real,0,1)
        return inner_prod

    def normalize(self,vec):
        if vec.ndim!=1:
            vec=vec.flatten()
        norm2=(vec.conj().T@vec).real
        if norm2>0:
            return vec/np.sqrt(norm2)
        else:
            return vec

    def R_tensor(self,vec,n,i):
        vec=self.P_tensor(vec,n,i)
        if self.feedback and n==1:
            vec=self.X_tensor(vec,i)
        return vec
    
    def X_tensor(self,vec,i):
        if vec.ndim!=self.L:
            vec=vec.reshape((2,)*self.L)
        vec=np.roll(vec,1,axis=i)
        return vec

    def U_layer(self,vec,even=True):
        '''apply one layers of U gates, even= True (2i,2i+1), False (2i+1,2i+2)'''
        idx=0 if even else 1
        for i in range(self.L//2):
            vec=self.U_tensor(vec,idx,rng=self.rng_C)
            idx+=2
        return vec
    
    def P_layer(self,vec,p_proj):
        '''apply one layers of P gates'''
        for i in range(self.L)[::-1]:
            if self.rng_C.random()<p_proj:
                p_0=self.inner_prob(vec,i=i,n=0)
                if self.rng.random()<p_0:
                    vec=self.P_tensor(vec,n=0,i=i)
                else:
                    vec=self.P_tensor(vec,n=1,i=i)
                vec=self.normalize(vec)
        return vec

    def C_layer(self,vec,p_proj):
        ''' apply one layer of Control (C) gates, C=U R, where R is the measurement with feedback'''
        for i in range(self.L)[::-1]:
            if self.rng_C.random()<p_proj:
                p_0=self.inner_prob(vec,i=i,n=0)
                if self.rng.random()<p_0:
                    vec=self.R_tensor(vec,n=0,i=i)
                else:
                    vec=self.R_tensor(vec,n=1,i=i)
                vec=self.normalize(vec)
                vec=self.adder_cpu(vec,i)
        return vec

    def U_layer_staircase(self,vec,start):
        '''apply the unitary in a staircase fashion, starting with (-1,0), to (L-2,L-1)'''
        for idx in range(start,start+self.L):
            i=idx%self.L
            print(f'Apply U to {i,(i+1)%self.L}')
            vec=self.U_tensor(vec,i%self.L,rng=self.rng_C)
            # print(vec.flatten())
        return vec,(i+1)%self.L
    
    def C_layer_binomial(self,vec,p_proj,start):
        '''apply the control map from right to left for n times, where n obeys binomial distribution'''
        n=self.rng_C.binomial(self.L,p_proj)
        # for i in range(self.L-n,self.L)[::-1]:
        if n>0:
            for idx in range(start,start-n,-1):
                i=idx%self.L
                p_0=self.inner_prob(vec,i=i,n=0)
                outcome= 0 if self.rng.random()<=p_0 else 1
                print(f'Apply control to {i} with {outcome}')
                vec=self.R_tensor(vec,n=outcome,i=i)
                vec=self.normalize(vec)
                # print(np.where(vec.flatten()))
                vec=self.adder_cpu(vec,i)
                # print(np.where(vec.flatten()))
                # print(vec.flatten())
            return vec,(i-1)%self.L
        else:
            return vec,start


    def random_projection(self,p_proj):
        ''' a global adder after a layer of projection'''
        vec=self.vec_history[-1].copy()
        vec=self.U_layer(vec,even=True)
        vec=self.P_layer(vec,p_proj)
        vec=self.adder_cpu(vec)
        vec=self.U_layer(vec,even=False)
        vec=self.P_layer(vec,p_proj)
        self.update_history(vec,)

    def random_projection_2(self,p_proj):
        ''' an adder right after each projection'''
        vec=self.vec_history[-1].copy()
        vec=self.U_layer(vec,even=True)
        vec=self.C_layer(vec,p_proj)
        vec=self.U_layer(vec,even=False)
        vec=self.C_layer(vec,p_proj)
        self.update_history(vec,)
    
    def random_projection_3(self,p_proj,start):
        ''' the control map should be applied successively, from left to right, without skipping any qubit'''
        vec=self.vec_history[-1].copy()
        vec,start=self.U_layer_staircase(vec,start=start)
        vec,start=self.C_layer_binomial(vec,p_proj,start=start)
        self.update_history(vec,)
        return start

    def update_history(self,vec):
        if vec is not None:
            vec=vec.copy().flatten()
            if self.store_vec:
                self.vec_history.append(vec)
            else:
                self.vec_history=[vec]
        
        


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
        vec : np.array, shape=(2**L,) or (2,)*L, optional
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
            S_A=self.von_Neumann_entropy_pure(subregion_A,vec=vec,n=n,threshold=threshold)
            S_B=self.von_Neumann_entropy_pure(subregion_B,vec=vec,n=n,threshold=threshold)
            S_C=self.von_Neumann_entropy_pure(subregion_C,vec=vec,n=n,threshold=threshold)
            S_AB=self.von_Neumann_entropy_pure(np.concatenate([subregion_A,subregion_B]),vec=vec,n=n,threshold=threshold)
            S_AC=self.von_Neumann_entropy_pure(np.concatenate([subregion_A,subregion_C]),vec=vec,n=n,threshold=threshold)
            S_BC=self.von_Neumann_entropy_pure(np.concatenate([subregion_B,subregion_C]),vec=vec,n=n,threshold=threshold)
            S_ABC=self.von_Neumann_entropy_pure(np.concatenate([subregion_A,subregion_B,subregion_C]),vec=vec,n=n,threshold=threshold)
            # print(S_A+ S_B , S_C,S_AB,S_AC,S_BC,S_ABC)
            return S_A+ S_B + S_C-S_AB-S_AC-S_BC+S_ABC

    def half_system_entanglement_entropy(self,vec=None,selfaverage=False,n=1,threshold=1e-10):
        """Calculate the half-system entanglement entropy, where the state vector is `vec`. The half-system entanglement entropy is defined as \sum_{i=0..L/2-1}S_([i,i+L/2)) / (L/2), where S_([i,i+L/2)) is the von Neumann entropy of the subregion [i,i+L/2).

        Parameters
        ----------
        vec : np.array, shape=(2**L,) or (2,)*L, optional
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
            return self.von_Neumann_entropy_pure(np.arange(self.L//2),vec,n=n,threshold=threshold)

    def von_Neumann_entropy_pure(self,subregion,vec=None,n=1,threshold=1e-10):
        """Calculate the von Neumann entropy of a pure state, where the state vector is `vec` and the subregion is `subregion`. Using the Schmidt decomposition, the von Neumann entropy is -\sum_i \lambda_i^2 \log \lambda_i^2, where \lambda_i is the singular value of the reshaped state vector `vec`.

        Parameters
        ----------
        subregion : list of int or np.array
            The spatial subregion to calculate the von Neumann entropy
        vec : np.array, shape=(2**L,) or (2,)*L, optional
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
            vec=self.vec_history[-1].copy()
        vec_tensor=vec.reshape((2,)*(self.L))
        subregion=list(subregion)
        not_subregion=[i for i in range(self.L) if i not in subregion]
        vec_tensor_T=vec_tensor.transpose(np.hstack([subregion , not_subregion]))
        S=np.linalg.svd(vec_tensor_T.reshape((2**len(subregion),2**len(not_subregion))),compute_uv=False)
        if threshold is not None:
            S_pos2=np.clip(S,threshold,None)**2
        else:
            S_pos2=S**2
        if n==1:
            return -np.sum(np.log(S_pos2)*S_pos2)
        elif n==0:
            return np.log((S_pos2>threshold).sum())
        elif n==np.inf:
            return -np.log(np.max(S_pos2))
        else:
            return np.log(np.sum(S_pos2**n))/(1-n)

    def order_parameter(self,vec=None,kind='AFM'):
        if vec is None:
            vec=self.vec_history[-1].copy()
        if kind=='AFM':
            O=self.ZZ_tensor(vec)
        else:
            raise ValueError(f'kind {kind} not supported')
        return O.real
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
        if vec.ndim != (2,)*self.L:
            vec=vec.reshape((2,)*self.L)
        rs=0
        for i in range(self.L):
            for zi in range(2):
                for zj in range(2):
                    # idx_list=[slice(None)]*self.L
                    # idx_list[i],idx_list[(i+1)%self.L]=zi,zj
                    inner_prod=self.inner_prob(vec, i=[i,(i+1)%self.L],n=[zi,zj])
                    exp=1-2*(zi^zj) # expectation-- zi^zj is xor of two bits which is only one when zi!=zj
                    # vec_i=vec[tuple(idx_list)].flatten()
                    # rs+=vec_i.conj()@vec_i*exp
                    rs+=inner_prod*exp
        return -rs/self.L


        
