import numpy as np
from functools import reduce
import scipy.sparse as sp
import scipy
from fractions import Fraction
from functools import partial, lru_cache
import torch

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

class CT_quantum:
    def __init__(self,L,history=False,seed=None,x0=None,xj=set([Fraction(1,3),Fraction(2,3)]),_eps=1e-10, ancilla=False):
        '''save using an array of 2^L'''
        self.L=L # physical L, excluding ancilla
        self.L_T=L+1 if ancilla else L # tensor L, ancilla
        self.history=history
        self.rng=np.random.default_rng(seed)
        self.x0=self.rng.random() if x0 is None else x0
        self.op_history=[]  # control: true, Bernoulli: false
        self.ancilla=ancilla
        self.vec=self._initialize_vector()
        self.vec_history=[self.vec]
        self._eps=_eps
        self.xj=set(xj)
    
    def _initialize_vector(self):
        '''save using an array of 2^L
        if ancilla qubit: it should be put to the last qubit, positioned as L, entangled with L-1'''
        if not self.ancilla:
            vec_int=int(''.join(map(str,dec2bin(self.x0,self.L))),2)
            vec=np.zeros((2**self.L,),dtype=complex)
            vec[vec_int]=1
        else:
            # Simply create a GHZ state, (|0...0> + |1...1> )/sqrt(2)
            vec=np.zeros((2**(self.L+1),),dtype=complex)
            vec[0]=1/np.sqrt(2)
            vec[-1]=1/np.sqrt(2)
            # Randomize it 
            for _ in range(self.L**2):
                vec=self.Bernoulli_map(vec)
        return vec

    def Bernoulli_map(self,vec):
        # vec=T(self.L,left=True)@vec
        vec=self.T_tensor(vec,left=True)
        # vec=S(self.L,rng=self.rng)@vec
        vec=self.S_tensor(vec,rng=self.rng)
        return vec
    
    def control_map(self,vec,bL):
        '''control map depends on the outcome of the measurement of bL'''
        # projection on the last bits
        # P_cached=P(self.L,bL)
        # vec=P_cached@vec
        vec=self.P_tensor(vec,bL)
        if bL==1:
            vec=self.XL_tensor(vec)
        vec=self.normalize(vec)
        # right shift 
        # vec=T(self.L,left=False)@vec
        vec=self.T_tensor(vec,left=False)

        assert np.abs(vec[vec.shape[0]//2:]).sum() == 0, f'first qubit is not zero ({np.abs(vec[vec.shape[0]//2:]).sum()}) after right shift '

        # Adder
        if not self.ancilla:
            vec=self.adder()@vec
        else:
            vec=(self.adder()@vec.reshape((2**self.L,2))).flatten()
        
        return vec

    def projection_map(self,vec,pos,n):
        '''projection to `pos` with outcome of `n`
        note that here is 0-index, and pos=L-1 is the last bit'''
        # vec=P(self.L,n=n,pos=pos)@vec
        vec=self.P_tensor(vec,n,pos)
        vec=self.normalize(vec)

        # proj to any axis
        # U_2=U(2,self.rng)
        # # if not self.ancilla:
        # vec_tensor=vec.reshape((2,)*self.L_T)
        # idx_list=np.arange(self.L_T)
        # idx_list[pos],idx_list[0]=idx_list[0],idx_list[pos]
        # vec_tensor=vec_tensor.transpose(idx_list).reshape((2,2**(self.L_T-1)))
        # vec=(U_2@vec_tensor).reshape((2,)*self.L_T).transpose(idx_list).flatten()

        return vec

    def random_control(self,p_ctrl,p_proj):
        '''
        p_ctrl: the control probability
        p_proj: the projection probability
        This is not the desired protocol, too strong measurement.
        '''
        vec=self.vec_history[-1].copy()

        p={}
        p[("L",0)]= vec.conj()@P(self.L,n=0,pos=self.L-1)@vec
        p[("L",1)] = vec.conj()@P(self.L,n=1,pos=self.L-1)@vec
        p[("L-1",0)] = vec.conj()@P(self.L,n=0,pos=self.L-2)@vec
        p[("L-1",1)] = vec.conj()@P(self.L,n=1,pos=self.L-2)@vec
        
        for key, val in p.items():
            assert np.abs(val.imag)<self._eps, f'probability for {key} is not real {val}'
            p[key]=val.real

        pool = ["C0","C1","PL0","PL1","PL-10","PL-11","chaotic"]
        probabilities = [p_ctrl * p[("L",0)], p_ctrl * p[("L",1)], p_proj * p[("L",0)], p_proj *  p[("L",1)], p_proj * p[("L-1",0)], p_proj * p[("L-1",1)], 1- p_ctrl-2*p_proj]

        op = self.rng.choice(pool,p=probabilities)

        op_list= {"C0":partial(self.control_map,bL=0),
                  "C1":partial(self.control_map,bL=1),
                  "PL0":partial(self.projection_map,pos=self.L-1,n=0),
                  "PL1":partial(self.projection_map,pos=self.L-1,n=1),
                  "PL-10":partial(self.projection_map,pos=self.L-2,n=0),
                  "PL-11":partial(self.projection_map,pos=self.L-2,n=1),
                  "chaotic":self.Bernoulli_map
                  }

        vec=op_list[op](vec)
        
        if self.history:
            self.vec_history.append(vec)
            self.op_history.append(op)
        else:
            self.vec_history=[vec]
            self.op_history=[op]

    def random_control_2(self,p_ctrl,p_proj):
        '''the competition between chaotic and random, where the projection can only be applied after the unitary
        Notation: L-1 is the last digits'''
        vec=self.vec_history[-1].copy()
        
        p= self.get_prob_tensor([self.L-1],vec)

        pool = ["C0","C1","chaotic"]
        probabilities = [p_ctrl * p[(self.L-1,0)], p_ctrl * p[(self.L-1,1)],  1- p_ctrl]

        op = self.rng.choice(pool,p=probabilities)

        op_list= {"C0":partial(self.control_map,bL=0),
                  "C1":partial(self.control_map,bL=1),
                  f"P{self.L-1}0":partial(self.projection_map,pos=self.L-1,n=0),
                  f"P{self.L-1}1":partial(self.projection_map,pos=self.L-1,n=1),
                  f"P{self.L-2}0":partial(self.projection_map,pos=self.L-2,n=0),
                  f"P{self.L-2}1":partial(self.projection_map,pos=self.L-2,n=1),
                  "chaotic":self.Bernoulli_map,
                  "I":lambda x:x
                  }
        vec=op_list[op](vec)
        self.update_history(vec,op)

        if op=="chaotic":
            for pos in [self.L-1,self.L-2]:
                p_2=self.get_prob_tensor([pos], vec)
                pool_2=["I",f"P{pos}0",f"P{pos}1"]
                probabilities_2=[1-p_proj, p_proj * p_2[(pos,0)], p_proj *  p_2[(pos,1)],]
                op_2 = self.rng.choice(pool_2,p=probabilities_2)
                vec=op_list[op_2](vec)
                self.update_history(vec,op_2)

    def random_control_3(self,p_ctrl,p_proj):
        '''This is the same protocol as `random_control_2`, the only difference is the way to generate rng, and document op'''
        vec=self.vec_history[-1].copy()
        op_list= {"C0":partial(self.control_map,bL=0),
                  "C1":partial(self.control_map,bL=1),
                  f"P{self.L-1}0":partial(self.projection_map,pos=self.L-1,n=0),
                  f"P{self.L-1}1":partial(self.projection_map,pos=self.L-1,n=1),
                  f"P{self.L-2}0":partial(self.projection_map,pos=self.L-2,n=0),
                  f"P{self.L-2}1":partial(self.projection_map,pos=self.L-2,n=1),
                  "chaotic":self.Bernoulli_map,
                  "I":lambda x:x
                  }

        if self.rng.random()<=p_ctrl:
            # control 
            p_0=self.inner_prob(vec, pos=self.L-1,)
            op='C0' if self.rng.random()<=p_0 else 'C1'
        else:
            # chaotic
            op='chaotic'
        
        vec=op_list[op](vec)

        if op=="chaotic":
            for pos in [self.L-1,self.L-2]:
                if self.rng.random()<p_proj:
                    # projection
                    p_2=self.inner_prob(vec, pos=pos,)
                    op_2=f"P{pos}0" if self.rng.random()<p_2 else f"P{pos}1"
                    vec=op_list[op_2](vec)
                    op=op+'_'+op_2
        self.update_history(vec,op)



    
    def order_parameter(self,vec=None):
        if vec is None:
            vec=self.vec_history[-1].copy()
        # O=(vec.conj().T@ZZ(self.L)@vec).toarray()[0,0]
        # O=(vec.conj().T@ZZ(self.L)@vec)
        if self.xj== set([Fraction(1,3),Fraction(2,3)]):
            O=self.ZZ_tensor(vec)
        elif self.xj == set([0]):
            O=self.Z_tensor(vec)


        assert np.abs(O.imag)<self._eps, f'<O> is not real ({val}) '
        return O.real
    
    def von_Neumann_entropy_pure(self,subregion,vec=None):
        '''`subregion` the spatial dof
        this version uses Schmidt decomposition, which is easier for pure state'''
        if vec is None:
            vec=self.vec_history[-1].copy()
        vec_tensor=vec.reshape((2,)*(self.L_T))
        subregion=list(subregion)
        not_subregion=[i for i in range(self.L_T) if i not in subregion]
        vec_tensor_T=vec_tensor.transpose(np.hstack([subregion , not_subregion]))
        S=np.linalg.svd(vec_tensor_T.reshape((2**len(subregion),2**len(not_subregion))),compute_uv=False)
        S_pos=S[S>1e-18]
        return -np.sum(np.log(S_pos**2)*S_pos**2)

    def half_system_entanglement_entropy(self,vec=None):
        '''\sum_{i=0..L/2-1}S_([i,i+L/2)) / (L/2)'''
        if vec is None:
            vec=self.vec_history[-1].copy()
        # S_A=[self.von_Neumann_entropy_pure(np.arange(i,i+self.L//2),vec) for i in range(self.L//2)]
        # return (S_A)
        S_A=self.von_Neumann_entropy_pure(np.arange(self.L//2),vec)
        return S_A

    def tripartite_mutual_information(self,subregion_A,subregion_B, subregion_C,selfaverage=False,vec=None):
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

    def update_history(self,vec=None,op=None):
        if self.history:
            if vec is not None:
                self.vec_history.append(vec)
            if op is not None:
                self.op_history.append(op)
        else:
            if vec is not None:
                self.vec_history=[vec]
            if op is not None:
                self.op_history=[op]

    def get_prob(self,L_list,vec):
        '''get the probability of measuring 0 at site L_list'''
        # prob={(pos,n):(vec.conj().T@P(self.L,n=n,pos=pos)@vec).toarray()[0,0] for pos in L_list for n in [0,1]}
        prob={(pos,n):(vec.conj().T@P(self.L,n=n,pos=pos)@vec) for pos in L_list for n in [0,1]}
        for key, val in prob.items():
            assert np.abs(val.imag)<self._eps, f'probability for {key} is not real {val}'
            prob[key]=val.real
        return prob
    
    def get_prob_tensor(self,L_list,vec):
        prob={(pos,0):self.inner_prob(vec,pos) for pos in L_list}
        prob.update({(pos,1):1-prob[(pos,0)] for pos in L_list})
        return prob

    def inner_prob(self,vec,pos):
        '''probability of `vec` of measuring 0 at L
        convert the vector to tensor (2,2,..), take about the specific pos-th index, and flatten to calculate the inner product'''
        if vec.ndim == 1:
            vec_tensor=vec.reshape((2,)*self.L_T)
        elif vec.ndim== self.L_T:
            vec_tensor=vec
        idx_list=[slice(None)]*self.L_T
        idx_list[pos]=0
        vec_0=vec_tensor[tuple(idx_list)].flatten()
        inner_prod=vec_0.conj()@vec_0
        assert np.abs(inner_prod.imag)<self._eps, f'probability for outcome 0 is not real {inner_prod}'
        inner_prod=inner_prod.real
        assert inner_prod>-self._eps, f'probability for outcome 0 is not positive {inner_prod}'
        inner_prod=max(0,inner_prod)
        assert inner_prod<1+self._eps, f'probability for outcome 1 is not smaller than 1 {inner_prod}'
        inner_prod=min(inner_prod,1)
        return inner_prod


    def normalize(self,vec):
        # normalization after projection
        # norm=np.sqrt(vec.conj().T@vec).toarray()[0,0]
        norm=np.sqrt(vec.conj().T@vec)
        assert norm != 0 , f'Cannot normalize: norm is zero {norm}'
        return vec/norm
        
    def XL_tensor(self,vec):
        '''directly swap 0 and 1'''
        if not self.ancilla:
            vec_tensor=vec.reshape((2,)*self.L_T)[...,[1,0]]
        else:
            vec_tensor=vec.reshape((2,)*self.L_T)[...,[1,0],:]
        return vec_tensor.flatten()

    def P_tensor(self,vec,n,pos=None):
        '''directly set zero at tensor[...,0] =0 for n==1 and tensor[...,1] =0 for n==0'''
        vec_tensor=vec.reshape((2,)*self.L_T)
        if pos is None or pos==self.L-1:
            # project the last site
            if not self.ancilla:
                vec_tensor[...,1-n]=0
            else:
                vec_tensor[...,1-n,:]=0
        if pos == self.L-2:
            if not self.ancilla:
                vec_tensor[...,1-n,:]=0
            else:
                vec_tensor[...,1-n,:,:]=0
        return vec_tensor.flatten()

    def T_tensor(self,vec,left=True):
        '''directly transpose the index of tensor'''
        vec_tensor=vec.reshape((2,)*self.L_T)
        idx_list=np.arange(self.L_T)
        # shift=-1 if left else 1
        if left:
            idx_list_2=list(range(1,self.L))+[0]
        else:
            idx_list_2=[self.L-1]+list(range(self.L-1))
        if self.ancilla:
            idx_list_2.append(self.L)
        return vec_tensor.transpose(idx_list_2).flatten()

    def S_tensor(self,vec,rng):
        '''directly convert to tensor and apply to the last two indices'''
        U_4=U(4,rng)
        if not self.ancilla:
            vec_tensor=vec.reshape((2**(self.L-2),2**2)).T  
            return (U_4@vec_tensor).T.flatten()
        else:
            vec_tensor=vec.reshape((2**(self.L-2),2**2,2)).transpose((1,0,2)).reshape((2**2,2**(self.L-1)))
            return (U_4@vec_tensor).reshape((2**2,2**(self.L-2),2)).transpose((1,0,2)).flatten()


    def ZZ_tensor(self,vec):
        vec_tensor=vec.reshape((2,)*self.L_T)
        rs=0
        for i in range(self.L):
            for zi in range(2):
                for zj in range(2):
                    idx_list=[slice(None)]*self.L_T
                    idx_list[i],idx_list[(i+1)%self.L]=zi,zj
                    exp=1-2*(zi^zj) # expectation-- zi^zj is xor of two bits which is only one when zi!=zj
                    vec_i=vec_tensor[tuple(idx_list)].flatten()
                    rs+=vec_i.conj()@vec_i*exp
        return -rs/self.L
    
    def Z_tensor(self,vec):
        vec_tensor=vec.reshape((2,)*self.L_T)
        rs=0
        for i in range(self.L):
            P0=self.inner_prob(vec_tensor,i)
            rs+=P0*1+(1-P0)*(-1)
        return rs/self.L

        
    @lru_cache(maxsize=None)
    def adder(self):
        ''' This is not a full adder, which assume the leading digit in the input bitstring is zero (because of the T^{-1}R_L, the leading bit should always be zero).'''
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

    
def construct_density_matrix(vec):
    return np.tensordot(vec.conj(),vec,axes=0)
    
def partial_trace( rho, L, subregion):
    '''for a density matrix rho, of a system size L, (dim: 2**L), trace out the dof in `subregion`'''
    assert L<=16, f'L ({L}) cannot be longer than 16 because np.ndarray is restricted to 32 rank'
    rho_tensor=rho.reshape((2,)*(2*L))
    idx_list=np.arange(2*L)
    idx_list[subregion+L]=idx_list[subregion]
    mask=np.ones(2*L,dtype=bool)
    mask[subregion]=False
    mask[subregion+L]=False
    out_idx_list=idx_list[mask]
    rho_reduce=np.einsum(rho_tensor,idx_list,out_idx_list).reshape((2**(L-subregion.shape[0]),2**(L-subregion.shape[0])))
    return rho_reduce

def minus_rho_log_rho(rho):
    vals=np.linalg.eigvalsh(rho)
    vals_positive=vals[vals>0]
    return np.sum(-np.log(vals_positive)*vals_positive)

def dec2bin(x,L):
    '''
    convert a float number x in [0,1) to the binary form with maximal length of L, where the leading 0 as integer part is truncated
    Example, 1/3 is 010101...
    '''
    assert 0<=x<1, f'{x} is not in [0,1)'
    bits=[]
    for _ in range(L):
        x*=2
        bits.append(int(x))
        x-=int(x)
    return np.array(bits,dtype=int)

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

kron_list=lambda x: reduce(sp.kron,x)


def U(n,rng=None,size=1):
    '''Generate Haar random U(n)
    Return: dense matrix of Haar random U(4) `Q`'''
    if rng is None:
        rng=np.random.default_rng(None)
    return scipy.stats.unitary_group.rvs(n,random_state=rng,size=size)

def S(L,rng):
    '''construct quantum scrambler, Haar random U(4) applies to the last two digits only
    Return : sparse matrix'''

    I2=sp.eye(2**(L-2))
    U_4=scipy.stats.unitary_group.rvs(4,random_state=rng)

    return sp.kron(I2,U_4)



@lru_cache(maxsize=None)
def bin_pad(x,L):
    '''convert a int to binary form with 0 padding to the left'''
    return (bin(x)[2:]).rjust(L,'0')



class CT_tensor:
    def __init__(self,L,history=False,seed=None,x0=None,xj=set([Fraction(1,3),Fraction(2,3)]),_eps=1e-10, ancilla=False,gpu=False,complex128=True,ensemble=None):
        '''the tensor is saved as (0,1,...L-1, ancilla, ensemble)
        complex128: True:complex128 or False: complex64, default `complex128`
        if `seed` is a list, use numpy rng generator, the ensemble size is the same as `len(seed)`
        if `seed` is a number, use torch random, the ensemble size is given by `ensemble`
        '''
        self.L=L # physical L, excluding ancilla
        self.L_T=L+1 if ancilla else L # tensor L, ancilla
        self.history=history
        self.gpu=gpu
        self.device=self._initialize_device()
        self.ensemble=ensemble
        self.rng=self._initialize_random_seed(seed)

        self.x0=self._initialize_x0(x0)
        self.op_history=[]  
        self.ancilla=ancilla
        self.dtype={'numpy':np.complex128,'torch':torch.complex128} if complex128 else {'numpy':np.complex64,'torch':torch.complex64}
        self.vec=self._initialize_vector()
        self.vec_history=[self.vec]
        self._eps=_eps
        self.xj=set(xj)
        self.op_list=self._initialize_op()
        self.new_idx,self.old_idx, self.not_new_idx=self.adder_gpu()
        self.tensor_true,self.tensor_false=torch.tensor(True,device=self.device), torch.tensor(False,device=self.device) # this is a workaround for generate_binary, because loop over tensor returns a tensor with a single value

    def _initialize_device(self):
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
        if self.ensemble is None:
            return np.array([np.random.default_rng(s) for s in seed])
        else:
            torch.manual_seed(seed)

    def _initialize_x0(self,x0):
        if self.ensemble is None:
            return [rng.random() if x is None else x for rng,x in zip(self.rng,([x0]*len(self.rng)) if x0 is None else x0)]
        else:
            if x0 is None:
                return torch.randint(0,2**self.L,(self.ensemble,),device=self.device)
            else:
                return x0

    def _initialize_vector(self):
        '''save using an array of 2^L
        if ancilla qubit: it should be put to the last qubit, positioned as L, entangled with L-1'''
        if not self.ancilla:
            vec=torch.zeros((2,)*(self.L)+(len(self.x0),),dtype=self.dtype['torch'],device=self.device)
            if self.ensemble is None:
                vec_int=np.array([np.hstack([dec2bin(x0,self.L),[idx]]) for idx,x0 in enumerate(self.x0)])
                vec[tuple((vec_int).T)]=1
            else:
                vec_v=vec.view((-1,self.ensemble))
                vec_v[self.x0,torch.arange(self.ensemble,device=self.device)]=1

        else:
            # Simply create a GHZ state, (|0...0> + |1...1> )/sqrt(2)
            vec=torch.zeros((2,)*(self.L_T)+(len(self.x0),),dtype=self.dtype['torch'],device=self.device)
            vec[(0,)*self.L_T]=1/np.sqrt(2)
            vec[(1,)*self.L_T]=1/np.sqrt(2)

            # Randomize it 
            if self.ensemble is None:
                for _ in range(self.L**2):
                    vec=self.Bernoulli_map(vec,self.rng)
            else:
                for _ in range(self.L**2):
                    vec=self.Bernoulli_map(vec,len(self.x0))
        return vec
    
    def _initialize_op(self):
        return {"C0":partial(self.control_map,bL=0),
                "C1":partial(self.control_map,bL=1),
                f"P{self.L-1}0":partial(self.projection_map,pos=self.L-1,n=0),
                f"P{self.L-1}1":partial(self.projection_map,pos=self.L-1,n=1),
                f"P{self.L-2}0":partial(self.projection_map,pos=self.L-2,n=0),
                f"P{self.L-2}1":partial(self.projection_map,pos=self.L-2,n=1),
                "chaotic":self.Bernoulli_map,
                "I":lambda x:x
                }

    def Bernoulli_map(self,vec,rng):
        vec=self.T_tensor(vec,left=True)
        vec=self.S_tensor(vec,rng=rng)
        return vec

    def control_map(self,vec,bL):
        '''control map depends on the outcome of the measurement of bL'''
        # projection on the last bits
        self.P_tensor_(vec,bL)
        if bL==1:
            vec=self.XL_tensor_(vec)
        self.normalize_(vec)
        # right shift 
        vec=self.T_tensor(vec,left=False)

        # Adder
        
        if not vec.is_contiguous():
            vec=vec.contiguous()
        self.adder_tensor_(vec)
        
        return vec
    
    def projection_map(self,vec,pos,n):
        '''projection to `pos` with outcome of `n`
        note that here is 0-index, and pos=L-1 is the last bit'''
        self.P_tensor_(vec,n,pos)
        self.normalize_(vec)
        return vec

    def random_control(self,p_ctrl,p_proj,vec=None):
        '''the competition between chaotic and random, where the projection can only be applied after the unitary
        Notation: L-1 is the last digits'''
        if vec is None:
            vec=self.vec

        ctrl_idx_dict=self.generate_binary(torch.arange(self.rng.shape[0] if self.ensemble is None else self.ensemble,device=self.device), p_ctrl)
        ctrl_0_idx_dict={}
        if len(ctrl_idx_dict[True])>0:
            vec_ctrl=vec[...,ctrl_idx_dict[True]]
            p_0= self.inner_prob(vec=vec_ctrl,pos=[self.L-1],n_list=[0]) # prob for 0
            ctrl_0_idx_dict=self.generate_binary(ctrl_idx_dict[True], p_0)
            for key,idx in ctrl_0_idx_dict.items():
                if len(idx)>0:
                    vec[...,idx]=self.op_list[f'C{0*key+1*(1-key)}'](vec[...,idx])

        proj_idx_dict={} # {pos: {True: .., False:..}} whether pos is projected
        proj_0_idx_dict={} # {pos: {True:.., False: ..}} if projected, whether it is projected to 0 
        if len(ctrl_idx_dict[False])>0:
            rng_ctrl=self.rng[ctrl_idx_dict[False].cpu().numpy()] if self.ensemble is None else ctrl_idx_dict[False].shape[0]
            vec[...,ctrl_idx_dict[False]]=self.op_list['chaotic'](vec[...,ctrl_idx_dict[False]],rng_ctrl)
            for pos in [self.L-1,self.L-2]:
                proj_idx_dict[pos]=self.generate_binary(ctrl_idx_dict[False], p_proj)
                if len(proj_idx_dict[pos][True])>0:
                    vec_p=vec[...,proj_idx_dict[pos][True]]
                    p_2 = self.inner_prob(vec=vec_p,pos=[pos], n_list=[0])
                    proj_0_idx_dict[pos]=self.generate_binary(proj_idx_dict[pos][True], p_2)
                    for key,idx in proj_0_idx_dict[pos].items():
                        if len(idx)>0:
                            vec[...,idx]=self.op_list[f'P{pos}{0*key+1*(1-key)}'](vec[...,idx])
        self.update_history(vec,ctrl_idx_dict,ctrl_0_idx_dict,proj_idx_dict,proj_0_idx_dict)

    def order_parameter(self,vec=None):
        if vec is None:
            vec=self.vec
        if self.xj== set([Fraction(1,3),Fraction(2,3)]):
            O=self.ZZ_tensor(vec)
        elif self.xj == set([0]):
            O=self.Z_tensor(vec)
        return O  

    def von_Neumann_entropy_pure(self,subregion,vec=None,driver='gesvd'):
        '''`subregion` the spatial dof
        this version uses Schmidt decomposition, which is easier for pure state'''
        if vec is None:
            vec=self.vec
        if not self.gpu:
            driver=None
        subregion=list(subregion)
        not_subregion=[i for i in range(self.L_T) if i not in subregion]
        vec=vec.permute([self.L_T]+subregion+not_subregion)
        vec_=vec.contiguous().view((self.rng.shape[0] if self.ensemble is None else self.ensemble,2**len(subregion),2**len(not_subregion)))

        S=torch.linalg.svdvals(vec_,driver=driver)
        S_pos=torch.clamp(S,min=1e-18)
        return torch.sum(-torch.log(S_pos**2)*S_pos**2,axis=1)

        # vNE=torch.empty((self.rng.shape[0] if self.ensemble is None else self.ensemble,),dtype=torch.float)
        # for i in range(self.rng.shape[0] if self.ensemble is None else self.ensemble):
        #     S=torch.linalg.svdvals(vec_[i,:,:],driver=driver)
        #     S_pos=S[S>1e-18]
        #     vNE[i]=(-torch.sum(torch.log(S_pos**2)*S_pos**2).item())
        # return vNE

    def half_system_entanglement_entropy(self,vec=None,selfaverage=False):
        '''\sum_{i=0..L/2-1}S_([i,i+L/2)) / (L/2)'''
        if vec is None:
            vec=self.vec
        if selfaverage:
            S_A=np.mean([self.von_Neumann_entropy_pure(np.arange(i,i+self.L//2),vec) for i in range(self.L//2)])
        else:
            S_A=self.von_Neumann_entropy_pure(np.arange(self.L//2),vec)
        return S_A

    def tripartite_mutual_information(self,subregion_A,subregion_B, subregion_C,selfaverage=False,vec=None):
        assert np.intersect1d(subregion_A,subregion_B).size==0 , "Subregion A and B overlap"
        assert np.intersect1d(subregion_A,subregion_C).size==0 , "Subregion A and C overlap"
        assert np.intersect1d(subregion_B,subregion_C).size==0 , "Subregion B and C overlap"
        if vec is None:
            vec=self.vec
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
    
    def update_history(self,vec=None,ctrl_idx_dict=None,ctrl_0_idx_dict=None,proj_idx_dict=None,proj_0_idx_dict=None):
        if self.history:
            if vec is not None:
                self.vec_history.append(vec.cpu().clone())
            if ctrl_idx_dict is not None and ctrl_0_idx_dict is not None and proj_idx_dict is not None and proj_0_idx_dict is not None:
                op_map=np.empty((self.rng.shape[0] if self.ensemble is None else self.ensemble,),dtype='<U20')
                op_map[ctrl_0_idx_dict[True]]='C0'
                op_map[ctrl_0_idx_dict[False]]='C1'
                op_map[ctrl_idx_dict[False]]='chaotic'

                for pos,idx_dict in proj_0_idx_dict.items():
                    if len(idx_dict[True])>0:
                        op_map[idx_dict[True]]=np.char.add(op_map[idx_dict[True]],f'_P{pos}0')
                    if len(idx_dict[False])>0:
                        op_map[idx_dict[False]]=np.char.add(op_map[idx_dict[False]],f'_P{pos}1')

                self.op_history.append(op_map)

    def normalize_(self,vec):
        '''normalization after projection'''
        # norm=torch.sqrt(torch.tensordot(vec.conj(),vec,dims=(list(range(self.L_T)),list(range(self.L_T)))))
        norm=torch.sqrt(torch.einsum(vec.conj(),[...,0],vec,[...,0],[0]))

        # assert torch.all(norm != 0) , f'Cannot normalize: norm is zero {norm}'
        vec/=norm
    
    def inner_prob(self,vec,pos,n_list):
        '''probability of `vec` of measuring `n_list` at `pos`
        convert the vector to tensor (2,2,..), take about the specific pos-th index, and flatten to calculate the inner product'''
        idx_list=np.array([slice(None)]*self.L_T)
        # for p,n in zip(pos,n_list):
        #     idx_list[p]=n
        idx_list[pos]=n_list
        vec_0=vec[tuple(idx_list)]
        inner_prod=torch.einsum(vec_0.conj(),[...,0],vec_0,[...,0],[0])

        # assert torch.all(torch.abs(inner_prod.imag)<self._eps), f'probability for outcome 0 is not real {inner_prod}'
        inner_prod=inner_prod.real
        # assert torch.all(inner_prod>-self._eps), f'probability for outcome 0 is not positive {inner_prod}'
        # inner_prod=torch.maximum(0,inner_prod)
        # assert torch.all(inner_prod<1+self._eps), f'probability for outcome 1 is not smaller than 1 {inner_prod}'
        # inner_prod=torch.minimum(inner_prod,1)
        inner_prod=torch.clamp_(inner_prod,min=0,max=1)
        return inner_prod

    def XL_tensor_(self,vec):
        '''directly swap 0 and 1
        A new version using roll seems much faster than the in-place operation which is to my surprise'''
        vec=torch.roll(vec,1,dims=self.L-1)
        # if not self.ancilla:
        #     vec[...,[0,1],:]=vec[...,[1,0],:]
            

        # else:
        #     vec[...,[0,1],:,:]=vec[...,[1,0],:,:]

        return vec

    def P_tensor_(self,vec,n,pos=None):
        '''directly set zero at tensor[...,0] =0 for n==1 and tensor[...,1] =0 for n==0'
        This is an in-placed operation
        '''
        # vec_tensor=vec.reshape((2,)*self.L_T)
        if pos is None or pos==self.L-1:
            # project the last site
            if not self.ancilla:
                vec[...,1-n,:]=0
            else:
                vec[...,1-n,:,:]=0
        if pos == self.L-2:
            if not self.ancilla:
                vec[...,1-n,:,:]=0
            else:
                vec[...,1-n,:,:,:]=0
        # return vec

    def T_tensor(self,vec,left=True):
        '''directly transpose the index of tensor
        There could be an alternative way of whether using tensor operation'''
        
        idx_list=torch.arange(self.L_T)
        # shift=-1 if left else 1
        if left:
            idx_list_2=list(range(1,self.L))+[0]
        else:
            idx_list_2=[self.L-1]+list(range(self.L-1))
        if self.ancilla:
            idx_list_2.append(self.L)
        idx_list_2.append(self.L_T)
        return vec.permute(idx_list_2)

    def S_tensor(self,vec,rng):
        '''Scrambler only applies to the last two indices
        This is a bit confusing, because when using np.rng, `rng` is interpreted as a list of `rng`, but when using torch seed, `rng` is reused as the len of list, CHANGE IT FOR CONSISTENCY LATER'''
        if not isinstance(rng, int):
            U_4=torch.from_numpy(np.array([U(4,rng).astype(self.dtype['numpy']).reshape((2,)*4) for rng in rng]))
            if self.gpu:
                U_4=U_4.cuda()
        else:
            # U_4=U(4,size=rng).astype(self.dtype['numpy'])
            # U_4=torch.tensor(U_4,device=self.device).view((rng,2,2,2,2))

            # Another advantage is the reproducibility
            U_4=self.U(4,size=rng).view((rng,2,2,2,2))


        if not self.ancilla:
            # vec=torch.tensordot(vec,U_4,dims=([self.L-2,self.L-1],[2,3])).permute(list(range(self.L-2))+[self.L-1,self.L]+[self.L-2])
            vec=torch.einsum(vec,[...,0,1,2],U_4,[2,3,4,0,1],[...,3,4,2])
            return vec
        else:
            # vec=torch.tensordot(vec,U_4,dims=([self.L-2,self.L-1],[2,3])).permute(list(range(self.L-2))+[self.L,self.L+1]+[self.L-2,self.L-1])
            vec=torch.einsum(vec,[...,0,1,2,3],U_4,[3,4,5,0,1],[...,4,5,2,3])
            return vec

    def ZZ_tensor(self,vec):
        rs=0
        for i in range(self.L):
            for zi in range(2):
                for zj in range(2):
                    inner_prod=self.inner_prob(vec, [i,(i+1)%self.L],[zi,zj])
                    exp=1-2*(zi^zj) # expectation-- zi^zj is xor of two bits which is only one when zi!=zj
                    rs+=inner_prod*exp
        return -rs/self.L
    
    def Z_tensor(self,vec):
        rs=0
        for i in range(self.L):
            P0=self.inner_prob(vec,[i],[0])
            rs+=P0*1+(1-P0)*(-1)
        return rs/self.L

    def adder_gpu(self):
        ''' This is not a full adder, which assume the leading digit in the input bitstring is zero (because of the T^{-1}R_L, the leading bit should always be zero).'''
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
        new_idx=self.new_idx.flatten()
        old_idx=self.old_idx.flatten()
        not_new_idx=self.not_new_idx.flatten()
        if (new_idx).shape[0]>0 and (old_idx).shape[0]>0:
            vec_flatten=vec.view((-1,vec.shape[-1]))    
            vec_flatten[new_idx,:]=vec_flatten[old_idx,:]
            vec_flatten[not_new_idx,:]=0

    def generate_binary(self,idx_list,p):
        '''Generate boolean list, given probability `p` and seed `self.rng[idx]`
        scalar `p` is verbose, but this is for consideration of speed'''
        if self.ensemble is None:
            # idx_dict={True:[],False:[]}
            true_list=[]
            false_list=[]
            if isinstance(p, float) or isinstance(p, int):
                for idx in idx_list:
                    random=self.rng[idx].random()
                    # boolean=(random<=p)
                    # idx_dict[boolean].append(idx)
                    if random<=p:
                        true_list.append(idx)
                    else:
                        false_list.append(idx)
            else:
                assert len(idx_list) == len(p), f'len of idx_list {len(idx_list)} is not same as len of p {len(p)}'
                for idx,p in zip(idx_list,p):
                    random=self.rng[idx].random()
                    # boolean=torch.equal((random<=p),self.tensor_true)
                    # idx_dict[boolean].append(idx)
                    if random<=p:
                        true_list.append(idx)
                    else:
                        false_list.append(idx)

            idx_dict={True:torch.tensor(true_list,dtype=int,device=self.device),False:torch.tensor(false_list,dtype=int,device=self.device)}
            return idx_dict
        else:
            if isinstance(p, float) or isinstance(p, int):
                p=p*torch.ones((idx_list.shape[0],),device=self.device)

            p_rand=torch.bernoulli(p)
            true_idx=torch.nonzero(p_rand,as_tuple=True)[0]
            false_idx=torch.nonzero(1-p_rand,as_tuple=True)[0]
            # p_rand=torch.rand((idx_list.shape[0],),device=self.device)
            
            true_tensor=idx_list[true_idx]
            false_tensor=idx_list[false_idx]
            idx_dict={True:true_tensor,False:false_tensor}
            return idx_dict


    def U(self,n,size,):
        dtype=torch.float64 if self.dtype['torch']==torch.complex128 else torch.float32
        im = torch.randn((size,n, n), device=self.device,dtype=dtype)
        re = torch.randn((size,n, n), device=self.device,dtype=dtype)
        z=torch.complex(re,im)
        Q,R=torch.linalg.qr(z)
        r_diag=torch.diagonal(R,dim1=-2,dim2=-1)
        Lambda=torch.diag_embed(r_diag/torch.abs(r_diag))
        Q=torch.einsum(Q,[0,1,2],Lambda,[0,2,3],[0,1,3])
        return Q
