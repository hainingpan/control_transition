import numpy as np
from functools import reduce
import scipy.sparse as sp
import scipy
from fractions import Fraction
from functools import partial, lru_cache

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
        self.xj=xj
    
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
        vec=normalize(vec)
        # right shift 
        # vec=T(self.L,left=False)@vec
        vec=self.T_tensor(vec,left=False)

        assert np.abs(vec[vec.shape[0]//2:]).sum() == 0, f'first qubit is not zero ({np.abs(vec[vec.shape[0]//2:]).sum()}) after right shift '

        # Adder
        if not self.ancilla:
            vec=adder(self.L,self.xj)@vec
        else:
            vec=(adder(self.L,self.xj)@vec.reshape((2**self.L,2))).flatten()
        
        return vec

    def projection_map(self,vec,pos,n):
        '''projection to `pos` with outcome of `n`
        note that here is 0-index, and pos=L-1 is the last bit'''
        # vec=P(self.L,n=n,pos=pos)@vec
        vec=self.P_tensor(vec,n,pos)
        vec=normalize(vec)

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


    
    def order_parameter(self,vec=None):
        if vec is None:
            vec=self.vec_history[-1].copy()
        # O=(vec.conj().T@ZZ(self.L)@vec).toarray()[0,0]
        # O=(vec.conj().T@ZZ(self.L)@vec)
        O=self.ZZ_tensor(vec)

        assert np.abs(O.imag)<self._eps, f'<O> is not real ({val}) '
        return O.real

    # def von_Neumann_entropy(self,subregion,vec=None):
    #     '''`subregion` the spatial dof'''
    #     if vec is None:
    #         vec=self.vec_history[-1].copy()
    #     subregion=np.array(subregion)
    #     rho=construct_density_matrix(vec)
    #     rho_reduce=partial_trace(rho,self.L_T,subregion)
    #     return minus_rho_log_rho(rho_reduce)
    
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
        vec_tensor=vec.reshape((2,)*self.L_T)
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
                    exp=1-2*(zi^zj) # zi^zj is xor which is only one when zi!=zj
                    vec_i=vec_tensor[tuple(idx_list)].flatten()
                    rs+=vec_i.conj()@vec_i*exp
        return -rs/self.L
            
    

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

# @lru_cache(maxsize=None)
# def T(L,left=True):
#     '''
#     circular right shift the computational basis `vec` : 
#     b_0 b_1 ... b_{L-1} -> right shift -> b_{L-1} b_0 ... b_{L-2}
#     b_0 b_1 ... b_{L-1} -> left shift ->  b_1 b_2... b_{L-2} b_{L-1}
#     Return: sparse matrix
#     '''
#     SWAP=sp.csr_array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],dtype=int)
#     I2=sp.eye(2,dtype=int)
#     op_list=[I2]*(L-1)
    
#     rs=sp.eye(2**L,dtype=int)
#     idx_list=np.arange(L-1)[::-1] if left else np.arange(L-1)
#     for i in idx_list:
#         op_list[i]=SWAP
#         rs=rs@kron_list(op_list)
#         op_list[i]=I2
#     return rs

def U(n,rng=None):
    '''Generate Haar random U(n)
    Return: dense matrix of Haar random U(4) `Q`'''
    if rng is None:
        rng=np.random.default_rng(None)
    # re=rng.normal(size=(n,n))
    # im=rng.normal(size=(n,n))
    # z=re+1j*im
    # Q,R=np.linalg.qr(z)
    # r_diag=np.diag(R)
    # Lambda=np.diag(r_diag/np.abs(r_diag))
    # Q=Q@Lambda
    # # R=Lambda.conj()@R
    # return Q
    return scipy.stats.unitary_group.rvs(n,random_state=rng)

def S(L,rng):
    '''construct quantum scrambler, Haar random U(4) applies to the last two digits only
    Return : sparse matrix'''
    # I2=sp.eye(2,dtype=int)
    # U_4=U(4,rng)
    # op_list=[I2]*(L-2)+[U_4]
    # return kron_list(op_list)

    I2=sp.eye(2**(L-2))
    # U_4=U(4,rng)
    U_4=scipy.stats.unitary_group.rvs(4,random_state=rng)

    return sp.kron(I2,U_4)

# @lru_cache(maxsize=None)
# def P(L,n,pos=None):
#     '''projection to n=0 or 1
#     Return : sparse matrix'''
#     if pos is None:
#         pos=L-1
#     PL=sp.diags([1-n,n])
#     I2=sp.eye(2,dtype=int)
#     op_list=[I2]*(L)
#     op_list[pos]=PL
#     return kron_list(op_list)

# @lru_cache(maxsize=None)
# def XL(L):
#     '''X_L for the last digits'''
#     sigma_x=sp.csr_matrix([[0,1],[1,0]],dtype=int)
#     I2=sp.eye(2,dtype=int).tocsr()
#     op_list=[I2]*(L-1)+[sigma_x]
#     return kron_list(op_list)

@lru_cache(maxsize=None)
def adder(L,xj):
    ''' This is not a full adder, which assume the leading digit in the input bitstring is zero (because of the T^{-1}R_L, the leading bit should always be zero).'''
    if xj==set([Fraction(1,3),Fraction(1,6)]):
        bin_1_6=dec2bin(Fraction(1,6), L)
        bin_1_6[-1]=1
        bin_1_3=dec2bin(Fraction(1,3), L)
        int_1_6=int(''.join(map(str,bin_1_6)),2)
        int_1_3=int(''.join(map(str,bin_1_3)),2)
        old_idx=np.arange(2**(L-1))
        adder_idx=np.array([int_1_6]*2**(L-2)+[int_1_3]*2**(L-2))    
        new_idx=old_idx+adder_idx
        # handle the extra attractors, if 1..0x1, then 1..0(1-x)1, if 0..1x0, then 0..1(1-x)0 [shouldn't enter this branch..]
        mask_1=(new_idx&(1<<L-1) == (1<<L-1)) & (new_idx&(1<<2) == (0)) & (new_idx&(1) == (1))
        mask_2=(new_idx&(1<<L-1) == (0)) & (new_idx&(1<<2) == (1<<2)) & (new_idx&(1) == (0))

        new_idx[mask_1+mask_2]=new_idx[mask_1+mask_2]^(0b10)

        ones=np.ones(2**(L-1))
        return sp.coo_matrix((ones,(new_idx,old_idx)),shape=(2**L,2**L))
    if xj=set([0]):
        return sp.eye(2**L)

def normalize(vec):
    # normalization after projection
    # norm=np.sqrt(vec.conj().T@vec).toarray()[0,0]
    norm=np.sqrt(vec.conj().T@vec)
    assert norm != 0 , f'Cannot normalize: norm is zero {norm}'
    return vec/norm

# @lru_cache(maxsize=None)
# def ZZ(L):
#     '''Z |0> = -|0>
#     Z |1> = |1>'''
#     sigma_Z= sp.csr_matrix([[-1,0],[0,1]],dtype=int)
#     I2=sp.eye(2,dtype=int)
#     op_list=[I2]*(L)
#     rs=0
#     for i in range(L):
#         # a little dumb here, swap can be used but anyway, profiling is a later step
#         # another simplied way is to note basis is the eigenvector of ZZ
#         op_list[i],op_list[(i+1)%L]=sigma_Z,sigma_Z
#         rs=rs+kron_list(op_list)
#         op_list[i],op_list[(i+1)%L]=I2,I2
#     return -rs/L


@lru_cache(maxsize=None)
def bin_pad(x,L):
    '''convert a int to binary form with 0 padding to the left'''
    return (bin(x)[2:]).rjust(L,'0')

