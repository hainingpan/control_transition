import numpy as np
from functools import reduce
import scipy.sparse as sp
class CT_classical:
    def __init__(self,L,history=False,seed=None,x0=None):
        '''
        
        if classical is False: save using an array of 2^L
        '''
        self.L=L
        self.history=history
        self.rng=np.random.default_rng(seed)
        self.x0=self.rng.random() if x0 is None else x0
        self.op_history=[]  # control: true, Bernoulli: false
        self.binary=self._initialize_binary([1/6,1/3])

        self.vec=self._initialize_vector()
        self.vec_history=[self.vec]
    def _initialize_vector(self):
        '''save using an array of L'''
        vec=dec2bin(self.x0,self.L)
        return vec
    def _initialize_binary(self,x_list):
        return {x:dec2bin(x, self.L) for x in x_list}

    def Bernoulli_map(self,vec):
        vec=np.roll(vec,-1)
        vec[-2:]=self.rng.permutation(vec[-2:])        
        return vec
    
    def control_map(self,vec):
        vec[-1]=0
        vec=np.roll(vec,1)
        if vec[1]==0:
            # attract to 1/3
            vec=add_binary(vec,self.binary[1/6])
        else:
            # attract to 2/3
            vec=add_binary(vec,self.binary[1/3])
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
    def __init__(self,L,history=False,seed=None,x0=None):
        '''save using an array of 2^L'''
        self.L=L
        self.history=history
        self.rng=np.random.default_rng(seed)
        self.x0=self.rng.random() if x0 is None else x0
        self.op_history=[]  # control: true, Bernoulli: false
        self.vec=self._initialize_vector()
        self.vec_history=[self.vec]
        self.T={'L':T(self.L,left=True),'R':T(self.L,left=False)}
    
    def _initialize_vector(self):
        '''save using an array of 2^L'''
        vec_int=int(''.join(map(str,dec2bin(self.x0,self.L))),2)
        vec=np.zeros((2**self.L))
        vec[vec_int]=1
        return vec

    def Bernoulli_map(self,vec):
        vec=self.T['L']@vec
        vec=T(vec)
        vec=S(m,vec)
        return vec
    
    def control_map(self,vec):
        vec[-1]=0
        vec=np.roll(vec,1)
        if vec[1]==0:
            # attract to 1/3
            vec=add_binary(vec,self.binary[1/6])
        else:
            # attract to 2/3
            vec=add_binary(vec,self.binary[1/3])
        return vec

def dec2bin(x,L):
    '''
    convert a float number x in [0,1) to the binary form with maximal length of L, where the leading 0 is truncated
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
        return np.array(vec1_sum[1:]) #drop carry
    else:
        return np.array(vec1_sum)

def T(L,left=True):
    '''
    circular right shift the computational basis `vec` : 
    a_{L-1} a_{L-2}.. a_{1} a_{0} -> right shift -> a_{0} a_{L-1} a_{L-2}.. a_{1} 
    a_{L-1} a_{L-2}.. a_{1} a_{0} -> left shift -> a_{L-2}.. a_{1} a_{0} a_{L-1}
    '''
    SWAP=sp.csr_array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],dtype=int)
    I2=sp.eye(2,dtype=int)
    op_list=[I2]*(L-1)
    kron_list=lambda x: reduce(sp.kron,x)
    rs=sp.eye(2**L,dtype=int)
    idx_list=np.arange(L-1)[::-1] if left else np.arange(L-1)
    for i in idx_list:
        op_list[i]=SWAP
        rs=rs@kron_list(op_list)
        op_list[i]=I2
    return rs
