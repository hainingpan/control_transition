import numpy as np
import torch

class bricklayer_tensor:
    def __init__(self,L,store_vec=False,store_op=False,store_prob=False,seed=None,seed_vec=None,seed_C=None,x0=None,_eps=1e-10,add_x=0,debug=False):
        self.L=L
        self.store_vec=store_vec
        self.store_op=store_op
        self.store_prob=store_prob
        self.rng=self._initialize_random_seed(seed)
        self.rng_vec=self._initialize_random_seed(seed_vec) if seed_vec is not None else self.rng
        self.rng_C=np.random.default_rng(seed_C) if seed_C is not None else self.rng
        self.rng_C=self._initialize_random_seed(seed_C) if seed_C is not None else self.rng
        self.x0=x0
        self.op_history=[] # store the history of operators applied to the circuit
        self.prob_history=[]  # store the history of each probability at projective measurement
        self.vec=self._initialize_vector() # initialize the state vector
        self.vec_history=[self.vec] # store the history of state vector
        self._eps=_eps
        self.add_x=add_x
        self.debug=debug

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
        k=1
        factor=np.sqrt(1/k)
        if self.x0 is None:
            if self.ensemble is None:
                from .utils import Haar_state
                vec=factor*torch.from_numpy(np.array([Haar_state(self.L,ensemble=1,rng=rng,k=k).astype(self.dtype['numpy']).reshape((2,)*(self.L_T)) for rng in self.rng_vec])).permute(list(range(1,self.L_T+1))+[0]).unsqueeze(-1)
                if self.gpu:
                    vec=vec.cuda()
            else:
                # Though verbose, but optimize for GPU RAM usage, squeeze() tends to reserve huge memory
                vec=factor*self.Haar_state(k=k,ensemble=self.ensemble,rng=self.rng_vec)
                if k == 1:
                    vec=vec[...,0,:]
                vec=vec.unsqueeze(-1).repeat(*(1,)*self.L_T+(1,self.ensemble_m,))
        else:
            vec=torch.zeros((2,)*(self.L_T)+(self.ensemble,self.ensemble_m,),device=self.device,dtype=self.dtype['torch'])
            vec_int=dec2bin(self.x0,self.L)
            vec[tuple(int(i) for i in bin(vec_int)[2:].zfill(self.L))+(slice(None),slice(None))]=1
        return vec