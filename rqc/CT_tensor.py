 
import numpy as np
from .utils import U
from .utils import dec2bin
from fractions import Fraction
from functools import partial
import torch
class CT_tensor:
    def __init__(self,L,store_vec=False,store_op=False,store_prob=False,seed=None,seed_vec=None,seed_C=None,x0=None,xj=frozenset([Fraction(1,3),Fraction(2,3)]),_eps=1e-10, ancilla=False,gpu=False,complex128=True,ensemble=None,ensemble_m=1,debug=False,add_x=0,feedback=True):
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
        xj : frozenset of Fractions, optional
            the frozenset of attractors using Fractions, by default frozenset([Fraction(1,3),Fraction(2,3)])
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
        feedback: bool, optional
            if false, the feedback in the measurement is turned off
        """
        self.L=L # physical L, excluding ancilla
        self.L_T=L+1 if ancilla else L # tensor L, ancilla
        self.store_vec=store_vec
        self.store_op=store_op
        self.store_prob=store_prob
        self.gpu=gpu
        self.add_x=add_x
        self.device=self._initialize_device()
        self.ensemble=ensemble
        self.ensemble_m=ensemble_m
        self.rng=self._initialize_random_seed(seed)
        self.rng_vec=self._initialize_random_seed(seed_vec) if seed_vec is not None else self.rng
        self.rng_C=self._initialize_random_seed(seed_C) if seed_C is not None else self.rng
        self.x0=x0
        self.op_history=[]  
        self.prob_history=[]  
        self.ancilla=ancilla
        self.dtype={'numpy':np.complex128,'torch':torch.complex128} if complex128 else {'numpy':np.complex64,'torch':torch.complex64}
        self.vec=self._initialize_vector()
        self.vec_history=[self.vec]
        self._eps=_eps
        self.xj=frozenset(xj)
        self.op_list=self._initialize_op()
        # self.new_idx,self.old_idx, self.not_new_idx=self.adder_gpu()
        self.new_idx,self.old_idx, self.not_new_idx=self.initialize_adder()
        self.debug=debug
        self.feedback=feedback

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
    
    def _initialize_op(self):
        """Initialize the operators inthe circuit. `C` is the control map, `P` is the projection, `B` is the Bernoulli map, `I` is the identity map. The second element in the tuple is the outcome. The number after "P" is the position of projection (0-index).

        Returns
        -------
        dict of operatiors
            possible operators in the circuit
        """
        return {("C",0):partial(self.control_map,pos=[self.L-1],n=[0]),
                ("C",1):partial(self.control_map,pos=[self.L-1],n=[1]),
                ("C",0,0):partial(self.control_map,pos=[0,self.L-1],n=[0,0]),
                ("C",0,1):partial(self.control_map,pos=[0,self.L-1],n=[0,1]),
                ("C",1,0):partial(self.control_map,pos=[0,self.L-1],n=[1,0]),
                ("C",1,1):partial(self.control_map,pos=[0,self.L-1],n=[1,1]),
                (f"P{self.L-1}",0):partial(self.projection_map,pos=[self.L-1],n=[0]),
                (f"P{self.L-1}",1):partial(self.projection_map,pos=[self.L-1],n=[1]),
                (f"P{self.L-2}",0):partial(self.projection_map,pos=[self.L-2],n=[0]),
                (f"P{self.L-2}",1):partial(self.projection_map,pos=[self.L-2],n=[1]),
                ("B",):self.Bernoulli_map,
                ("I",):lambda x:x
                }

    def initialize_adder(self):
        new_idx,old_idx, not_new_idx={},{},{}
        if self.xj == frozenset([Fraction(1,3),Fraction(2,3),Fraction(-1,3)]):
            for xj in [frozenset([Fraction(1,3),Fraction(2,3)]), frozenset([Fraction(1,3),Fraction(-1,3)])]:
                new_idx[xj],old_idx[xj],not_new_idx[xj]=self.adder_gpu(xj)
        else:
            new_idx[self.xj],old_idx[self.xj],not_new_idx[self.xj]=self.adder_gpu(self.xj)
        return new_idx,old_idx,not_new_idx
        
    
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

        # Adder
        if not vec.is_contiguous():
            vec=vec.contiguous()
        if self.xj==frozenset([Fraction(1,3),Fraction(2,3),Fraction(-1,3)]):
            if len(n)==1:
                self.adder_tensor_(vec,xj=frozenset([Fraction(1,3),Fraction(2,3)]))
            elif len(n)==2:
                self.adder_tensor_(vec,xj=frozenset([Fraction(1,3),Fraction(-1,3)]))
            else:
                raise NotImplementedError(f"control map with len(n)={len(n)} not supported")
        else:
            self.adder_tensor_(vec,xj=self.xj)
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
        
    def random_control(self,p_ctrl,p_proj,vec=None,p_global=None):
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
        op_idx_dict=self.generate_binary(idx_list=[torch.arange(self.rng.shape[0] if self.ensemble is None else self.ensemble,device=self.device),], p=torch.tensor([p_ctrl,],device=self.device),rng=self.rng_C,label=['C','B'])  # op_idx_dict['C'] for control map, op_idx_dict['B'] for Bernoulli map
        ctrl_outcome_idx_dict={} # whether projected to zero in control map
        p_0=None
        if (op_idx_dict['C']).numel()>0:
            # apply control map
            vec_ctrl=vec[...,op_idx_dict['C'][:,0],:]
            ctrl_idx=[op_idx_dict['C'][:,0], torch.arange(self.ensemble_m,device=self.device)]
            if self.xj in [frozenset([Fraction(1,3),Fraction(2,3)]), frozenset([0]),frozenset([1]),frozenset([-1])]:
                p_0= self.inner_prob(vec=vec_ctrl,pos=[self.L-1],n_list=[0])[None] # prob for 0
                ctrl_outcome_idx_dict=self.generate_binary(idx_list=ctrl_idx, p=p_0,rng=self.rng,label=[0,1])
                for key,idx in ctrl_outcome_idx_dict.items():
                    if len(idx)>0:
                        vec[...,idx[:,0],idx[:,1]]=self.op_list[("C",key)](vec[...,idx[:,0],idx[:,1]])
            elif self.xj==frozenset([Fraction(1,3),Fraction(-1,3)]):
                p_0=torch.stack([self.inner_prob(vec_ctrl,pos=[0,self.L-1],n_list=n) for n in [(0,0),(0,1),(1,0)]])
                ctrl_outcome_idx_dict=self.generate_binary(ctrl_idx, p=p_0,rng=self.rng,label=[(0,0),(0,1),(1,0),(1,1)])
                for key,idx in ctrl_outcome_idx_dict.items():
                    if len(idx)>0:
                        vec[...,idx[:,0],idx[:,1]]=self.op_list[('C',)+key](vec[...,idx[:,0],idx[:,1]])
            elif self.xj==frozenset([Fraction(1,3),Fraction(2,3),Fraction(-1,3)]):
                assert p_global is not None, "p_global should not be None"
                op2ctrl_idx_dict=self.generate_binary(idx_list=ctrl_idx, p=torch.tensor([p_global,],device=self.device),rng=self.rng_C,label=['C-Global','C-Local'])
                # del op_idx_dict['C']
                op_idx_dict.update(op2ctrl_idx_dict)
                for key,idx in op_idx_dict.items():
                    if len(idx)>0:
                        vec_ctrl=vec[...,op_idx_dict[key][:,0],:]
                        ctrl_idx=[op_idx_dict[key][:,0], torch.arange(self.ensemble_m,device=self.device)]
                        if key == 'C-Global':
                            p_0= self.inner_prob(vec=vec_ctrl,pos=[self.L-1],n_list=[0])[None] # prob for 0
                            ctrl_outcome_idx_dict=self.generate_binary(idx_list=ctrl_idx, p=p_0,rng=self.rng,label=[0,1])
                            for key,idx in ctrl_outcome_idx_dict.items():
                                if len(idx)>0:
                                    vec[...,idx[:,0],idx[:,1]]=self.op_list[("C",key)](vec[...,idx[:,0],idx[:,1]])
                        elif key == 'C-Local':
                            p_0=torch.stack([self.inner_prob(vec_ctrl,pos=[0,self.L-1],n_list=n) for n in [(0,0),(0,1),(1,0)]])
                            ctrl_outcome_idx_dict=self.generate_binary(ctrl_idx, p=p_0,rng=self.rng,label=[(0,0),(0,1),(1,0),(1,1)])
                            for key,idx in ctrl_outcome_idx_dict.items():
                                if len(idx)>0:
                                    vec[...,idx[:,0],idx[:,1]]=self.op_list[('C',)+key](vec[...,idx[:,0],idx[:,1]])


        proj_idx_dict={} # {pos: {True: .., False:..}} whether pos is projected site
        proj_outcome_idx_dict={} # {pos: {True:.., False: ..}} if projected, whether it is projected to 0 
        p_2_dict={}
        if (op_idx_dict['B']).numel()>0:
            # apply Bernoulli map
            rng_C=self.rng_C[op_idx_dict['B'][:,0].cpu().numpy()] if self.ensemble is None else None
            size=op_idx_dict['B'].shape[0] if self.ensemble is not None else None
            vec[...,op_idx_dict['B'][:,0],:]=self.op_list[("B",)](vec[...,op_idx_dict['B'][:,0],:],size=size,rng=rng_C)
            for pos in [self.L-1,self.L-2]:
                proj_idx_dict[pos]=self.generate_binary([op_idx_dict['B'][:,0],], p=torch.tensor([p_proj,],device=self.device),rng=self.rng_C,label=[True,False])
                if (proj_idx_dict[pos][True]).numel()>0:
                    vec_p=vec[...,proj_idx_dict[pos][True][:,0],:]
                    p_2 = self.inner_prob(vec=vec_p,pos=[pos], n_list=[0])[None]
                    proj_idx=[proj_idx_dict[pos][True][:,0], torch.arange(self.ensemble_m,device=self.device)]
                    p_2_dict[pos]=p_2
                    proj_outcome_idx_dict[pos]=self.generate_binary(idx_list=proj_idx, p=p_2,rng=self.rng,label=[0,1])
                    for key,idx in proj_outcome_idx_dict[pos].items():
                        if len(idx)>0:
                            vec[...,idx[:,0],idx[:,1]]=self.op_list[(f'P{pos}',key)](vec[...,idx[:,0],idx[:,1]])
                            
        self.update_history(vec,(op_idx_dict,ctrl_outcome_idx_dict,proj_idx_dict,proj_outcome_idx_dict),(p_0,p_2_dict))

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
        op_idx_dict,ctrl_outcome_idx_dict,proj_idx_dict,proj_outcome_idx_dict=op
        self.generate_binary([torch.arange(self.rng.shape[0] if self.ensemble is None else self.ensemble,device=self.device),], p=None,rng=self.rng_C,dummy=True,label=None) # dumm run for controvl vs Bernoulli
        p_0=None
        if (op_idx_dict['C']).numel()>0:
            vec_ctrl=vec[...,op_idx_dict['C'][:,0],:]
            ctrl_idx=[op_idx_dict['C'][:,0], torch.arange(self.ensemble_m,device=self.device)]
            if self.xj in [frozenset([Fraction(1,3),Fraction(2,3)]), frozenset([0]),frozenset([1]),frozenset([-1])]:
                p_0= self.inner_prob(vec=vec_ctrl,pos=[self.L-1],n_list=[0])[None]
                self.generate_binary(ctrl_idx, p=None,rng=self.rng,dummy=True,label=None)
                for key, idx in ctrl_outcome_idx_dict.items():
                    if len(idx)>0:
                        vec[...,idx[:,0],idx[:,1]]=self.op_list[("C",key)](vec[...,idx[:,0],idx[:,1]])
            elif self.xj==frozenset([Fraction(1,3),Fraction(-1,3)]):
                p_0=torch.stack([self.inner_prob(vec_ctrl,pos=[0,self.L-1],n_list=n) for n in [(0,0),(0,1),(1,0)]])
                self.generate_binary(ctrl_idx, p=p_0,rng=self.rng,dummy=True,label=None)
                for key,idx in ctrl_outcome_idx_dict.items():
                    if len(idx)>0:
                        vec[...,idx[:,0],idx[:,1]]=self.op_list[('C',)+key](vec[...,idx[:,0],idx[:,1]])

        p_2_dict={}
        if (op_idx_dict['B']).numel()>0:
            rng_C=self.rng_C[op_idx_dict['B'][:,0].cpu().numpy()] if self.ensemble is None else None
            size=op_idx_dict['B'].shape[0] if self.ensemble is not None else None
            vec[...,op_idx_dict['B'][:,0],:]=self.op_list[("B",)](vec[...,op_idx_dict['B'][:,0],:],size=size,rng=rng_C)
            for pos in [self.L-1,self.L-2]:
                self.generate_binary([op_idx_dict['B'][:,0],], p=None,rng=self.rng_C,dummy=True,label=None)
                if (proj_idx_dict[pos][True]).numel()>0:
                    vec_p=vec[...,proj_idx_dict[pos][True][:,0],:]
                    p_2 = self.inner_prob(vec=vec_p,pos=[pos], n_list=[0])[None]
                    proj_idx=[proj_idx_dict[pos][True][:,0], torch.arange(self.ensemble_m,device=self.device)]
                    p_2_dict[pos]=p_2
                    self.generate_binary(proj_idx, p=None,rng=self.rng,dummy=True,label=None)
                    for key,idx in proj_outcome_idx_dict[pos].items():
                        if len(idx)>0:
                            vec[...,idx[:,0],idx[:,1]]=self.op_list[(f'P{pos}',key)](vec[...,idx[:,0],idx[:,1]])
        self.update_history(vec,(op_idx_dict,ctrl_outcome_idx_dict,proj_idx_dict,proj_outcome_idx_dict),(p_0,p_2_dict))

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
        if self.xj in [frozenset([Fraction(1,3),Fraction(2,3)]), frozenset([Fraction(1,3),Fraction(-1,3)]),frozenset([Fraction(1,3),Fraction(2,3),Fraction(-1,3)])]:
            O=self.ZZ_tensor(vec)
        elif self.xj in [frozenset([0]),frozenset([-1])]:
            O=self.Z_tensor(vec)
        elif self.xj == frozenset([1]):
            O=-self.ZZ_tensor(vec)
        else:
            raise NotImplementedError(f"Order parameter of {self.xj} is not implemented")
        return O  

    def von_Neumann_entropy_pure(self,subregion,vec=None,driver='gesvd',n=1,threshold=None,sv=False):
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
        # There might be a problem here: the normalization of all sv, i.e., sum(sv**2)=1, may need to check, this may not affect n=1, because, 0log0 = 1 log 1 =0 anyway. but it has a singular effect on n=0. An post-workaround is to test the norm, if norm >1 , which is problematic, then either only keep the first one, or drop it
        
        S=torch.linalg.svdvals(vec_,driver=driver)
        if sv:
            return S

        S_pos=torch.clamp(S,min=1e-18)
        S_pos2=S_pos**2
        if n==1:
            return torch.sum(-torch.log(S_pos2)*S_pos2,axis=-1)
        elif n==0:
            if threshold is None:
                # this threshold may be too stringent.
                threshold=torch.finfo(self.dtype['torch']).eps
            return torch.log((S_pos2>threshold**2).sum(axis=-1))
        elif n==np.inf:
            return -torch.log(torch.max(S_pos2,axis=-1)[0])
        else:
            return torch.log((S_pos2**n).sum(axis=-1))/(1-n)

    def half_system_entanglement_entropy(self,vec=None,selfaverage=False,n=1,threshold=None,sv=False):
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
            S_A=torch.mean([self.von_Neumann_entropy_pure(np.arange(i,i+self.L//2),vec,n=n,threshold=threshold) for i in range(self.L//2)])
        else:
            S_A=self.von_Neumann_entropy_pure(np.arange(self.L//2),vec,n=n,threshold=threshold,sv=sv)
        return S_A

    def bipartite_mutual_information(self,subregion_A,subregion_B,selfaverage=False,vec=None,n=1,threshold=1e-10,sv=False):
        """Calculate bipartite entanglement entropy. The bipartite entanglement entropy is defined as S_A+S_B-S_AB. 
        
        Parameters
        ----------
        subregion_A : list of int or torch.tensor
            subregion A
        subregion_B : list of int or torch.tensor
            subregion B
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
        if vec is None:
            vec=self.vec
        if selfaverage:
            return torch.mean([self.ipartite_mutual_information((subregion_A+shift)%self.L,(subregion_B+shift)%self.L,selfaverage=False) for shift in range(len(subregion_A))])
        else:
            S_A=self.von_Neumann_entropy_pure(subregion_A,vec=vec,n=n,threshold=threshold,sv=sv)
            S_B=self.von_Neumann_entropy_pure(subregion_B,vec=vec,n=n,threshold=threshold,sv=sv)
            S_AB=self.von_Neumann_entropy_pure(np.concatenate([subregion_A,subregion_B]),vec=vec,n=n,threshold=threshold,sv=sv)
        if sv:
            return {'S_A':S_A,'S_B':S_B,'S_AB':S_AB}
        return S_A+ S_B -S_AB


    def tripartite_mutual_information(self,subregion_A,subregion_B, subregion_C,selfaverage=False,vec=None,n=1,threshold=1e-10,sv=False):
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
            S_A=self.von_Neumann_entropy_pure(subregion_A,vec=vec,n=n,threshold=threshold,sv=sv)
            S_B=self.von_Neumann_entropy_pure(subregion_B,vec=vec,n=n,threshold=threshold,sv=sv)
            S_C=self.von_Neumann_entropy_pure(subregion_C,vec=vec,n=n,threshold=threshold,sv=sv)
            S_AB=self.von_Neumann_entropy_pure(np.concatenate([subregion_A,subregion_B]),vec=vec,n=n,threshold=threshold,sv=sv)
            S_AC=self.von_Neumann_entropy_pure(np.concatenate([subregion_A,subregion_C]),vec=vec,n=n,threshold=threshold,sv=sv)
            S_BC=self.von_Neumann_entropy_pure(np.concatenate([subregion_B,subregion_C]),vec=vec,n=n,threshold=threshold,sv=sv)
            S_ABC=self.von_Neumann_entropy_pure(np.concatenate([subregion_A,subregion_B,subregion_C]),vec=vec,n=n,threshold=threshold,sv=sv)
            if sv:
                return {'S_A':S_A,'S_B':S_B,'S_C':S_C,'S_AB':S_AB,'S_AC':S_AC,'S_BC':S_BC,'S_ABC':S_ABC}
            
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
        '''Maybe I should abandon the previous way of storing history, why not just use the four *_idx_dict to store the history directly? This will also create convenience when call it'''
        if vec is not None:
            if self.store_vec:
                self.vec_history.append(vec.cpu().clone())
            else:
                # Reserved for later use
                pass
        if op is not None:
            if self.store_op:
                self.op_history.append(op)
            else:
                pass
        if p is not None:
            if self.store_prob:
                p_0,p_2_dict=p

                dtype=torch.float64 if self.dtype['torch']==torch.complex128 else torch.float32
                op_idx_dict,ctrl_outcome_idx_dict,proj_idx_dict,proj_outcome_idx_dict=op
                p_map=torch.ones((3,self.rng.shape[0] if self.ensemble is None else self.ensemble,self.ensemble_m),device=self.device,dtype=dtype)
                if op_idx_dict['C'].numel()>0:
                    p_0=torch.concat((p_0,1-p_0.sum(axis=0)[None]))
                    p_ctrl_map=torch.ones((p_0.shape[0],self.rng.shape[0] if self.ensemble is None else self.ensemble,self.ensemble_m),device=self.device,dtype=dtype)
                    p_ctrl_map[:,op_idx_dict['C'][:,0]]=-p_0
                    outcome_idx=0
                    for key,val in ctrl_outcome_idx_dict.items():
                        p_ctrl_map[outcome_idx][tuple(val.T)] = (-1)*p_ctrl_map[outcome_idx][tuple(val.T)]
                        outcome_idx+=1
                    p_map[0]=p_ctrl_map.max(dim=0)[0]
                    
                for idx,key in enumerate(p_2_dict.keys()):
                    p_2_dict[key]=torch.concat((p_2_dict[key],1-p_2_dict[key].sum(axis=0)[None]))
                    p_proj_map=torch.ones((p_2_dict[key].shape[0],self.rng.shape[0] if self.ensemble is None else self.ensemble,self.ensemble_m),device=self.device,dtype=dtype)
                    p_proj_map[:,proj_idx_dict[key][True][:,0]]=-p_2_dict[key]
                    if key in proj_outcome_idx_dict:
                        outcome_idx=0
                        for key,val in proj_outcome_idx_dict[key].items():
                            p_proj_map[outcome_idx][tuple(val.T)] = (-1)*p_proj_map[outcome_idx][tuple(val.T)]
                            outcome_idx+=1
                    p_map[idx+1]=p_proj_map.max(dim=0)[0]
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
        """Directly frozenset zero at tensor[...,0,:] =0 for n==1 and tensor[...,1,:] =0 for n==0. This works both one and two ensemble indices. 

        Parameters
        ----------
        vec : torch.tensor
            state vector
        n : int, {0,1}
            out come of the projection
        pos : int, {0,1,...,L-1}
            position to apply the projection.
        """
        '''directly frozenset zero at tensor[...,0] =0 for n==1 and tensor[...,1] =0 for n==0'
        This is an in-placed operation
        '''
        for n_i,pos_i in zip(n,pos):
            idx_list=[slice(None)]*vec.dim()
            idx_list[pos_i]=1-n_i
            vec[tuple(idx_list)]=0

    def R_tensor(self,vec,n,pos):
        self.P_tensor_(vec,n,pos)
        # if self.xj in [frozenset([Fraction(1,3),Fraction(2,3)]), frozenset([0]),frozenset([1]),frozenset([-1])]:
        if len(n)==1:
            if n[0]==1:
                if self.feedback:
                    vec=self.XL_tensor(vec)
        # elif self.xj==frozenset([Fraction(1,3),Fraction(-1,3)]):
        elif len(n)==2: 
            if n[0]^n[1]==0:
                if self.feedback:
                    vec=self.XL_tensor(vec)
        else:
            raise NotImplementedError(f"Reset of {self.xj} is not implemented")
        self.normalize_(vec)
        return vec

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

    def adder_gpu(self,xj):
        """Calculate the shift adder index, namely, `old_index` -> `new_index`, and `not_new_index` should be all zero. This is not a full adder, which assume the leading digit in the input bitstring is zero (because of the T^{-1}R_L, the leading bit should always be zero). 

        Returns
        -------
        tuple of torch.tensor
            new_idx, old_idx, not_new_idx
        """
        if xj==frozenset([Fraction(1,3),Fraction(2,3)]):
            int_1_6=(int(Fraction(1,6)*2**self.L)|1)
            int_1_3=(int(Fraction(1,3)*2**self.L))
            old_idx=torch.arange(2**(self.L-1),device=self.device).view((2,-1))
            adder_idx=torch.tensor([[int_1_6],[int_1_3]],device=self.device)
            new_idx=(old_idx+adder_idx)
            # handle the extra attractors, if 1..0x1, then 1..0(1-x)1, if 0..1x0, then 0..1(1-x)0 [shouldn't enter this branch..]
            mask_1=(new_idx&(1<<self.L-1) == (1<<self.L-1)) & (new_idx&(1<<2) == (0)) & (new_idx&(1) == (1))
            mask_2=(new_idx&(1<<self.L-1) == (0)) & (new_idx&(1<<2) == (1<<2)) & (new_idx&(1) == (0))
            new_idx[mask_1+mask_2]=new_idx[mask_1+mask_2]^(0b10)
        elif xj in [frozenset([0]),frozenset([Fraction(1,3),Fraction(-1,3)])]:
            return torch.tensor([]), torch.tensor([]),torch.tensor([])
        elif xj==frozenset([1]):
            int_1=(1<<self.L)-1
            int_1_2=(1<<(self.L-1))+1
            old_idx=torch.arange(2**(self.L-1),device=self.device).view((2,-1))
            adder_idx=torch.tensor([[int_1],[int_1_2]],device=self.device)
            new_idx=(old_idx+adder_idx)%(1<<self.L)
        elif xj==frozenset([-1]):
            old_idx=torch.arange(2**(self.L),device=self.device).view((2,-1))
            new_idx=(old_idx+self.add_x)%(1<<self.L)
        else:
            raise NotImplementedError(f"{xj} is not implemented")
        if self.ancilla:
            new_idx=torch.hstack((new_idx<<1,(new_idx<<1)+1))
            old_idx=torch.hstack((old_idx<<1,(old_idx<<1)+1))
        not_new_idx=torch.ones(2**(self.L_T),dtype=bool,device=self.device)
        not_new_idx[new_idx]=False
        return new_idx, old_idx, not_new_idx

    def adder_tensor_(self,vec,xj):
        """Apply the index shuffling of `vec` using the `old_idx` and `new_idex` generated from `adder_gpu`. 

        Parameters
        ----------
        vec : torch.tensor
            state vector 
        """
        new_idx=self.new_idx[xj].flatten()
        old_idx=self.old_idx[xj].flatten()
        not_new_idx=self.not_new_idx[xj].flatten()
        if (new_idx).shape[0]>0 and (old_idx).shape[0]>0:
            vec_flatten=vec.view((2**self.L_T,-1))    
            vec_flatten[new_idx]=vec_flatten[old_idx]
            vec_flatten[not_new_idx]=0

    def generate_binary(self,idx_list,p,rng,label,dummy=False,):
        """Generate boolean list, given probability `p` and seed `rng`.

        Parameters
        ----------
        idx_list : list of list
            [[ensemble...],[ensemble_m...]], the second list could be None if no slice on `ensemble_m` axis needed.
        p : float or torch.tensor
            if float, this is used in np.rng, if torch.tensor, this is used in torch.rng, the shape should be (idx_list[0].shape[0], ensemble_m), make it list of prob, which corresponds to label[:-1], the last one is 1-sum(p), p.shape = (outcome, ensemble,ensemble_m) or (outcome,)/list
        rng : list of np.rng or torch.Generator
            Random generators to use. 
        dummy : bool, optional
            If false, no dictionary will be generated, and exists for the reason of reproducibility, by default False

        Returns
        -------
        dict
            A dictionary to store the index of operators.
        """
        idx_dict={}
        if self.ensemble is None:
            random=torch.tensor([rng[idx].random() for idx in idx_list[0]],device=self.device)
        else:
            random=torch.rand(*(idx_list[i].shape[0] for i in range(len(idx_list))),device=self.device,generator=rng)
        if not dummy:
            if p.dim()==1:
                # p=p.reshape((-1,1,1)).expand(1,random.shape[0],random.shape[1])
                p=p.reshape((-1,)+(1,)*random.dim()).expand(1,*random.shape)
            outcome=(p.cumsum(axis=0)<random).sum(axis=0)
            for lb_idx,lb in enumerate(label):
                outcome_idx=torch.nonzero(outcome==lb_idx)
                idx_dict[lb]=torch.stack([idx_list[i][outcome_idx[:,i]] for i in range(outcome_idx.shape[-1])]).T
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
