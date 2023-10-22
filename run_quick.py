# A dirty quick way to convert parallel run in ipcluster to CMD, not using argparse
from CT import *
import numpy as np
import pandas as pd
from tqdm import tqdm
from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI
def prob_log(CT):
    if len(CT.prob_history)>0:
        return np.log(np.hstack(CT.prob_history)).sum()
    else:
        return 0
def run_xeb(inputs):
    try:
        L,p_ctrl,p_proj,seed_C,seed=inputs
        ct_q=CT_quantum(L=L,seed=seed+9000,seed_vec=10*seed_C+9000,seed_C=seed_C+9000,x0=None,ancilla=False,store_vec=False,store_op=True,store_prob=True,normalization=True)
        for _ in range(2*ct_q.L**2):
            ct_q.random_control(p_ctrl, p_proj)
        ct_r=CT_quantum(L=L,seed=seed+9000,seed_vec=20*seed_C+9000,seed_C=seed_C+9000,x0=None,ancilla=False,store_vec=False,store_op=True,store_prob=True,normalization=True)
        ct_r.reference_control(ct_q.op_history)
        ct_r_=CT_quantum(L=L,seed=seed+9000,seed_vec=20*seed_C+9000,seed_C=seed_C+9000,x0=None,ancilla=False,store_vec=False,store_op=True,store_prob=True,normalization=True)
        for _ in range(2*ct_q.L**2):
            ct_r_.random_control(p_ctrl, p_proj)
        return prob_log(CT=ct_q),prob_log(CT=ct_r),prob_log(CT=ct_r_)
    except:
        return np.nan,np.nan,np.nan


if __name__=="__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    with MPIPoolExecutor() as executor:
        seed_list=np.arange(0,1000)
        L_list=np.array([10,])
        p_ctrl_list=np.array([0.,0,1])
        p_proj_list=np.linspace(0.1,0.25,11)

        # p_ctrl_list=np.array([0.4,.6,11])
        # p_proj_list=np.linspace(0.,0.,1)
        inputs=[(L,p_ctrl,p_proj,seed_C,seed) for L in L_list for p_ctrl in p_ctrl_list for p_proj in p_proj_list for seed_C in np.arange(0,int(20*((L/L_list[0])*(p_proj/p_proj_list[0]))**2)) for seed in seed_list]

        results=list(tqdm(executor.map(run_xeb,inputs),total=len(inputs)))
        inputs_index=pd.MultiIndex.from_tuples(inputs,names=['L','p_ctrl','p_proj','seed_C','seed'])
        df=pd.DataFrame((results),columns=['log_p_q','log_p_r','log_p_r_'],index=inputs_index)
        df.to_pickle(path=f'CT_quantum_xeb_{L_list[0]}_pctrl0_10.pkl')