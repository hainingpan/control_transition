# A dirty quick way to convert parallel run in ipcluster to CMD, not using argparse
from rqc.bricklayer import bricklayer
import numpy as np
import pandas as pd
from tqdm import tqdm
from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI
import argparse

def run_bl(inputs):
    L,p,add_x,idx=inputs
    bl=bricklayer(L=L,seed=idx,add_x=add_x,store_vec=False)
    for _ in range(bl.L):
        bl.random_projection(p)
    SA=bl.half_system_entanglement_entropy()
    TMI=bl.tripartite_mutual_information(np.arange(L//4),np.arange(L//4)+L//4,np.arange(L//4)+(L//4)*2,selfaverage=False)
    return SA, TMI


if __name__=="__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    parser=argparse.ArgumentParser()
    parser.add_argument('--nu','-nu',default=0,type=int,help='numertor')
    parser.add_argument('--de','-de',default=1,type=int,help='denominator')
    args=parser.parse_args()


    L_list=np.array([8,12,])
    p_list=np.linspace(0,.6,21)
    ensemble=2000

    inputs=[(L,p,2**L*args.nu/args.de,idx) for L in L_list for p in p_list for idx in range(ensemble)]
    with MPIPoolExecutor() as executor:
        results=list(tqdm(executor.map(run_bl,inputs),total=len(inputs)))
    
    with open(f'bricklayer_adder_{args.nu}_{args.de}.pickle','wb') as f:
        pickle.dump(results,f)