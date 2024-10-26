from rqc.APT import APT
import numpy as np
from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI
from time import time
import argparse
import pickle
from tqdm import tqdm
from fractions import Fraction


def run(inputs):
    L, p_m,p_f,seed  = inputs
    apt=APT(L=L,x0=Fraction(2**L-1,2**L),seed=seed,seed_C=None,seed_vec=None,store_op=False)
    for i in range(2*apt.L):
        apt.random_cicuit(p_m=p_m,p_f=p_f,even=True)
        apt.random_cicuit(p_m=p_m,p_f=p_f,even=False)
    OP=apt.order_parameter()
    TMI=apt.tripartite_mutual_information(np.arange(apt.L//4),np.arange(apt.L//4)+apt.L//4,np.arange(apt.L//4)+apt.L//2,selfaverage=True)
    return OP,TMI
    # return OP



if __name__=="__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    # rank = comm.Get_rank()
    print(f'Total size:{size}',flush=True)
    parser=argparse.ArgumentParser()
    parser.add_argument('--L','-L',type=int,default=12,help='Parameters for L')
    parser.add_argument('--es','-es',default=10,type=int,help='Ensemble size (default: 10).')
    parser.add_argument('--p_m','-p_m',type=float,nargs=3,default=[0,1,11],help='Parameters for p_m in the form [start, stop, num] to generate values with np.linspace (default: [0, 1, 11]).')
    parser.add_argument('--p_f','-p_f',type=float,nargs=3,default=[0,1,11],help='Parameters for p_f in the form [start, stop, num] to generate values with np.linspace (default: [0, 1, 11]). If the third parameter is -1, then use the same value of p_m')
    args=parser.parse_args()
    p_m_list=np.linspace(args.p_m[0],args.p_m[1],int(args.p_m[2]))
    if args.p_f[2]==-1:
        p_f_list=np.array([-1])
        inputs=[(args.L, p_m,p_m,idx) for p_m in p_m_list for p_f in p_f_list for idx in range(args.es)]
        
    else:
        p_f_list=np.linspace(args.p_f[0],args.p_f[1],int(args.p_f[2]))
        inputs=[(args.L, p_m,p_f,idx) for p_m in p_m_list for p_f in p_f_list for idx in range(args.es)]
    st=time()

    with MPIPoolExecutor() as executor:
        results=list(tqdm(executor.map(run,inputs),total=len(inputs)))
    # results=list(map(run,inputs))
    
    rs=np.array(results).reshape((p_m_list.shape[0],np.abs(p_f_list.shape[0]),args.es,2))
    O_map,TMI_map=rs[...,0],rs[...,1]

    with open('APT_En{:d}_pm({:.2f},{:.2f},{:.0f})_pf({:.2f},{:.2f},{:.0f})_L{:d}.pickle'.format(args.es,*args.p_m,*args.p_f,args.L),'wb') as f:
        pickle.dump({"O":O_map,"TMI":TMI_map,"args":args}, f)
    
    print('Time elapsed: {:.4f}'.format(time()-st))