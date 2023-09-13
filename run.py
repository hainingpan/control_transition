from CT import *
import numpy as np
import pickle
import argparse
from mpi4py.futures import MPIPoolExecutor


def run_quantum(inputs):
    L,p_ctrl,p_proj,xj,idx=inputs
    ct=CT_quantum(L=L,x0=None,seed=idx,history=False,xj=xj)
    for _ in range(2*ct.L**2):
        ct.random_control_2(p_ctrl=p_ctrl,p_proj=p_proj)
    O=ct.order_parameter()
    SA=ct.half_system_entanglement_entropy()
    TMI=ct.tripartite_mutual_information(np.arange(ct.L//4),np.arange(ct.L//4)+ct.L//4,np.arange(ct.L//4)+(ct.L//4)*2,selfaverage=False)

    return O,SA, TMI

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--es','-es',default=10,type=int)
    args=parser.parse_args()
    ensemble=args.es
    L_list=np.array([10,12,14,16])

    p_ctrl_list=np.linspace(0,1,11)
    p_proj_list=np.linspace(0,1,11)

    inputs=[(L,p_ctrl,p_proj,[Fraction(1,3),Fraction(2,3)],idx) for L in L_list for p_ctrl in p_ctrl_list for p_proj in p_proj_list for idx in range(ensemble)]

    with MPIPoolExecutor() as executor:
        results=(executor.map(run_quantum,inputs))

    # results=list(map(run_quantum,inputs))

    rs=np.array(list(results)).reshape((L_list.shape[0],p_ctrl_list.shape[0],p_proj_list.shape[0],ensemble,3))
    O_map,EE_map,TMI_map=rs[:,:,:,:,0],rs[:,:,:,:,1],rs[:,:,:,:,2]
    with open('CT_En{:d}.pickle'.format(args.es),'wb') as f:
        pickle.dump({"O":O_map,"EE":EE_map,"TMI":TMI_map}, f)
