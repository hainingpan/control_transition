from rqc.CT_classical import CT_classical
import numpy as np
import pickle
import argparse
import time
from tqdm import tqdm
from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI
from fractions import Fraction
import h5py

def convert_to_fraction(fraction_str):
    fractions = []
    for item in fraction_str.split(","):
        if "/" in item:
            num, denom = item.split("/")
            fractions.append(Fraction(int(num), int(denom)))
        else:
            fractions.append(int(item))
    return fractions
from functools import lru_cache
@lru_cache(maxsize=None)
def convert_bitstring_to_dw(L,ZZ=False):
    # The distance of the leftmost "1" to the right
    dw=[]
    for bitstring in range(2**L):
        pos=L-1
        if ZZ:
            bitstring=~(bitstring^(bitstring>>1))
            pos-=1
        while pos>-1 and bitstring&(1<<pos)==0:
            pos-=1
        dw.append(pos+1)
    return np.array(dw)

def run_quantum(inputs,):
    (L_idx,L),(p_ctrl_idx,p_ctrl),xj,seed,save_T,x0=inputs
    ct=CT_classical(L=L,seed=seed,xj=xj,x0=x0,store_vec=save_T)
    idx=0
    dw=convert_bitstring_to_dw(L,ZZ=False if 0 in xj else True)
    for _ in range(2*ct.L**2):
        ct.random_control(p=p_ctrl)

    O=ct.order_parameter()
    return O,ct.vec_history

if __name__=="__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    # rank = comm.Get_rank()
    print(f'Total size:{size}',flush=True)
    parser=argparse.ArgumentParser()
    parser.add_argument('--es','-es',default=10,type=int,help='Ensemble size (default: 10).')
    parser.add_argument('--p_ctrl','-p_ctrl',type=float,nargs=3,default=[0,1,11],help='Parameters for p_ctrl in the form [start, stop, num] to generate values with np.linspace (default: [0, 1, 11]).')
    parser.add_argument('--L','-L',type=int,nargs=3,default=[10,16,2],help='Parameters for L in the form [start, stop, step] to generate values with np.arange (default: [10, 16, 2]).')
    parser.add_argument('--xj','-xj',type=str, help="List of fractions or 0 in the format num1/denom1,num2/denom2,... or 0. For example: 1/2,2/3")
    parser.add_argument('--x0','-x0',type=float,default=None, help="Initial value in fraction")
    parser.add_argument('--save_T','-save_T',action='store_true', help="add --save to save the time evolution of the wavefunction")

    args=parser.parse_args()
    xj = convert_to_fraction(args.xj)
    L_list=np.arange(args.L[0],args.L[1],args.L[2])
    p_ctrl_list=np.linspace(args.p_ctrl[0],args.p_ctrl[1],int(args.p_ctrl[2]))
    st=time.time()
    inputs=[((L_idx,L),(p_ctrl_idx,p_ctrl),xj,seed,args.save_T,args.x0) for L_idx,L in enumerate(L_list) for p_ctrl_idx,p_ctrl in enumerate(p_ctrl_list) for seed in range(args.es)]

    with h5py.File('CT_En{:d}_pctrl({:.2f},{:.2f},{:.0f})_L({:d},{:d},{:d})_xj({:s})_C{:s}.hdf5'.format(args.es,*args.p_ctrl,*args.L,args.xj.replace('/','-'),'_T'*args.save_T),'w') as f:
        if args.save_T:
            # Save all history of wavefunction (as a function of time)
            datasets={L:f.create_dataset(f'wf_{L}',((len(p_ctrl_list),2*L**2+1,args.es)),dtype=int) for L in L_list}
        else:
            # Save only wavefunction (at the end of time evolution)
            datasets={L:f.create_dataset(f'wf_{L}',((len(p_ctrl_list),1,args.es)),dtype=int) for L in L_list}
        metric_datasets={metric:f.create_dataset(f'{metric}',((len(p_ctrl_list),len(L_list),args.es)),dtype=float) for metric in ['O',]}

        # for param in tqdm(inputs):
        #     results=run_quantum(param)
        results=tqdm(map(run_quantum,inputs),total=len(inputs))
        # with MPIPoolExecutor() as executor:
        #     results=list(tqdm(executor.map(run_quantum,inputs),total=len(inputs)))
        
        for result,input_ in zip(results,inputs):
            O,vec_history=result
            (L_idx,L),(p_ctrl_idx,p_ctrl),xj,seed,save_T,x0=input_
            datasets[L][p_ctrl_idx,:,seed]=vec_history
            metric_datasets['O'][p_ctrl_idx,L_idx,seed]=O
            

    print('Time elapsed: {:.4f}'.format(time.time()-st))