from CT import CT_quantum
import numpy as np
import argparse
import time
from tqdm import tqdm
from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI
from fractions import Fraction
from tqdm import tqdm
import pickle

def convert_to_fraction(fraction_str):
    fractions = []
    for item in fraction_str.split(","):
        if "/" in item:
            num, denom = item.split("/")
            fractions.append(Fraction(int(num), int(denom)))
        else:
            fractions.append(int(item))
    return fractions

def run_quantum(inputs):
    L,xj,idx=inputs
    ct=CT_quantum(L=L,seed=idx,xj=xj)
    for _ in range(100):
        ct.encoding()
    # return ct.order_parameter()
    return ct.half_system_entanglement_entropy()

if __name__=="__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    parser=argparse.ArgumentParser()
    parser.add_argument('--es','-es',default=10,type=int,help='Ensemble size (default: 10).')
    parser.add_argument('--L','-L',default=10,type=int,help='System size (default: 10).')
    parser.add_argument('--xj','-xj',type=str, help="List of fractions or 0 in the format num1/denom1,num2/denom2,... or 0. For example: 1/2,2/3")
    args=parser.parse_args()
    xj = convert_to_fraction(args.xj)
    inputs=[(args.L,xj,idx) for idx in range(args.es)]

    st=time.time()
    with MPIPoolExecutor() as executor:
        results=list(tqdm(executor.map(run_quantum,inputs),total=len(inputs)))
    print('T={:.4f}s'.format(time.time()-st))
    with open('CT_En{:d}_pctrl(0)_pproj(0)_L({:d})_xj({:s}).pickle'.format(args.es,args.L,args.xj.replace('/','-')),'wb') as f:
        pickle.dump(results, f)

