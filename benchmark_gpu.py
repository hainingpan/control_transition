from CT import CT_tensor
import numpy as np
import argparse
import torch
from fractions import Fraction
import time
from tqdm import tqdm

def convert_to_fraction(fraction_str):
    fractions = []
    for item in fraction_str.split(","):
        if "/" in item:
            num, denom = item.split("/")
            fractions.append(Fraction(int(num), int(denom)))
        else:
            fractions.append(int(item))
    return fractions

def run_tensor(inputs):
    L,xj,seed,ensemble=inputs
    ct=CT_tensor(L=L,seed=seed,xj=xj,gpu=True,complex128=True,ensemble=ensemble,ancilla=False)
    for _ in tqdm(range(100)):
        ct.encoding()
        torch.cuda.empty_cache()
    O=ct.order_parameter()
    return O

if __name__=="__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('--es','-es',default=10,type=int,help='Ensemble size (default: 10).')
    parser.add_argument('--seed','-seed',default=0,type=int,help='Random seed (default: 0).')
    parser.add_argument('--L','-L',default=10,type=int,help='System size (default: 10).')
    parser.add_argument('--xj','-xj',type=str,default="1/3,2/3", help="List of fractions or 0 in the format num1/denom1,num2/denom2,... or 0. For example: 1/2,2/3")
    args=parser.parse_args()
    xj = convert_to_fraction(args.xj)

    st=time.time()
    _=run_tensor((args.L,xj,args.seed,args.es))
    print('T={:.4f}s'.format(time.time()-st))