from rqc.CT_tensor import CT_tensor
import numpy as np
import pickle
import argparse
import time
from tqdm import tqdm
from fractions import Fraction
import torch
import gc


def convert_to_fraction(fraction_str):
    fractions = []
    for item in fraction_str.split(","):
        if "/" in item:
            num, denom = item.split("/")
            fractions.append(Fraction(int(num), int(denom)))
        else:
            fractions.append(int(item))
    return fractions

def convert_x0(xj,L):
    """for FM, it is used to create a 'k=1' initial state or 'k=L/2'"""
    if xj == [Fraction(1,3),Fraction(2,3)]:
        return Fraction(((int(Fraction(1,3)*(2<<(L//2-1)))<<(L//2))+((1-(L//2%2))<<(L//2-1))),2**L)
    elif xj==[0]:
        # return Fraction(1<<(L//2-1),1<<L)
        return Fraction(1,1<<L)

def get_total_coherence(wf):
    # wf.shape = (2,2,...,2,C,M), assume M=1
    L=len(wf.shape)-2
    return torch.einsum(torch.abs(wf[...,0]),range(L+1),[L]).square_()-1

    
def run_tensor_final(inputs):
    L,p_ctrl,p_proj,xj,complex128,seed,ensemble,x0=inputs
    assert xj == [0], f'xj= {xj}, the AFM has to be created, because the domain wall support is different'
    ct=CT_tensor(L=L,seed=seed,xj=xj,gpu=True,complex128=complex128,_eps=1e-5,ensemble=ensemble,ancilla=False,x0=x0)
    T_max=2*ct.L**2

    for t_idx in range(T_max):
        ct.random_control(p_ctrl=p_ctrl,p_proj=p_proj)
    
    coherece_matrix_map = get_total_coherence(ct.vec)
    
    return coherece_matrix_map

if __name__=="__main__":
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(),flush=True)
    parser=argparse.ArgumentParser()
    parser.add_argument('--es','-es',default=10,type=int,help='Ensemble size (default: 10).')
    parser.add_argument('--seed','-seed',default=0,type=int,help='Random seed (default: 0).')
    parser.add_argument('--p_ctrl','-p_ctrl',type=float,nargs=3,default=[0,1,3],help='Parameters for p_ctrl in the form [start, stop, num] to generate values with np.linspace (default: [0, 1, 11]).')
    parser.add_argument('--p_proj','-p_proj',type=float,nargs=3,default=[0,1,1],help='Parameters for p_proj in the form [start, stop, num] to generate values with np.linspace (default: [0, 1, 11]).')
    parser.add_argument('--L','-L',type=int,nargs=3,default=[10,12,2],help='Parameters for L in the form [start, stop, step] to generate values with np.arange (default: [10, 16, 2]).')
    parser.add_argument('--xj','-xj',type=str,default="1/3,2/3", help="List of fractions or 0 in the format num1/denom1,num2/denom2,... or 0. For example: 1/2,2/3")
    parser.add_argument('--x0','-x0',type=float,default=None, help="Initial value in fraction. Using -1 for the initial value of the first domain wall right in the middle of the chain.")
    parser.add_argument('--complex128','-complex128',action='store_true', help="add --complex128 to have precision of complex128")

    args=parser.parse_args()

    xj = convert_to_fraction(args.xj)
    print(xj)

    L_list=np.arange(args.L[0],args.L[1],args.L[2])

    p_ctrl_list=np.linspace(args.p_ctrl[0],args.p_ctrl[1],int(args.p_ctrl[2]))
    p_proj_list=np.linspace(args.p_proj[0],args.p_proj[1],int(args.p_proj[2]))
    st=time.time()
    inputs=[(L,p_ctrl,p_proj,xj,args.complex128,args.seed,args.es,convert_x0(xj,L) if args.x0 == -1 else args.x0) for L in (L_list) for p_ctrl in (p_ctrl_list) for p_proj in (p_proj_list)]

    results=[]
    for param in tqdm(inputs):
        result=run_tensor_final(param)
        results.append(result.cpu())
        del result
        gc.collect()
        torch.cuda.empty_cache()
    
    results=torch.cat(results)
    L0=args.L[0]
    rs=results.reshape((L_list.shape[0],p_ctrl_list.shape[0],p_proj_list.shape[0],args.es))

    with open('CT_En{:d}_pctrl({:.2f},{:.2f},{:.0f})_pproj({:.2f},{:.2f},{:.0f})_L({:d},{:d},{:d})_xj({:s})_seed{:d}{:s}_coherence_final.pickle'.format(args.es,*args.p_ctrl,*args.p_proj,*args.L,args.xj.replace('/','-'),args.seed,'_128' if args.complex128 else '_64'),'wb') as f:
        pickle.dump(rs, f)
    print('Time elapsed: {:.4f}'.format(time.time()-st))


