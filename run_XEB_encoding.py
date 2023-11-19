from CT import *
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

def prob_log(CT):
    if len(CT.prob_history)>0:
        return torch.log(torch.stack(CT.prob_history)).sum(axis=0)
    else:
        return 0
def run_tensor(inputs):
    L,p_ctrl,p_proj,xj,complex128,seed,seed_C,ancilla,ensemble,ensemble_m=inputs
    ct_q=CT_tensor(L=L,seed=seed,seed_vec=seed_C,seed_C=seed_C,ancilla=ancilla,store_vec=False,store_op=True,store_prob=True,complex128=complex128,gpu=True,ensemble=ensemble,ensemble_m=ensemble_m,xj=xj)
    for _ in range(L**2*2):
        ct_q.random_control(p_ctrl=0, p_proj=0)
    for _ in range(L**2*2):
        ct_q.random_control(p_ctrl, p_proj)
    op_history=ct_q.op_history.copy()
    log_q=prob_log(ct_q)
    tensor_size=ct_q.vec.numel()*ct_q.vec.element_size()/1024**2
    del ct_q
    gc.collect()
    torch.cuda.empty_cache()
    ct_r=CT_tensor(L=L,seed=seed,seed_vec=1+seed_C,seed_C=seed_C,ancilla=False,store_vec=False,store_op=True,store_prob=True,complex128=complex128,gpu=True,ensemble=ensemble,ensemble_m=ensemble_m,xj=xj)
    for op in op_history:
        ct_r.reference_control(op)
    log_r=prob_log(ct_r)
    del ct_r
    gc.collect()
    torch.cuda.empty_cache()
    ct_r_=CT_tensor(L=L,seed=seed,seed_vec=1+seed_C,seed_C=seed_C,ancilla=False,store_vec=False,store_op=True,store_prob=True,complex128=complex128,gpu=True,ensemble=ensemble,ensemble_m=ensemble_m,xj=xj)
    for _ in range(L**2*2):
        ct_r_.random_control(p_ctrl=0, p_proj=0)
    for _ in range(L**2*2):
        ct_r_.random_control(p_ctrl, p_proj)
    log_r_=prob_log(ct_r_)
    del ct_r_
    gc.collect()
    torch.cuda.empty_cache()
    return log_q,log_r,log_r_

if __name__=="__main__":
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(),flush=True)
    parser=argparse.ArgumentParser()
    parser.add_argument('--es','-es',default=10,type=int,help='Ensemble size for circuit (default: 10).')
    parser.add_argument('--es_m','-es_m',default=10,type=int,help='Ensemble size for outcome (default: 10).')
    parser.add_argument('--seed','-seed',default=0,type=int,help='Random seed (default: 0).')
    parser.add_argument('--seed_C','-seed_C',default=0,type=int,help='Random seed_C (default: 0).')
    parser.add_argument('--p_ctrl','-p_ctrl',type=float,nargs=3,default=[0,1,3],help='Parameters for p_ctrl in the form [start, stop, num] to generate values with np.linspace (default: [0, 1, 11]).')
    parser.add_argument('--p_proj','-p_proj',type=float,nargs=3,default=[0,1,1],help='Parameters for p_proj in the form [start, stop, num] to generate values with np.linspace (default: [0, 1, 11]).')
    parser.add_argument('--L','-L',type=int,nargs=3,default=[10,12,2],help='Parameters for L in the form [start, stop, step] to generate values with np.arange (default: [10, 16, 2]).')
    parser.add_argument('--xj','-xj',type=str,default="1/3,2/3", help="List of fractions or 0 in the format num1/denom1,num2/denom2,... or 0. For example: 1/2,2/3")
    parser.add_argument('--complex128','-complex128',action='store_true', help="add --complex128 to have precision of complex128")
    parser.add_argument('--ancilla','-ancilla',action='store_true', help="add --ancilla to have ancilla qubit")

    args=parser.parse_args()

    xj = convert_to_fraction(args.xj)

    L_list=np.arange(args.L[0],args.L[1],args.L[2])

    p_ctrl_list=np.linspace(args.p_ctrl[0],args.p_ctrl[1],int(args.p_ctrl[2]))
    p_proj_list=np.linspace(args.p_proj[0],args.p_proj[1],int(args.p_proj[2]))
    st=time.time()
    inputs=[(L,p_ctrl,p_proj,xj,args.complex128,args.seed,args.seed_C,args.ancilla,args.es,args.es_m) for L in L_list for p_ctrl in p_ctrl_list for p_proj in p_proj_list]

    # results=list(tqdm(map(run_tensor,inputs),total=len(inputs)))
    results=[]
    for param in tqdm(inputs):
        result=run_tensor(param)
        result_cpu=[r.cpu() for r in result]
        results.append(result_cpu)
        del result,result_cpu
        gc.collect()
        torch.cuda.empty_cache()


    results=torch.cat([torch.cat(tensors) for tensors in results])


    rs=results.reshape((L_list.shape[0],p_ctrl_list.shape[0],p_proj_list.shape[0],3,args.es,args.es_m))
    log_q_map,log_r_map,log_r__map=rs[:,:,:,0,:],rs[:,:,:,1,:],rs[:,:,:,2,:]
    save_dict={"log_q":log_q_map,"log_r":log_r_map,"log_r_":log_r__map,"args":args}


    with open('CT_En{:d}_Enm{:d}_pctrl({:.2f},{:.2f},{:.0f})_pproj({:.2f},{:.2f},{:.0f})_L({:d},{:d},{:d})_xj({:s})_seed{:d}_seedC{:d}{:s}{:s}_XEB_Haar_encoding.pickle'.format(args.es,args.es_m,*args.p_ctrl,*args.p_proj,*args.L,args.xj.replace('/','-'),args.seed,args.seed_C,'_128' if args.complex128 else '_64','_anc'*args.ancilla),'wb') as f:
        pickle.dump(save_dict, f)

    print('Time elapsed: {:.4f}'.format(time.time()-st))
