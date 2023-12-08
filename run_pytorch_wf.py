# from CT import *
from rqc.CT_tensor import CT_tensor
import numpy as np
import pickle
import argparse
import time
from tqdm import tqdm
from fractions import Fraction
import torch
import gc
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

def run_tensor(inputs,datasets,metric_datasets):
    (L_idx,L),(p_ctrl_idx,p_ctrl),(p_proj_idx,p_proj),xj,complex128,seed,ancilla,ensemble,save_T,x0=inputs
    ct=CT_tensor(L=L,seed=seed,xj=xj,gpu=True,complex128=complex128,_eps=1e-5,ensemble=ensemble,ancilla=ancilla,x0=x0)
    T_max=ct.L**2//2 if ancilla else 2*ct.L**2
    idx=0
    if save_T:
        vec_np = ct.vec.cpu().numpy()
        datasets[L][p_ctrl_idx,p_proj_idx,idx]=vec_np
        idx+=1
    for t_idx in range(T_max):
        ct.random_control(p_ctrl=p_ctrl,p_proj=p_proj)
        if save_T or t_idx==T_max-1:
            vec_np = ct.vec.cpu().numpy()
            datasets[L][p_ctrl_idx,p_proj_idx,idx]=vec_np
            idx+=1
        torch.cuda.empty_cache()

    if not ancilla:
        O=ct.order_parameter()
        EE=ct.half_system_entanglement_entropy()
        TMI=ct.tripartite_mutual_information(np.arange(ct.L//4),np.arange(ct.L//4)+ct.L//4,np.arange(ct.L//4)+(ct.L//4)*2,selfaverage=False)
        metric_datasets['O'][p_ctrl_idx,p_proj_idx,L_idx]=O.cpu().numpy().real
        metric_datasets['EE'][p_ctrl_idx,p_proj_idx,L_idx]=EE.cpu().numpy().real
        metric_datasets['TMI'][p_ctrl_idx,p_proj_idx,L_idx]=TMI.cpu().numpy().real
        # return O,SA, TMI
    else:
        SA=ct.von_Neumann_entropy_pure([ct.L])
        # return SA,

    

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
    parser.add_argument('--x0','-x0',type=float,default=None, help="Initial value in fraction")
    parser.add_argument('--complex128','-complex128',action='store_true', help="add --complex128 to have precision of complex128")
    parser.add_argument('--ancilla','-ancilla',action='store_true', help="add --ancilla to have ancilla qubit")
    parser.add_argument('--save_T','-save_T',action='store_true', help="add --save to save the time evolution of the wavefunction")

    args=parser.parse_args()

    xj = convert_to_fraction(args.xj)

    L_list=np.arange(args.L[0],args.L[1],args.L[2])

    p_ctrl_list=np.linspace(args.p_ctrl[0],args.p_ctrl[1],int(args.p_ctrl[2]))
    p_proj_list=np.linspace(args.p_proj[0],args.p_proj[1],int(args.p_proj[2]))
    st=time.time()
    inputs=[((L_idx,L),(p_ctrl_idx,p_ctrl),(p_proj_idx,p_proj),xj,args.complex128,args.seed,args.ancilla,args.es,args.save_T,args.x0) for L_idx,L in enumerate(L_list) for p_ctrl_idx,p_ctrl in enumerate(p_ctrl_list) for p_proj_idx,p_proj in enumerate(p_proj_list)]

    with h5py.File('CT_En{:d}_pctrl({:.2f},{:.2f},{:.0f})_pproj({:.2f},{:.2f},{:.0f})_L({:d},{:d},{:d})_xj({:s})_seed{:d}{:s}{:s}_wf{:s}.hdf5'.format(args.es,*args.p_ctrl,*args.p_proj,*args.L,args.xj.replace('/','-'),args.seed,'_128' if args.complex128 else '_64','_anc'*args.ancilla,'_T'*args.save_T),'w') as f:
        
        datasets={L:f.create_dataset(f'wf_{L}',((len(p_ctrl_list),len(p_proj_list),2*L**2+1 if args.save_T else 1)+(2,)*L+(args.es,1)),dtype=complex) for L in L_list}
        metric_datasets={metric:f.create_dataset(f'{metric}',((len(p_ctrl_list),len(p_proj_list),len(L_list),args.es,1)),dtype=float) for metric in ['O','EE','TMI']}
        print('SS'*10)
        for param in tqdm(inputs):
            result=run_tensor(param,datasets,metric_datasets)
            del result
            gc.collect()
            torch.cuda.empty_cache()

    print('Time elapsed: {:.4f}'.format(time.time()-st))
