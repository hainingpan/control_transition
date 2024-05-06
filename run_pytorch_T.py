from rqc.CT_tensor import CT_tensor
import numpy as np
import pickle
import argparse
import time
from tqdm import tqdm
from fractions import Fraction
import torch
import gc

"""this code has main part same as run_pytorch.py. The only difference is that it tracks the change of TMI, EE, and O.
L can only be scalar , i.e., L_list[0] is used only 
start with a product state , say x0=0
"""

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
    L,p_ctrl,p_proj,xj,complex128,seed,ancilla,ensemble,add_x,no_feedback=inputs
    
    ct=CT_tensor(L=L,seed=seed,xj=xj,gpu=True,complex128=complex128,_eps=1e-5,ensemble=ensemble,ancilla=ancilla,add_x=add_x,feedback=(not no_feedback),x0=None if ancilla else 0)
    T_max=2*ct.L**2
    dtype=torch.float64 if complex128 else torch.float32
    if not ancilla:
        O_list=torch.zeros((T_max+1,ensemble), device='cuda',dtype=dtype)
        SA_list=torch.zeros((T_max+1,ensemble), device='cuda',dtype=dtype)
        TMI_list=torch.zeros((T_max+1,ensemble), device='cuda',dtype=dtype)
        O_list[0]=ct.order_parameter()[:,0]
        SA_list[0]=ct.half_system_entanglement_entropy()[:,0]
        TMI_list[0]=ct.tripartite_mutual_information(np.arange(ct.L//4),np.arange(ct.L//4)+ct.L//4,np.arange(ct.L//4)+(ct.L//4)*2,selfaverage=False)[:,0]
    else:
        SA_list=torch.zeros((T_max+1,ensemble), device='cuda',dtype=dtype)
        SA_list[0]=ct.von_Neumann_entropy_pure([ct.L])[:,0]

    for T_idx in tqdm(range(T_max),total=T_max):
        ct.random_control(p_ctrl=p_ctrl,p_proj=p_proj)
        torch.cuda.empty_cache()

        if not ancilla:
            O_list[T_idx+1]=ct.order_parameter()[:,0]
            SA_list[T_idx+1]=ct.half_system_entanglement_entropy()[:,0]
            TMI_list[T_idx+1]=ct.tripartite_mutual_information(np.arange(ct.L//4),np.arange(ct.L//4)+ct.L//4,np.arange(ct.L//4)+(ct.L//4)*2,selfaverage=False)[:,0]
        else:
            SA_list[T_idx+1]=ct.von_Neumann_entropy_pure([ct.L])[:,0]

    if not ancilla:
        return O_list,SA_list, TMI_list
    else:
        return SA_list,

    

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
    parser.add_argument('--complex128','-complex128',action='store_true', help="add --complex128 to have precision of complex128")
    parser.add_argument('--ancilla','-ancilla',action='store_true', help="add --ancilla to have ancilla qubit")
    parser.add_argument('--no_feedback','-no_feedback',action='store_true', help="add --no_feedback to remove feedback")
    parser.add_argument('--add_x','-add_x',type=int,default=0, help="add x")


    args=parser.parse_args()

    xj = convert_to_fraction(args.xj)
    L_list=np.arange(args.L[0],args.L[1],args.L[2])
    assert len(L_list) == 1, "Only L scalar is supported"

    p_ctrl_list=np.linspace(args.p_ctrl[0],args.p_ctrl[1],int(args.p_ctrl[2]))
    p_proj_list=np.linspace(args.p_proj[0],args.p_proj[1],int(args.p_proj[2]))
    st=time.time()
    inputs=[(L,p_ctrl,p_proj,xj,args.complex128,args.seed,args.ancilla,args.es,args.add_x,args.no_feedback) for L in L_list for p_ctrl in p_ctrl_list for p_proj in p_proj_list]

    # results=list(tqdm(map(run_tensor,inputs),total=len(inputs)))
    results=[]
    for param in (inputs):
        result=run_tensor(param)
        result_cpu=[r.cpu() for r in result]
        results.append(result_cpu)
        del result,result_cpu
        gc.collect()
        torch.cuda.empty_cache()


    results=torch.cat([torch.cat(tensors) for tensors in results])

    if not args.ancilla:
        # print(results.shape)
        rs=results.reshape((L_list.shape[0],p_ctrl_list.shape[0],p_proj_list.shape[0],3,2*L_list[0]**2+1,args.es))
        O_map,EE_map,TMI_map=rs[:,:,:,0,:,:],rs[:,:,:,1,:,:],rs[:,:,:,2,:,:]
        save_dict={"O":O_map,"EE":EE_map,"TMI":TMI_map,"args":args}
    else:
        rs=results.reshape((L_list.shape[0],p_ctrl_list.shape[0],p_proj_list.shape[0],1,2*L_list[0]**2+1,args.es))
        SA_map=rs[:,:,:,0,:,:]
        save_dict={"SA":SA_map,"args":args}


    with open('CT_En{:d}_pctrl({:.2f},{:.2f},{:.0f})_pproj({:.2f},{:.2f},{:.0f})_L({:d},{:d},{:d})_xj({:s})_seed{:d}{:s}{:s}{:s}{:s}_T.pickle'.format(args.es,*args.p_ctrl,*args.p_proj,*args.L,args.xj.replace('/','-'),args.seed,'_128' if args.complex128 else '_64','_anc'*args.ancilla,f'_x{args.add_x}'*(args.add_x!=0),'_nFB'*(args.no_feedback)),'wb') as f:
        pickle.dump(save_dict, f)

    print('Time elapsed: {:.4f}'.format(time.time()-st))