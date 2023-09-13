from CT import *
import numpy as np
import pickle
import argparse
# from mpi4py.futures import MPIPoolExecutor

from fractions import Fraction

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
    parser.add_argument('--es','-es',default=10,type=int,help='Ensemble size (default: 10).')
    parser.add_argument('--p_ctrl','-p_ctrl',type=float,nargs=3,default=[0,1,11],help='Parameters for p_ctrl in the form [start, stop, num] to generate values with np.linspace (default: [0, 1, 11]).')
    parser.add_argument('--p_proj','-p_proj',type=float,nargs=3,default=[0,1,11],help='Parameters for p_proj in the form [start, stop, num] to generate values with np.linspace (default: [0, 1, 11]).')
    parser.add_argument('--L','-L',type=int,nargs=3,default=[10,16,2],help='Parameters for L in the form [start, stop, step] to generate values with np.arange (default: [10, 16, 2]).')
    parser.add_argument('--xj','-xj',type=str, help="List of fractions or 0 in the format num1/denom1,num2/denom2,... or 0. For example: 1/2,2/3")

    
    args=parser.parse_args()

    fractions = convert_to_fraction(args.xj)


    # L_list=np.array([10,12,14,16])
    L_list=np.arange(args.L[0],args.L[1],args.L[2])

    # p_ctrl_list=np.linspace(0,1,11)
    # p_proj_list=np.linspace(0,1,11)
    p_ctrl_list=np.linspace(args.p_ctrl[0],args.p_ctrl[1],int(args.p_ctrl[2]))
    p_proj_list=np.linspace(args.p_proj[0],args.p_proj[1],int(args.p_proj[2]))

    inputs=[(L,p_ctrl,p_proj,[Fraction(1,3),Fraction(2,3)],idx) for L in L_list for p_ctrl in p_ctrl_list for p_proj in p_proj_list for idx in range(args.es)]

    # with MPIPoolExecutor() as executor:
    #     results=(executor.map(run_quantum,inputs))

    results=list(map(run_quantum,inputs))

    rs=np.array(list(results)).reshape((L_list.shape[0],p_ctrl_list.shape[0],p_proj_list.shape[0],args.es,3))
    O_map,EE_map,TMI_map=rs[:,:,:,:,0],rs[:,:,:,:,1],rs[:,:,:,:,2]
    with open('CT_En{:d}_pctrl({:.2f},{:.2f},{:.0f})_pproj({:.2f},{:.2f},{:.0f})_L({:d},{:d},{:d})_xj({:s}).pickle'.format(args.es,*args.p_ctrl,*args.p_proj,*args.L,args.xj.replace('/','-')),'wb') as f:
        pickle.dump({"O":O_map,"EE":EE_map,"TMI":TMI_map,"args":args}, f)
