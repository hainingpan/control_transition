from rqc_sv.APT import APT
import numpy as np
from joblib import Parallel, delayed
from time import time
import argparse
import pickle
from tqdm import tqdm
from fractions import Fraction
import os


def run(inputs):
    L, p_m,p_f,seed, seed_C  = inputs
    apt=APT(L=L,x0=Fraction(2**L-1,2**L),seed=seed,seed_C=seed_C,seed_vec=None,store_op=False)
    # for i in range(2*apt.L):
    tf = int(10*apt.L**1.6)
    OP_list=[]
    OP2_list=[]
    for i in range(tf):
        apt.random_circuit(p_m=p_m,p_f=p_f,even=True)
        apt.random_circuit(p_m=p_m,p_f=p_f,even=False)
        result = apt.order_parameter(moment=(1,2))
        OP_list.append(result['OP'])
        OP2_list.append(result['OP2'])
    return {"OP": OP_list, "OP2": OP2_list}



if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--L','-L',type=int,default=12,help='Parameters for L')
    parser.add_argument('--es','-es',default=[1,10],type=int,nargs=2,help='Ensemble size (default: 10) for measurement outcome.')
    parser.add_argument('--es_C','-es_C',default=[1,10],type=int,nargs=2,help='Ensemble size for circuit (default: 10).')
    parser.add_argument('--p_m','-p_m',type=float,nargs=3,default=[0,1,11],help='Parameters for p_m in the form [start, stop, num] to generate values with np.linspace (default: [0, 1, 11]).')
    parser.add_argument('--p_f','-p_f',type=float,nargs=3,default=[0,1,11],help='Parameters for p_f in the form [start, stop, num] to generate values with np.linspace (default: [0, 1, 11]). If the third parameter is -1, then use the same value of p_m')
    parser.add_argument('--n_jobs','-n',type=int,default=-1,help='Number of parallel jobs (-1 for all available cores, default: -1).')
    args=parser.parse_args()
    p_m_list=np.linspace(args.p_m[0],args.p_m[1],int(args.p_m[2]))
    es_list=np.arange(args.es[0],args.es[1])
    es_C_list=np.arange(args.es_C[0],args.es_C[1])
    if args.p_f[2]==-1:
        p_f_list=np.array([-1])
        inputs=[(args.L, p_m,p_m,idx,idx_C) for p_m in p_m_list for p_f in p_f_list for idx in es_list for idx_C in es_C_list]

    else:
        p_f_list=np.linspace(args.p_f[0],args.p_f[1],int(args.p_f[2]))
        inputs=[(args.L, p_m,p_f,idx,idx_C) for p_m in p_m_list for p_f in p_f_list for idx in es_list for idx_C in es_C_list]
    st=time()

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(run)(input_data) for input_data in tqdm(inputs, desc="Processing")
    )

    keys = results[0].keys()
    shape = (p_m_list.shape[0], np.abs(p_f_list.shape[0]), es_list.shape[0], es_C_list.shape[0], -1)
    data = {key: np.array([r[key] for r in results]).reshape(shape) for key in keys}
    data["args"] = args

    with open(os.environ.get('WORKDIR', '..') + '/control_transition/APT_OP_T/APT_En({:d},{:d})_EnC({:d},{:d})_pm({:.3f},{:.3f},{:.0f})_pf({:.3f},{:.3f},{:.0f})_L{:d}_OP_T.pickle'.format(*args.es,*args.es_C,*args.p_m,*args.p_f,args.L),'wb') as f:
        pickle.dump(data, f)

    print('Time elapsed: {:.4f}'.format(time()-st))
