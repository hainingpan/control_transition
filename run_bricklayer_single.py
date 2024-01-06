from rqc.bricklayer import bricklayer
import numpy as np
import argparse
import pickle

def run_bl(inputs):
    L,p,add_x,idx=inputs
    bl=bricklayer(L=L,seed=idx,add_x=add_x,store_vec=False)
    for _ in range(bl.L):
        bl.random_projection(p)
    EE=bl.half_system_entanglement_entropy()
    TMI=bl.tripartite_mutual_information(np.arange(L//4),np.arange(L//4)+L//4,np.arange(L//4)+(L//4)*2,selfaverage=False)
    return EE, TMI


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--nu','-nu',default=0,type=int,help='numertor')
    parser.add_argument('--de','-de',default=1,type=int,help='denominator')
    parser.add_argument('--p','-p',default=0,type=float,help='measurement rate')
    parser.add_argument('--L','-L',default=8,type=int,help='system size')
    parser.add_argument('--seed','-seed',default=0,type=int,help='random seed')

    args=parser.parse_args()
    results=run_bl((args.L,args.p,int(2**args.L*args.nu/args.de),args.seed))
    
    with open(f'bricklayer_adder_({args.nu}-{args.de})_L{args.L}_p{args.p:.2f}_s{args.seed}.pickle','wb') as f:
        pickle.dump({'EE':results[0],'TMI':results[1],'args':args},f)