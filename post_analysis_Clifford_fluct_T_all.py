# This script can run P(sigma^2=0), <sigma^2>, and <(sigma^2)^2> - <sigma^2>^2
# For each sigma, we have traj, state, shot variance
# So total 9 quantities are calculated and saved in a pickle file
import sys
# dir_path='../control_transition'
# sys.path.append(dir_path)
# import matplotlib.pyplot as plt

from tqdm import tqdm
# from plot_utils import *
import numpy as np
import rqc
import pickle

batch_config = {
    # L=16: max es*es_C = 864000/0.225 = 3.84M, use 500*500=250k (1 job per p_m)
    16: {'total_es': 500, 'total_es_C': 500, 'es_batch': 500, 'es_C_batch': 500},

    # L=32: max es*es_C = 864000/0.944 = 915k, use 500*500=250k (1 job per p_m)
    32: {'total_es': 500, 'total_es_C': 500, 'es_batch': 500, 'es_C_batch': 500},

    # L=64: max es*es_C = 864000/6.626 = 130k, use 500*100=50k (5 jobs per p_m)
    64: {'total_es': 500, 'total_es_C': 500, 'es_batch': 500, 'es_C_batch': 100},

    # L=128: max es*es_C = 864000/89.7 = 9632, use 500*10=5k (50 jobs per p_m)
    128: {'total_es': 500, 'total_es_C': 500, 'es_batch': 500, 'es_C_batch': 10},

    # L=256: max es*es_C = 864000/1499 = 576, use 125*1=125 (2000 jobs per p_m)
    256: {'total_es': 500, 'total_es_C': 500, 'es_batch': 125, 'es_C_batch': 1},
}

def run(L, alpha, p_m, ob):
    # ob="DW"
    # ob="O"
    ob1=ob+''
    ob2=ob+'2'

    cfg = batch_config[L]
    total_es = cfg['total_es']
    total_es_C = cfg['total_es_C']
    es_batch = cfg['es_batch']
    es_C_batch = cfg['es_C_batch']

    # Verify divisibility
    assert total_es % es_batch == 0, f"L={L}: es_batch={es_batch} must divide total_es={total_es}"
    assert total_es_C % es_C_batch == 0, f"L={L}: es_C_batch={es_C_batch} must divide total_es_C={total_es_C}"

    num_es_batches = total_es // es_batch
    num_es_C_batches = total_es_C // es_C_batch

    # Pre-compute all es_range and es_C_range tuples
    es_ranges = [(es_batch_idx * es_batch + 1, (es_batch_idx + 1) * es_batch + 1)
                for es_batch_idx in range(num_es_batches)]
    es_C_ranges = [(es_C_batch_idx * es_C_batch + 1, (es_C_batch_idx + 1) * es_C_batch + 1)
                for es_C_batch_idx in range(num_es_C_batches)]

    fixed_params = {
        'L': L,
        'alpha': alpha,
    }
    p_m_list = [p_m]
    L_list = [L]
    vary_params = {
        'p_m': p_m_list,
        'es_range': es_ranges,
        'es_C_range': es_C_ranges,
    }

    params_list = [(fixed_params, vary_params)]

    data_dict = {'fn': set()}
    for fixed_params, vary_params in params_list:
        rqc.generate_params(
            fixed_params=fixed_params,
            vary_params=vary_params,
            fn_template='Clifford_En({es_range[0]},{es_range[1]})_EnC({es_C_range[0]},{es_C_range[1]})_pm({p_m:.3f},{p_m:.3f},1)_alpha{alpha:.1f}_L{L}_T.pickle',
            fn_dir_template='Clifford',
            input_params_template='--L {L} --p_m {p_m:.3f} {p_m:.3f} 1 --alpha {alpha:.1f} --es {es_range[0]} {es_range[1]} --es_C {es_C_range[0]} {es_C_range[1]}',
            load_data=rqc.load_pickle,
            filename=None,
            load=True,
            data_dict=data_dict,
        )
    df=rqc.convert_pd(data_dict, names=['Metrics', 'L', 'p_m', 'es_m', 'es_C'])
    
    def process_each_traj(df,L,p_m,sC,p_proj=0, threshold=1e-8, ob1='OP', ob2='OP2'):
        # data_ob1.shape = (num_state, num_timepoints), ob1 means "first moment of ob"
        # data_ob2.shape = (num_state, num_timepoints), ob2 means "second moment of ob"
        # traj_var = E_traj[E_state[ob]^2] - (E_traj[E_state[ob]])^2
        # state_var = E_state[<ob^2>] - E_state[<ob>]^2
        # shot_var = E_traj[E_state[ob^2]] - (E_traj[E_state[ob]])^2
        data = df['observations'].xs(sC,level='es_C').xs(p_m,level='p_m').xs(L,level='L')
        data_ob1=np.stack(data.xs(ob1,level='Metrics'))
        data_ob2=np.stack(data.xs(ob2,level='Metrics'))
        sigma_mc=data_ob1.var(axis=0)
        traj_var = sigma_mc
        state_var = data_ob2-data_ob1**2
        shot_var = data_ob2.mean(axis=0) - data_ob1.mean(axis=0)**2
        traj_weight = (traj_var<threshold).astype(float)
        state_weight = (state_var<threshold).sum(axis=0)
        shot_weight = (shot_var<threshold).astype(float)
        num_state = state_var.shape[0]
        return num_state, traj_weight, state_weight, shot_weight, traj_var, state_var, shot_var
        
    traj_weight_list = {}
    state_weight_list = {}
    shot_weight_list = {}
    traj_mean_list = {}
    state_mean_list = {}
    shot_mean_list = {}
    traj_var_list = {}
    state_var_list = {}
    shot_var_list = {}
    for p in tqdm(p_m_list):
        for L in L_list:
            print(p,L)
            num_state=0
            num_traj=0
            traj_weight_sum=0
            state_weight_sum=0
            shot_weight_sum=0
            traj_mean_sum=0
            state_mean_sum=0
            shot_mean_sum=0
            traj_sq_sum=0
            state_sq_sum=0
            shot_sq_sum=0
            for sC in range(1,batch_config[L]['total_es_C']+1):
                # try:
                num_state_, traj_weight, state_weight, shot_weight, traj_var, state_var, shot_var = process_each_traj(df,L=L,p_m=p,sC=sC,p_proj=0)
                num_traj +=1
                traj_weight_sum +=traj_weight
                state_weight_sum +=state_weight
                shot_weight_sum +=shot_weight
                traj_mean_sum += traj_var
                state_mean_sum += state_var.sum(axis=0)
                shot_mean_sum += shot_var
                traj_sq_sum += (traj_var**2)
                state_sq_sum += (state_var**2).sum(axis=0)
                shot_sq_sum += (shot_var**2)
                num_state += num_state_
                # except:
                #     pass
            traj_weight_list[(p,L)]=traj_weight_sum/num_traj
            state_weight_list[(p,L)]=state_weight_sum/num_state
            shot_weight_list[(p,L)]=shot_weight_sum/num_traj
            traj_mean_list[(p,L)]=traj_mean_sum/num_traj
            state_mean_list[(p,L)]=state_mean_sum/num_state
            shot_mean_list[(p,L)]=shot_mean_sum/num_traj
            traj_var_list[(p,L)] = traj_sq_sum/num_traj - (traj_mean_sum/num_traj)**2
            state_var_list[(p,L)] = state_sq_sum/num_state - (state_mean_sum/num_state)**2
            shot_var_list[(p,L)] = shot_sq_sum/num_traj - (shot_mean_sum/num_traj)**2


    with open(f'traj_state_var_{p_m:.3f}_{ob}_L{L}_Clifford.pickle','wb') as f:
        pickle.dump({
            'traj_weight': traj_weight_list,
            'state_weight': state_weight_list,
            'shot_weight': shot_weight_list,
            'traj_mean': traj_mean_list,
            'state_mean': state_mean_list,
            'shot_mean': shot_mean_list,
            'traj_var': traj_var_list,
            'state_var': state_var_list,
            'shot_var': shot_var_list,
            },f)
    # return traj_weight_list, state_weight_list

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--L', type=int, required=True, help='System size parameter')
    parser.add_argument('--p_m', type=float, required=True, help='Measurement probability') 
    parser.add_argument('--alpha', type=float, required=True, help='Alpha parameter controlling the range') 

    parser.add_argument('--ob', type=str, required=True, help='Observable, DW or O')
    args = parser.parse_args()


    run(args.L, args.alpha, args.p_m, args.ob)
