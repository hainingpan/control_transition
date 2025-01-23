from plot_utils import *
from tqdm import tqdm

sC_list=range(2000)
sm_list=range(500)
# p_m_list=[0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15]
# p_m_list=[0.07,0.08,0.085,0.087,0.089,0.09,0.091,0.093,0.095,0.1,0.105,0.11,]
# p_m_list=[0.09]

# p_m_list= [0.29,0.295,0.3,0.305,0.31]
p_m_list = [0.29,0.295,0.297,0.3,0.303,0.305,0.31]

def trajvar(df,sC):
    data=df.xs(sC,level='sC')
    return data.var(ddof=0)
    
sC_traj_var={}
for p_m in p_m_list:
    data_APT_dict={'fn':set()}
    
    params_list=[
    # ({'L':12,'per_esC0':50},{'esC0':range(0,2000,50),'p_m':[p_m]}),
    # ({'L':14,'per_esC0':25},{'esC0':range(0,2000,25),'p_m':[p_m]}),
    # ({'L':16,'per_esC0':10},{'esC0':range(0,2000,10),'p_m':[p_m]}),
    ({'L':18,'per_esC0':5},{'esC0':range(0,2000,5),'p_m':[p_m]}),
    # ({'L':20,'per_esC0':2},{'esC0':range(0,2000,2),'p_m':[p_m]}),
    ]
    
    L=params_list[0][0]['L']
    print(L,p_m)
    for fixed_params,vary_params in params_list:
        data_APT_dict=generate_params(
            fixed_params=fixed_params,
            vary_params=vary_params,
            # fn_template='APT_EnC({esC0},{esC0+per_esC0})_Enm(0,500)_pm({p_m:.3f},{p_m:.3f},1)_pf(1.000,1.000,1)_L{L}_Tf.pickle',
            fn_template='APT_EnC({esC0},{esC0+per_esC0})_Enm(0,500)_pm({p_m:.3f},{p_m:.3f},1)_pf(1.000,1.000,-1)_L{L}_Tf.pickle',
            # fn_dir_template='APT_Tf',
            # fn_dir_template='/home/jake/Data/APT_Tf',
            # fn_dir_template='APT_Tf_diag',
            fn_dir_template='/mnt/e/Control_Transition/APT/APT_Tf_diag',
            input_params_template='{p:.3f} {L} {seed} {ancilla}',
            load_data=load_pickle,
            filename=None,
            filelist=None,
            load=True,
            data_dict=data_APT_dict,
            # data_dict_file='APT_T.pickle', 
        )
    data_APT=convert_pd(data_APT_dict,names=['Metrics','L','p_m','p_f','sC','sm'])

    data_APT=data_APT.xs(L,level='L').xs(p_m,level='p_m').xs(1,level='p_f').xs('O',level='Metrics')['observations']
    sC_traj_var[(p_m,L)]=[]
    for sC in tqdm(sC_list):
        try:
            sC_traj_var[(p_m,L)].append(trajvar(data_APT,sC=sC))
        except:
            pass
    sC_traj_var[(p_m,L)]=np.array(sC_traj_var[(p_m,L)])
    print(sC_traj_var[(p_m,L)].shape)

# with open(f'traj_var_C_m_Tf_APT_L{L}.pickle','wb') as f:
with open(f'traj_var_C_m_Tf_APT_diag_L{L}.pickle','wb') as f:
    pickle.dump(sC_traj_var,f)
        