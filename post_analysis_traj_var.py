from plot_utils import *
from tqdm import tqdm

sC_list=range(500)
sm_list=range(500)
# p_m_list=[0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15]
# p_m_list=[0.07,0.08,0.085,0.087,0.089,0.09,0.091,0.093,0.095,0.1,0.105,0.11,]
p_m_list=[0.09]
params_list=[
# ({'L':12,'per_es0':50},{'es0':range(0,500,50),'p_m':p_m_list}),
# ({'L':14,'per_es0':10},{'es0':range(0,500,10),'p_m':p_m_list}),
({'L':16,'per_es0':5},{'es0':range(0,500,5),'p_m':p_m_list}),
# ({'L':18,'per_es0':2},{'es0':range(0,500,2),'p_m':p_m_list}),

]
L=params_list[0][0]['L']
p_m_list=params_list[0][1]['p_m']
print(L,p_m_list)
data_APT_dict={'fn':set()}
for fixed_params,vary_params in params_list:
    data_APT_dict=generate_params(
        fixed_params=fixed_params,
        vary_params=vary_params,
        fn_template='APT_En({es0},{es0+per_es0})_EnC(0,500)_pm({p_m:.3f},{p_m:.3f},1)_pf(1.000,1.000,1)_L{L}_T.pickle',
        # fn_dir_template='APT_T',
        fn_dir_template='/home/jake/Data/APT_T',
        input_params_template='{p:.3f} {L} {seed} {ancilla}',
        load_data=load_pickle,
        filename=None,
        filelist=None,
        load=True,
        data_dict=data_APT_dict,
        # data_dict_file='APT_T.pickle', 
    )
data_APT=convert_pd(data_APT_dict,names=['Metrics','L','p_m','p_f','sC','sm'])

def trajvar(df,L,p_m,sC):
    data=df.xs(L,level='L').xs(p_m,level='p_m').xs(1,level='p_f').xs(sC,level='sC')

    # single=[data.xs(sm,level='sm').loc['DW1']['observations'] for sm in range((params_list[0][1]['sm']).shape[0])]
    # single=[data.xs(sm,level='sm').loc['O']['observations'] for sm in sm_list]
    single = np.vstack(data.xs('O',level='Metrics')['observations'])

    return single.var(axis=0)

sC_traj_var={}
for p in p_m_list:
    print(p)
    sC_traj_var[(p,L)]=[]
    for sC in tqdm(range(500)):
        try:
            sC_traj_var[(p,L)].append(trajvar(data_APT,L=L,p_m=p,sC=sC))
        except:
            pass
    sC_traj_var[(p,L)]=np.array(sC_traj_var[(p,L)])
    print(sC_traj_var[(p,L)].shape)

with open(f'traj_var_C_m_T_APT_L{L}.pickle','wb') as f:
    pickle.dump(sC_traj_var,f)
        