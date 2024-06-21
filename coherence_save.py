from coherence import *
def run(L):
    f_0={}
    dirpath='.'
    f_0[L]=h5py.File(os.path.join(dirpath,f'CT_En2000_pctrl(0.00,1.00,21)_pproj(0.00,0.00,1)_L({L},{L+2},2)_xj(0)_seed0_64_wf.hdf5'))
    # save_reduced_dm(f_0,L=L,)
    save_reduced_dm(f_0,L=L,internal_coherence=True)
    # save_reduced_dm_swap(f_0,L=L)

def run2():
    # bootstrap_size_list=[100,]
    bootstrap_size_list=[100,500,1000,2000]
    T_list=range(129)
    f_T_s={}
    dirpath='.'
    L=8
    for s in range(5):
        f_T_s[s]=h5py.File(os.path.join(dirpath,f'CT_En20000_pctrl(0.50,0.75,2)_pproj(0.00,0.00,1)_L({L},{L+2},2)_xj(0)_seed{s}_64_wf_T_all.hdf5'))


    red_dm_list_map,red_dm_per_list_map=resample(f_T_s,L=8,T_list=T_list,i_list=[0,1],ensemble_size=20,bootstrap_size_list=bootstrap_size_list,seed_max=5,internal_coherence=True,swap=True)
    with open('rho_ave_8_all_swap.pickle','wb') as f:
        pickle.dump([red_dm_list_map,red_dm_per_list_map],f)

if __name__  == "__main__":
    # for L in [16]:
    # for L in [8,]:
        # run(L)
    run2()
    print('finished')
