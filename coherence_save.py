from coherence import *
def run(L):
    f_0={}
    dirpath='.'
    f_0[L]=h5py.File(os.path.join(dirpath,f'CT_En2000_pctrl(0.00,1.00,21)_pproj(0.00,0.00,1)_L({L},{L+2},2)_xj(0)_seed0_64_wf.hdf5'))
    # save_reduced_dm(f_0,L=L,)
    # save_reduced_dm(f_0,L=L,internal_coherence=True)
    # save_reduced_dm_swap(f_0,L=L)
    save_coherence_matrix(f_0,L=L,i_list=range(21),order='ave_coh')

def run_T(L):
    f_0={}
    dirpath='.'
    f_0[L]=h5py.File(os.path.join(dirpath,f'CT_En2000_pctrl(0.00,1.00,21)_pproj(0.00,0.00,1)_L({L},{L+2},2)_xj(0)_seed0_64_wf_T_all.hdf5'))
    save_coherence_matrix_T(f_0,L=L,i_list=range(21),order='ave_coh')
    # 'CT_En2000_pctrl(0.00,1.00,21)_pproj(0.00,0.00,1)_L(8,10,2)_xj(0)_seed0_64_wf_T_all.hdf5'
def run2():
    # bootstrap_size_list=[100,]
    bootstrap_size_list=[500]
    T_list=range(129)
    dirpath='.'
    # L=8
    # f_T_s={}
    # for s in range(5):
    #     f_T_s[s]=h5py.File(os.path.join(dirpath,f'CT_En20000_pctrl(0.50,0.75,2)_pproj(0.00,0.00,1)_L({L},{L+2},2)_xj(0)_seed{s}_64_wf_T_all.hdf5'))

    f_T={}
    for L in [8,]:
        f_T[L]=h5py.File(os.path.join(dirpath,f'CT_En2000_pctrl(0.00,1.00,21)_pproj(0.00,0.00,1)_L({L},{L+2},2)_xj(0)_seed0_64_wf_T_all.hdf5'))


    red_dm_list_map,red_dm_per_list_map=resample(f_T,L=8,T_list=T_list,i_list=[0,5,10,.15,19,20],ensemble_size=20,bootstrap_size_list=bootstrap_size_list,seed_max=1,internal_coherence=True,swap=True)
    with open('rho_ave_8_all_swap_special_p.pickle','wb') as f:
        pickle.dump([red_dm_list_map,red_dm_per_list_map],f)

if __name__  == "__main__":
    # for L in [16]:
    for L in [8,10]:
        # run(L)
        run_T(L)
    # run2()
    print('finished')
