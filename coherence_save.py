from coherence import *
def run(L):
    f_0={}
    dirpath='.'
    f_0[L]=h5py.File(os.path.join(dirpath,f'CT_En2000_pctrl(0.00,1.00,21)_pproj(0.00,0.00,1)_L({L},{L+2},2)_xj(0)_seed0_64_wf.hdf5'))
    # save_reduced_dm(f_0,L=L,)
    save_reduced_dm(f_0,L=L,internal_coherence=True)
    # save_reduced_dm_swap(f_0,L=L)

# def run2(L):
#     red_dm_list_map,red_dm_per_list_map=resample(f_T_s,L=8,T_list=T_list,i_list=[0],ensemble_size=20,bootstrap_size_list=bootstrap_size_list,seed_max=5,internal_coherence=True)
#     with open('rho_ave')

if __name__  == "__main__":
    for L in [16]:
    # for L in [8,]:
        run(L)
    print('finished')
