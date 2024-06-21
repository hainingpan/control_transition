import h5py
import numpy as np
import os
# import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import pickle
from opt_einsum import contract
import dask.array as da
import torch

def save_reduced_dm(f_0, L,idx_max=21,hdf5=False,internal_coherence=False):
    red_dm_list=np.zeros((idx_max,L+1,L+1),dtype=np.float64)
    red_dm_per_list=np.zeros((idx_max,L+1,L+1),dtype=np.float64)
    for idx in tqdm(range(idx_max)):
        rho_av=get_rho_av(f_0,L,idx)
        red_dm=get_reduced_dm(rho_av,internal_coherence=internal_coherence)
        red_dm_list[idx]=(red_dm)
        red_dm_per_list[idx]=(get_reduced_dm_per_basis(red_dm,internal_coherence=internal_coherence))
    output_fn=f'rho_av_{L}'
    if internal_coherence:
        output_fn+='_internal'
    if hdf5:
        with h5py.File(output_fn+'.hdf5','w') as f:
            f.create_dataset('red_dm',data=red_dm_list)
            f.create_dataset('red_dm_per',data=red_dm_per_list)
    else:
        with open(output_fn+'.pickle','wb') as f:
            pickle.dump({'red_dm':red_dm_list,'red_dm_per':red_dm_per_list},f)


def save_reduced_dm_swap(f_0, L,s_max=2000,idx_max=21,hdf5=False,internal_coherence=False):
    """ \sum_m p_m  C(\rho_m)"""

    red_dm_list=np.zeros((idx_max,L+1,L+1),dtype=np.float64)
    red_dm_per_list=np.zeros((idx_max,L+1,L+1),dtype=np.float64)
    for idx in tqdm(range(idx_max)):
        red_dm=[get_reduced_dm(get_rho_av(f_0,L,idx,s),internal_coherence=internal_coherence) for s in range(s_max)]
        red_dm_list[idx]=np.mean(red_dm,axis=0)
        red_dm_per_list[idx]=get_reduced_dm_per_basis(red_dm_list[idx],internal_coherence=internal_coherence)
    output_fn=f'C_av_{L}'
    if internal_coherence:
        output_fn+='_internal'
    if hdf5:
        with h5py.File(output_fn+'.hdf5','w') as f:
            f.create_dataset('red_dm',data=red_dm_list)
            f.create_dataset('red_dm_per',data=red_dm_per_list)
    else:
        with open(output_fn+'.pickle','wb') as f:
            pickle.dump({'red_dm':red_dm_list,'red_dm_per':red_dm_per_list},f)

def save_reduced_dm_T(f_T, L,T_max=None, idx_max=21,hdf5=False,internal_coherence=False):
    """ \sum_m p_m  C(\rho_m)"""
    if T_max is None:
        T_max=1+2*L**2
    red_dm_list=np.zeros((idx_max,T_max,L+1,L+1),dtype=np.float64)
    red_dm_per_list=np.zeros((idx_max,T_max,L+1,L+1),dtype=np.float64)
    for idx in tqdm(range(idx_max)):
        red_dm=np.array([get_reduced_dm(get_rho_av_T(f_T,L=L,i=idx,T=T),internal_coherence=internal_coherence) for T in range(T_max)])
        red_dm_list[idx]=(red_dm)
        red_dm_per_list[idx]=np.array([get_reduced_dm_per_basis(rho,internal_coherence=internal_coherence) for rho in red_dm_list[idx]])
    if hdf5:
        with h5py.File(f'rho_T_av_{L}.hdf5','w') as f:
            f.create_dataset('red_dm',data=red_dm_list)
            f.create_dataset('red_dm_per',data=red_dm_per_list)
    else:
        with open(f'rho_T_av_{L}.pickle','wb') as f:
            pickle.dump({'red_dm':red_dm_list,'red_dm_per':red_dm_per_list},f)

def save_reduced_dm_T_seed(f_T, L,seed_range, T_list=None, i_list=None, hdf5=False,bootstrap=False,rng=None,save=True,internal_coherence=False):
    if T_list is None:
        T_list=range(0,1+2*L**2)
    if i_list is None:
        i_list=range(21)

    red_dm_list=np.zeros((len(i_list),len(T_list),L+1,L+1),dtype=np.float64)
    red_dm_per_list=np.zeros((len(i_list),len(T_list),L+1,L+1),dtype=np.float64)
    for i_idx,i in (enumerate(i_list)):
    # for i_idx,i in tqdm(enumerate(i_list)):
        red_dm=np.array([get_reduced_dm(get_rho_av_T_seed(f_T,L=L,i=i,T=T,seed_range=seed_range,bootstrap=bootstrap,rng=rng),internal_coherence=internal_coherence) for T_idx,T in enumerate(T_list)])
        red_dm_list[i_idx]=(red_dm)
        red_dm_per_list[i_idx]=np.array([get_reduced_dm_per_basis(rho,internal_coherence=internal_coherence) for rho in red_dm_list[i_idx]])
    if save:
        with open(f'rho_T_av_{L}_all.pickle','wb') as f:
                pickle.dump({'red_dm':red_dm_list,'red_dm_per':red_dm_per_list},f)
    else:
        return red_dm_list, red_dm_per_list

def save_reduced_dm_T_seed_swap(f_T, L,seed_range, T_list=None, i_list=None, bootstrap=False,rng=None,save=True,internal_coherence=False):
    if T_list is None:
        T_list=range(0,1+2*L**2)
    if i_list is None:
        i_list=range(21)
    # if bootstrap is None:
    #     bootstrap=1
    
    red_dm_list=np.zeros((len(i_list),len(T_list),L+1,L+1),dtype=np.float64)
    red_dm_per_list=np.zeros((len(i_list),len(T_list),L+1,L+1),dtype=np.float64)
    for i_idx,i in (enumerate(i_list)):
        red_dm=np.array([
            get_reduced_dm(
            get_rho_av_T_seed_swap(
                f_T,L=L,i=i,T=T,seed_range=seed_range,bootstrap=bootstrap,rng=rng
                ),internal_coherence=internal_coherence
                ) for T_idx,T in enumerate(T_list)])
        red_dm_list[i_idx]=red_dm
        red_dm_per_list[i_idx]=np.array([get_reduced_dm_per_basis(rho,internal_coherence=internal_coherence) for rho in red_dm_list[i_idx]])
    if save:
        with open(f'rho_T_av_{L}_all.pickle','wb') as f:
                pickle.dump({'red_dm':red_dm_list,'red_dm_per':red_dm_per_list},f)
    else:
        return red_dm_list, red_dm_per_list



def l1_coherence(rho,k,normalization=False,average=False):
    L=len(rho.shape)//2
    if k == 0:
        ket_idx=(0,)*L
    else:
        ket_idx=(0,)*(L-k)+(1,)+(slice(None),)*(k-1)

    bra_idx=(0,)*(L-k-1)+(1,)+(slice(None),)*(k)
    tr=trace(rho[ket_idx+ket_idx])+trace(rho[bra_idx+bra_idx])

    coh=np.abs(rho[ket_idx+bra_idx]).sum()

    if normalization:
        coh/=tr.real
    if average:
        coh/=np.prod(rho[ket_idx+bra_idx].shape)
    return coh

def l1_coherence_2(rho,k1,k2,internal_coherence=False):
    L=len(rho.shape)//2
    if k1 == 0:
        ket_idx=(0,)*L
    else:
        ket_idx=(0,)*(L-k1)+(1,)+(slice(None),)*(k1-1)
    if k2 == 0:
        bra_idx=(0,)*L
    else:
        bra_idx=(0,)*(L-k2)+(1,)+(slice(None),)*(k2-1)
    rho_=rho[ket_idx+bra_idx]
    if k1 == k2:
        if internal_coherence:
            return (torch.abs(rho_).sum()-trace(rho_))
        else:
            return trace(rho_)
    else:
        return torch.abs(rho_).sum()

def trace(rho):
    L=len(rho.shape)
    if L>0:
        return contract(rho,list(range(L//2))*2)
    else:
        return rho

def get_rho_av(f_0,L,i,s=None):
    if s is None:
        wf=torch.from_numpy(f_0[L][f'wf_{L}'][i,0,0,...,:,0])
        
        rho_av=torch.abs(contract(wf,list(range(L))+[2*L],torch.conj(wf),list(range(L,2*L))+[2*L],list(range(2*L)))/f_0[L][f'wf_{L}'].shape[-2])
        return rho_av
    else:
        wf=torch.from_numpy(f_0[L][f'wf_{L}'][i,0,0,...,s,0])
        rho=torch.abs(contract(wf,list(range(L)),torch.conj(wf),list(range(L,2*L)),list(range(2*L))))
        return rho
        
def get_rho_av_T(f,L,i,T,s=None):
    if s is None:
        wf=torch.from_numpy(f[L][f'wf_{L}'][i,0,T,...,:,0])
        # rho_av=(contract(wf,np.r_[np.arange(0,L),2*L],np.conj(wf),np.r_[np.arange(L,2*L),2*L],np.arange(2*L))/f[L][f'wf_{L}'].shape[-2])
        rho_av=torch.abs(contract(wf,np.r_[np.arange(0,L),2*L],np.conj(wf),np.r_[np.arange(L,2*L),2*L],np.arange(2*L))/f[L][f'wf_{L}'].shape[-2])
        return rho_av
    else:
        wf=torch.from_numpy(f[L][f'wf_{L}'][i,0,T,...,s,0])
        rho=torch.abs(contract(wf,list(range(L)),torch.conj(wf),list(range(L,2*L)),list(range(2*L))))
        return rho

def get_rho_av_T_swap(f,L,i,T,s=None):
    if s is None:
        wf=torch.from_numpy(f[L][f'wf_{L}'][i,0,T,...,:,0])
        # rho_av=(contract(wf,np.r_[np.arange(0,L),2*L],np.conj(wf),np.r_[np.arange(L,2*L),2*L],np.arange(2*L))/f[L][f'wf_{L}'].shape[-2])
        rho_av=torch.abs(contract(wf,np.r_[np.arange(0,L),2*L],np.conj(wf),np.r_[np.arange(L,2*L),2*L],np.arange(2*L))/f[L][f'wf_{L}'].shape[-2])
        return rho_av
    else:
        wf=torch.from_numpy(f[L][f'wf_{L}'][i,0,T,...,s,0])
        rho=torch.abs(contract(wf,list(range(L)),torch.conj(wf),list(range(L,2*L)),list(range(2*L))))
        return rho

def get_rho_av_T_seed(f,L,i,T,seed_range, bootstrap=False,rng=None):
    for seed in seed_range:
        if seed == seed_range[0]:
            wf=torch.from_numpy(f[seed][f'wf_{L}'][i,0,T,...,:,0])
        else:
            wf=torch.cat((wf,torch.from_numpy(f[seed][f'wf_{L}'][i,0,T,...,:,0])),dim=-1)
    if bootstrap is not False:
        wf=resample_last_axis(wf,bootstrap,rng)
    rho_av=torch.abs(contract(wf,np.r_[np.arange(0,L),2*L],np.conj(wf),np.r_[np.arange(L,2*L),2*L],np.arange(2*L))/wf.shape[-1])
    return rho_av

def get_rho_av_T_seed_swap(f,L,i,T,seed_range, bootstrap=False,rng=None):
    for seed in seed_range:
        if seed == seed_range[0]:
            wf=torch.from_numpy(f[seed][f'wf_{L}'][i,0,T,...,:,0])
        else:
            wf=torch.cat((wf,torch.from_numpy(f[seed][f'wf_{L}'][i,0,T,...,:,0])),dim=-1)
    if bootstrap is not False:
        wf=resample_last_axis(wf,bootstrap,rng)
    wf=torch.abs(wf)
    rho_av=(contract(wf,np.r_[np.arange(0,L),2*L],np.conj(wf),np.r_[np.arange(L,2*L),2*L],np.arange(2*L))/wf.shape[-1])
    return rho_av


        
def resample_last_axis(tensor, num_samples, rng=None):
    """
    Resamples the last axis of a tensor with replacement.

    Parameters:
    tensor (torch.Tensor): The input tensor.
    num_samples (int): The number of samples to draw along the last axis.

    Returns:
    torch.Tensor: The resampled tensor.
    """
    # Generate random indices with replacement
    if rng is not None:
        torch.manual_seed(rng)

    indices = torch.randint(0, tensor.size(-1), (num_samples,))
    # Resample the tensor along the last axis
    resampled_tensor = tensor[..., indices]
    return resampled_tensor

def get_reduced_dm(rho,internal_coherence=False):
    L=len(rho.shape)//2
    red_dm=np.zeros((L+1,L+1),dtype=np.float64)
    for i in range(L+1):
        for j in range(i,L+1):
            red_dm[i,j]=l1_coherence_2(rho,i,j,internal_coherence=internal_coherence)
            red_dm[j,i]=red_dm[i,j]
    return red_dm

def get_reduced_dm_per_basis(rho, internal_coherence=False):
    L=rho.shape[0]-1
    number_state=(np.r_[1,2**np.arange(L)])
    number_state_map=np.outer(number_state,number_state)
    if internal_coherence:
        np.fill_diagonal(number_state_map,number_state**2-number_state)
    else:
        np.fill_diagonal(number_state_map,number_state)
    
    return rho/number_state_map


def plot_reduced_dm(rho,ax=None,label=r'$|{\rho}|$'):
    import matplotlib.pyplot as plt

    if ax is None:
        fig,ax=plt.subplots()
    im=ax.imshow(np.log10(rho+1e-6),cmap='Blues')
    axins=ax.inset_axes([1.05,0,0.05,1],transform=ax.transAxes)
    im=plt.colorbar(im,cax=axins,label=label)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('FDW')
    ax.set_ylabel('FDW')
    ax.set_xticks(np.arange(rho.shape[0]))
    ax.set_yticks(np.arange(rho.shape[0]))

def plot_coherence(rho,diag=False,ax=None,idx=(0,12)):
    if ax is None:
        fig,ax=plt.subplots()
    L=len(rho.shape)//2
    rho_abs=np.abs(rho).reshape((2**L,2**L))
    if diag:
        im=ax.imshow(rho_abs[idx[0]:idx[1],idx[0]:idx[1]])
    else:
        im=ax.imshow(rho_abs[idx[0]:idx[1],idx[0]:idx[1]]- np.diag(np.diag(rho_abs[idx[0]:idx[1],idx[0]:idx[1]])))
    axins=ax.inset_axes([1.05,0,0.05,1],transform=ax.transAxes)
    im=plt.colorbar(im,cax=axins,label=r'$|{\rho}|$')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

def resample(f_T_s,L,T_list,i_list,ensemble_size,bootstrap_size_list,seed_max=8,internal_coherence=False,swap=False):
    for bs_idx,bootstrap_size in tqdm(enumerate(bootstrap_size_list),total=len(bootstrap_size_list)):
        for idx in (range(ensemble_size)):
            if not swap:
                red_dm_list, red_dm_per_list=save_reduced_dm_T_seed(f_T_s,L,seed_range=range(seed_max),T_list=T_list,i_list=i_list,rng=idx,save=False,bootstrap=bootstrap_size,internal_coherence=internal_coherence)
            else:
                red_dm_list, red_dm_per_list=save_reduced_dm_T_seed_swap(f_T_s,L,seed_range=range(seed_max),T_list=T_list,i_list=i_list,bootstrap=bootstrap_size,rng=idx,save=False,internal_coherence=internal_coherence)
                
            if idx ==0 and bs_idx==0:
                red_dm_list_map=np.zeros((len(bootstrap_size_list),ensemble_size,)+red_dm_list.shape)
                red_dm_per_list_map=np.zeros((len(bootstrap_size_list),ensemble_size,)+red_dm_per_list.shape)
            red_dm_list_map[bs_idx,idx]=red_dm_list
            red_dm_per_list_map[bs_idx,idx]=red_dm_per_list
    return red_dm_list_map,red_dm_per_list_map