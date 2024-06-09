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

def save_reduced_dm(f_0, L,idx_max=21,hdf5=False):
    red_dm_list=np.zeros((idx_max,L+1,L+1),dtype=np.float64)
    red_dm_per_list=np.zeros((idx_max,L+1,L+1),dtype=np.float64)
    for idx in tqdm(range(idx_max)):
        rho_av=get_rho_av(f_0,L,idx)
        red_dm=get_reduced_dm(rho_av)
        red_dm_list[idx]=(red_dm)
        red_dm_per_list[idx]=(get_reduced_dm_per_basis(red_dm))
    if hdf5:
        with h5py.File(f'rho_av_{L}.hdf5','w') as f:
            f.create_dataset('red_dm',data=red_dm_list)
            f.create_dataset('red_dm_per',data=red_dm_per_list)
    else:
        with open(f'rho_av_{L}.pickle','wb') as f:
            pickle.dump({'red_dm':red_dm_list,'red_dm_per':red_dm_per_list},f)


def save_reduced_dm_swap(f_0, L,s_max=2000,idx_max=21,hdf5=False):
    """ \sum_m p_m  C(\rho_m)"""

    red_dm_list=np.zeros((idx_max,L+1,L+1),dtype=np.float64)
    red_dm_per_list=np.zeros((idx_max,L+1,L+1),dtype=np.float64)
    for idx in tqdm(range(idx_max)):
        red_dm=[get_reduced_dm(get_rho_av(f_0,L,idx,s)) for s in range(s_max)]
        red_dm_list[idx]=np.mean(red_dm,axis=0)
        red_dm_per_list[idx]=get_reduced_dm_per_basis(red_dm_list[idx])
    if hdf5:
        with h5py.File(f'C_av_{L}_tt.hdf5','w') as f:
            f.create_dataset('red_dm',data=red_dm_list)
            f.create_dataset('red_dm_per',data=red_dm_per_list)
    else:
        with open(f'C_av_{L}_tt.pickle','wb') as f:
            pickle.dump({'red_dm':red_dm_list,'red_dm_per':red_dm_per_list},f)

def save_reduced_dm_T(f_0, L,s_max=2000,idx_max=21,hdf5=False):
    """ \sum_m p_m  C(\rho_m)"""

    red_dm_list=np.zeros((idx_max,L+1,L+1),dtype=np.float64)
    red_dm_per_list=np.zeros((idx_max,L+1,L+1),dtype=np.float64)
    for idx in tqdm(range(idx_max)):
        red_dm=[get_reduced_dm(get_rho_av(f_0,L,idx,s)) for s in range(s_max)]
        red_dm_list[idx]=np.mean(red_dm,axis=0)
        red_dm_per_list[idx]=get_reduced_dm_per_basis(red_dm_list[idx])
    if hdf5:
        with h5py.File(f'C_av_{L}.hdf5','w') as f:
            f.create_dataset('red_dm',data=red_dm_list)
            f.create_dataset('red_dm_per',data=red_dm_per_list)
    else:
        with open(f'C_av_{L}.pickle','wb') as f:
            pickle.dump({'red_dm':red_dm_list,'red_dm_per':red_dm_per_list},f)

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

def l1_coherence_2(rho,k1,k2):
    L=len(rho.shape)//2
    if k1 == 0:
        ket_idx=(0,)*L
    else:
        ket_idx=(0,)*(L-k1)+(1,)+(slice(None),)*(k1-1)
    if k2 == 0:
        bra_idx=(0,)*L
    else:
        bra_idx=(0,)*(L-k2)+(1,)+(slice(None),)*(k2-1)
    if k1 == k2:
        return trace(rho[ket_idx+bra_idx])
    else:
        return torch.abs(rho[ket_idx+bra_idx]).sum()

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
        rho=torch.abs(contract(wf,list(range(L)),np.conj(wf),list(range(L,2*L)),list(range(2*L))))
        return rho
        
def get_rho_av_T(f,L,i,T):
    wf=torch.from_numpy(f[L][f'wf_{L}'][i,0,T,...,:,0])
    rho_av=np.abs(contract(wf,np.r_[np.arange(0,L),2*L],np.conj(wf),np.r_[np.arange(L,2*L),2*L],np.arange(2*L))/f[L][f'wf_{L}'].shape[-2])
    return rho_av


def get_reduced_dm(rho):
    L=len(rho.shape)//2
    red_dm=np.zeros((L+1,L+1),dtype=np.float64)
    for i in range(L+1):
        for j in range(i,L+1):
            red_dm[i,j]=l1_coherence_2(rho,i,j)
            red_dm[j,i]=red_dm[i,j]
    return red_dm

def get_reduced_dm_per_basis(rho):
    L=rho.shape[0]-1
    number_state=(np.r_[1,2**np.arange(L)])
    number_state_map=np.outer(number_state,number_state)
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

