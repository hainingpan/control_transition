import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import pickle
from opt_einsum import contract
import dask.array as da
import torch

def save_coherence_matrix(f_0, L,i_list, order):
    """wf -> density matrix -> average->coherence
    |\sum_m p_m rho_m| : coh(ave(..))"""
    assert order in ['coh_ave','ave_coh'], 'order must be either coh_ave or ave_coh'
    coherence_matrix_list=np.zeros((len(i_list),L+1,L+1),dtype=np.float64)
    coherence_matrix_per_list=np.zeros((len(i_list),L+1,L+1),dtype=np.float64)
    fdw_list=np.zeros((len(i_list),L+1),dtype=np.float64)
    fdw_per_list=np.zeros((len(i_list),L+1),dtype=np.float64)
    for idx in tqdm(i_list):
        if order == 'coh_ave':
            rho_av=get_rho_av(f_0,L,idx)
        elif order == 'ave_coh':
            rho_av=get_rho_av(f_0,L,idx,abs=True)
        coherence_matrix, fdw_list[idx]=get_coherence_matrix(rho_av)
        coherence_matrix_list[idx]=(coherence_matrix)
        coherence_matrix_per_list[idx]=(get_coherence_matrix_per_basis(coherence_matrix,))
        fdw_per_list[idx]=get_coherence_matrix_per_basis(fdw_list[idx])

    output_fn=f'{order}_L{L}'
    with open(output_fn+'.pickle','wb') as f:
        pickle.dump({
            'coherence_matrix':coherence_matrix_list,'coherence_matrix_per':coherence_matrix_per_list,
            'fdw':fdw_list,
            'fdw_per':fdw_per_list},f)

def save_coherence_matrix_T(f_T, L,order, T_list=None, i_list=None,):
    """ \sum_m p_m  C(\rho_m)"""
    if T_list is None:
        T_list=range(0,1+2*L**2)
    if i_list is None:
        i_list=range(21)

    coherence_matrix_list=np.zeros((len(i_list),len(T_list),L+1,L+1),dtype=np.float64)
    coherence_matrix_per_list=np.zeros((len(i_list),len(T_list),L+1,L+1),dtype=np.float64)
    fdw_list=np.zeros((len(i_list),len(T_list),L+1),dtype=np.float64)
    fdw_per_list=np.zeros((len(i_list),len(T_list),L+1),dtype=np.float64)

    for idx,i in tqdm((enumerate(i_list)),total=len(i_list)):
        for T_idx,T in (enumerate(T_list)):
            if order == 'coh_ave':
                rho_av=get_rho_av_T(f_T,L=L,i=idx,T=T)
            elif order == 'ave_coh':
                rho_av=get_rho_av_T(f_T,L=L,i=idx,T=T,abs=True)
            coherence_matrix, fdw = get_coherence_matrix(rho_av)

            coherence_matrix_list[idx,T_idx]=(coherence_matrix)
            coherence_matrix_per_list[idx,T_idx]=(get_coherence_matrix_per_basis(coherence_matrix,))
            fdw_list[idx,T_idx]=fdw
            fdw_per_list[idx,T_idx]=get_coherence_matrix_per_basis(fdw)

    output_fn=f'{order}_L{L}_T'
    with open(output_fn+'.pickle','wb') as f:
        pickle.dump({
            'coherence_matrix':coherence_matrix_list,'coherence_matrix_per':coherence_matrix_per_list,
            'fdw':fdw_list,
            'fdw_per':fdw_per_list},f)

def save_coherence_matrix_T_seed(f_T, L,seed_range, T_list=None, i_list=None, hdf5=False,bootstrap=False,rng=None,save=True,internal_coherence=False):
    if T_list is None:
        T_list=range(0,1+2*L**2)
    if i_list is None:
        i_list=range(21)

    red_dm_list=np.zeros((len(i_list),len(T_list),L+1,L+1),dtype=np.float64)
    red_dm_per_list=np.zeros((len(i_list),len(T_list),L+1,L+1),dtype=np.float64)
    for i_idx,i in (enumerate(i_list)):
    # for i_idx,i in tqdm(enumerate(i_list)):
        red_dm=np.array([get_coherence_matrix(get_rho_av_T_seed(f_T,L=L,i=i,T=T,seed_range=seed_range,bootstrap=bootstrap,rng=rng),internal_coherence=internal_coherence) for T_idx,T in enumerate(T_list)])
        red_dm_list[i_idx]=(red_dm)
        red_dm_per_list[i_idx]=np.array([get_coherence_matrix_per_basis(rho,internal_coherence=internal_coherence) for rho in red_dm_list[i_idx]])
    if save:
        with open(f'rho_T_av_{L}_all.pickle','wb') as f:
                pickle.dump({'red_dm':red_dm_list,'red_dm_per':red_dm_per_list},f)
    else:
        return red_dm_list, red_dm_per_list

def save_coherence_matrix_T_seed_swap(f_T, L,seed_range, T_list=None, i_list=None, bootstrap=False,rng=None,save=True,internal_coherence=False):
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
            get_coherence_matrix(
            get_rho_av_T_seed_swap(
                f_T,L=L,i=i,T=T,seed_range=seed_range,bootstrap=bootstrap,rng=rng
                ),internal_coherence=internal_coherence
                ) for T_idx,T in enumerate(T_list)])
        red_dm_list[i_idx]=red_dm
        red_dm_per_list[i_idx]=np.array([get_coherence_matrix_per_basis(rho,internal_coherence=internal_coherence) for rho in red_dm_list[i_idx]])
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

def l1_coherence_2(rho,k1,k2,):
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
        return ((torch.abs(rho_).sum()-trace(rho_)),trace(rho_))
    else:
        return torch.abs(rho_).sum()

def trace(rho):
    L=len(rho.shape)
    if L>0:
        return contract(rho,list(range(L//2))*2)
    else:
        return rho

def get_rho_av(f_0,L,i,s=None, abs=False):
    """ dm = wf.conj() @ wf; if abs= False
        dm = abs(wf) @ abs(wf); if abs= True"""
    if s is None:
        wf=torch.from_numpy(f_0[L][f'wf_{L}'][i,0,0,...,:,0])
        ensemble_size=f_0[L][f'wf_{L}'].shape[-2]
        index_1=list(range(L))+[2*L]
        index_2=list(range(L,2*L))+[2*L]
        index_final=list(range(2*L))
        if abs:
            wf=wf.abs()
            rho_av=(contract(wf,index_1,wf,index_2,index_final)/ensemble_size)
        else:
            rho_av=(contract(wf,index_1,(wf).conj(),index_2,index_final).abs()/ensemble_size)
        return rho_av
    else:
        # need to update but may not be very useful now, could be deleted
        wf=torch.from_numpy(f_0[L][f'wf_{L}'][i,0,0,...,s,0])
        rho=torch.abs(contract(wf,list(range(L)),torch.conj(wf),list(range(L,2*L)),list(range(2*L))))
        return rho
        
def get_rho_av_T(f,L,i,T,s=None,abs=False):
    if s is None:
        wf=torch.from_numpy(f[L][f'wf_{L}'][i,0,T,...,:,0])
        ensemble_size=f[L][f'wf_{L}'].shape[-2]
        index_1=list(range(L))+[2*L]
        index_2=list(range(L,2*L))+[2*L]
        index_final=list(range(2*L))
        if abs:
            wf=wf.abs()
            rho_av=(contract(wf,index_1,wf,index_2,index_final)/ensemble_size)
        else:
            rho_av=(contract(wf,index_1,wf.conj(),index_2,index_final).abs()/ensemble_size)
        return rho_av
    else:
        # This is not used now, could be deleted
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

def get_coherence_matrix(rho,):
    """ obtain the coherence matrix from a density matrix
    the diagonal (off-diagonal) elements are intra-FDW (inter-FDW) coherence
    """
    L=len(rho.shape)//2
    coherence_matrix=np.zeros((L+1,L+1),dtype=np.float64)
    fdw=np.zeros((L+1,),dtype=np.float64)
    for i in range(L+1):
        for j in range(i,L+1):
            if i == j:
                coherence_matrix[i,j], fdw[i]= l1_coherence_2(rho,i,j,)
            else:
                coherence_matrix[i,j]=l1_coherence_2(rho,i,j,)
                coherence_matrix[j,i]=coherence_matrix[i,j]
    return coherence_matrix, fdw

def get_coherence_matrix_per_basis(rho, ):
    L=rho.shape[0]-1
    number_state=(np.r_[1,2**np.arange(L)])
    if len(rho.shape) == 2:
        number_state_map=np.outer(number_state,number_state)
        np.fill_diagonal(number_state_map,number_state**2-number_state)
        return rho/number_state_map
    elif len(rho.shape) == 1:
        # np.fill_diagonal(number_state_map,number_state)
        return rho/number_state

def plot_reduced_dm(rho,ax=None,label=r'$|{\rho}|$'):

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
                red_dm_list, red_dm_per_list=save_coherence_matrix_T_seed(f_T_s,L,seed_range=range(seed_max),T_list=T_list,i_list=i_list,rng=idx,save=False,bootstrap=bootstrap_size,internal_coherence=internal_coherence)
            else:
                red_dm_list, red_dm_per_list=save_coherence_matrix_T_seed_swap(f_T_s,L,seed_range=range(seed_max),T_list=T_list,i_list=i_list,bootstrap=bootstrap_size,rng=idx,save=False,internal_coherence=internal_coherence)
                
            if idx ==0 and bs_idx==0:
                red_dm_list_map=np.zeros((len(bootstrap_size_list),ensemble_size,)+red_dm_list.shape)
                red_dm_per_list_map=np.zeros((len(bootstrap_size_list),ensemble_size,)+red_dm_per_list.shape)
            red_dm_list_map[bs_idx,idx]=red_dm_list
            red_dm_per_list_map[bs_idx,idx]=red_dm_per_list
    return red_dm_list_map,red_dm_per_list_map

# Fitting


def generate_fitting_data(rho_av,idx,L_list,kind,idx_min=1,):
    # idx=0
    L_list_=[]
    k_list_=[]
    C_list_=[]
    for L in (L_list):
        k_list=np.arange(idx_min,L+1)
        if kind == 'inter':
            C=rho_av[L]['coherence_matrix_per'][idx][0,idx_min:]
        elif kind == 'intra':
            C=np.diag(rho_av[L]['coherence_matrix_per'][idx])[idx_min:]
        elif kind == 'fdw':
            C=rho_av[L]['fdw_per'][idx,idx_min:]

        L_list_.extend([L]*len(k_list))
        k_list_.extend(k_list)
        C_list_.extend(C)
        # ax.plot(k_list,,'.',label=f'L={L}',color=color)

    L_list_=np.array(L_list_)
    k_list_=np.array(k_list_)
    C_list_=np.array(C_list_)
    return k_list_,L_list_,C_list_

def remove_zero(k_list_,L_list_,C_list_,threshold=1e-10):
    mask= (C_list_>threshold)
    return k_list_[mask],L_list_[mask],C_list_[mask]

def fit_params(rho_av,L_list,kind='inter'):
    alpha_list=[]
    beta_list=[]
    A_list=[]
    alpha_error_list=[]
    beta_error_list=[]
    A_error_list=[]

    for idx in range(20):
        k_list_,L_list_,C_list_=generate_fitting_data(rho_av,idx,L_list=L_list,idx_min=2,kind=kind)
        # print(k_list_.shape,L_list_.shape,C_list_.shape)
        k_list_,L_list_,C_list_=remove_zero(k_list_,L_list_,C_list_)
        try:
            alpha,beta,A,alpha_err, beta_err, A_err=linear_regression_2d(k_list_,L_list_,np.log2(C_list_))
        except:
            alpha,beta,A,alpha_err, beta_err, A_err=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
        alpha_list.append(alpha)
        beta_list.append(beta)
        A_list.append(A)
        alpha_error_list.append(alpha_err)
        beta_error_list.append(beta_err)
        A_error_list.append(A_err)
    return alpha_list,beta_list,A_list,alpha_error_list,beta_error_list,A_error_list,

import statsmodels.api as sm
def linear_regression_2d(k, L, y):
    # Create the design matrix
    X = np.column_stack((L, k))
    X = sm.add_constant(X)  # Adds a constant term (A) to the model

    # Fit the model
    model = sm.OLS(y, X).fit()

    # Extract the parameters and standard errors
    alpha, beta, A = model.params[1], model.params[2], model.params[0]
    alpha_err, beta_err, A_err = model.bse[1], model.bse[2], model.bse[0]

    return alpha, beta, A, alpha_err, beta_err, A_err


def plot_spread(ave_coh_T, L,kind, idx,per='',ax=None,offset=1e-8,t=20,i=None,vmin=None,vmax=None,colorbar=True):
    if ax is None:
        fig,ax=plt.subplots()
    if kind == 'fdw':
        data = ave_coh_T[L]['fdw'+per][idx,:] + offset
        label = r'$\bar{f}(k)$'
        ax.set_xticks(np.arange(0,L+1,2))
    elif kind == 'inter':
        data = ave_coh_T[L]['coherence_matrix'+per][idx,:,i,:]
        label = rf'$\bar{{\mathcal{{C}}}}({i},k)$'
        ax.set_xlim(1,L)
        ax.set_xticks(np.arange(1,L+1,2))
    elif kind == 'intra':
        data = contract((ave_coh_T[L]['coherence_matrix'+per][idx,:]),[0,1,1],[0,1])+offset
        label= r'$\bar{\mathcal{C}}(k,k)$'
        ax.set_xlim(2,L)
        ax.set_xticks(np.arange(2,L+1,2))
    im=ax.imshow(np.log10(data),cmap='Blues',vmin=vmin,vmax=vmax)
    ax.set_ylim(0,t)
    ax.set_ylabel('t')
    ax.set_xlabel('k')
    if colorbar:
        plt.colorbar(im,label=label)

def plot_growth(ave_coh_T,L,idx,k,tmin=None,tmax=40,ax=None,log=False,per='',fit=False,part='all'):
    if ax is None:
        fig,ax=plt.subplots()
    color_list=['r','b','cyan']
    if tmin is None:
        tmin=L//2-1
    t_list=np.arange(tmin,tmax)
    ax.plot((t_list-tmin),(ave_coh_T[L]['fdw'+per][idx,tmin:tmax,k]),label='fdw'+per,color=color_list[0])

    ax.plot((t_list-tmin),(ave_coh_T[L]['coherence_matrix'+per][idx,tmin:tmax,k,k]),label=f'intra'+per,color=color_list[1])

    ax.plot((t_list-tmin),(ave_coh_T[L]['coherence_matrix'+per][idx,tmin:tmax,k,k+1]),label=f'inter'+per,color=color_list[2])
    
    ax.set_xlabel('t')
    if log:
        ax.set_yscale('log')
        ax.set_xscale('log')


    if fit:
        x=np.log((t_list-tmin)[1:])
        if part == 'all':
            slice_=slice(None)
        elif part == 'even':
            slice_=slice(0,None,2)
        elif part == 'odd':
            slice_=slice(1,None,2)

        fit_growth(
            x=np.log((t_list-tmin)[1:])[slice_], 
            y=np.log((ave_coh_T[L]['fdw'+per][idx,tmin+1:tmax,k]))[slice_],
            color=color_list[0],
            label='fdw'+per,
            ax=ax)

        fit_growth(
            x=np.log((t_list-tmin)[1:])[slice_], 
            y=np.log((ave_coh_T[L]['coherence_matrix'+per][idx,tmin+1:tmax,k,k]))[slice_],
            color=color_list[1],
            label='intra'+per,
            ax=ax)

        fit_growth(
            x=np.log((t_list-tmin)[1:])[slice_], 
            y=np.log((ave_coh_T[L]['coherence_matrix'+per][idx,tmin+1:tmax,k,k+1]))[slice_],
            color=color_list[2],
            label='inter'+per,
            ax=ax)
        
    ax.legend()

def fit_growth(x, y, color, label, ax):
    # Mask to filter out NaN and Inf values
    mask = (~np.isinf(x)) & (~np.isinf(y))
    x = x[mask]
    y = y[mask]
    
    # Adding a constant term to the independent variable for statsmodels
    x_with_const = sm.add_constant(x)
    
    # Fit the regression model
    model = sm.OLS(y, x_with_const)
    results = model.fit()
    
    # Extracting slope and its standard error
    slope = results.params[1]
    slope_std_err = results.bse[1]
    
    # Generate fitted line
    x_fine = np.linspace(x[0], x[-1], 101)
    x_fine_with_const = sm.add_constant(x_fine)
    y_fitted = results.predict(x_fine_with_const)
    
    # Plotting
    ax.plot(np.exp(x_fine), np.exp(y_fitted), '--', color=color, label=f'{label}:{slope:.2f} ± {slope_std_err:.2f}')
    
    # Returning the results including the slope and its standard error
    return slope, slope_std_err