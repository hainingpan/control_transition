import numpy as np
import os
import sys
import importlib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import rqc

def sem(arr, axis=None, ddof=1):
    """Standard error of the mean, ignoring NaN values."""
    arr = np.asarray(arr)
    # Count non-NaN values along the axis
    n = np.sum(~np.isnan(arr), axis=axis)
    return np.nanstd(arr, axis=axis, ddof=ddof) / np.sqrt(n)

def stack_with_nan_padding(arrays):
    """Stack arrays of potentially different lengths, padding shorter ones with NaN.

    Args:
        arrays: Iterable of 1D arrays (can have different lengths)

    Returns:
        2D numpy array with shape (n_arrays, max_length), padded with NaN
    """
    arrays = list(arrays)
    if len(arrays) == 0:
        return np.array([])
    max_len = max(len(arr) for arr in arrays)
    result = np.full((len(arrays), max_len), np.nan)
    for i, arr in enumerate(arrays):
        result[i, :len(arr)] = arr
    return result

def _load_pickle_with_swap(fn):
    data = rqc.load_pickle(fn)
    coherence = data.get('coherence')
    # if coherence is not None and getattr(coherence, 'ndim', 0) == 5:
        # stored as (p_m, p_f, es_m, es_C, T); swap axes to match parse_APT_T expectation
        # data['coherence'] = np.swapaxes(coherence, 2, 3)
        # pass
    return data


def _load_zip_pickle_with_swap(fn, z):
    data = rqc.load_zip_pickle(fn, z)
    coherence = data.get('coherence')
    # if coherence is not None and getattr(coherence, 'ndim', 0) == 5:
    #     data['coherence'] = np.swapaxes(coherence, 2, 3)
    return data

# This is the old version before changing to es_C batching, can be dumped later
# def load_apt_coherence(use_zip=True, L_list=(12, 14, 16, 18, 20, 22, 24), p_f=1, p_f_int=1,zipfilename='APT_coherence_T_pf1.zip', 
#                        p_m_list = np.hstack([np.arange(0, 0.08, 0.01), np.arange(0.08, 0.101, 0.005),np.arange(0.11, 0.2, 0.01)]),
#     BATCH_CONFIG = {
#         12: {'es_batch': 2000, 'num_batches': 1},
#         14: {'es_batch': 2000, 'num_batches': 1},
#         16: {'es_batch': 2000, 'num_batches': 1},
#         18: {'es_batch': 2000, 'num_batches': 1},
#         20: {'es_batch': 2000, 'num_batches': 1},
#         22: {'es_batch': 1000, 'num_batches': 2},
#         24: {'es_batch': 100, 'num_batches': 20},
#     }):
#     """Load APT coherence pickles into a parsed dictionary using rqc.generate_params."""
#     load_data = _load_zip_pickle_with_swap if use_zip else _load_pickle_with_swap
#     data_dict = {'fn': set()}
#     FN_TEMPLATE = (
#     'APT_En(1,2)_EnC({es_C_range[0]},{es_C_range[1]})_pm({p_m:.3f},{p_m:.3f},1)_pf({p_f:.3f},{p_f:.3f},{p_f_int:d})_L{L}_coherence_T.pickle'
# )   
#     ZIP_PATH = os.path.expandvars(f'$WORKDIR/control_transition/{zipfilename}')



#     for L in L_list:
#         cfg = BATCH_CONFIG[L]
#         es_ranges = []
#         for batch_idx in range(cfg['num_batches']):
#             es_start = 1 + batch_idx * cfg['es_batch']
#             es_end = min(1 + (batch_idx + 1) * cfg['es_batch'], 2001)
#             es_ranges.append((es_start, es_end))

#         data_dict = rqc.generate_params(
#             # fixed_params={'es_start': 1, 'es_end': 2, 'p_f': -1, 'L': L},
#             fixed_params={'es_start': 1, 'es_end': 2, 'p_f': p_f, 'L': L, 'p_f_int': p_f_int},
#             vary_params={'p_m': p_m_list, 'es_C_range': es_ranges},
#             fn_template=FN_TEMPLATE,
#             fn_dir_template='.',
#             input_params_template='',
#             load_data=load_data,
#             filename=None,
#             filelist=None,
#             load=True,
#             data_dict=data_dict,
#             zip_fn=ZIP_PATH if use_zip else None,
#         )

#     return data_dict


def apt_coherence_to_df(data_dict):
    """Convert the loaded coherence dictionary to a pandas DataFrame."""
    return rqc.convert_pd(data_dict, names=['Metrics', 'L', 'p_m', 'p_f', 'es_m', 'es_C'])


### Plot utilities

def plot_apt_coherence_T_vs_steps(data_df, fix_={'L': 12} , ax =None, z=0, delta =0):
    if ax is None:
        fig, ax =plt.subplots(figsize=(5, 5))
    if 'L' in fix_:
        L = fix_['L']
        iter_var = 'p_m'
        fix_var = 'L'
        fix_value = L
        iter_list = data_df.index.get_level_values(iter_var).unique()
    elif 'p_m' in fix_:
        p_m = fix_['p_m']
        iter_var = 'L'
        fix_var = 'p_m'
        fix_value = p_m
        iter_list = data_df.index.get_level_values(iter_var).unique()
    color_list = plt.cm.Blues(np.linspace(.4, 1, len(iter_list)))
    idx_=1
    fix_var = 'L' if iter_var != 'L' else 'p_m'
    for  idx, iter_,  in enumerate(iter_list):

        data_ = np.nanmean(stack_with_nan_padding(data_df.xs(iter_, level=iter_var).xs(fix_value,level=fix_var)['observations']), axis=0)
        x = np.arange(len(data_))
        y = data_
        ax.plot(x[idx_:]/iter_**z, y[idx_:] * iter_**(delta), label=f'{iter_var}={iter_}', color=color_list[idx])
    if fix_var == 'L':
        ax.axhline(np.pi/4 * 2**(L), color='k', linestyle='--', )
    elif fix_var == 'p_m':
        for L in iter_list:
            ax.axhline(np.pi/4 * 2**(L), color='k', linestyle='--', )
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim(None, x.max())
    ax.grid(True, which='both', alpha=0.3)
    ax.minorticks_on()
    ax.tick_params(top=True, labeltop=True)
    ax.set_xlabel('Time step (one layer)')
    ax.set_ylabel('L1 coherence')


def plot_apt_coherence_T_vs_L(data_df, p_m, ax=None,idx_min=0, idx_max=2, min_func =lambda L: int(L**1.6),max_func =lambda L: -1, L_list = np.arange(12,25,2)):
    if ax is None:
        fig, ax =plt.subplots(figsize=(4, 4))
    x = L_list
    y = [np.nanmean(stack_with_nan_padding(data_df.xs(p_m, level='p_m').xs(L,level='L')['observations'])[:,min_func(L):max_func(L)]) for L in L_list]
    # y = [np.nanmean(stack_with_nan_padding(data_df.xs(p_m, level='p_m').xs(L,level='L')['observations']), axis=0)[-1:].mean() for L in L_list]
    yerr = [sem(stack_with_nan_padding(data_df.xs(p_m, level='p_m').xs(L,level='L')['observations'])[:,min_func(L):max_func(L)]) for L in L_list]
    ax.errorbar(x, y, yerr=yerr, fmt='.-', capsize=5)
    ax.set_yscale('log')
    ax.set_title(f'$p_m$={p_m:.3f}')

    # Here I want to use the first two points of "(x,y)", then fit linear in the linear-log scale, and plot it, using numpy,

    # Take the first two points
    x_fit = x[idx_min:idx_max]
    y_fit = y[idx_min:idx_max]

    # Fit a line in linear-log scale
    log_y = np.log10(y_fit)
    slope, intercept = np.polyfit(x_fit, log_y, 1)

        # Create a function for the fitted line
    def fitted_line(L):
        return 10**(intercept + slope * L)

    # Plot the original data and the fitted line
    ax.plot(x, y, '.-')
    ax.plot(x, fitted_line(np.array(x)), 'r--')
    ax.set_xlabel('L')
    ax.set_ylabel('L1 coherence')


def plot_apt_coherence_T_vs_steps_fixedL(data_df, L, ax =None, idx_=1):
    if ax is None:
        fig, ax =plt.subplots(figsize=(4,4))
    p_m_list = data_df.index.get_level_values('p_m').unique()
    color_list = plt.cm.Blues(np.linspace(.4, 1, len(p_m_list)))
    
    for  idx, p_m,  in enumerate(p_m_list):

        data_ = np.nanmean(stack_with_nan_padding(data_df.xs(p_m, level='p_m').xs(L,level='L')['observations']), axis=0)
        x = np.arange(len(data_))
        y = data_
        ax.plot(x[idx_:], y[idx_:], label=f'p_m={p_m:.3f}', color=color_list[idx])
    ax.axhline(np.pi/4 * 2**(L), color='k', linestyle='--', )
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim(None, x.max())
    ax.grid(True, which='both', alpha=0.3)
    ax.minorticks_on()
    ax.tick_params(top=True, labeltop=True)
    ax.set_xlabel('Time step (one layer)')
    ax.set_ylabel('L1 coherence')

def plot_apt_coherence_T_vs_steps_fixedp_m(data_df, p_m, ax =None, idx_=1, prefactor=None, z=1.6, average_log=False, theory_line=False):
    if ax is None:
        fig, ax =plt.subplots(figsize=(4,4))
    L_list = sorted(data_df.index.get_level_values('L').unique())
    color_list = plt.cm.Blues(np.linspace(.4, 1, len(L_list)))

    for  idx, L,  in enumerate(L_list):

        if average_log:
            data_ = np.exp(np.nanmean(np.log(stack_with_nan_padding(data_df.xs(p_m, level='p_m').xs(L,level='L')['observations'])), axis=0))
        else:
            data_ = np.nanmean(stack_with_nan_padding(data_df.xs(p_m, level='p_m').xs(L,level='L')['observations']), axis=0)
            print(len(data_df.xs(p_m, level='p_m').xs(L,level='L')['observations']))
        x = np.arange(len(data_))
        y = data_
        ax.plot(x[idx_:], y[idx_:], label=f'L={L}', color=color_list[idx])
        if theory_line:
            ax.axhline(np.pi/4 * 2**(L), color=color_list[idx], linestyle='--', alpha=0.5)

    # Add guided lines showing t = prefactor * L**z
    if prefactor is not None:
        prefactor_list = [prefactor] if not isinstance(prefactor, (list, tuple)) else prefactor
        guide_color_list = plt.cm.Reds(np.linspace(.4, 1, len(prefactor_list)))
        for pf_idx, pf in enumerate(prefactor_list):
            guide_t = []
            guide_y = []
            for L in L_list:
                if average_log:
                    data_ = np.exp(np.nanmean(np.log(stack_with_nan_padding(data_df.xs(p_m, level='p_m').xs(L, level='L')['observations'])), axis=0))
                else:
                    data_ = np.nanmean(stack_with_nan_padding(data_df.xs(p_m, level='p_m').xs(L, level='L')['observations']), axis=0)
                t_idx = int(pf * L**z)
                if t_idx < len(data_):
                    guide_t.append(t_idx)
                    guide_y.append(data_[t_idx])
            ax.plot(guide_t, guide_y, color=guide_color_list[pf_idx], linestyle='--', label=f'${pf}L^{{{z}}}$')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim(None, x.max())
    ax.grid(True, which='both', alpha=0.3)
    ax.minorticks_on()
    ax.tick_params(top=True, labeltop=True)
    ax.set_xlabel('Time step (one layer)')
    ax.set_ylabel('L1 coherence')

def plot_apt_coherence_T_vs_L(data_df, p_m, ax=None,idx_min=0, idx_max=2, min_func =lambda L: int(L**1.6), max_func=None, average_log=False, color='b', label=None):
    if ax is None:
        fig, ax =plt.subplots(figsize=(4, 4))
    L_list = sorted(data_df.index.get_level_values('L').unique())
    x = L_list
    label = label if label is not None else f'p_m={p_m}'
    y = []
    yerr = []
    for L in L_list:
        obs = stack_with_nan_padding(data_df.xs(p_m, level='p_m').xs(L, level='L')['observations'])
        max_idx = max_func(L) if max_func is not None else None
        obs_slice = obs[:, min_func(L):max_idx]
        if average_log:
            sample_means = np.exp(np.nanmean(np.log(obs_slice), axis=1))
        else:
            sample_means = np.nanmean(obs_slice, axis=1)
        y.append(np.nanmean(sample_means))
        yerr.append(sem(sample_means))
    ax.errorbar(x, y, yerr=yerr, fmt='.-', capsize=5, color=color, label=label)
    ax.set_yscale('log')

    # Here I want to use the first two points of "(x,y)", then fit linear in the linear-log scale, and plot it, using numpy,

    # Take the first two points
    x_fit = x[idx_min:idx_max]
    y_fit = y[idx_min:idx_max]

    # Fit a line in linear-log scale
    log_y = np.log10(y_fit)
    slope, intercept = np.polyfit(x_fit, log_y, 1)

        # Create a function for the fitted line
    def fitted_line(L):
        return 10**(intercept + slope * L)

    # Plot the original data and the fitted line
    ax.plot(x, y, '.-')
    ax.plot(x, fitted_line(np.array(x)), 'r--')
    ax.set_xlabel('L')
    ax.set_ylabel('L1 coherence')

def load_apt_coherence(use_zip=True, L_list=(12, 14, 16, 18, 20, 22, 24), p_f=1, p_f_int=1,zipfilename='APT_coherence_T_pf1.zip', 
                       p_m_list = np.hstack([np.arange(0, 0.08, 0.01), np.arange(0.08, 0.101, 0.005),np.arange(0.11, 0.2, 0.01)]),
BATCH_CONFIG = {
    12: {'es_C_batch': 2000, 'num_batches': 2},
    14: {'es_C_batch': 2000, 'num_batches': 2},
    16: {'es_C_batch': 2000, 'num_batches': 2},
    18: {'es_C_batch': 2000, 'num_batches': 2},
    20: {'es_C_batch': 1000, 'num_batches': 4},
    22: {'es_C_batch': 24*10, 'num_batches': 4000//(24*10)+1},
    24: {'es_C_batch': 24*2, 'num_batches': 4000//(24*2)+1}
}):
    """Load APT coherence pickles into a parsed dictionary using rqc.generate_params."""
    load_data = _load_zip_pickle_with_swap if use_zip else _load_pickle_with_swap
    data_dict = {'fn': set()}
    FN_TEMPLATE = (
    'APT_En(1,2)_EnC({es_C_range[0]},{es_C_range[1]})_pm({p_m:.3f},{p_m:.3f},1)_pf({p_f:.3f},{p_f:.3f},{p_f_int:d})_L{L}_coherence_T.pickle'
)   
    ZIP_PATH = os.path.expandvars(f'$WORKDIR/control_transition/{zipfilename}')



    for L in L_list:
        cfg = BATCH_CONFIG[L]
        es_ranges = []
        for batch_idx in range(cfg['num_batches']):
            es_start = 1 + batch_idx * cfg['es_C_batch']
            # es_end = min(1 + (batch_idx + 1) * cfg['es_C_batch'], 2001)
            es_end = 1 + (batch_idx + 1) * cfg['es_C_batch']
            es_ranges.append((es_start, es_end))

        data_dict = rqc.generate_params(
            # fixed_params={'es_start': 1, 'es_end': 2, 'p_f': -1, 'L': L},
            fixed_params={'es_start': 1, 'es_end': 2, 'p_f': p_f, 'L': L, 'p_f_int': p_f_int},
            vary_params={'p_m': p_m_list, 'es_C_range': es_ranges},
            fn_template=FN_TEMPLATE,
            fn_dir_template='.',
            input_params_template='',
            load_data=load_data,
            filename=None,
            filelist=None,
            load=True,
            data_dict=data_dict,
            zip_fn=ZIP_PATH if use_zip else None,
        )

    return data_dict


# def apt_coherence_to_df(data_dict):
#     """Convert the loaded coherence dictionary to a pandas DataFrame."""
#     return rqc.convert_pd(data_dict, names=['Metrics', 'L', 'p_m', 'p_f', 'es_C', 'es_m'])


def aggregate_over_samples(data_df):
    """
    Aggregate observations over es_C and es_m, computing mean and SEM for each (p_m, L).

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame with MultiIndex (Metrics, L, p_m, p_f, es_C, es_m) and 'observations' column
        containing arrays/lists of values.

    Returns
    -------
    pd.DataFrame
        DataFrame with MultiIndex (p, L) and columns:
        - 'mean': mean over all es_C and es_m samples
        - 'sem': standard error of the mean
        - 'log_mean': log of the mean
        - 'se_log_mean': standard error of log(mean) via chain rule: sem / mean
    """
    # Get unique (p_m, L) combinations
    p_m_vals = data_df.index.get_level_values('p_m').unique()
    L_vals = data_df.index.get_level_values('L').unique()

    results = []
    for p_m in p_m_vals:
        for L in L_vals:
            try:
                # Get all observations for this (p_m, L) combination
                subset = data_df.xs(p_m, level='p_m').xs(L, level='L')
                # Stack all observation arrays, padding shorter ones with NaN
                obs_stack = stack_with_nan_padding(subset['observations'].values)
                # Compute mean and SEM across all samples
                mean_val = np.nanmean(obs_stack, axis=0)
                sem_val = sem(obs_stack, axis=0)
                # Log2 of mean and its standard error via chain rule: d(log2(x))/dx = 1/(x*ln(2))
                log_mean_val = np.log2(mean_val)
                se_log_mean_val = sem_val / (mean_val * np.log(2))
                results.append({
                    'p': p_m,
                    'L': L,
                    'mean': mean_val,
                    'sem': sem_val,
                    'log_mean': log_mean_val,
                    'se_log_mean': se_log_mean_val
                })
            except KeyError:
                # Skip if this (p, L) combination doesn't exist
                continue

    result_df = pd.DataFrame(results)
    result_df = result_df.set_index(['p', 'L'])
    return result_df
