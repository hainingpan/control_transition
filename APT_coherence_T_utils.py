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
    """Standard error of the mean."""
    arr = np.asarray(arr)
    n = arr.shape[axis] if axis is not None else arr.size
    return arr.std(axis=axis, ddof=ddof) / np.sqrt(n)

def _load_pickle_with_swap(fn):
    data = rqc.load_pickle(fn)
    coherence = data.get('coherence')
    if coherence is not None and getattr(coherence, 'ndim', 0) == 5:
        # stored as (p_m, p_f, es_m, es_C, T); swap axes to match parse_APT_T expectation
        data['coherence'] = np.swapaxes(coherence, 2, 3)
    return data


def _load_zip_pickle_with_swap(fn, z):
    data = rqc.load_zip_pickle(fn, z)
    coherence = data.get('coherence')
    if coherence is not None and getattr(coherence, 'ndim', 0) == 5:
        data['coherence'] = np.swapaxes(coherence, 2, 3)
    return data


def load_apt_coherence(use_zip=True, L_list=(12, 14, 16, 18, 20, 22, 24), p_f=1, p_f_int=1,zipfilename='APT_coherence_T_pf1.zip', 
                       p_m_list = np.hstack([np.arange(0, 0.08, 0.01), np.arange(0.08, 0.101, 0.005),np.arange(0.11, 0.2, 0.01)]),
    BATCH_CONFIG = {
        12: {'es_batch': 2000, 'num_batches': 1},
        14: {'es_batch': 2000, 'num_batches': 1},
        16: {'es_batch': 2000, 'num_batches': 1},
        18: {'es_batch': 2000, 'num_batches': 1},
        20: {'es_batch': 2000, 'num_batches': 1},
        22: {'es_batch': 1000, 'num_batches': 2},
        24: {'es_batch': 100, 'num_batches': 20},
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
            es_start = 1 + batch_idx * cfg['es_batch']
            es_end = min(1 + (batch_idx + 1) * cfg['es_batch'], 2001)
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


def apt_coherence_to_df(data_dict):
    """Convert the loaded coherence dictionary to a pandas DataFrame."""
    return rqc.convert_pd(data_dict, names=['Metrics', 'L', 'p_m', 'p_f', 'es_C', 'es_m'])


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

        data_ = np.stack(data_df.xs(iter_, level=iter_var).xs(fix_value,level=fix_var)['observations']).mean(axis=0)
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
    y = [np.stack(data_df.xs(p_m, level='p_m').xs(L,level='L')['observations'])[:,min_func(L):max_func(L)].mean() for L in L_list]
    # y = [np.stack(data_df.xs(p_m, level='p_m').xs(L,level='L')['observations']).mean(axis=0)[-1:].mean() for L in L_list]
    yerr = [sem(np.stack(data_df.xs(p_m, level='p_m').xs(L,level='L')['observations'])[:,min_func(L):max_func(L)]) for L in L_list]
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
