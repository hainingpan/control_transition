import numpy as np
import matplotlib.pyplot as plt

def sem(arr, axis=None, ddof=1):
    """Standard error of the mean."""
    arr = np.asarray(arr)
    n = arr.shape[axis] if axis is not None else arr.size
    return arr.std(axis=axis, ddof=ddof) / np.sqrt(n)

def plot_apt_coherence_T_vs_steps_fixedL(data_df, L, ax =None, idx_=1):
    if ax is None:
        fig, ax =plt.subplots(figsize=(4,4))
    p_m_list = data_df.index.get_level_values('p_m').unique()
    color_list = plt.cm.Blues(np.linspace(.4, 1, len(p_m_list)))
    
    for  idx, p_m,  in enumerate(p_m_list):

        data_ = np.stack(data_df.xs(p_m, level='p_m').xs(L,level='L')['observations']).mean(axis=0)
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
            data_ = np.exp(np.log(np.stack(data_df.xs(p_m, level='p_m').xs(L,level='L')['observations'])).mean(axis=0))
        else:
            data_ = np.stack(data_df.xs(p_m, level='p_m').xs(L,level='L')['observations']).mean(axis=0)
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
                    data_ = np.exp(np.log(np.stack(data_df.xs(p_m, level='p_m').xs(L, level='L')['observations'])).mean(axis=0))
                else:
                    data_ = np.stack(data_df.xs(p_m, level='p_m').xs(L, level='L')['observations']).mean(axis=0)
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
        obs = np.stack(data_df.xs(p_m, level='p_m').xs(L, level='L')['observations'])
        max_idx = max_func(L) if max_func is not None else None
        obs_slice = obs[:, min_func(L):max_idx]
        if average_log:
            sample_means = np.exp(np.log(obs_slice).mean(axis=1))
        else:
            sample_means = obs_slice.mean(axis=1)
        y.append(sample_means.mean())
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
