import matplotlib.pyplot as plt
import numpy as np

def plot_metric_T_vs_steps_fixedL(data_df, L, metric, ax =None, idx_=1, ylabel=None, ylim=None,p_m_list=None, cmap=plt.cm.Blues, yscale='log', xscale='log'):
    if ax is None:
        fig, ax =plt.subplots(figsize=(4,4))
    if p_m_list is None:
        p_m_list = data_df.index.get_level_values('p_m').unique()

    color_list = cmap(np.linspace(.4, 1, len(p_m_list)))
    
    for  idx, p_m,  in enumerate(p_m_list):

        data_ = data_df.xs((metric, p_m, L))['observations']
        x = np.arange(len(data_))
        y =data_
        ax.plot(x[idx_:], y[idx_:], label=f'p_m={p_m:.3f}', color=color_list[idx], )
    # ax.axhline(np.pi/4 * 2**(L), color='k', linestyle='--', )
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim(None, x.max())
    ax.set_ylim(ylim)
    ax.grid(True, which='both', alpha=0.3)
    ax.minorticks_on()
    ax.tick_params(top=True, labeltop=True)
    ax.set_xlabel('Time step (one layer)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'L={L}')


def plot_metric_T_vs_steps_fixedp_m(data_df, p_m, metric, ax =None, idx_=1, prefactor=None, z=1.62, average_log=False, theory_line=False, ylabel=None, L_list=None,ylim=None, yscale='log', xscale='log'):
    if ax is None:
        fig, ax =plt.subplots(figsize=(4,4))
    if L_list is None:
        L_list = sorted(data_df.index.get_level_values('L').unique())
    color_list = plt.cm.Blues(np.linspace(.4, 1, len(L_list)))

    for  idx, L,  in enumerate(L_list):
        if average_log:
            raise NotImplementedError("average_log=True is not updated yet.")
            data_ = np.exp(np.log(np.stack(data_df.xs(p_m, level='p_m').xs(L,level='L')['observations'])).mean(axis=0))
        else:
            # data_ = np.stack(data_df.xs(p_m, level='p_m').xs(L,level='L')['observations']).mean(axis=0)
            data_ = data_df.xs((metric, p_m, L), level = ('Metrics','p_m','L'))['observations'].iloc[0]
        x = np.arange(len(data_))
        y = data_
        ax.plot(x[idx_:], y[idx_:], label=f'L={L}', color=color_list[idx])
        # if theory_line:
        #     ax.axhline(np.pi/4 * 2**(L), color=color_list[idx], linestyle='--', alpha=0.5)

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
                    data_ = data_df.xs((metric, p_m, L), level = ('Metrics','p_m','L'))['observations'].iloc[0]
                t_idx = int(pf * L**z)
                if t_idx < len(data_):
                    guide_t.append(t_idx)
                    guide_y.append(data_[t_idx])
            ax.plot(guide_t, guide_y, color=guide_color_list[pf_idx], linestyle='--', label=f'${pf}L^{{{z}}}$')

    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim(None, x.max())
    ax.set_ylim(ylim)
    ax.grid(True, which='both', alpha=0.3)
    ax.minorticks_on()
    ax.tick_params(top=True, labeltop=True)
    ax.set_xlabel('Time step (one layer)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'$p_m$={p_m:.3f}')

def simple_linearfit(x, y, xfunc=lambda x: x, yfunc=lambda x: x, ax=None, idx_min=2, idx_max=100):
        
    # idx_min = 2
    # idx_max = 100
    x_fit = x[idx_min:idx_max]
    y_fit = y[idx_min:idx_max]

    # Fit a line in linear-log scale
    log_x = xfunc(x_fit)
    log_y = yfunc(y_fit)
    slope, intercept = np.polyfit(log_x, log_y, 1)

    func_inv = {np.log: np.exp,
                 np.exp: np.log,
                lambda x: x: lambda x: x
                }
    def fitted_line(L):
        return func_inv[yfunc](intercept + slope * xfunc(L))

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x,(fitted_line((np.array(x)))), 'r--', label=f'Fit: slope={slope:.3f}')
    return slope, intercept
    # ax.plot(x, np.exp(fitted_line(np.log(np.array(x)))), 'r--')