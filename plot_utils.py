import os

def load_json(fn):
    import json
    with open(fn, "r") as file:
        data = json.load(file)
    return data

def load_pickle(fn):
    import pickle
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    return data

def visualize_dataset(df,xlabel,ylabel,params={'Metrics':'EE',}):
    """Visualize the ensemble size for two axis

    Parameters
    ----------
    df : DataFrame
        DataFrame to check
    xlabel : str
        str for the x-axis
    ylabel : str
        str for the y-axis
    params : dict, optional
        which metrics to look at, by default {'Metrics':'EE',}
    """    
    import matplotlib.pyplot as plt
    df=df.xs(params.values(),level=list(params.keys()))
    y=df.index.get_level_values(ylabel).values
    x=df.index.get_level_values(xlabel).values
    ensemblesize=df['observations'].apply(len).values
    fig,ax=plt.subplots()
    cm=ax.scatter(x,y,100,ensemblesize,marker='s',cmap='inferno')
    plt.colorbar(cm)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def add_to_dict(data_dict,data,filename,fixed_params_keys={},skip_keys=set(['offset'])):
    """Add an entry to the dictionary

    Parameters
    ----------
    data_dict : Dict
        Dictionary to be added to
    data : Dict
        Dictionary for a single entry
    filename : str
        Filename of the data
    """    
    import argparse
    data_dict['fn'].add(filename)

    assert not ('EE' in data and 'SA' in data), f'{filename} is problematic'

    if 'args' in data:
        if isinstance(data['args'],argparse.Namespace):
            iterator=data['args'].__dict__
        elif isinstance(data['args'],dict):
            iterator=data['args']
    else:
        raise ValueError(f'{filename} does not have args')
    
    # if 'offset' in iterator and iterator['offset']>0:
    #     print(iterator)

    if filename.split('.')[-1] == 'pickle':
        L_list=np.arange(*data['args'].L)
        p_ctrl_list=np.round(np.linspace(data['args'].p_ctrl[0],data['args'].p_ctrl[1],int(data['args'].p_ctrl[2])),3)
        p_proj_list=np.round(np.linspace(data['args'].p_proj[0],data['args'].p_proj[1],int(data['args'].p_proj[2])),3)
        

    for metric in set(data.keys())-set(['args']):
        if filename.split('.')[-1] == 'pickle':
            for L_idx,L in enumerate(L_list):
                for p_ctrl_idx,p_ctrl in enumerate(p_ctrl_list):
                    for p_proj_idx,p_proj in enumerate(p_proj_list):
                        observations=data[metric][L_idx,p_ctrl_idx,p_proj_idx]
                        if torch.is_tensor(observations):
                            observations=observations.cpu().tolist()
                            observations=[obs for obs in observations if not np.isnan(obs)]
                        params=(metric,L,p_ctrl,p_proj)
                        if params in data_dict:
                            data_dict[params]=data_dict[params]+observations
                        else:
                            data_dict[params]=observations
        else:
            params=(metric,)+tuple(val for key,val in iterator.items() if key != 'seed' and key not in fixed_params_keys and key not in skip_keys)

            if params in data_dict:
                data_dict[params].append(data[metric])
            else:
                data_dict[params]=[data[metric]]


def convert_pd(data_dict,names):
    """Convert the dictionary to a pandas dataframe

    Parameters
    ----------
    data_dict : Dict
        Dictionary
    names : List[str]
        List of names for the MultiIndex, example: ['Metrics','adder', 'L', 'p']

    Returns
    -------
    DataFrame
        Pandas DataFrame
    """    
    import pandas as pd
    index = pd.MultiIndex.from_tuples([key for key in data_dict.keys() if key!='fn'], names=names)
    df = pd.DataFrame({'observations': [val for key,val in data_dict.items() if key!='fn']}, index=index)
    return df           

def generate_params(
    fixed_params,
    vary_params,
    fn_template,
    fn_dir_template,
    input_params_template,
    load_data,
    filename='params.txt',
    filelist=None,
    load=False,
    data_dict=None,
    data_dict_file=None,
    fn_dir='auto',
    exist=False,
):
    """Generate params to running and loading

    Parameters
    ----------
    fixed_params : Dict 
        Fixed parameters, example: {'nu':0,'de':1}
    vary_params : Dict of list
        Returns the cartesian product of the lists, example: {'L':[8,10],'p':[0,0.5,1]}
    fn_template : str
        Template of filename, example: 'MPS_({nu},{de})_L{L}_p{p:.3f}_s{s}_a{a}.json'
    fn_dir_template : str
        Template of directory, example: 'MPS_{nu}-{de}'
    input_params_template : str
        Template of input parameters, example: '{p:.3f} {L} {seed} {ancilla}'
    load_data : Function,
        the function to load data, currently `load_json` and `load_pickle` are supported
    filename : str, optional
        _description_, by default 'params.txt'
    filelist : str, None, optional
        If true, read the `filelist` as a list of existing files, by default None
    load : bool, optional
        Load files into data_dict, by default False
    data_dict : Dict, optional
        The Dictionary to load into, the format should be {'fn':set(),...}, by default None
    data_dict_file : str, optional
        The filename of the cache file data_dict, if None, then use Dict provided by data_dict, else try to load file `data_dict_file`, if not exist, create a new Dict and save to disk, by default None
    fn_dir : str, 'auto', optional
        The search directory, if 'auto', use the template provided by `fn_dir_template`, by default 'auto'
    exist : bool, optional
        If true, print , by default False

    Returns
    -------
    List or Dict
        List of parameters to submit (if `exist` is False, and `load` is False) or existing files (if `exist` is True, and `load` is False), or the data_dict (if `load` is True)
    """
    from itertools import product
    import numpy as np
    from tqdm import tqdm
    import pickle


    params_text=[]
    if fn_dir=='auto':
        # fn_dir=fn_dir_template.format(**fixed_params)
        fn_dir=eval(f"f'{fn_dir_template}'", {},  {**locals(),**fixed_params,**vary_params})
        # eval(f"f'{filename_template}'", {},  locals())
    
    inputs=product(*vary_params.values())
    # vary_params.values()
    total=np.product([len(val) for val in vary_params.values()])


    if data_dict_file is not None:
        cache_fn=eval(f"f'{data_dict_file}'", {},  {**locals(),**fixed_params,**vary_params})
        data_dict_fn=os.path.join(fn_dir,cache_fn)
        if os.path.exists(data_dict_fn):
            print(f'Loading data_dict {data_dict_fn}')
            with open(data_dict_fn,'rb') as f:
                data_dict=pickle.load(f)
        else:
            print(f'Creating new data_dict {data_dict_fn}')
            data_dict={'fn':set()}

    for input0 in tqdm(inputs,mininterval=1,desc='generate_params',total=total):
        dict_params={key:val for key,val in zip(vary_params.keys(),input0)}
        dict_params.update(fixed_params)
        # fn=fn_template.format(**dict_params)
        fn=eval(f"f'{fn_template}'", {},  {**locals(),**dict_params})

        if load:
            if fn not in data_dict['fn']:
                if os.path.exists(os.path.join(fn_dir,fn)):
                    try:
                        data=load_data(os.path.join(fn_dir,fn))
                    except:
                        print(f'Error loading {fn}')
                        continue
                    add_to_dict(data_dict,data,fn,fixed_params_keys=fixed_params.keys())
        else:
            if filelist is None:
                file_exist = os.path.exists(os.path.join(fn_dir,fn))
            else:
                with open(filelist,'r') as f:
                    fn_list=f.read().split('\n')
                file_exist = fn in fn_list
            
            if not file_exist:
                params_text.append(input_params_template.format(**dict_params))
            elif exist:
                params_text.append(fn)
    if load:
        if data_dict_file is not None:
            with open(data_dict_fn,'wb') as f:
                pickle.dump(data_dict,f)
        return data_dict
    else:
        if filename is not None:
            with open(filename,'a') as f:
                f.write('\n'.join(params_text)+'\n')
        return params_text
    
    
import numpy as np
def plot_line(
    df,
    x_name,
    ax=None,
    params={'Metrics':'O','p':0,},
    L_list=None,
    yscale=None,
    ylim=None,
    method=np.mean,
    errorbar=False
    ):
    """Plot a single line"""
    import matplotlib.pyplot as plt
    from functools import partial
    from scipy.stats import moment

    if ax is None:
        fig,ax=plt.subplots()
    assert method in {np.mean,np.var}, f'the method should be either np.mean or np.var. {method} is not currently supported.'
    # x_name='p'
    # title_name=
    op_str={np.mean:r'\overline',np.var:r'Var~'}
    ylabel_name={'O':rf'${op_str[method]}{{\langle O \rangle}}$','EE':rf'${op_str[method]}{{ S_{{L/2}} }}$','TMI':rf'${op_str[method]}{{I_3}}$','SA':rf'${op_str[method]}{{ S_{{anc}} }}$','max_bond':rf'${op_str[method]}{{\chi}}$'}
    df=df.xs(params.values(),level=list(params.keys()))
    if L_list is None:
        L_list=np.sort(df.index.get_level_values('L').unique())
    colormap = (plt.cm.Blues(0.4+0.6*(i/L_list.shape[0])) for i in range(L_list.shape[0]))
    for L in sorted(L_list):
        dd=df.xs(key=L,level='L')['observations'].apply(method)
        if errorbar:
            if method is np.mean:
                dd_se=df.xs(key=L,level='L')['observations'].apply(np.std).values/np.sqrt(df.xs(key=L,level='L')['observations'].apply(len).values)
            if method is np.var:
                mu4=df.xs(key=L,level='L')['observations'].apply(partial(moment,moment=4)).values
                mu2=df.xs(key=L,level='L')['observations'].apply(partial(moment,moment=2)).values
                n=(df.xs(key=L,level='L')['observations'].apply(len).values)
                dd_se=np.sqrt((mu4-(n-3)/(n-1)*mu2**2)/n)
                
        x=dd.index.get_level_values(x_name)
        arg_sort=x.argsort()
        if yscale == 'log' and params['Metrics']== 'TMI':
            dd_sort=np.abs(dd.values[arg_sort])
            ylabel_name['TMI']=rf'${op_str[method]}{{|I_3|}}$'

        else:
            dd_sort=dd.values[arg_sort]
        if errorbar:
            ax.errorbar(x[arg_sort],dd_sort,yerr=dd_se[arg_sort],label=f'L={L}',lw=1,color=colormap.__next__(),capsize=2)
        else:
            ax.plot(x[arg_sort],dd_sort,'.-',label=f'L={L}',lw=1,color=colormap.__next__())
    ax.legend()
    if ylim is not None:
        ax.set_ylim(ylim)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.set_ylabel(ylabel_name[params['Metrics']])
    ax.set_xlabel(x_name)
    # ax.set_title(f'{title_name}={params[title_name]:.2f}')

def plot_inset(
    data,
    ax,
    xlim,
    ylim,
    ax_inset_pos,
    L_list,
    params,
    yscale,
    method,
    x_name
    ):
    """Plot inset"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    

    # .27,.3
    # .6,.63
    axins = ax.inset_axes(ax_inset_pos,transform=ax.transAxes)
    plot_line(data,params=params,ax=axins,L_list=L_list,yscale=yscale,method=method,x_name=x_name)
    axins.grid('on')
    axins.set_xlim(xlim)
    axins.set_ylim(ylim)
    axins.legend().remove()
    axins.set_ylabel('')
    axins.set_title('')
    axins.set_xlabel('')
    rect=mpatches.Rectangle((xlim[0],ylim[0]),xlim[1]-xlim[0],ylim[1]-ylim[0],ls='dashed',fill=None,lw=0.5,zorder=10)
    ax.add_patch(rect)
    if xlim[0]>=0.6*(ax_inset_pos[0]) and xlim[1]>=0.6*(ax_inset_pos[0]+ax_inset_pos[1]):
        dashed_coord_1=(xlim[0],ylim[1])
        inset_coord_1=(0,1)
        dashed_coord_2=(xlim[1],ylim[0])
        inset_coord_2=(1,0)
    elif xlim[0]>=0.6*(ax_inset_pos[0]) and xlim[1]<0.6*(ax_inset_pos[0]+ax_inset_pos[1]):
        dashed_coord_1=(xlim[0],ylim[1])
        inset_coord_1=(0,1)
        dashed_coord_2=(xlim[1],ylim[1])
        inset_coord_2=(1,1)
    elif xlim[0]<0.6*(ax_inset_pos[0]) and xlim[1]<0.6*(ax_inset_pos[0]+ax_inset_pos[1]):
        dashed_coord_1=(xlim[0],ylim[0])
        inset_coord_1=(0,0)
        dashed_coord_2=(xlim[1],ylim[1])
        inset_coord_2=(1,1)

    line1=mpatches.ConnectionPatch(dashed_coord_1, inset_coord_1, coordsA='data',coordsB='axes fraction',axes=ax,axesB=axins,ls='dashed',lw=0.5)
    ax.add_patch(line1)
    line2=mpatches.ConnectionPatch(dashed_coord_2, inset_coord_2, coordsA='data',coordsB='axes fraction',axes=ax,axesB=axins,ls='dashed',lw=0.5)
    ax.add_patch(line2)

def plot_line_inset(
    df_anc,
    L_list,
    xlim1,
    xlim2,
    ylim1,
    ylim2,
    ax_inset_pos1,
    ax_inset_pos2,
    metrics,
    x_name='p',
    fixed_params={},
    inset1=False,
    inset2=False,
    yscale=None,
    filename=None,
    dirpath='Fig',
    ylim=None,
    errorbar=False,
    method=np.mean,
    filename_template='{metrics}_{method_name[method]}_L({L_list[0]},{L_list[-1]}){"_log" if yscale else ""}.png'
    ):
    """plot lines and inset

    Parameters
    ----------
    df_anc : DataFrame
        DataFrame to plot
    L_list : np.array
        List of L to plot, example: np.array([8,10,12])
    xlim1 : List
        xlim for inset1, example: [.28,.32]
    xlim2 : List
        xlim for inset2, example: [0.48,0.52]
    ylim1 : List
        ylim for inset1, example: [0.1,0.3]
    ylim2 : List
        ylim for inset2, example: [0.4,0.6]
    ax_inset_pos1 : List
        ax_inset_pos for inset1, example: [.13,.45,.4,.3]
    ax_inset_pos2 : List
        ax_inset_pos for inset2, example: [.1,.1,.3,.4]
    metrics : str
        which metrics to plot, example: 'EE'
    x_name : str, optional
        which x_name to plot, example: 'p'
    fixed_params : dict, optional
        other params which should be fixed in plotting, by default None
    inset1 : bool, optional
        whether to plot inset1, by default False
    inset2 : bool, optional
        whether to plot inset1, by default False
    yscale : None, str, optional
        If none, use linear scale; If `log`, use log scale for yaxis, by default None
    filename : str, optional
        If 'auto', use `filename_template` to generate filename, otherwise use `filename` itself, by default None
    dirpath : str, optional
        _description_, by default 'Fig'
    ylim : List, optional
        set ylim, by default None
    errorbar : bool, optional
        whether errorbar should show, by default False
    method : func, optional
        if method is `np.mean`, compute mean; if method is `np.var` compute variance, otherwise unsupported, by default np.mean
    filename_template : str, optional
        use f-string template to control the output, by default '{metrics}_{method_name[method]}_L({L_list[0]},{L_list[-1]}){"_log" if yscale else ""}.png'
    """    
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6.8,
    5))
    params={'Metrics':metrics,}
    params.update(fixed_params)
    
    plot_line(df_anc,params=params,ax=ax,L_list=L_list,yscale=yscale,ylim=ylim,errorbar=errorbar,method=method,x_name=x_name)
    ax.grid('on')
    ax.set_xlim(0,0.6)
    if inset1:
        plot_inset(df_anc,ax,xlim=xlim1,ylim=ylim1,ax_inset_pos=ax_inset_pos1,params=params,L_list=L_list,yscale=yscale,method=method,x_name=x_name)

    if inset2:
        plot_inset(df_anc,ax,xlim=xlim2,ylim=ylim2,ax_inset_pos=ax_inset_pos2,params=params,L_list=L_list,yscale=yscale,method=method,x_name=x_name)
    
    fstr= lambda template: eval(f"f'{template}'")

    if filename is not None:
        if filename== 'auto':
            method_name={np.mean:'mean',np.var:'var'}
            filename= eval(f"f'{filename_template}'", {},  locals())
        print(filename)
        # plt.subplots_adjust(left=(.8)/fig.get_size_inches()[0],right=1-(.1)/fig.get_size_inches()[0],bottom=.5/fig.get_size_inches()[1],top=1-.2/fig.get_size_inches()[1])
        # fig.savefig(os.path.join(dirpath,filename),)

from matplotlib.colors import LogNorm
import torch
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
class Optimizer:
    def __init__(self,p_c,nu,df,params={'Metrics':'O',},p_range=[-0.1,0.1],Lmin=None,Lmax=None,bootstrap=False,gaussian_check=False):
        self.p_c=torch.tensor([p_c],requires_grad=False)
        self.nu=torch.tensor([nu],requires_grad=False)
        self.p_range=p_range
        self.Lmin=0 if Lmin is None else Lmin
        self.Lmax=100 if Lmax is None else Lmax
        self.bootstrap=bootstrap
        self.gaussian_check=gaussian_check
        self.params=params
        self.df=self.load_dataframe(df,params)
        self.L_i,self.p_i,self.d_i,self.y_i = self.load_data()

    
    def load_dataframe(self,df,params):
        df=df.xs(params.values(),level=list(params.keys()))['observations']
        df=df[(df.index.get_level_values('p')<=self.p_c.item()+self.p_range[1]) & (self.p_c.item()+self.p_range[0]<=df.index.get_level_values('p'))]
        df=df[(df.index.get_level_values('L')<=self.Lmax) & (self.Lmin<=df.index.get_level_values('L'))]
        if self.bootstrap:
            df=df.apply(lambda x: list(np.random.choice(x,size=len(x),replace=True)))
        if self.gaussian_check:
            print(df.apply(scipy.stats.shapiro))
        return df
    
    def load_data(self):
        L_i=torch.from_numpy(self.df.index.get_level_values('L').values)
        p_i=torch.from_numpy(self.df.index.get_level_values('p').values)
        d_i=torch.from_numpy(self.df.apply(np.std).values)/np.sqrt(self.df.apply(len).values)
        y_i=torch.from_numpy(self.df.apply(np.mean).values)
        assert p_i.unique().shape[0]>=4, f'not enough data points {p_i.unique().shape[0]}'
        return L_i,p_i,d_i,y_i

    def loss(self,p_c,nu,MLE=True):
        x_i=(self.p_i-p_c)*(self.L_i)**(1/nu)
        order=x_i.argsort()
        x_i_ordered=x_i[order]
        y_i_ordered=self.y_i[order]
        d_i_ordered=self.d_i[order]
        x={i:x_i_ordered[1+i:x_i_ordered.shape[0]-1+i] for i in [-1,0,1]}
        d={i:d_i_ordered[1+i:d_i_ordered.shape[0]-1+i] for i in [-1,0,1]}
        y={i:y_i_ordered[1+i:y_i_ordered.shape[0]-1+i] for i in [-1,0,1]}
        x_post_ratio=(x[1]-x[0])/(x[1]-x[-1])
        x_pre_ratio=(x[-1]-x[0])/(x[1]-x[-1])
        y_var=d[0]**2+(x_post_ratio*d[-1])**2+(x_pre_ratio*d[1])**2
        y_bar=x_post_ratio*y[-1]-x_pre_ratio*y[1]
        # return torch.sum((y[0]-y_bar)**2/y_var)
        if MLE:
            return self.MLE(y[0],y_bar,y_var)
        else:
            return self.chi2(y[0],y_bar,y_var)
    
    def loss_drift(self,p_c,nu,omega,beta):
        x_i=(self.p_i-p_c)*(self.L_i)**(1/nu)
        y_var=self.d_i**2
        self.y_i_fitted=0
        for i,b in enumerate(beta[:-1]):
            self.y_i_fitted+=b*x_i**i
        self.y_i_fitted+=beta[-1]/self.L_i**omega
        self.p_c=p_c
        self.nu=nu
        self.omega=omega
        self.beta=beta
        
        # self.y_i_fitted=a+b*x_i+c*x_i**2+d/self.L_i**omega
        return self.chi2(self.y_i_fitted,self.y_i,y_var)

    
    def loss_drift_sample(self,p_c,nu,omega,b1,b2,a):
        """return the residual of each sample, 
        a.shape(n1+1,n2+1), a=[[a00, a01=1, a02, a03, ...],
                               [a10=1, a11, a12, a13, ...],
                               [a20, a21,   a22, a23, ...],
                               ...]
        b1.shape=(m1,), b1=[b10=0,b11,b12,...]
        b2.shape=(m2+1,), b2=[b20,b21,b22,...]
        w_i.shape=(n_sample,)"""
        w_i=((self.p_i-p_c)/p_c)    # (n_sample,)
        u1_i=torch.tensor(b1)@w_i**torch.arange(1,b1.shape[0]+1)[:,None]    # (n_sample,) because b10=0 to ensure u1(w=0)=0
        u2_i=torch.tensor(b2)@w_i**torch.arange(b2.shape[0])[:,None]  # (n_sample,)
        phi_1=u1_i*(self.L_i)**(1/nu)    # (n_sample,)
        phi_2=u2_i*(self.L_i)**(-omega)  # (n_sample,)
        phi_1_=phi_1 ** torch.arange(a.shape[0])[:,None]    # (n1+1,n_sample)
        phi_2_=phi_2 ** torch.arange(a.shape[1])[:,None]    # (n2+1,n_sample)
        self.y_i_fitted=torch.einsum('ij,ik,kj->j',phi_1_,torch.tensor(a),phi_2_)


        self.p_c=p_c
        self.nu=nu
        self.omega=omega
        
        return (self.y_i_fitted-self.y_i)/self.d_i

    def chi2(self,y,y_fitted,sigma2):
        return 0.5*torch.sum((y-y_fitted)**2/sigma2)
    
    def MLE(self,y,y_fitted,sigma2):
        return 0.5*torch.sum((y-y_fitted)**2/sigma2)+0.5*torch.sum(torch.log(sigma2))

    
    def visualize(self,p_c_range,nu_range,trajectory=False,fig=True,ax=None,mapfunc=lambda x:x):
        p_c_list=np.linspace(*p_c_range,82)
        nu_list=np.linspace(*nu_range,80)
        loss_map=np.array([[self.loss(torch.tensor([p_c]),torch.tensor([nu]),MLE=False).item() for p_c in p_c_list] for nu in nu_list])
        if fig:
            if ax is None:
                fig, ax = plt.subplots()
            cm=ax.contourf(p_c_list,nu_list,mapfunc(loss_map),levels=20)
            ax.set_xlabel(r'$p_c$')
            ax.set_ylabel(r'$\nu$')
            plt.colorbar(cm)
            if trajectory:
                ax.scatter(self.p_c_history,self.nu_history,s=np.linspace(3,1,len(self.p_c_history))**2,)
            ct=ax.contour(p_c_list,nu_list,mapfunc(loss_map),levels=[mapfunc(self.loss(self.p_c,self.nu,MLE=False).item()*1.3),],colors='k',linestyles='dashed')
        else:
            ct=plt.contour(p_c_list,nu_list,mapfunc(loss_map),levels=[mapfunc(self.loss(self.p_c,self.nu,MLE=False).item()*1.3),],colors='k',linestyles='dashed');
        params_range=ct.collections[0].get_paths()[0].vertices
        return params_range[:,0].min(),params_range[:,0].max(),params_range[:,1].min(),params_range[:,1].max()

    def optimize(self,tolerance=1e-10):
        """Optimize using pytorch, Gradient Descent method"""
        p_c_prime = torch.tensor([torch.logit(self.p_c)],requires_grad=True)
        nu_prime = torch.tensor([torch.log(self.nu)],requires_grad=True)
        optimizer=torch.optim.Adam([p_c_prime,nu_prime],)
        # optimizer=torch.optim.Adam([self.p_c,self.nu],)
        prev_loss=float('inf')
        current_loss=0
        self.loss_history=[]
        # self.p_c_history=[self.p_c.item()]
        # self.nu_history=[self.nu.item()]
        self.p_c_history=[torch.sigmoid(p_c_prime).item()]
        self.nu_history=[torch.exp(nu_prime).item()]
        iteration=0
        while abs(prev_loss-current_loss)>tolerance and iteration<10000:
            p_c_transformed = torch.sigmoid(p_c_prime)
            nu_transformed = torch.exp(nu_prime)

            loss_ = self.loss(p_c_transformed, nu_transformed,MLE=False)
            # loss_=self.loss(self.p_c,self.nu)
            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()
            prev_loss=current_loss
            current_loss=loss_.item()
            self.loss_history.append(current_loss)
            self.p_c_history.append(p_c_transformed.item())
            self.nu_history.append(nu_transformed.item())
            # self.p_c_history.append(self.p_c.item())
            # self.nu_history.append(self.nu.item())
            iteration+=1
        self.p_c = torch.sigmoid(p_c_prime)
        self.nu = torch.exp(nu_prime)
        Hessian= torch.tensor(torch.autograd.functional.hessian(self.loss,(self.p_c,self.nu)))
        self.se=torch.sqrt(torch.diag(torch.inverse(Hessian)))
        
        return {'p_c':self.p_c.item(),'nu':self.nu.item(),'loss':current_loss*2/(self.y_i.shape[0]-2),'se':self.se.detach().numpy()}

    def optimize_scipy(self):
        """Optimize using scipy.minimize"""
        func=lambda x: self.loss(torch.tensor([x[0]]),torch.tensor([x[1]]),MLE=False).item()
        res=scipy.optimize.minimize(func,[self.p_c.item(),self.nu.item()],method='Nelder-Mead',bounds=[(0,1),(0,2)])
        # res=scipy.optimize.minimize(func,[self.p_c.item(),self.nu.item()],method='L-BFGS-B',bounds=[(0,1),(0,5)])
        # 'L-BFGS-B',bounds=[(0,1),(0,5)]
        Hessian= torch.tensor(torch.autograd.functional.hessian(self.loss,(torch.tensor(res.x[0]),torch.tensor(res.x[1]))))
        se=torch.sqrt(torch.diag(torch.inverse(Hessian)))
        self.p_c=torch.tensor([res.x[0]])
        self.nu=torch.tensor([res.x[1]])
        return res,res.fun*2/(self.y_i.shape[0]-2),se

    def optimize_drift(self,omega,a,b,c,d,tolerance=1e-10,):
        """Optimize using pytorch, Gradient Descent method with consideration of drifting of crossing point. Scaling function is Talyor expansion in  PHYS. REV. X 12, 041002 (2022)"""
        p_c_prime = torch.tensor([torch.logit(self.p_c)],requires_grad=True)
        nu_prime = torch.tensor([torch.log(self.nu)],requires_grad=True)
        omega=torch.tensor([omega],requires_grad=True,dtype=torch.float32)
        a=torch.tensor([a],requires_grad=True,dtype=torch.float32)
        b=torch.tensor([b],requires_grad=True,dtype=torch.float32)
        c=torch.tensor([c],requires_grad=True,dtype=torch.float32)
        d=torch.tensor([d],requires_grad=True,dtype=torch.float32)
        optimizer=torch.optim.Adam([p_c_prime,nu_prime,omega,a,b,c,d],)
        prev_loss=float('inf')
        current_loss=0
        self.loss_history=[]
        self.p_c_history=[torch.sigmoid(p_c_prime).item()]
        self.nu_history=[torch.exp(nu_prime).item()]
        iteration=0
        while abs(prev_loss-current_loss)>tolerance and iteration<100000:
            p_c_transformed = torch.sigmoid(p_c_prime)
            nu_transformed = torch.exp(nu_prime)

            loss_ = self.loss_drift(p_c_transformed, nu_transformed,omega,a,b,c,d)
            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()
            prev_loss=current_loss
            current_loss=loss_.item()
            self.loss_history.append(current_loss)
            self.p_c_history.append(p_c_transformed.item())
            self.nu_history.append(nu_transformed.item())
            iteration+=1
        self.p_c = torch.sigmoid(p_c_prime)
        self.nu = torch.exp(nu_prime)
        return {'p_c':self.p_c.item(),'nu':self.nu.item(),'omega':omega.item(),'a':a.item(),'b':b.item(),'c':c.item(),'d':d.item(),'loss':current_loss,'chi-square_nu':current_loss*2/(self.y_i.shape[0]-7)}

    def optimize_drift_scipy(self,omega,a,b,c,d):
        """Optimize using scipy.minimize, using taylor expansion PHYS. REV. X 12, 041002 (2022)"""
        # omega,a,b,c,d=
        func=lambda x: self.loss_drift(*tuple(x),d=d).item()
        res=scipy.optimize.minimize(func,[self.p_c.item(),self.nu.item(),omega,a,b,c],method='Nelder-Mead')
        Hessian= torch.tensor(torch.autograd.functional.hessian(lambda x: self.loss_drift(*x,d=d),torch.tensor(res.x)))
        # se=torch.sqrt(torch.diag(torch.inverse(Hessian)))
        # self.p_c=torch.tensor([res.x[0]])
        # self.nu=torch.tensor([res.x[1]])
        return res,res.fun*2/(self.y_i.shape[0]-7),

    def linear_least_square(self,p_c,nu,omega,n):
        """n is the order of relevant parts"""
        X=torch.zeros((self.y_i.shape[0],n+2),dtype=torch.float64)
        X[:,0]=torch.ones_like(self.y_i)
        x_i=(self.p_i-p_c)*(self.L_i)**(1/nu)
        for j in range(1,n+1):
            X[:,j]=x_i**j
        X[:,n+1]=1/self.L_i**omega
        Y=self.y_i
        Sigma_inv=torch.diag(1/self.d_i**2)
        XY=X.T @ Sigma_inv @ Y
        XX=X.T @ Sigma_inv @ X
        # beta=torch.linalg.solve(XX,XY)
        beta=torch.linalg.inv(XX)@XY
        return beta
    def optimize_drift_lsq(self,omega,n=2):
        """generalized version of PHYS. REV. X 12, 041002 (2022), to n-th order"""
        def func(x):
            beta= self.linear_least_square(p_c=x[0],nu=x[1],omega=x[2],n=n)
            return self.loss_drift(p_c=x[0],nu=x[1],omega=x[2],beta=beta).item()

        res=scipy.optimize.minimize(func,[self.p_c.item(),self.nu.item(),omega],method='Nelder-Mead',bounds=[(min(self.p_i),max(self.p_i)),(.2,3),(1e-4,None)],)
        self.p_c=torch.tensor([res.x[0]])
        self.nu=torch.tensor([res.x[1]])
        return res, res.fun*2/(self.y_i.shape[0]-3), 

    def optimize_drift_nonlinear_lsq(self,omega, m1,m2,n1,n2,x0=None):
        """m1, m2 controls the order of (p-p_c)/p_c, in the relevant and irrelevant operaor.
        n1, n2 controls the order of phi=u((p-p_c)/p_c) * L^{1/nu} and  phi=u((p-p_c)/p_c) * L^{-omega}   
        n2 can be zero while n1 cannot be zero, 

        TODO: the mixing of torch and numpy is messy, should use a clean version

        """ 
        assert n1>0, 'n1 should be greater than 0'

        def func(x):
            p_c,nu,omega=x[0],x[1],x[2]
            b1=x[3:3+m1]
            if n2>0:
                b2=x[3+m1:3+m1+m2+1]
            else:
                b2=np.array([])
            a=torch.zeros((n1+1,n2+1),dtype=torch.float64)
            ## This is not correct, why m2 is there when n2 is zero?
            a[0,0]=x[3+m1+m2+1]
            if n2>0:
                a[0,1]=1
                if n2>1:
                    a[0,2:n2+1]=torch.tensor([x[3+m1+m2+2:3+m1+m2+2+n2-1]])
            if n1>0:
                a[1,0]=1
                a[1,1:n2+1]=torch.tensor(x[3+m1+m2+2+n2-1:3+m1+m2+2+n2-1+n2])
                if n1>1:
                    a[2:,:]=torch.tensor(x[3+m1+m2+2+n2-1+n2:].reshape(n1-1,n2+1))

            self.x0=x

            return self.loss_drift_sample(p_c,nu,omega,b1,b2,a)
        if x0 is None:
            x0=[self.p_c.item(),self.nu.item(),omega]+[0]*(m1+m2+1+(n1+1)*(n2+1)-2)
        else:
            x0=[self.p_c.item(),self.nu.item(),omega]+x0
        res=scipy.optimize.least_squares(func,x0,method='lm',)
        return res, res.cost*2/(self.y_i.shape[0]-len(x0))



    def plot_loss(self):
        if hasattr(self, 'loss_history'):
            fig,ax=plt.subplots()
            ax.plot(self.loss_history,'.-')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('O')
    
    def plot_data_collapse(self,ax=None,drift=False):
        x_i=(self.p_i-self.p_c)*(self.L_i)**(1/self.nu)
        # x_i=self.p_i
        if ax is None:
            fig,ax = plt.subplots()
        L_list=self.df.index.get_level_values('L').unique().sort_values().values
        idx_list=[0]+(np.cumsum([self.df.xs(key=L,level='L').shape[0] for L in L_list])).tolist()
        L_dict={L:(start_idx,end_idx) for L,start_idx,end_idx in zip(L_list,idx_list[:-1],idx_list[1:])}
        # color_iter=iter(plt.cm.rainbow(np.linspace(0,1,len(L_list))))
        color_iter = iter(plt.cm.Blues(0.4+0.6*(i/L_list.shape[0])) for i in range(L_list.shape[0]))
        for L,(start_idx,end_idx) in L_dict.items():
            color=next(color_iter)
            if drift:
                ax.errorbar(self.p_i.detach().numpy()[start_idx:end_idx], self.y_i.detach().numpy()[start_idx:end_idx], label=f'{L}', color=color, yerr=self.d_i.detach().numpy()[start_idx:end_idx], capsize=2, fmt='x',linestyle="None")
                ax.plot(self.p_i.detach().numpy()[start_idx:end_idx],self.y_i_fitted.detach().numpy()[start_idx:end_idx],label=f'{L}',color=color)
                
            else:
                ax.scatter(x_i.detach().numpy()[start_idx:end_idx],self.y_i.detach().numpy()[start_idx:end_idx],label=f'{L}',color=color)
                # ax.plot(x_i.detach().numpy()[start_idx:end_idx],self.y_i_fitted.detach().numpy()[start_idx:end_idx],label=f'{L}')
                


        if drift:
            ax.set_xlabel(r'$p_i$')
            ax.set_title(rf'$p_c={self.p_c.item():.3f},\nu={self.nu.item():.3f},\omega = {self.omega.item():.3f}$')
        else:
            ax.set_xlabel(r'$(p_i-p_c)L^{1/\nu}$')
            ax.set_title(rf'$p_c={self.p_c.item():.3f},\nu={self.nu.item():.3f}$')
        ax.set_ylabel(r'$y_i$')
        ax.legend()
        ax.grid('on')

        # adder=self.df.index.get_level_values('adder').unique().tolist()[0]
        # print(f'{self.params["Metrics"]}_Scaling_L({L_list[0]},{L_list[-1]})_adder({adder[0]}-{adder[1]}).png')
        
    
    def plot_line(self):
        fig,ax=plt.subplots()
        ax.plot(self.p_i,self.y_i)

from lmfit import minimize, Parameters
class DataCollapse:
    """DataCollapse class, use scipy"""
    def __init__(self,df,params={'Metrics':'O',},p_range=[-0.1,0.1],Lmin=None,Lmax=None,p_dim=1):
        self.p_range=p_range
        self.Lmin=0 if Lmin is None else Lmin
        self.Lmax=1000 if Lmax is None else Lmax
        self.params=params
        self.p_='p' if p_dim==1 else ('p_ctrl' if 'p_proj' in params else 'p_proj')
        self.df=self.load_dataframe(df,params,p_dim=p_dim)
        self.L_i,self.p_i,self.d_i,self.y_i = self.load_data()
    
    def load_dataframe(self,df,params,p_dim):
        df=df.xs(params.values(),level=list(params.keys()))['observations']
        df=df[(df.index.get_level_values(self.p_)<=self.p_range[1]) & (self.p_range[0]<=df.index.get_level_values(self.p_))]
        df=df[(df.index.get_level_values('L')<=self.Lmax) & (self.Lmin<=df.index.get_level_values('L'))]
        return df.sort_index(level=["L",self.p_])

    def load_data(self):
        L_i=(self.df.index.get_level_values('L').values)
        p_i=(self.df.index.get_level_values(self.p_).values)
        d_i=(self.df.apply(np.std).values)/np.sqrt(self.df.apply(len).values)
        # d_i=(self.df.apply(np.std).values)
        y_i=(self.df.apply(np.mean).values)
        assert np.unique(p_i).shape[0]>=4, f'not enough data points {np.unique(p_i).shape[0]}'
        return L_i,p_i,d_i,y_i    
    
    def loss(self,p_c,nu):
        x_i=(self.p_i-p_c)*(self.L_i)**(1/nu)
        order=x_i.argsort()
        x_i_ordered=x_i[order]
        y_i_ordered=self.y_i[order]
        d_i_ordered=self.d_i[order]
        x={i:x_i_ordered[1+i:x_i_ordered.shape[0]-1+i] for i in [-1,0,1]}
        d={i:d_i_ordered[1+i:d_i_ordered.shape[0]-1+i] for i in [-1,0,1]}
        y={i:y_i_ordered[1+i:y_i_ordered.shape[0]-1+i] for i in [-1,0,1]}
        x_post_ratio=(x[1]-x[0])/(x[1]-x[-1])
        x_pre_ratio=(x[-1]-x[0])/(x[1]-x[-1])
        y_var=d[0]**2+(x_post_ratio*d[-1])**2+(x_pre_ratio*d[1])**2
        y_bar=x_post_ratio*y[-1]-x_pre_ratio*y[1]
        return (y[0]-y_bar)/np.sqrt(y_var)

    def loss_with_drift(self,p_c,nu,y,b1,b2,a):
        """p_c: critical point
        nu: relevent critical exponent
        y: irrelevant scaling function
        b1: b1=[b10=0,b11,b12,...] in relevent part , b1.shape=(m1,)
        b2: b2=[b20,b21,b22,...] in irrelevant part, b2.shape=(m2+1,)
        a: a=[[a00, a01=1, a02, a03, ...],
            [a10=1, a11, a12, a13, ...],
            [a20, a21,   a22, a23, ...],
            ...]
        w_i.shape=(n_sample,)
        """
        w_i=((self.p_i-p_c))
        # w_i=((self.p_i-p_c)/p_c)
        u1_i=(b1)@w_i**np.arange(len(b1))[:,np.newaxis]    # (n_sample,) because b10=0 to ensure u1(w=0)=0
        u2_i=(b2)@w_i**np.arange(len(b2))[:,np.newaxis]  # (n_sample,)
        phi_1=u1_i*(self.L_i)**(1/nu)    # (n_sample,)
        phi_2=u2_i*(self.L_i)**(-y)  # (n_sample,)
        self.phi_1_=phi_1 ** np.arange(a.shape[0])[:,np.newaxis]    # (n1+1,n_sample)
        self.phi_2_=phi_2 ** np.arange(a.shape[1])[:,np.newaxis]    # (n2+1,n_sample)
        self.a=a
        self.y_i_fitted=np.einsum('ij,ik,kj->j',self.phi_1_,self.a,self.phi_2_)
        
        return (self.y_i-self.y_i_fitted)/self.d_i
    
    def datacollapse(self,p_c=None,nu=None,**kwargs):
        """data collapse without drift, x_i=(p_i-p_c)L^{1/nu}, and try to make x_i vs y_i collapse to a smooth line"""

        params=Parameters()
        params.add('p_c',value=p_c,min=0,max=1)
        params.add('nu',value=nu,min=0,max=2)
        def residual(params):
            p_c,nu=params['p_c'],params['nu']
            return self.loss(p_c,nu)

        res=minimize(residual,params,**kwargs)
        self.p_c=res.params['p_c'].value
        self.nu=res.params['nu'].value
        self.res=res
        return res

    def datacollapse_with_drift(self,m1,m2,n1,n2,p_c=None,nu=None,y=None,b1=None,b2=None,a=None,p_c_vary=True,nu_vary=True,y_vary=True,seed=None,**kwargs):
        params=Parameters()
        params.add('p_c',value=p_c,min=0,max=1,vary=p_c_vary)
        params.add('nu',value=nu,min=0,max=2,vary=nu_vary)
        params.add('y',value=y,min=0,vary=y_vary)
        rng=np.random.default_rng(seed)
        if b1 is None:
            # b1=[0]*(m1+1)
            b1=rng.normal(size=(m1+1))
        if b2 is None:
            # b2=[0]*(m2+1)
            b2=rng.normal(size=(m2+1))
        if a is None:
            # a=np.array([[0]*(n2+1)]*(n1+1))
            a=rng.normal(size=(n1+1,n2+1))
        for i in range(m1+1):
            if i == 0:
                params.add(f'b_1_{i}',value=0,vary=False)
            else:
                params.add(f'b_1_{i}',value=b1[i])
        for i in range(m2+1):
            params.add(f'b_2_{i}',value=b2[i])
        for i in range(n1+1):
            for j in range(n2+1):
                if (i==1 and j==0) or (i==0 and j==1):
                    params.add(f'a_{i}_{j}',value=1,vary=False)
                else:
                    params.add(f'a_{i}_{j}',value=a[i,j])
        def residual(params):
            return self.loss_with_drift(params['p_c'],params['nu'],params['y'],[params[f'b_1_{i}'] for i in range(m1+1)],[params[f'b_2_{i}'] for i in range(m2+1)],np.array([[params[f'a_{i}_{j}'] for j in range(n2+1)] for i in range(n1+1)]))
        res=minimize(residual,params,**kwargs)
        self.p_c=res.params['p_c'].value
        self.nu=res.params['nu'].value
        self.y=res.params['y'].value
        self.res=res

        self.x_i=(self.p_i-self.p_c)*(self.L_i)**(1/nu)
        
        self.y_i_minus_irrelevant=self.y_i-np.einsum('ij,ik,kj->j',self.phi_1_,self.a[:,1:],self.phi_2_[1:,:])
        return res
    
    def loss_with_drift_GSL(self,p_c,nu,y,n1,n2):
        x_i=(self.p_i-p_c)*(self.L_i)**(1/nu)
        ir_i=self.L_i**(-y) # irrelevant scaling
        j1,j2=np.meshgrid(np.arange(n1+1),np.arange(n2+1),indexing='ij')
        self.X=(x_i**j1.flatten()[:,np.newaxis] * ir_i**j2.flatten()[:,np.newaxis]).T
        Y=self.y_i
        Sigma_inv=np.diag(1/self.d_i**2)
        XX=self.X.T@ Sigma_inv @ self.X
        XY=self.X.T@ Sigma_inv @ Y
        self.beta=np.linalg.inv(XX)@XY
        self.y_i_fitted=self.X @ self.beta
        return (self.y_i-self.y_i_fitted)/self.d_i

    
    def datacollapse_with_drift_GSL(self,n1,n2,p_c=None,nu=None,y=None,**kwargs):
        """fit the coefficient of the taylor expansion of the scaling function, using generalized least square"""
        params=Parameters()
        params.add('p_c',value=p_c,min=0,max=1)
        params.add('nu',value=nu,min=0,max=2)
        params.add('y',value=y,min=0)

        def residual(params):
            return self.loss_with_drift_GSL(params['p_c'],params['nu'],params['y'],n1,n2)
        res=minimize(residual,params,**kwargs)
        self.p_c=res.params['p_c'].value
        self.nu=res.params['nu'].value
        self.y=res.params['y'].value
        self.res=res
        self.x_i=(self.p_i-self.p_c)*(self.L_i)**(1/nu)
        if n2>0:
            self.y_i_minus_irrelevant=self.y_i- self.X.reshape((-1,n1+1,n2+1))[:,:,1:].reshape((-1,(n1+1)*n2))@self.beta.reshape((n1+1,n2+1))[:,1:].flatten()
            self.y_i_irrelevant=self.X.reshape((-1,n1+1,n2+1))[:,:,1:].reshape((-1,(n1+1)*n2))@self.beta.reshape((n1+1,n2+1))[:,1:].flatten()
        else:
            self.y_i_minus_irrelevant=self.y_i
            self.y_i_irrelevant=0
        self.res=res
        return res
        



    def plot_data_collapse(self,ax=None,drift=False,driftcollapse=False):
        x_i=(self.p_i-self.p_c)*(self.L_i)**(1/self.nu)
        # x_i=self.p_i
        if ax is None:
            fig,ax = plt.subplots()
        L_list=self.df.index.get_level_values('L').unique().sort_values().values
        idx_list=[0]+(np.cumsum([self.df.xs(key=L,level='L').shape[0] for L in L_list])).tolist()
        L_dict={L:(start_idx,end_idx) for L,start_idx,end_idx in zip(L_list,idx_list[:-1],idx_list[1:])}
        # color_iter=iter(plt.cm.rainbow(np.linspace(0,1,len(L_list))))
        color_iter = iter(plt.cm.Blues(0.4+0.6*(i/L_list.shape[0])) for i in range(L_list.shape[0]))
        color_r_iter = iter(plt.cm.Reds(0.4+0.6*(i/L_list.shape[0])) for i in range(L_list.shape[0]))
        if drift and driftcollapse:
            ax2=ax.twinx()
            ax2.set_ylabel(r'$y_{irre}$')
        for L,(start_idx,end_idx) in L_dict.items():
            color=next(color_iter)
            if drift:
                if not driftcollapse:
                    ax.errorbar(self.p_i[start_idx:end_idx], self.y_i[start_idx:end_idx], label=f'{L}', color=color, yerr=self.d_i[start_idx:end_idx], capsize=2, fmt='x',linestyle="None")
                    ax.plot(self.p_i[start_idx:end_idx],self.y_i_fitted[start_idx:end_idx],label=f'{L}',color=color)
                else:
                    ax.scatter(x_i[start_idx:end_idx],self.y_i_minus_irrelevant[start_idx:end_idx],label=f'{L}',color=color)
                    
                    color_r=next(color_r_iter)
                    ax2.scatter(x_i[start_idx:end_idx],self.y_i_irrelevant[start_idx:end_idx],label=f'{L}',color=color_r)

            else:
                ax.scatter(x_i[start_idx:end_idx],self.y_i[start_idx:end_idx],label=f'{L}',color=color)
                

        ax.set_ylabel(r'$y_i$')
        if drift:
            if not driftcollapse:
                ax.set_xlabel(r'$p_i$')
                ax.set_title(rf'$p_c={self.p_c:.3f},\nu={self.nu:.3f},y = {self.y:.3f}$')
            else:
                ax.set_xlabel(r'$x_i$')
                ax.set_title(rf'$p_c={self.p_c:.3f},\nu={self.nu:.3f},y = {self.y:.3f}$')
                ax.set_ylabel(r'$y_i-y_{irre}$')
        else:
            ax.set_xlabel(r'$(p_i-p_c)L^{1/\nu}$')
            ax.set_title(rf'$p_c={self.p_c:.3f},\nu={self.nu:.3f}$')
        
        ax.legend()
        ax.grid('on')

        # adder=self.df.index.get_level_values('adder').unique().tolist()[0]
        # print(f'{self.params["Metrics"]}_Scaling_L({L_list[0]},{L_list[-1]})_adder({adder[0]}-{adder[1]}).png')


def grid_search(n1_list,n2_list,p_c,nu,y,verbose=False,**kwargs):
    """grid search for the best n1 and n2
    provided arguments: 
    df=df_0_1
    params={'Metrics':'O',}
    Lmin,Lmax
    p_range=[0.45,0.55]
    """
    # red_chi2_list=np.zeros((len(n1_list),len(n2_list)))
    model_dict={}

    n_list=[(n1,n2) for n1 in n1_list for n2 in n2_list]
    for (n1,n2) in (n_list):
        if verbose:
            print(n1,n2)
        dc=DataCollapse(**kwargs)
        try:
            res0=dc.datacollapse_with_drift_GSL(n1=n1,n2=n2,p_c=p_c,nu=nu,y=y,)
        except:
            print(f'Fitting Failed for (n1={n1},n2={n2})')
        model_dict[(n1,n2)]=dc
        
    return model_dict

def plot_chi2_ratio(model_dict,L1=False):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    color_list=['r','b','c','m','y','k','g']

    n1_list=[]
    n2_list=[]
    for key in model_dict.keys():
        if key[0] not in n1_list:
            n1_list.append(key[0])
        if key[1] not in n2_list:
            n2_list.append(key[1])

    for n2 in n2_list:
        ax.plot(n1_list,[(model_dict[n1,n2].res.redchi if hasattr(model_dict[n1,n2],"res") else np.nan) for n1 in n1_list],label=f'$n_2$={n2}',color=color_list[n2])
        
    ax.set_yscale('log')
    ax.axhline(1,color='k',ls='dotted',lw=0.5)
    ax.legend()

    ax2=ax.twinx()
    for n2 in n2_list:
        if L1:
            ratio=[np.abs(model_dict[n1,n2].y_i_irrelevant/model_dict[n1,n2].y_i_minus_irrelevant).mean() if hasattr(model_dict[n1,n2],"res") else np.nan for n1 in n1_list]
        else:
            ratio=[np.var(model_dict[n1,n2].y_i_irrelevant)/np.var(model_dict[n1,n2].y_i) if hasattr(model_dict[n1,n2],"res") else np.nan for n1 in n1_list]
        ax2.plot(n1_list,ratio,label=f'$n_2$={n2}',color=color_list[n2],ls='--')
        ax2.set_ylim([0,1.05])

    ax.set_xlabel('$n_1$')
    ax.set_ylabel(r'$\chi_{\nu}^2$')
    ax2.set_ylabel('mean(|irre/re|)')
        