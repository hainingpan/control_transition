import os
import torch
import numpy as np
import json
import orjson
import zipfile
def load_json(fn):
    # with open(fn, "r") as file:
    #     data = json.load(file)
    with open(fn, "rb") as file:
        file_content = file.read()
        data = orjson.loads(file_content)
    return data

def load_zip_json(fn,z):
    return orjson.loads(z.open(fn).read())
def load_torch_pt(fn):
    return torch.load(fn, map_location='cpu')

import pickle
def load_pickle(fn):
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    return data

def visualize_dataset(df,xlabel,ylabel,params={'Metrics':'EE',},**kwargs):
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
    cm=ax.scatter(x,y,100,ensemblesize,marker='.',cmap='inferno',**kwargs)
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
    from itertools import product
    import argparse

    data_dict['fn'].add(filename)
    # data_dict_fn_set.add(filename)

    assert not ('EE' in data and 'SA' in data), f'{filename} is problematic'

    if 'args' in data:
        if isinstance(data['args'],argparse.Namespace):
            iterator=data['args'].__dict__
        elif isinstance(data['args'],dict):
            iterator=data['args']
    else:
        raise ValueError(f'{filename} does not have args')
    
    if filename.split('.')[-1] == 'pickle':
        if not hasattr(data['args'],'p_m') and hasattr(data['args'],'p_f'):
            L_list=np.arange(*data['args'].L)
        if hasattr(data['args'],'p_ctrl'):
            p_ctrl_list=np.round(np.linspace(data['args'].p_ctrl[0],data['args'].p_ctrl[1],int(data['args'].p_ctrl[2])),3)
        if hasattr(data['args'],'p_proj'):
            p_proj_list=np.round(np.linspace(data['args'].p_proj[0],data['args'].p_proj[1],int(data['args'].p_proj[2])),3)
        if hasattr(data['args'],'p_global'):
            p_global_list=np.round(np.linspace(data['args'].p_global[0],data['args'].p_global[1],int(data['args'].p_global[2])),3)
        if hasattr(data['args'],'p_m') and hasattr(data['args'],'p_f'):
            p_m_list=np.round(np.linspace(data['args'].p_m[0],data['args'].p_m[1],int(data['args'].p_m[2])),3)
            p_f_list=np.round(np.linspace(data['args'].p_f[0],data['args'].p_f[1],abs(int(data['args'].p_f[2]))),3)
            es_C_list=np.arange(*data['args'].es_C)
            es_m_list=np.arange(*data['args'].es)
            # T_list=np.arange(1,40*data['args'].L+1)   # for efficiency consideration

    for metric in set(data.keys())-set(['args']):
        # Save each metrics
        if filename.split('.')[-1] == 'pickle':
            if hasattr(data['args'],'p_m') and hasattr(data['args'],'p_f'):
                iteration_list=product(enumerate([data['args'].L]),enumerate(p_m_list),enumerate(p_f_list),enumerate(es_C_list),enumerate(es_m_list))
            elif hasattr(data['args'],'p_global'):
                iteration_list=product(enumerate(L_list),enumerate(p_ctrl_list),enumerate(p_proj_list),enumerate(p_global_list))
            else:
                iteration_list=product(enumerate(L_list),enumerate(p_ctrl_list),enumerate(p_proj_list))
                
            for iteration in iteration_list:
                # Iterate over parameters
                if not hasattr(data['args'],'p_global'):
                    # (L_idx,L),(p_ctrl_idx,p_ctrl),(p_proj_idx,p_proj)=iteration
                    if 'APT' in filename and 'T' in filename:
                        parse_APT_T(data_dict,data,metric,iteration)
                    
                    elif filename.split('.')[-2][-2:]=='_T':
                        parse_T(data_dict,data,metric,iteration)
                    

                    elif isinstance(data[metric],dict):
                        # For singular value, TMI
                        parse_TMI_sv(data_dict,data,metric,iteration)
                        

                    elif data[metric].dim()>=5:
                        # For singular value, EE
                        parse_EE_sv(data_dict,data,metric,iteration)
                    else:
                       parse(data_dict,data,metric,iteration)
                else:
                    parse_global(data_dict,data,metric,iteration)

        elif filename.split('.')[-1] == 'json':
            if not '_sC' in filename:
                if '_DW' in filename or '_T' in filename or '_coherence' in filename:
                # if '_T' in filename or '_coherence' in filename:
                    parse_json_T(data_dict,data,metric,iterator,fixed_params_keys,skip_keys)
                else:
                    parse_json(data_dict,data,metric,iterator,fixed_params_keys,skip_keys)
            else:
                parse_json(data_dict,data,metric,iterator,fixed_params_keys,skip_keys)
        elif filename.split('.')[-1] == 'pt':
            if 'Lx' in data['args']:
                iteration_list = [(data['args'].Lx,data['args'].Ly,data['args'].nshell,data['args'].mu,data['args'].sigma,data['args'].seed0)]
            elif 'sigma' in data['args']:
                iteration_list = [(data['args'].L,data['args'].nshell,data['args'].mu,data['args'].sigma,data['args'].seed0)]
            else:
                iteration_list = [(data['args'].L,data['args'].nshell,data['args'].mu,data['args'].seed0)]
            for iteration in iteration_list:
                parse_pt(data_dict,data,metric,iteration)
def parse_pt(data_dict,data,metric,iteration):
    """parse pytorch tensor"""
    observations=data[metric]
    params=(metric,)+iteration
    data_dict[params]=(observations).numpy()

def parse_T(data_dict,data,metric,iteration):
    """parse data as a function of time T"""
    # For the metrics as a function of T, the data index is (L,p_ctrl,p_proj,time,ensemble)
    (L_idx,L),(p_ctrl_idx,p_ctrl),(p_proj_idx,p_proj)=iteration
    observations=data[metric][L_idx,p_ctrl_idx,p_proj_idx]
    observations=(observations.cpu().numpy())
    for T_idx in range(observations.shape[0]):
        params=(metric,L,p_ctrl,p_proj,T_idx)
        observations_T_idx=observations[T_idx]
        observations_T_idx=observations_T_idx[~np.isnan(observations_T_idx)]
        if params in data_dict:
            data_dict[params]=np.concatenate([data_dict[params],observations_T_idx])
        else:
            data_dict[params]=observations_T_idx
def parse_APT_T(data_dict,data,metric,iteration):
    """parse APT (absorbind transition) data as a function of time T"""
    (L_idx,L),(p_m_idx,p_m),(p_f_idx,p_f),(es_C_idx,es_C),(es_m_idx,es_m)=iteration
    # This is reversed ``es_m_idx,es_C_idx`` for historical reason
    # Now I fix it
    observations=data[metric][p_m_idx,p_f_idx,es_C_idx,es_m_idx]
    params=(metric,L,p_m,p_f,es_C,es_m)
    data_dict[params]=observations

def parse_TMI_sv(data_dict,data,metric,iteration):
    (L_idx,L),(p_ctrl_idx,p_ctrl),(p_proj_idx,p_proj)=iteration
    for key,val in data[metric].items():
        observations=data[metric][key][L_idx,p_ctrl_idx,p_proj_idx]
        # assert not torch.isnan(observations).any(), "The TMI contains NaN values."
        params=(metric+'_'+key,L,p_ctrl,p_proj)
        add_attach_dict(data_dict,params,observations,axis=1,drop_nan=False)

def parse_EE_sv(data_dict,data,metric,iteration):
    (L_idx,L),(p_ctrl_idx,p_ctrl),(p_proj_idx,p_proj)=iteration
    observations=data[metric][L_idx,p_ctrl_idx,p_proj_idx]
    # assert not torch.isnan(observations).any(), "The EE contains NaN values."
    params=(metric,L,p_ctrl,p_proj)
    add_attach_dict(data_dict,params,observations,axis=1,drop_nan=False)

def parse(data_dict,data,metric,iteration):
    # normal parse (original one)
    (L_idx,L),(p_ctrl_idx,p_ctrl),(p_proj_idx,p_proj)=iteration
    observations=data[metric][L_idx,p_ctrl_idx,p_proj_idx]
    params=(metric,L,p_ctrl,p_proj)
    add_attach_dict(data_dict,params,observations)


def parse_global(data_dict,data,metric,iteration):
    # parse data with p_global
    (L_idx,L),(p_ctrl_idx,p_ctrl),(p_proj_idx,p_proj),(p_global_idx,p_global)=iteration
    observations=data[metric][L_idx,p_ctrl_idx,p_proj_idx,p_global_idx]
    params=(metric,L,p_ctrl,p_proj,p_global)
    add_attach_dict(data_dict,params,observations)

def parse_json(data_dict,data,metric,iterator,fixed_params_keys,skip_keys):
    params=(metric,)+tuple(val for key,val in iterator.items() if key != 'seed' and key not in fixed_params_keys and key not in skip_keys)
    if params in data_dict:
        # data_dict[params].append(data[metric])
        data_dict[params]=np.vstack([data_dict[params],data[metric]])
    else:
        data_dict[params]=np.array(data[metric])

from collections import defaultdict
def parse_json_T(data_dict,data,metric,iterator,fixed_params_keys,skip_keys):
    # params=(metric,)+tuple(val for key,val in iterator.items() if key != 'seed' and key not in fixed_params_keys and key not in skip_keys)
    params=(metric,iterator['L'],iterator['p_ctrl'],iterator['p_proj'])
    observations=data[metric]
    coherence_flag=isinstance(observations[0],list)
    
    if coherence_flag:
        # used for coherence with shape (L+1,L+1,T,)
        observations=np.array(observations)
        T_f=observations.shape[-1]
    else:
        # used for DW with shape (T,)
        T_f=len(observations)

    for T_idx in range(T_f):
        params_T=(*params, T_idx)
        if coherence_flag:
            observations_T_idx=observations[...,T_idx]
        else:
            observations_T_idx=observations[T_idx]
        if params_T in data_dict:
            data_dict[params_T].append(observations_T_idx)
        else:
            data_dict[params_T]=[observations_T_idx]

def add_attach_dict(data_dict,params,observations,axis=0,drop_nan=True):
    """if not in dict, add it
    if already in dict, atttach it
    if drop_nan: drop nan:
        else: replace nan with 0 (for the use of singular value)
    """
    if torch.is_tensor(observations):
        observations=observations.cpu().numpy()
        if drop_nan:
            observations=observations[~np.isnan(observations)]
        else:
            observations=np.nan_to_num(observations)
    if params in data_dict:
        if axis==0:
            data_dict[params]=np.concatenate((data_dict[params],(observations)))
        elif axis==1:
            data_dict[params]=np.vstack([data_dict[params],observations])
    else:
        data_dict[params]=observations


def tuple_to_string_key(t):
    return "_".join(map(str, t)) 
        
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

def convert_pd_0(data_dict,names,threshold):
    """Convert the dictionary to a pandas witha a threshold for compute the 0th Renyi entropy"""
    import pandas as pd

    data_dict_0={}

    for key, val in data_dict.items():
        if key=='fn':
            data_dict_0[key]=val
        elif 'O' in key:
            data_dict_0[key]=val
        elif 'EE' == key[0]:
            params=key[1:]
            if ('EE',)+params not in data_dict_0:
                EE=entropy(data_dict[('EE',)+params],threshold,n=0)
                data_dict_0[('EE',)+params]=EE[~np.isnan(EE)]
        elif 'TMI' in key[0]:
            params=key[1:]
            if ('TMI',)+params not in data_dict_0:
                TMI=tripartite_mutual_information(
                    S_A=data_dict[('TMI_S_A',)+params],
                    S_B=data_dict[('TMI_S_B',)+params],
                    S_C=data_dict[('TMI_S_C',)+params],
                    S_AB=data_dict[('TMI_S_AB',)+params],
                    S_AC=data_dict[('TMI_S_AC',)+params],
                    S_BC=data_dict[('TMI_S_BC',)+params],
                    S_ABC=data_dict[('TMI_S_ABC',)+params],
                    threshold=threshold,n=0)
                data_dict_0[('TMI',)+params]=TMI[~np.isnan(TMI)]
    df=convert_pd(data_dict_0,names)
    return df

def entropy(sv,threshold,n=0,postprocess='drop'):
    """compute n-th Renyi entropy from the singular value, the first axis is the ensemble and the second axis all singular value
    postprocess: 
    None: do nothing
    drop: drop non-normalized state
    enforce: set this to be 1
    """
    if n==0:
        S0=np.log(np.count_nonzero((sv>threshold),axis=1))
        # handle some issue that singular values are not normalized, i.e., sum(sv**2) != 1
        if postprocess is None:
            mask=slice(None)
        elif postprocess == 'drop':
            mask=(np.abs(np.sum(sv**2,axis=1)-1)>5.5e-15)
            S0[mask]=np.nan
        elif postprocess == 'enforce':
            # technically this is not correct
            mask=(np.abs(np.sum(sv**2,axis=1)-1)>5.5e-15)
            S0[mask]=0
        return S0
    else:
        raise NotImplementedError("Renyi entropy for n>0 is not yet implemented")

def tripartite_mutual_information(S_A,S_B,S_C,S_AB,S_AC,S_BC,S_ABC,threshold, n=0,postprocess='drop'):
    if n==0:
        return entropy(S_A,threshold,n=0,postprocess=postprocess)+entropy(S_B,threshold,n=0,postprocess=postprocess)+entropy(S_C,threshold,n=0,postprocess=postprocess)-entropy(S_AB,threshold,n=0,postprocess=postprocess)-entropy(S_AC,threshold,n=0,postprocess=postprocess)-entropy(S_BC,threshold,n=0,postprocess=postprocess)+entropy(S_ABC,threshold,n=0,postprocess=postprocess)
    else:
        raise NotImplementedError("tripartite MI for n>0 is not yet implemented")


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
    zip_fn=None,
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
            if data_dict_fn.split('.')[-1] == 'pickle':
                with open(data_dict_fn,'rb') as f:
                    data_dict=pickle.load(f)
            # if data_dict_fn.split('.')[-1] == 'json':
            #     with open(data_dict_fn,'rb') as f:
            #         data_dict=orjson.loads(f.read())
            #     data_dict['fn']=set(data_dict['fn'])
        else:
            print(f'Creating new data_dict {data_dict_fn}')
            data_dict={'fn':set()}

    if zip_fn is not None:
        z=zipfile.ZipFile(zip_fn, 'r')
        

    if load:
        if zip_fn is None:
            all_fns=set(os.listdir(fn_dir))
        else:
            all_fns=set(z.namelist())
    else:
        if filelist is None:
            all_fns=set(os.listdir(fn_dir))
        else:
            with open(filelist,'r') as f:
                all_fns=set(f.read().split('\n'))

    for input0 in tqdm(inputs,mininterval=1,desc='generate_params',total=total):
        dict_params={key:val for key,val in zip(vary_params.keys(),input0)}
        dict_params.update(fixed_params)
        fn=eval(f"f'{fn_template}'", {},  {**locals(),**dict_params})

        if load:
            if fn not in data_dict['fn']:
                fn_fullpath=os.path.join(fn_dir,fn)
                if fn in (all_fns):
                    try:
                        if zip_fn is None:
                            data=load_data(fn_fullpath)
                        else:
                            data=load_data(fn,z)
                    except:
                        print(f'Error loading {fn}')
                        continue
                    add_to_dict(data_dict,data,fn,fixed_params_keys=fixed_params.keys())
        else:
            file_exist = fn in all_fns
            
            if not file_exist:
                params_text.append(eval(f"f'{input_params_template}'", {},  {**locals(),**dict_params}))
            elif exist:
                params_text.append(fn)
    if load:
        if data_dict_file is not None:
            if data_dict_fn.split('.')[-1] == 'pickle':
                with open(data_dict_fn,'wb') as f:
                    pickle.dump(data_dict,f)
            # if data_dict_fn.split('.')[-1] == 'json':
            #     data_dict['fn']=list(data_dict['fn'])
            #     with open(data_dict_fn,'wb') as f:
            #         f.write(orjson.dumps(data_dict))
        return data_dict
    else:
        if filename is not None:
            with open(filename,'a') as f:
                f.write('\n'.join(params_text)+'\n')
        return params_text
    if zip_fn is not None:
        zip_fn.close()


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
    errorbar=False,
    capsize=3,
    capthick=None,
    lw=1,
    colormap=None,
    **kwargs,
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
    ylabel_name={'O':rf'${op_str[method]}{{\langle O \rangle}}$','EE':rf'${op_str[method]}{{ S_{{L/2}} }}$','TMI':rf'${op_str[method]}{{I_3}}$','SA':rf'${op_str[method]}{{ S_{{anc}} }}$','max_bond':rf'${op_str[method]}{{\chi}}$','MI':rf'${op_str[method]}{{I_2}}$'}
    df=df.xs(params.values(),level=list(params.keys()))
    if L_list is None:
        L_list=np.sort(df.index.get_level_values('L').unique())
    if colormap is None:
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
            ax.errorbar(x[arg_sort],dd_sort,yerr=dd_se[arg_sort],label=f'L={L}',lw=lw,color=colormap.__next__(),capsize=capsize,**kwargs)
        else:
            ax.plot(x[arg_sort],dd_sort,'.-',label=f'L={L}',lw=lw,color=colormap.__next__(),**kwargs)
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
    import numpy as np
    

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
    df,
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
    df : DataFrame
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
    import numpy as np
    fig,ax=plt.subplots(figsize=(6.8,
    5))
    params={'Metrics':metrics,}
    params.update(fixed_params)
    
    plot_line(df,params=params,ax=ax,L_list=L_list,yscale=yscale,ylim=ylim,errorbar=errorbar,method=method,x_name=x_name)
    ax.grid('on')
    ax.set_xlim(0,0.6)
    if inset1:
        plot_inset(df,ax,xlim=xlim1,ylim=ylim1,ax_inset_pos=ax_inset_pos1,params=params,L_list=L_list,yscale=yscale,method=method,x_name=x_name)

    if inset2:
        plot_inset(df,ax,xlim=xlim2,ylim=ylim2,ax_inset_pos=ax_inset_pos2,params=params,L_list=L_list,yscale=yscale,method=method,x_name=x_name)
    
    fstr= lambda template: eval(f"f'{template}'")

    if filename is not None:
        if filename== 'auto':
            method_name={np.mean:'mean',np.var:'var'}
            filename= eval(f"f'{filename_template}'", {},  locals())
        print(filename)
        # plt.subplots_adjust(left=(.8)/fig.get_size_inches()[0],right=1-(.1)/fig.get_size_inches()[0],bottom=.5/fig.get_size_inches()[1],top=1-.2/fig.get_size_inches()[1])
        # fig.savefig(os.path.join(dirpath,filename),)

