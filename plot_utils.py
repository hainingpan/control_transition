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
    for key in set(data.keys())-set(['args']):
        # params=(key,data['args']['ancilla'],data['args']['L'],data['args']['p'])
        if isinstance(data['args'],argparse.Namespace):
            iterator=data['args'].__dict__
        elif isinstance(data['args'],dict):
            iterator=data['args']
        
        if 'offset' in iterator and iterator['offset']>0:
            print(iterator)

        params=(key,)+tuple(val for key,val in iterator.items() if key != 'seed' and key not in fixed_params_keys and key not in skip_keys)
        if params in data_dict:
            data_dict[params].append(data[key])
        else:
            data_dict[params]=[data[key]]

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
        The Dictionary to load into, the format should be {'fn':{},...}, by default None
    data_dict_file : str, optional
        The filename of the data_dict, if None, then use Dict provided by data_dict, else try to load file `data_dict_file`, if not exist, create a new Dict and save to disk, by default None
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
        fn_dir=fn_dir_template.format(**fixed_params)
    
    inputs=product(*vary_params.values())
    vary_params.values()
    total=np.product([len(val) for val in vary_params.values()])


    if data_dict_file is not None:
        data_dict_fn=os.path.join(fn_dir,data_dict_file.format(**fixed_params))
        if os.path.exists(data_dict_fn):
            with open(data_dict_fn,'rb') as f:
                data_dict=pickle.load(f)
        else:
            data_dict={'fn':set()}

    for input0 in tqdm(inputs,mininterval=1,desc='generate_params',total=total):
        dict_params={key:val for key,val in zip(vary_params.keys(),input0)}
        dict_params.update(fixed_params)
        fn=fn_template.format(**dict_params)

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
    
    def loss_shift(self,p_c,nu,omega,a,b,c,d,):
        x_i=(self.p_i-p_c)*(self.L_i)**(1/nu)
        y_var=self.d_i**2
        self.y_i_fitted=a+b*x_i+c*x_i**2+d/self.L_i**omega
        return self.chi2(self.y_i_fitted,self.y_i,y_var)

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
        func=lambda x: self.loss(torch.tensor([x[0]]),torch.tensor([x[1]]),MLE=False).item()
        res=scipy.optimize.minimize(func,[self.p_c.item(),self.nu.item()],method='Nelder-Mead',bounds=[(0,1),(0,2)])
        # res=scipy.optimize.minimize(func,[self.p_c.item(),self.nu.item()],method='L-BFGS-B',bounds=[(0,1),(0,5)])
        # 'L-BFGS-B',bounds=[(0,1),(0,5)]
        Hessian= torch.tensor(torch.autograd.functional.hessian(self.loss,(torch.tensor(res.x[0]),torch.tensor(res.x[1]))))
        se=torch.sqrt(torch.diag(torch.inverse(Hessian)))
        self.p_c=torch.tensor([res.x[0]])
        self.nu=torch.tensor([res.x[1]])
        return res,res.fun*2/(self.y_i.shape[0]-2),se

    def optimize_shift(self,omega,a,b,c,d,tolerance=1e-10,):
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

            loss_ = self.loss_shift(p_c_transformed, nu_transformed,omega,a,b,c,d)
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

    def optimize_shift_scipy(self,omega,a,b,c,d):
        # omega,a,b,c,d=
        func=lambda x: self.loss_shift(*tuple(x),d=d).item()
        res=scipy.optimize.minimize(func,[self.p_c.item(),self.nu.item(),omega,a,b,c],method='Nelder-Mead')
        Hessian= torch.tensor(torch.autograd.functional.hessian(lambda x: self.loss_shift(*x),torch.tensor(res.x)))
        se=torch.sqrt(torch.diag(torch.inverse(Hessian)))
        self.p_c=torch.tensor([res.x[0]])
        self.nu=torch.tensor([res.x[1]])
        return res,res.fun*2/(self.y_i.shape[0]-7),se

    def plot_loss(self):
        if hasattr(self, 'loss_history'):
            fig,ax=plt.subplots()
            ax.plot(self.loss_history,'.-')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('O')
    
    def plot_data_collapse(self,ax=None):
        x_i=(self.p_i-self.p_c)*(self.L_i)**(1/self.nu)
        # x_i=self.p_i
        if ax is None:
            fig,ax = plt.subplots()
        L_list=self.df.index.get_level_values('L').unique().sort_values().values
        idx_list=[0]+(np.cumsum([self.df.xs(key=L,level='L').shape[0] for L in L_list])).tolist()
        L_dict={L:(start_idx,end_idx) for L,start_idx,end_idx in zip(L_list,idx_list[:-1],idx_list[1:])}
        for L,(start_idx,end_idx) in L_dict.items():
            ax.scatter(x_i.detach().numpy()[start_idx:end_idx],self.y_i.detach().numpy()[start_idx:end_idx],label=f'{L}')
            # ax.plot(x_i.detach().numpy()[start_idx:end_idx],self.y_i_fitted.detach().numpy()[start_idx:end_idx],label=f'{L}')
        ax.set_xlabel(r'$(p_i-p_c)L^{1/\nu}$')
        ax.set_ylabel(r'$y_i$')
        ax.legend()
        ax.grid('on')
        ax.set_title(rf'$p_c={self.p_c.item():.3f},\nu={self.nu.item():.3f}$')

        # adder=self.df.index.get_level_values('adder').unique().tolist()[0]
        # print(f'{self.params["Metrics"]}_Scaling_L({L_list[0]},{L_list[-1]})_adder({adder[0]}-{adder[1]}).png')
        
    
    def plot_line(self):
        fig,ax=plt.subplots()
        ax.plot(self.p_i,self.y_i)
