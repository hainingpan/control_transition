import os


def load_json(fn):
    import json
    with open(fn, "r") as file:
        data = json.load(file)
    return data

def load_pickle(fn):
    import pickle
    try:
        with open(fn, 'rb') as f:
            data = pickle.load(f)
    except:
        print(f'Error loading {fn}')

    return data

def visualize_dataset(df,xlabel,ylabel,params={'Metrics':'EE',}):
    """Visualize the ensemble size for two axis

    Parameters
    ----------
    df : DataFrame
        DataFrame to check
    xlabel : String
        String for the x-axis
    ylabel : String
        String for the y-axis
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
    filename : String
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
    names : List[String]
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
    fn_template : String
        Template of filename, example: 'MPS_({nu},{de})_L{L}_p{p:.3f}_s{s}_a{a}.json'
    fn_dir_template : String
        Template of directory, example: 'MPS_{nu}-{de}'
    input_params_template : String
        Template of input parameters, example: '{p:.3f} {L} {seed} {ancilla}'
    load_data : Function,
        the function to load data, currently `load_json` and `load_pickle` are supported
    filename : str, optional
        _description_, by default 'params.txt'
    filelist : String, None, optional
        If true, read the `filelist` as a list of existing files, by default None
    load : bool, optional
        Load files into data_dict, by default False
    data_dict : Dict, optional
        The Dictionary to load into, the format should be {'fn':{},...}, by default None
    data_dict_file : String, optional
        The filename of the data_dict, if None, then use Dict provided by data_dict, else try to load file `data_dict_file`, if not exist, create a new Dict and save to disk, by default None
    fn_dir : String, 'auto', optional
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
                    data=load_data(os.path.join(fn_dir,fn))
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
    
    

    