import h5py
import numpy as np
import os
from tqdm import tqdm

"""quick script to concatenate two 1000 sample datasets into a 2000 sample dataset"""

dirpath='.'
L=18
f=[h5py.File(os.path.join(dirpath,f'CT_En1000_pctrl(0.00,1.00,21)_pproj(0.00,0.00,1)_L({L},{L+2},2)_xj(0)_seed{s}_64_wf.hdf5')) for s in range(2)]

fname_out=os.path.join(dirpath,f'CT_En2000_pctrl(0.00,1.00,21)_pproj(0.00,0.00,1)_L({L},{L+2},2)_xj(0)_seed0_64_wf.hdf5')

chunk_size=50
total_chunk=2000//chunk_size
with h5py.File(fname_out,'w') as f_out:
    for key in f[0].keys():
        new_shape = list(f[0][key].shape) 
        new_shape[-2]*=2
        new_shape=tuple(new_shape)
        dset_out=f_out.create_dataset(key,shape=new_shape,dtype=f[0][key].dtype)
        if 'wf' not in key:
            dset_out[:]=np.concatenate([f[i][key] for i in range(2)],axis=-2)
        else:
            for i in tqdm(range(total_chunk)):
                start=i*chunk_size
                end=(i+1)*chunk_size
                idx=0
                if end > 1000:
                    start -= 1000
                    end -= 1000
                    idx +=1
                dset_out[...,start:end,:]=f[idx][key][...,start:end,:]
