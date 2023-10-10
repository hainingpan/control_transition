from CT import CT_tensor
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import time

# def run_tensor():
#     ct=CT_tensor(L=16,gpu=True,seed=list(range(100)),x0=None,ancilla=True,history=False)
#     ct.control_map(ct.vec,bL=0)
#     # for _ in range(2):
#     #     ct.random_control(1.0,0)

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,record_shapes=True,on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/Bernoulli'),with_stack=True) as prof:
#     run_tensor()


def run_tensor(gpu,complex128):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(torch.cuda.get_device_name(),flush=True)


    st0=time.time()
    # ct=CT_tensor(L=18,gpu=True,seed=list(range(200)),x0=None,ancilla=False,history=False,complex128=True)
    ct=CT_tensor(L=16,gpu=gpu,seed=0,x0=None,ancilla=True,history=False,complex128=complex128,ensemble=2000)
    init_timestamp=time.time()
    for _ in range(ct.L**2//2):
        ct.random_control(1,0)
        torch.cuda.empty_cache()
    evo_timestamp=time.time()
    _=ct.order_parameter()
    OP_timestamp=time.time()
    # _=ct.half_system_entanglement_entropy()
    _=ct.von_Neumann_entropy_pure([ct.L],)
    EE_timestamp=time.time()
    _=ct.tripartite_mutual_information(np.arange(ct.L//4),np.arange(ct.L//4)+ct.L//4,np.arange(ct.L//4)+ct.L//4*2)
    TMI_timestamp=time.time()
    peak_memory_MB = torch.cuda.max_memory_allocated()/ (1024 ** 2)
    return init_timestamp-st0,evo_timestamp-init_timestamp,OP_timestamp-evo_timestamp,EE_timestamp-OP_timestamp,TMI_timestamp-EE_timestamp,peak_memory_MB, ct.vec.numel()*ct.vec.element_size()/1024**2
 
for gpu in [torch.cuda.is_available(),]:
    for complex128 in [False,]:
        print(f'GPU:{gpu}, complex128: {complex128}')
        print('Initialization Time: {:.2f} s\nEvolution Time:{:.2f} s\nOP Time:{:.2f} s\nEE Time:{:.2f} s\nTMI Time:{:.2f} s\nPeak GPU RAM:{:.2f} MB\nTensor GPU Ram:{:.2f} MB'.format(*run_tensor(gpu,complex128)))

