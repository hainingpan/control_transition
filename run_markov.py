from rqc.CT_classical_markov import CT_classical_markov
import numpy as np
from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI
from time import time
import argparse
import pickle
from tqdm import tqdm

def run_markov(inputs):
    L, p, sC, sm = inputs
    ct = CT_classical_markov(L, seed_C=sC, seed=sm)
    tf = 2*L**2
    FDW_list = np.zeros((tf+1, 2))
    FDW_list[0] = ct.FDW(ct.vec)
    Z_list = np.zeros((tf+1, 2))
    Z_list[0] = ct.Z_tensor(ct.vec)
    for i in range(tf):
        ct.vec = ct.random_control(ct.vec, p)
        FDW_list[i+1] = ct.FDW(ct.vec)
        Z_list[i+1] = ct.Z(ct.vec)

    return FDW_list, Z_list

def collect_results(results, p_list, sC_list, sm_list, L):
    rs = np.array(list(results)).reshape((p_list.shape[0], sC_list.shape[0], sm_list.shape[0], 2, 2*L**2+1, 2))
    FDW_map, O_map = rs[..., 0, :, :], rs[..., 1, :, :]
    return FDW_map, O_map

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    print(f'Total size: {size}', flush=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--L', '-L', type=int, default=10, help='System size L')
    parser.add_argument('--sC', '-sC', type=int, default=500, help='Number of seed_C values')
    parser.add_argument('--sm', '-sm', type=int, default=500, help='Number of seed values')
    parser.add_argument('--p', '-p', type=float, nargs=3, default=[0.4, 0.6, 3], 
                        help='Parameters for p in the form [start, stop, num] to generate values with np.linspace')
    
    args = parser.parse_args()
    
    # Set up parameters
    L = args.L
    sC_list = np.arange(args.sC)
    sm_list = np.arange(args.sm)
    
    # Generate p values using np.linspace
    p_list = np.linspace(args.p[0], args.p[1], int(args.p[2]))
    
    # Create inputs
    inputs = [(L, p, sC, sm) for p in p_list for sC in sC_list for sm in sm_list]
    
    st = time()
    
    # Run with MPI
    with MPIPoolExecutor() as executor:
        results = list(tqdm(executor.map(run_markov, inputs), total=len(inputs)))
    
    # Collect and process results
    FDW_map, O_map = collect_results(results, p_list, sC_list, sm_list, L)
    
    # Auto-generate output filename based on L
    output_file = f"markov_L{L}_p{args.p[0]:.2f}.pickle"
    
    # Save results
    with open(output_file, 'wb') as f:
        pickle.dump((FDW_map, O_map), f)
    
    print(f'Time elapsed: {time() - st:.4f}s')
    print(f'Results saved to {output_file}')