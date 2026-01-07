from rqc_sv.APT import APT
import numpy as np
from joblib import Parallel, delayed
from time import time
import argparse
import pickle
from tqdm import tqdm
from fractions import Fraction
import os
import shutil


def get_tf(L):
    """Compute the number of time steps based on system size L."""
    return int(12 * L ** 1.6)


def run_local_storage(inputs):
    """Run simulation and save results to local temp file to avoid OOM."""
    L, p_m, p_f, seed, seed_C, flat_idx, temp_dir = inputs
    apt = APT(L=L, x0=Fraction(2**L-1, 2**L), seed=seed, seed_C=seed_C, seed_vec=None, store_op=False)
    tf = get_tf(L)

    # Use numpy arrays directly instead of lists
    OP_arr = np.zeros(tf, dtype=np.float64)
    OP2_arr = np.zeros(tf, dtype=np.float64)

    for i in range(tf):
        apt.random_circuit(p_m=p_m, p_f=p_f, even=True)
        apt.random_circuit(p_m=p_m, p_f=p_f, even=False)
        result = apt.order_parameter(moment=(1, 2))
        OP_arr[i] = result['OP']
        OP2_arr[i] = result['OP2']

    # Save to temp file using flat_idx for unique naming
    fname = os.path.join(temp_dir, f'result_{flat_idx}.npz')
    np.savez_compressed(fname, OP=OP_arr, OP2=OP2_arr)

    return flat_idx  # Return only the index


def run(inputs):
    L, p_m, p_f, seed, seed_C = inputs
    apt = APT(L=L, x0=Fraction(2**L-1, 2**L), seed=seed, seed_C=seed_C, seed_vec=None, store_op=False)
    tf = get_tf(L)
    OP_list = []
    OP2_list = []
    for i in range(tf):
        apt.random_circuit(p_m=p_m, p_f=p_f, even=True)
        apt.random_circuit(p_m=p_m, p_f=p_f, even=False)
        result = apt.order_parameter(moment=(1, 2))
        OP_list.append(result['OP'])
        OP2_list.append(result['OP2'])
    return {"OP": OP_list, "OP2": OP2_list}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--L', '-L', type=int, default=12, help='Parameters for L')
    parser.add_argument('--es', '-es', default=[1, 10], type=int, nargs=2, help='Ensemble size (default: 10) for measurement outcome.')
    parser.add_argument('--es_C', '-es_C', default=[1, 10], type=int, nargs=2, help='Ensemble size for circuit (default: 10).')
    parser.add_argument('--p_m', '-p_m', type=float, nargs=3, default=[0, 1, 11], help='Parameters for p_m in the form [start, stop, num] to generate values with np.linspace (default: [0, 1, 11]).')
    parser.add_argument('--p_f', '-p_f', type=float, nargs=3, default=[0, 1, 11], help='Parameters for p_f in the form [start, stop, num] to generate values with np.linspace (default: [0, 1, 11]). If the third parameter is -1, then use the same value of p_m')
    parser.add_argument('--n_jobs', '-n', type=int, default=-1, help='Number of parallel jobs (-1 for all available cores, default: -1).')
    parser.add_argument('--no_local_storage', action='store_true', help='Disable local storage mode (keeps results in memory, may OOM for large tf).')
    args = parser.parse_args()

    p_m_list = np.linspace(args.p_m[0], args.p_m[1], int(args.p_m[2]))
    es_list = np.arange(args.es[0], args.es[1])
    es_C_list = np.arange(args.es_C[0], args.es_C[1])

    if args.p_f[2] == -1:
        p_f_list = np.array([-1])
        use_same_pf = True
    else:
        p_f_list = np.linspace(args.p_f[0], args.p_f[1], int(args.p_f[2]))
        use_same_pf = False

    st = time()

    if not args.no_local_storage:
        # Local storage mode: save intermediate results to disk to avoid OOM
        # Create local temp directory with SLURM job ID to avoid conflicts between jobs
        job_id = os.environ.get('SLURM_JOB_ID', os.getpid())
        output_dir = os.environ.get('WORKDIR', '..')
        temp_dir = os.path.join(output_dir, 'control_transition', f'tmp_APT_results_{job_id}')
        os.makedirs(temp_dir, exist_ok=True)
        print(f'Using local storage mode. Temp dir: {temp_dir}')

        # Include flat_idx and temp_dir in inputs
        if use_same_pf:
            inputs = [(args.L, p_m, p_m, idx, idx_C, flat_idx, temp_dir)
                      for flat_idx, (p_m, _, idx, idx_C) in enumerate(
                          (p_m, p_f, idx, idx_C)
                          for p_m in p_m_list
                          for p_f in p_f_list
                          for idx in es_list
                          for idx_C in es_C_list)]
        else:
            inputs = [(args.L, p_m, p_f, idx, idx_C, flat_idx, temp_dir)
                      for flat_idx, (p_m, p_f, idx, idx_C) in enumerate(
                          (p_m, p_f, idx, idx_C)
                          for p_m in p_m_list
                          for p_f in p_f_list
                          for idx in es_list
                          for idx_C in es_C_list)]

        # Run parallel jobs - returns only indices
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(run_local_storage)(input_data) for input_data in tqdm(inputs, desc="Processing")
        )

        # Aggregate using memmap to avoid OOM
        tf = get_tf(args.L)
        shape = (p_m_list.shape[0], np.abs(p_f_list.shape[0]), es_list.shape[0], es_C_list.shape[0], tf)
        total_tasks = len(inputs)
        keys = ['OP', 'OP2']

        # Create memmap files for each metric
        mmap_files = {}
        mmaps = {}
        for key in keys:
            mmap_file = os.path.join(temp_dir, f'{key}.dat')
            mmap_files[key] = mmap_file
            mmaps[key] = np.memmap(mmap_file, dtype='float64', mode='w+', shape=shape)

        # Load one file at a time and write to memmap
        print('Aggregating results...')
        for flat_idx in tqdm(range(total_tasks), desc="Aggregating"):
            fname = os.path.join(temp_dir, f'result_{flat_idx}.npz')
            i_pm, i_pf, i_es, i_es_C = np.unravel_index(flat_idx, (p_m_list.shape[0], np.abs(p_f_list.shape[0]), es_list.shape[0], es_C_list.shape[0]))
            with np.load(fname) as f:
                for key in keys:
                    mmaps[key][i_pm, i_pf, i_es, i_es_C, :] = f[key]
            os.remove(fname)  # Delete temp file immediately

        # Flush and convert to regular arrays for pickle
        data = {}
        for key in keys:
            mmaps[key].flush()
            data[key] = np.array(mmaps[key])  # Copy to regular array
            del mmaps[key]
            os.remove(mmap_files[key])
        data["args"] = args

        # Clean up temp directory
        shutil.rmtree(temp_dir)
        print(f'Cleaned up temp dir: {temp_dir}')

    else:
        # Original mode: keep results in memory
        if use_same_pf:
            inputs = [(args.L, p_m, p_m, idx, idx_C) for p_m in p_m_list for p_f in p_f_list for idx in es_list for idx_C in es_C_list]
        else:
            inputs = [(args.L, p_m, p_f, idx, idx_C) for p_m in p_m_list for p_f in p_f_list for idx in es_list for idx_C in es_C_list]

        results = Parallel(n_jobs=args.n_jobs)(
            delayed(run)(input_data) for input_data in tqdm(inputs, desc="Processing")
        )

        keys = results[0].keys()
        shape = (p_m_list.shape[0], np.abs(p_f_list.shape[0]), es_list.shape[0], es_C_list.shape[0], -1)
        data = {key: np.array([r[key] for r in results]).reshape(shape) for key in keys}
        data["args"] = args

    output_dir = os.environ.get('WORKDIR', '..')
    filename = output_dir + '/control_transition/APT_OP_T/APT_En({:d},{:d})_EnC({:d},{:d})_pm({:.3f},{:.3f},{:.0f})_pf({:.3f},{:.3f},{:.0f})_L{:d}_OP_T.pickle'.format(*args.es, *args.es_C, *args.p_m, *args.p_f, args.L)
    print(f'Saving to: {filename}')

    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    print(f'Saved to: {filename}')
    print(f'Time elapsed: {time()-st:.4f}s')
