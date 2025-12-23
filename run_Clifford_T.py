from rqc_sv.clifford import Clifford
import numpy as np
from joblib import Parallel, delayed
from time import time
import argparse
import pickle
from tqdm import tqdm
import os
import shutil


def run_local_storage(inputs):
    """Run simulation and save results to local temp file to avoid OOM."""
    L, p_m, alpha, seed, seed_C, flat_idx, temp_dir = inputs
    cliff = Clifford(L=L, seed=seed, seed_C=seed_C, store_op=False, alpha=alpha)
    tf = int(256**1.62)

    # Use numpy arrays directly instead of lists
    OP_arr = np.zeros(tf + 1, dtype=np.float64)
    OP2_arr = np.zeros(tf + 1, dtype=np.float64)
    EE_arr = np.zeros(tf + 1, dtype=np.float64)
    coherence_arr = np.zeros(tf + 1, dtype=np.float64)

    OP_arr[0] = cliff.OP()
    OP2_arr[0] = cliff.OP2_adaptive(p_m)
    EE_arr[0] = cliff.half_system_entanglement_entropy()
    coherence_arr[0] = cliff.quantum_L1_coherence()

    for i in range(tf):
        cliff.random_circuit(p_m=p_m)
        OP_arr[i+1] = cliff.OP()
        OP2_arr[i+1] = cliff.OP2_adaptive(p_m)
        coherence_arr[i+1] = cliff.quantum_L1_coherence()
        EE_arr[i+1] = cliff.half_system_entanglement_entropy()

    # Save to temp file using flat_idx for unique naming
    fname = os.path.join(temp_dir, f'result_{flat_idx}.npz')
    np.savez_compressed(fname, OP=OP_arr, OP2=OP2_arr, EE=EE_arr, coherence=coherence_arr)

    return flat_idx  # Return only the index


def run(inputs):
    L, p_m, alpha, seed, seed_C = inputs
    cliff = Clifford(L=L, seed=seed, seed_C=seed_C, store_op=False, alpha=alpha)
    # tf = int(L**1.62)
    tf = int(256**1.62)
    OP_list = []
    OP2_list = []
    EE_list = []
    coherence_list = []

    OP_list.append(cliff.OP())
    OP2_list.append(cliff.OP2_adaptive(p_m))
    EE_list.append(cliff.half_system_entanglement_entropy())
    coherence_list.append(cliff.quantum_L1_coherence())
    for i in range(tf):
        cliff.random_circuit(p_m=p_m)
        OP_list.append(cliff.OP())
        OP2_list.append(cliff.OP2_adaptive(p_m))
        coherence_list.append(cliff.quantum_L1_coherence())
        EE_list.append(cliff.half_system_entanglement_entropy())
    # OP_list.append(cliff.OP())
    # OP2_list.append(cliff.OP2_adaptive(p_m))
    # EE_list.append(cliff.half_system_entanglement_entropy())
    return {"OP": OP_list, "OP2": OP2_list, "EE": EE_list, "coherence": coherence_list}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--L', '-L', type=int, default=12, help='System size (default: 12).')
    parser.add_argument('--es', '-es', default=[1, 10], type=int, nargs=2, help='Ensemble size range [start, end) (default: [1, 10]).')
    parser.add_argument('--es_C', '-es_C', default=[1, 10], type=int, nargs=2, help='Ensemble size for circuit [start, end) (default: [1, 10]).')
    parser.add_argument('--p_m', '-p_m', type=float, nargs=3, default=[0, 1, 11], help='Parameters for p_m in the form [start, stop, num] (default: [0, 1, 11]).')
    parser.add_argument('--alpha', '-a', type=float, default=2.0, help='Power-law exponent for variable-range control (default: 2.0).')
    parser.add_argument('--n_jobs', '-n', type=int, default=-1, help='Number of parallel jobs (-1 for all available cores, default: -1).')
    parser.add_argument('--no_local_storage', action='store_true', help='Disable local storage mode (keeps results in memory, may OOM for large tf).')
    args = parser.parse_args()

    p_m_list = np.linspace(args.p_m[0], args.p_m[1], int(args.p_m[2]))
    es_list = np.arange(args.es[0], args.es[1])
    es_C_list = np.arange(args.es_C[0], args.es_C[1])

    st = time()

    if not args.no_local_storage:
        # Local storage mode: save intermediate results to disk to avoid OOM
        # Create local temp directory with SLURM job ID to avoid conflicts between jobs
        job_id = os.environ.get('SLURM_JOB_ID', os.getpid())
        temp_dir = os.path.join(os.getcwd(), f'tmp_results_{job_id}')
        os.makedirs(temp_dir, exist_ok=True)
        print(f'Using local storage mode. Temp dir: {temp_dir}')

        # Include flat_idx and temp_dir in inputs
        inputs = [(args.L, p_m, args.alpha, idx, idx_C, flat_idx, temp_dir)
                  for flat_idx, (p_m, idx, idx_C) in enumerate(
                      (p_m, idx, idx_C)
                      for p_m in p_m_list
                      for idx in es_list
                      for idx_C in es_C_list)]

        # Run parallel jobs - returns only indices
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(run_local_storage)(input_data) for input_data in tqdm(inputs, desc="Processing")
        )

        # Aggregate using memmap to avoid OOM
        tf = int(256**1.62)
        shape = (p_m_list.shape[0], es_list.shape[0], es_C_list.shape[0], tf + 1)
        total_tasks = len(inputs)
        keys = ['OP', 'OP2', 'EE', 'coherence']

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
            i_pm, i_es, i_es_C = np.unravel_index(flat_idx, (p_m_list.shape[0], es_list.shape[0], es_C_list.shape[0]))
            with np.load(fname) as f:
                for key in keys:
                    mmaps[key][i_pm, i_es, i_es_C, :] = f[key]
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
        inputs = [(args.L, p_m, args.alpha, idx, idx_C)
                  for p_m in p_m_list
                  for idx in es_list
                  for idx_C in es_C_list]

        results = Parallel(n_jobs=args.n_jobs)(
            delayed(run)(input_data) for input_data in tqdm(inputs, desc="Processing")
        )

        # Automatically unpack all metrics from results
        keys = results[0].keys()
        data = {key: np.array([r[key] for r in results]).reshape((p_m_list.shape[0], es_list.shape[0], es_C_list.shape[0], -1)) for key in keys}
        data["args"] = args

    output_dir = os.environ.get('WORKDIR', '..')
    filename = f'{output_dir}/control_transition/Clifford/Clifford_En({args.es[0]:d},{args.es[1]:d})_EnC({args.es_C[0]:d},{args.es_C[1]:d})_pm({args.p_m[0]:.3f},{args.p_m[1]:.3f},{args.p_m[2]:.0f})_alpha{args.alpha:.1f}_L{args.L:d}_T.pickle'
    print(f'Saving to: {filename}')

    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    print(f'Saved to: {filename}')
    print(f'Time elapsed: {time()-st:.4f}s')

# OMP_NUM_THREADS=1 python run_Clifford_T.py -L 128 -es 1 2 -es_C 1 50 -p_m 0.5 .9 11 -a 3.5 -n -1