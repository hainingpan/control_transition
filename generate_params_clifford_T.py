#!/usr/bin/env python3
import numpy as np
import os

# Output directory where pickle files are saved
output_dir = os.path.join(os.environ.get('WORKDIR', '..'), 'control_transition/Clifford')

# Tunable parameter: p_m values sweep (one p_m per job)
p_m_values = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75])  # 6 values

# Tunable parameter: alpha (power-law exponent)
alpha = 0.5

output_filename = 'params_clifford_T.txt'

# Timing analysis (10 circuits, 1 trajectory each, single core):
# L=16:   2.25s total ->  0.225s per trajectory
# L=32:   9.44s total ->  0.944s per trajectory
# L=64:  66.26s total ->  6.626s per trajectory
# L=128: 896.9s total -> 89.7s per trajectory
# L=256: 1499s per trajectory (from timing test)

# 10-hour limit calculation (1 p_m per job):
# Time per job = (1 p_m * es_batch * es_C_batch * time_per_traj) / 24 cores
# For 10h: es_batch * es_C_batch <= 36000 * 24 / time_per_traj = 864000 / time_per_traj

# Divisors of 2000: 1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 125, 200, 250, 400, 500, 1000, 2000
# Pick largest divisor that fits the time constraint

# Batch configuration:
# - total_es: total trajectory seeds desired
# - total_es_C: total circuit seeds desired
# - es_batch: trajectory seeds per job (must divide total_es)
# - es_C_batch: circuit seeds per job (must divide total_es_C)

batch_config = {
    # L=16: max es*es_C = 864000/0.225 = 3.84M, use 500*500=250k (1 job per p_m)
    16: {'total_es': 500, 'total_es_C': 500, 'es_batch': 500, 'es_C_batch': 500},

    # L=32: max es*es_C = 864000/0.944 = 915k, use 500*500=250k (1 job per p_m)
    32: {'total_es': 500, 'total_es_C': 500, 'es_batch': 500, 'es_C_batch': 500},

    # L=64: max es*es_C = 864000/6.626 = 130k, use 500*100=50k (5 jobs per p_m)
    64: {'total_es': 500, 'total_es_C': 500, 'es_batch': 500, 'es_C_batch': 100},

    # L=128: max es*es_C = 864000/89.7 = 9632, use 500*10=5k (50 jobs per p_m)
    128: {'total_es': 500, 'total_es_C': 500, 'es_batch': 500, 'es_C_batch': 10},

    # L=256: max es*es_C = 864000/1499 = 576, use 125*1=125 (2000 jobs per p_m)
    256: {'total_es': 500, 'total_es_C': 500, 'es_batch': 125, 'es_C_batch': 1},
}

L_values = list(batch_config.keys())

# Build params_list for each (L, p_m, es_batch, es_C_batch) combination
params_list = []

for L in L_values:
    cfg = batch_config[L]
    total_es = cfg['total_es']
    total_es_C = cfg['total_es_C']
    es_batch = cfg['es_batch']
    es_C_batch = cfg['es_C_batch']

    # Verify divisibility
    assert total_es % es_batch == 0, f"L={L}: es_batch={es_batch} must divide total_es={total_es}"
    assert total_es_C % es_C_batch == 0, f"L={L}: es_C_batch={es_C_batch} must divide total_es_C={total_es_C}"

    num_es_batches = total_es // es_batch
    num_es_C_batches = total_es_C // es_C_batch

    for p_m in p_m_values:
        for es_batch_idx in range(num_es_batches):
            es_start = es_batch_idx * es_batch + 1
            es_end = (es_batch_idx + 1) * es_batch + 1

            for es_C_batch_idx in range(num_es_C_batches):
                es_C_start = es_C_batch_idx * es_C_batch + 1
                es_C_end = (es_C_batch_idx + 1) * es_C_batch + 1

                fixed_params = {
                    'L': L,
                    'es_start': es_start,
                    'es_end': es_end,
                    'es_C_start': es_C_start,
                    'es_C_end': es_C_end,
                    'alpha': alpha,
                    'p_m': p_m,
                }

                params_list.append(fixed_params)

# Generate parameters file
with open(output_filename, 'w') as f:
    for fixed_params in params_list:
        # Single p_m value: --p_m {value} {value} 1
        param_str = f"--L {fixed_params['L']} --p_m {fixed_params['p_m']:.3f} {fixed_params['p_m']:.3f} 1 --alpha {fixed_params['alpha']:.1f} --es {fixed_params['es_start']} {fixed_params['es_end']} --es_C {fixed_params['es_C_start']} {fixed_params['es_C_end']}"
        f.write(param_str + '\n')

print(f"Generated {output_filename} with {len(params_list)} parameter sets")
print(f"L values: {L_values}")
print(f"p_m values: {[f'{p:.2f}' for p in p_m_values]} ({len(p_m_values)} values)")
print(f"alpha: {alpha}")

# Print job summary
print("\nJob summary:")
total_all = 0
for L in L_values:
    cfg = batch_config[L]
    num_es_batches = cfg['total_es'] // cfg['es_batch']
    num_es_C_batches = cfg['total_es_C'] // cfg['es_C_batch']
    jobs_per_pm = num_es_batches * num_es_C_batches
    total_jobs = len(p_m_values) * jobs_per_pm
    traj_per_job = cfg['es_batch'] * cfg['es_C_batch']
    total_all += total_jobs
    print(f"  L={L}: {total_jobs} jobs ({len(p_m_values)} p_m x {num_es_batches} es_batches x {num_es_C_batches} es_C_batches)")
    print(f"         es_batch={cfg['es_batch']}, es_C_batch={cfg['es_C_batch']}")
    print(f"         {traj_per_job:,} trajectories per job")
print(f"\nTotal jobs: {total_all}")
