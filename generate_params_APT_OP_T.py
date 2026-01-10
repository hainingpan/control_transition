#!/usr/bin/env python3
import numpy as np
import os
from rqc import generate_params
from pathlib import Path

# Output directory where pickle files are saved
output_dir = os.path.join(os.environ.get('WORKDIR', '..'), 'control_transition/APT_OP_T')

# Tunable parameter: p_m values sweep (one p_m per job)
# p_m_values = [0.085,0.087,0.089,0.09,0.091,0.093,0.095, ]
p_m_values = [0.05,0.06,0.07,0.08,0.10,0.11,0.12, ]

# p_f parameters: [start, end, num] for linspace format
# [1.0, 1.0, 1] means p_f = linspace(1.0, 1.0, 1) = [1.0]
pf = [1.0, 1.0, 1]

output_filename = 'params_APT_OP_T.txt'

def scramble_params_file(path):
    """Randomly reorder generated parameter lines to vary job order."""
    if not os.path.isfile(path):
        return 0
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        return 0
    rng = np.random.default_rng()
    rng.shuffle(lines)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return len(lines)

# Batch configuration:
# - total_es: total trajectory seeds desired (500 for all L)
# - total_es_C: total circuit seeds desired (500 for all L)
# - es_batch: trajectory seeds per job (500 = all in one batch)
# - es_C_batch: circuit seeds per job (per_esC0 value)

batch_config = {
    12: {'total_es': 500, 'total_es_C': 500, 'es_batch': 500, 'es_C_batch': 50},
    14: {'total_es': 500, 'total_es_C': 500, 'es_batch': 500, 'es_C_batch': 25},
    16: {'total_es': 500, 'total_es_C': 500, 'es_batch': 500, 'es_C_batch': 10},
    # 18: {'total_es': 500, 'total_es_C': 500, 'es_batch': 500, 'es_C_batch': 2},
    # 20: {'total_es': 500, 'total_es_C': 500, 'es_batch': 500, 'es_C_batch': 2},
    # 22: {'total_es': 500, 'total_es_C': 500, 'es_batch': 500, 'es_C_batch': 1},
    # L=24: ~3.85h per traj×circuit, 100 traj × 1 circuit per job ≈ 16h
    # 24: {'total_es': 500, 'total_es_C': 500, 'es_batch': 100, 'es_C_batch': 1},
}

L_values = list(batch_config.keys())

# Build params_list for each L value (using vary_params to span all batch combinations)
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

    # Pre-compute all es_range and es_C_range tuples
    es_ranges = [(es_batch_idx * es_batch + 1, (es_batch_idx + 1) * es_batch + 1)
                 for es_batch_idx in range(num_es_batches)]
    es_C_ranges = [(es_C_batch_idx * es_C_batch + 1, (es_C_batch_idx + 1) * es_C_batch + 1)
                   for es_C_batch_idx in range(num_es_C_batches)]

    fixed_params = {
        'L': L,
        'pf1': pf[0],
        'pf2': pf[1],
        'pf3': pf[2],
    }

    vary_params = {
        'p_m': p_m_values,
        'es_range': es_ranges,
        'es_C_range': es_C_ranges,
    }

    params_list.append((fixed_params, vary_params))

# Generate parameters for each L value (one call per L)
for fixed_params, vary_params in params_list:
    generate_params(
        fixed_params=fixed_params,
        vary_params=vary_params,
        fn_template='APT_En({es_range[0]},{es_range[1]})_EnC({es_C_range[0]},{es_C_range[1]})_pm({p_m:.3f},{p_m:.3f},1)_pf(1.000,1.000,1)_L{L}_OP_T.pickle',
        fn_dir_template=Path(os.environ.get('WORKDIR', '..'))/'control_transition'/'APT_OP_T',
        input_params_template='--L {L} --p_m {p_m:.3f} {p_m:.3f} 1 --p_f {pf1:.1f} {pf2:.1f} {pf3:d} --es {es_range[0]} {es_range[1]} --es_C {es_C_range[0]} {es_C_range[1]}',
        load_data=lambda x: None,
        filename=output_filename,
        load=False,
        data_dict=None,
    )

# Scramble the final parameter order to distribute jobs more randomly
scrambled = scramble_params_file(output_filename)
print(f"Scrambled {scrambled} entries in {output_filename}")

print(f"Generated {output_filename}")
print(f"L values: {L_values}")
print(f"p_m values: {[f'{p:.3f}' for p in p_m_values]} ({len(p_m_values)} values)")
print(f"p_f: {pf}")

# Calculate total jobs
total_jobs = 0
for L in L_values:
    cfg = batch_config[L]
    num_es_batches = cfg['total_es'] // cfg['es_batch']
    num_es_C_batches = cfg['total_es_C'] // cfg['es_C_batch']
    jobs_for_L = len(p_m_values) * num_es_batches * num_es_C_batches
    print(f"L={L}: {jobs_for_L} jobs ({num_es_C_batches} es_C batches)")
    total_jobs += jobs_for_L
print(f"Total jobs: {total_jobs}")
