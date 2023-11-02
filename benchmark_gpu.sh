#!/bin/bash
L_list=( $(seq 8 2 26) )
es_list=( 921600 230400 57600 14400 3600 900 225 56 14 3)

# Ensure both arrays have the same length
if [ "${#L_list[@]}" -ne "${#es_list[@]}" ]; then
    echo "L_list and es_list have different lengths!"
    exit 1
fi

# Get the last index of the array
end_index=$((${#L_list[@]} - 1))

# Loop through the indices of the lists in reverse
for (( idx=$end_index; idx>=0; idx-- )); do
    L=${L_list[$idx]}
    es=${es_list[$idx]}

    echo "L=$L, es=$es"
    python benchmark_gpu.py -L $L -xj "1/3,2/3" -es $es -seed 0
done

