#!/bin/bash

# Configuration
# START_NUM=5092 # this is for params_0_anc.txt
START_NUM=1 # this is for params_1_3_ent.txt
END_NUM=2
TEMPLATE_FILE="run_pytorch_sweep2.sh"

# Function to submit job with a given ARR value
submit_job() {
  local arr_value=$1
  local job_file="${arr_value}.sh"

  # Create a new job file from template by replacing ARR placeholder
  sed "s/ARRARIDX/${arr_value}/g" "${TEMPLATE_FILE}" > "${job_file}"
  chmod +x "${job_file}"

  # Submit the job
  sbatch "${job_file}"
}

# Main loop
for (( ARR=START_NUM; ARR<=END_NUM; ARR++ )); do
  while : ; do
    # Check the number of pending jobs
    echo "checking ${ARR}"
    num_pending=$(squeue --states=PENDING -u hp636| tail -n +2 | wc -l)
    # num_pending=$(squeue -u hp636| tail -n +2 | wc -l)

    # If pending jobs are less than 150, try to submit a new job
    if (( num_pending < 150 )); then
      if submit_job $ARR; then
        echo "$(date +"%Y-%m-%d %T"):Job for ARR=${ARR} submitted."
        sleep 1
        break # Job submitted, exit the loop to submit next job
      else
        echo "$(date +"%Y-%m-%d %T"): Submission failed for ARR=${ARR}. Retrying..."
      fi
    else
      echo "$(date +"%Y-%m-%d %T"): There are currently ${num_pending} pending jobs. Waiting to submit job for ARR=${ARR}..."
    fi
    sleep 60 # Wait for 5 seconds before checking again
  done
done
