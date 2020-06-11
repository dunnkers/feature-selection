#!/bin/sh
# 
# Submits computation jobs for a chosen data collection. 
# 

./scripts/upload-files.sh

echo 
echo "Select a dataset collection to submit:"
select data_collection in ./data/*.csv; do
    [[ -n $data_collection ]] || { echo "Invalid choice,\
        choose again or press CTRL+C to quit job submission." >&2; continue; }
    break
done

# Parse data collection file to determine amount of datasets
n_datasets=$(cat $data_collection | sed '/^\s*$/d' | wc -l | grep -o "[0-9]\+")
((n_datasets-=1)) # skip csv header
job_name=${data_collection##*/}
echo "Submitting $n_datasets datasets from '${job_name%.csv}'..."

# Ask for number of desired tasks per job
read -p "CPU's (tasks) per node [24]:" cpus
cpus=${cpus:-24}

# Ask for desired amount of datasets to run per node
echo ""
echo "How many datasets would you like to assign to 1 node? Amount should probably \
be a multiple of CPU's ($cpus), for efficiency sake. Set to \`all\` to \
assign ALL datasets to 1 node."
read -p "Tasks per job [$cpus]:" datasets_per_job
datasets_per_job=${datasets_per_job:-$cpus}
if [[ $datasets_per_job -eq "all" ]]; then
    echo "Assigning ALL datasets to 1 node."
    ((datasets_per_job=$n_datasets))
fi

# Compute amount of jobs required
n_jobs=$(( ( $n_datasets + ( $datasets_per_job - 1 ) ) / $datasets_per_job ))
# -> writing division of (n / d) as ((n + (d - 1)) / d) performs `ceil`.

echo 
echo "Select a job to submit:"
select jobfile in ./scripts/jobs/*.sh; do
    [[ -n $jobfile ]] || { echo "Invalid choice,\
        choose again or press CTRL+C to quit job submission." >&2; continue; }
    break
done
echo "Chosen $jobfile."

# Send to peregrine
echo "Assigning datasets to $n_jobs jobs, $datasets_per_job datasets per job."
((n_jobs-=1)) # correct for array index of 0.
ssh $PEREGRINE_USERNAME@peregrine.hpc.rug.nl "sbatch \
    --job-name=${job_name%.csv} \
    --ntasks=$cpus \
    --export=data_collection=$data_collection \
    --array=0-$n_jobs \
    $jobfile"
echo "Submitted SLURM job array ✨"