#!/bin/sh
# 
# Submits computation jobs for ALL datasets. 
# 
# → Be considerate before submitting this many jobs to the cluster.
# 

./scripts/upload-files.sh

echo 
echo "Select a dataset collection to submit:"
select data_collection in ./data/*.csv
do
if [[ ! -z $data_collection ]]; then
    n_datasets=$(cat $data_collection | sed '/^\s*$/d' | wc -l | grep -o "[0-9]\+")
    ((n_datasets-=1)) # skip csv header
    job_name=${data_collection##*/}
    echo "Submitting $n_datasets datasets from '${job_name%.csv}'..."
    cpus=6
    cpus_per_task=1
    pool_size=$(( ( $cpus + ( $cpus_per_task-1 ) ) / $cpus_per_task )) # ceil'ed
    n_tasks=$(( ( $n_datasets + ( $pool_size-1 ) ) / $pool_size ))
    # ((n_tasks=2))
    echo "Pool size of $pool_size. Assigning datasets to $n_tasks tasks."
    ((n_tasks-=1)) # correct for array index of 0.
    ssh $PEREGRINE_USERNAME@peregrine.hpc.rug.nl "sbatch --job-name=${job_name%.csv} --export=data_collection=$data_collection --array=0-$n_tasks scripts/jobs/job-array.sh"
    echo "Submitted SLURM job array ✨"

    echo
    read -p "Submit another dataset? (y/n)" -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        break
    fi
else
    echo "Invalid choice, choose again or press CTRL+C to quit job submission."
fi
done
