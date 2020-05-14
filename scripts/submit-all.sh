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
    n_datasets=$(wc -l $data_collection | grep -o "[0-9]\+")-1
    job_name=${data_collection##*/}
    echo "Submitting $n_datasets datasets from '${job_name%.csv}'..."
    ssh $PEREGRINE_USERNAME@peregrine.hpc.rug.nl "sbatch --job-name=${job_name%.csv} --export=data_collection=$data_collection --array=0-$n_datasets scripts/jobs/job-array.sh"
    echo "Submitted $n_datasets in a SLURM job array ✨"

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
