#!/bin/sh
# 
# Submits computation jobs for ALL datasets apparent in system. 
# 
# → Be considerate before submitting this many jobs to the cluster.
# 

for i in `find ./data -type d `; do
    if [ -d $i ]; then
        j=`ls -1 $i | wc -l`;
        if [ $j -eq  30 ]; then
            echo "Submitting job for dataset: $i";
            ssh $PEREGRINE_USERNAME@peregrine.hpc.rug.nl "sbatch --job-name=${i} --export=dataset=$i scripts/jobs/any-job.sh"
            exit
        fi;
    fi;
done
