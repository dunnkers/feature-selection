#!/bin/sh
# 
# Submits computation jobs for ALL datasets apparent in system. 
# 
# → Be considerate before submitting this many jobs to the cluster.
# 

submitted_jobs=0
for i in `find ./data -type d `; do
    if [ -d $i ]; then
        j=`ls -1 $i | wc -l`;
        if [ $j -eq  30 ]; then
            if [ $submitted_jobs -gt 18 ]; then # limit seems to be 19.
                echo "[Job submission limit hit] Waiting a minute before submitting new jobs...";
                sleep 60;
                ((submitted_jobs=0))
            fi;
            echo "Submitting job for dataset: $i";
            ssh $PEREGRINE_USERNAME@peregrine.hpc.rug.nl "sbatch --job-name=${i} --export=dataset=$i scripts/jobs/any-job.sh"
            ((submitted_jobs=submitted_jobs+1))
        fi;
    fi;
done

echo "Submitted all jobs ✨"
