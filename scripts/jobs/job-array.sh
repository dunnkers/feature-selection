#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --mem=100000
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --partition=regular

if [[ -z $data_collection ]]; then
    echo 'No data collection provided.'
    exit
fi

echo "Array job $SLURM_ARRAY_JOB_ID with $SLURM_ARRAY_TASK_COUNT tasks."
echo "Running job $SLURM_JOB_ID, task $SLURM_ARRAY_TASK_ID."
echo "Using data collection = $data_collection"

module load Python/3.6.4-foss-2018a
pip install --quiet --user xgboost
pip install --quiet --user skrebate
pip install --quiet --user git+https://github.com/scikit-learn-contrib/stability-selection.git
# python -m fseval.ranking.rfe $data_collection --batch_id=$SLURM_ARRAY_JOB_ID --job_id=$SLURM_JOB_ID --task_id=$SLURM_ARRAY_TASK_ID
python -m fseval.ranking.multisurf $data_collection --batch_id=$SLURM_ARRAY_JOB_ID --job_id=$SLURM_JOB_ID --task_id=$SLURM_ARRAY_TASK_ID
python -m fseval.ranking.featboost $data_collection --batch_id=$SLURM_ARRAY_JOB_ID --job_id=$SLURM_JOB_ID --task_id=$SLURM_ARRAY_TASK_ID
python -m fseval.ranking.featboost_knn $data_collection --batch_id=$SLURM_ARRAY_JOB_ID --job_id=$SLURM_JOB_ID --task_id=$SLURM_ARRAY_TASK_ID
python -m fseval.ranking.featboost_deep $data_collection --batch_id=$SLURM_ARRAY_JOB_ID --job_id=$SLURM_JOB_ID --task_id=$SLURM_ARRAY_TASK_ID
# python -m fseval.ranking.stability_selection $data_collection --batch_id=$SLURM_ARRAY_JOB_ID --job_id=$SLURM_JOB_ID --task_id=$SLURM_ARRAY_TASK_ID
python -m fseval.ranking.extratrees $data_collection --batch_id=$SLURM_ARRAY_JOB_ID --job_id=$SLURM_JOB_ID --task_id=$SLURM_ARRAY_TASK_ID
