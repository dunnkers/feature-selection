#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=250
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --partition=short

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
python -m fseval.ranking.rfe $data_collection $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID
python -m fseval.ranking.featboost $data_collection $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID
python -m fseval.ranking.multisurf $data_collection $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID
python -m fseval.validation.knn $data_collection $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID