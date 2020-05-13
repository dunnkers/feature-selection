#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --job-name=any-job
#SBATCH --mem=3000
#SBATCH --output=logs/slurm-%j.out
#SBATCH --partition=regular
module load Python/3.6.4-foss-2018a
pip install --quiet --user xgboost
pip install --quiet --user skrebate
# echo "executing 'python jobs/run-rfe.py $dataset'..."
# python jobs/run-rfe.py $dataset
# echo "executing 'python jobs/run-featboost.py $dataset'..."
# python jobs/run-featboost.py $dataset
echo "executing 'python jobs/run-multisurf.py $dataset'..."
python jobs/run-multisurf.py $dataset
echo "executing 'python jobs/run-validation.py $dataset'..."
python jobs/run-validation.py $dataset