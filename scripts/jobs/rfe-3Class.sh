#!/bin/bash
#SBATCH --time=00:29:00
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --job-name=rfe-3Class
#SBATCH --mem=1000
#SBATCH --output=logs/slurm-%j.out
#SBATCH --partition=short
module load Python/3.6.4-foss-2018a
pip install --quiet --user xgboost
python jobs/run-rfe.py data/3Class_Datasets_Loc_2
python jobs/run-validation.py data/3Class_Datasets_Loc_2