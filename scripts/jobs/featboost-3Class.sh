#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --job-name=featboost-3Class
#SBATCH --mem=8000
#SBATCH --output=logs/slurm-%j.out
#SBATCH --partition=regular
module load Python/3.6.4-foss-2018a
pip install --quiet --user xgboost
python jobs/run-featboost.py data/3Class_Datasets_Loc_2
python jobs/run-validation.py data/3Class_Datasets_Loc_2