#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --job-name=both-6_bit_multiplexer
#SBATCH --mem=2000
#SBATCH --output=logs/slurm-%j.out
#SBATCH --partition=regular
module load Python/3.6.4-foss-2018a
pip install --quiet --user xgboost
python jobs/run-rfe.py data/6_bit_mutliplexer
python jobs/run-featboost.py data/6_bit_mutliplexer
python jobs/run-validation.py data/6_bit_mutliplexer