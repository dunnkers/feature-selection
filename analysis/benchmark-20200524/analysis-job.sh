#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --mem=1000
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --partition=short

echo "Running analysis jobs. Using data collection = '$data_collection'."

module load Python/3.6.4-foss-2018a
pip install --quiet --user tqdm
python -m fseval.analysis.roc $data_collection --batch_id=11714894_subsets
python -m fseval.analysis.roc $data_collection --batch_id=11824789