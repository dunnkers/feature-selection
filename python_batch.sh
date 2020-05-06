#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=3
#SBATCH --ntasks=10
#SBATCH --job-name=python_cpu
#SBATCH --mem=8000
module load Python/3.6.4-foss-2018a
python python_cpu.py
