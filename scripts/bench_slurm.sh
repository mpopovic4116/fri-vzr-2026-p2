#!/bin/bash
#SBATCH --job-name=lenia_benchmark
#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 # no multi-gpu requirement or impl yet
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread                # as in example
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --output=slurm_lenia_bench.out 
#SBATCH --error=slurm_lenia_bench.err

module load GCCcore/14.2.0
module load Python/3.13.1-GCCcore-14.2.0
module load CUDA/12.8.0 

cd $pwd/fri-2026-vzr-p2
source .venv/bin/activate

./scripts/bench.sh