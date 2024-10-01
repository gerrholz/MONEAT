#!/usr/bin/env bash
#SBATCH --time=48:00:00
#SBATCH --output=../sweep_%j.out
#SBATCH --mem=16G
dir=$(pwd)
srun nix develop "$dir" --command python "$dir/sweeps.py"
