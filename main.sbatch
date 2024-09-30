#!/usr/bin/env bash
#SBATCH --time=14:00:00
#SBATCH --output=../%j.out
#SBATCH --mem=16G
dir=$(pwd)
srun nix develop "$dir" --command python "$dir/main.py"
