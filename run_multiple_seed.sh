#!/usr/bin/env bash

for seed in {100..150}
do
    sbatch run_sbatch.sh "$seed"
done