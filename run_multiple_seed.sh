#!/usr/bin/env bash

for seed in {100..110}
do
    sbatch main.sbatch "$seed"
done
