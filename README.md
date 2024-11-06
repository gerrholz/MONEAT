# MONEAT

This repository contains an implemenation of a multi-objective Version for NEAT (MONEAT), used for my Bachelor's thesis.

## File Structure

This repository has following structure:

### nsga2/

This folder contains the necessary adaptation of the components from the neat-python libary, to support multi-objective problems. For selection the NSGA-II algorithm was used.

### stats/

This folder contains the reporters for stats and wandb.


### evaluation/

This folder contains all scripts and data used for evaluation (plots, fronts etc.)

### configs/

Contains config files for different problems from the mo-gymnasium library.

### main.py

Contains a script which is able to run MONEAT on different problesm from mo-gymnasium.

### sweeps.py

Contains a script for automatic hyperparameter tuning.


## Results

The results can be found on wandb

### PQL on Deep-Sea-Treasure

[WANDB](https://wandb.ai/gerrit-holzbaur-thesis/PQL-swimmer)

### PGMORL for Swimmer-v4

[WANDB](https://wandb.ai/gerrit-holzbaur-thesis/PGMORL-swimmer/)


### PGMORl, GPI-LS and CAPQL for Half-Cheetah-v4

For these runs, the runs from morl-baselines were used

[WANDB](https://wandb.ai/openrlbenchmark/MORL-Baselines?nw=nwuseraraffin)

### MONEAT on Deep-Sea-Treasure

[WANDB](https://wandb.ai/gerrit-holzbaur-thesis/moneat_evaluated_deep/)

### MONEAT on Half-Cheetah-v4

[WANDB](https://wandb.ai/gerrit-holzbaur-thesis/moneat_evaluated_halfcheetah/overview)

### MONEAT on Swimmer-v4

[WANDB](https://wandb.ai/gerrit-holzbaur-thesis/moneat_evaluated_swimmer/overview)