import mo_gymnasium as gym
import wandb
import numpy as np
import random

import neat
import neat.config
from nsga2.fitness import NSGA2Fitness
from nsga2.population import NSGA2Population
from nsga2.reproduction import NSGA2Reproduction
from stats.moreporter import MOReporter
from stats.performance_indicators import hypervolume

from dotenv import load_dotenv
import os
import io
import tempfile

def set_seed(seed=42):
    random.seed(seed)
    #gym.seed(seed)


load_dotenv()
api_key = os.getenv("WANDB_API")
wandb.login(key=api_key)

def create_neat_config(config):
    # Creates a temporary config file for NEAT and replaces the hyperparameters with the ones provided
    with open('configs/blanks/moneat_deepsea_hyperparams.config', 'r') as file:
        data = file.readlines()

    for i, line in enumerate(data):
        if '#population_size' in line:
            data[i] = data[i].replace('#population_size', str(config.population_size))
        elif '#conn_add_rate' in line:
            data[i] = data[i].replace('#conn_add_rate', str(config.conn_add_rate))
        elif '#conn_remove_rate' in line:
            data[i] = data[i].replace('#conn_remove_rate', str(config.conn_remove_rate))
        elif '#node_add_rate' in line:
            data[i] = data[i].replace('#node_add_rate', str(config.node_add_rate))
        elif '#node_remove_rate' in line:
            data[i] = data[i].replace('#node_remove_rate', str(config.node_remove_rate))
        elif '#num_generations' in line:
            data[i] = data[i].replace('#num_generations', str(config.num_generations))
        elif '#survival_threshold' in line:
            data[i] = data[i].replace('#survival_threshold', str(config.survival_threshold))
        elif '#weight_mutation_rate' in line:
            data[i] = data[i].replace('#weight_mutation_rate', str(config.weight_mutation_rate))
        elif '#weight_replace_rate' in line:
            data[i] = data[i].replace('#weight_replace_rate', str(config.weight_replace_rate))
        

    in_memory_config = "".join(data)

    with tempfile.NamedTemporaryFile(delete=False) as temp_config_file:
        temp_config_file.write(in_memory_config.encode())
        temp_config_file_path = temp_config_file.name

    neat_config = neat.config.Config(neat.DefaultGenome, NSGA2Reproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation, temp_config_file_path)
    
    os.remove(temp_config_file_path)
    return neat_config

env = gym.make("deep-sea-treasure-mirrored-v0")

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = NSGA2Fitness(0.0, [0.0, 0.0, 0.0])

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        observation, info =env.reset()

        done = False
        fitness = np.zeros(2)
        while not done:
            output = net.activate(observation)
            #action = np.clip(np.array(output), -1, 1)
            action = np.argmax(output)
            observation, vector_reward, terminated, truncated, info = env.step(action)
            fitness = np.add(fitness, vector_reward)

            if terminated or truncated:
                break

        genome.fitness.values = fitness
        env.close()


def objective(config):
    set_seed()
    neat_config = create_neat_config(config)

    p = NSGA2Population(neat_config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(MOReporter(ref_point=np.array([-100, -100])))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winners, non_dominant = p.run(eval_genomes, config.num_generations)

    return hypervolume(np.array([0, -25]), [g.fitness.values for g in non_dominant])


def main():
    wandb.init(project="moneat_sweep")
    hypervolume = objective(wandb.config)
    wandb.log({"hypervolume": hypervolume})


sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'hypervolume',
        'goal': 'maximize'
    },
    'parameters': {
        'population_size': {
            'values': [10, 30, 50, 80, 100, 150, 170, 200, 210, 230, 250, 270, 300]
        },
        'conn_add_rate': {
            'min': 0.1,
            'max': 0.7,
        },
        'conn_remove_rate': {
            'min': 0.1,
            'max': 0.7,
        },
        'node_add_rate': {
            'min': 0.1,
            'max': 0.7,
        },
        'node_remove_rate': {
            'min': 0.1,
            'max': 0.7,
        },
        'num_generations': {
            'values': [10, 50, 80, 100, 130, 150, 180, 200, 230, 250, 270, 300, 330, 350, 380, 400]
        },
        'survival_threshold': {
            'max': 0.5,
            'min': 0.1
        },
        'weight_mutation_rate': {
            'min': 0.1,
            'max': 0.9,
        },
        'weight_replace_rate': {
            'min': 0.1,
            'max': 0.9,
        },
    }
}

sweep_id = wandb.sweep(sweep_config, project="moneat_sweep")
wandb.agent(sweep_id, function=main)
        