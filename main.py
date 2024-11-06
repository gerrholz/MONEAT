import mo_gymnasium as gym
import neat
import random
import argparse

import neat.config
from nsga2.fitness import NSGA2Fitness
from nsga2.population import NSGA2Population
from nsga2.reproduction import NSGA2Reproduction
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import wandb
from stats.moreporter import MOReporter

from dotenv import load_dotenv
import os
import time
import warnings
import gymnasium as s_gym


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

ENV_ID = "mo-swimmer-v4"

env = gym.make(ENV_ID)
reward_dim = env.unwrapped.reward_space.shape[0]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = NSGA2Fitness(0.0, [0.0, 0.0, 0.0])

        net = neat.nn.RecurrentNetwork.create(genome, config)
        observation, info =env.reset()

        done = False
        fitness = np.zeros(2)
        while not done:
            output = net.activate(observation)
            action = np.clip(np.array(output), -1, 1)
            #action = np.argmax(output)
            observation, vector_reward, terminated, truncated, info = env.step(action)
            fitness = np.add(fitness, vector_reward)

            if terminated or truncated:
                break

        genome.fitness.values = fitness
        env.close()


def setup_wandb(project, entity, seed, config):
    # Load wandb api key from .env file
    load_dotenv()
    config_dict = {
    'pop_size': config.pop_size,
    'fitness_threshold': config.fitness_threshold,
    'reset_on_extinction': config.reset_on_extinction,
    'genome_config': config.genome_config.__dict__,
    'reproduction_config': config.reproduction_config.__dict__,
    'species_set_config': config.species_set_config.__dict__,
    'stagnation_config': config.stagnation_config.__dict__,
    }
    config = {
        "env_id": ENV_ID,
        "seed": seed,
        "config": config_dict
    }
    api_key = os.getenv("WANDB_API")
    wandb.login(key=api_key)
    full_name = f"MONEAT_{ENV_ID}_{seed}_{int(time.time())}"
    wandb.init(project=project, config=config, name=full_name, monitor_gym=True, save_code=True)
    #wandb.define_metric("*", step_metric="generation")

def close_wandb():
    wandb.finish()

# main method
def main(seed):
    set_seed(seed)
    config_path = 'configs/tuned/moneat_ant.config'
    config = neat.config.Config(neat.DefaultGenome, NSGA2Reproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    # Create the population, which is the top-level object for a NEAT run.

    setup_wandb("moneat_evaluated_ant", ENV_ID, seed, config)
    p = NSGA2Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(MOReporter(ref_point=np.array([-100, -100])))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 300 generations.
    winners, non_dominant = p.run(eval_genomes, 600)

    # Print best 10 genomes as points in a 2d space with the objective values as coordinates using matplotlib
    x = [g.fitness.values[0] for g in non_dominant]
    y = [g.fitness.values[1] for g in non_dominant]
    plt.scatter(y, x)
    plt.xlabel("Time reward")
    plt.ylabel("Treasure reward")
    plt.title("MONEAT Pareto Front")



    wandb.log({"final_front_plot": plt})
    close_wandb()
    #plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MONEAT with a specified seed.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    main(args.seed)
