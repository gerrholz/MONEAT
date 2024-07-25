import mo_gymnasium as gym
import neat
import random

import neat.config
from nsga2.fitness import NSGA2Fitness
from nsga2.population import NSGA2Population
from nsga2.reproduction import NSGA2Reproduction
from matplotlib import pyplot as plt
import numpy as np
import wandb
from stats.moreporter import MOReporter

from dotenv import load_dotenv
import os

def set_seed(seed=42):
    random.seed(seed)
    #gym.seed(seed)

env = gym.make("deep-sea-treasure-concave-v0")


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = NSGA2Fitness(0.0, [0.0, 0.0])

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        observation, info =env.reset()

        done = False
        fitness = np.zeros(2)
        while not done:
            output = net.activate(observation)
            action = np.argmax(output)
            observation, vector_reward, terminated, truncated, info = env.step(action)
            fitness = np.add(fitness, vector_reward)

            if terminated or truncated:
                break

        genome.fitness.values = fitness
        env.close()


def setup_wandb(project, entity, config):
    # Load wandb api key from .env file
    load_dotenv()
    api_key = os.getenv("WANDB_API")
    wandb.login(key=api_key)
    wandb.init(project=project, config=config, monitor_gym=True)

def close_wandb():
    wandb.finish()

# main method
def main():
    set_seed()
    config_path = 'moneat_deepsea.config'
    config = neat.config.Config(neat.DefaultGenome, NSGA2Reproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    # Create the population, which is the top-level object for a NEAT run.

    setup_wandb("moneat", "deep-sea-treasure-concave-v0", config)
    p = NSGA2Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(MOReporter(ref_point=np.array([0, -25])))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 300 generations.
    winners, non_dominant = p.run(eval_genomes, 100)

    

    # Print best 10 genomes as points in a 2d space with the objective values as coordinates using matplotlib
    x = [g.fitness.values[0] for g in non_dominant]
    y = [g.fitness.values[1] for g in non_dominant]
    plt.scatter(y, x)
    plt.xlabel("Time reward")
    plt.ylabel("Treasure reward")

    wandb.log({"final_front_plot": plt})
    close_wandb()
    plt.show()




if __name__ == '__main__':
    main()
