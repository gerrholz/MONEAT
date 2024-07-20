import mo_gymnasium as gym
import neat
import random
from nsga2.fitness import NSGA2Fitness
from nsga2.population import NSGA2Population
from nsga2.reproduction import NSGA2Reproduction
from matplotlib import pyplot as plt
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    #gym.seed(seed)

env = gym.make("mo-halfcheetah-v4")


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = NSGA2Fitness(0.0, [0.0, 0.0])

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        observation, _ =env.reset()

        done = False
        fitness = np.zeros(2)
        while not done:
            output = net.activate(observation)
            action = np.array(np.clip(output, 0, 1))
            next_obs, vector_reward, terminated, truncated, info = env.step(action)
            fitness = np.add(fitness, vector_reward)

            if terminated or truncated:
                break

        if(fitness[0] > 50):
            print(fitness)
            raise RuntimeError("Fitness values are greater than 50")

        genome.fitness.values = fitness
        env.close()

# main method
def main():
    set_seed()
    config_path = 'moneat.config'
    config = neat.config.Config(neat.DefaultGenome, NSGA2Reproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    
    # Create the population, which is the top-level object for a NEAT run.
    p = NSGA2Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 300 generations.
    winners, best_10 = p.run(eval_genomes, 10)

    # Print best 10 genomes as points in a 2d space with the objective values as coordinates using matplotlib
    x = [g.fitness.values[0] for g in best_10]
    y = [g.fitness.values[1] for g in best_10]
    plt.scatter(y, x)
    plt.xlabel("Time reward")
    plt.ylabel("Treasure reward")
    plt.show()




if __name__ == '__main__':
    main()
