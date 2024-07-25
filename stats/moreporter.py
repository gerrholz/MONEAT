from neat.reporting import BaseReporter
import wandb
from .performance_indicators import hypervolume, sparsity, cardinality

import time

class MOReporter(BaseReporter):


    def __init__(self, ref_point) -> None:
        self.geration_start_time = None
        self.generation_times = []
        self.generation = None
        self.cur_hyper_volume = 0
        self.cur_cardinality = 0
        self.cur_sparsity = 0
        self.ref_point = ref_point


    def start_generation(self, generation):
        self.generation = generation
        print('\n ****** Running generation {0} ****** \n'.format(generation))
        self.geration_start_time = time.time()

    def post_evaluate(self, config, population, species, best_genome):
        # Get the non-dominated solutions
        non_dominated = [g for g in population.values() if g.fitness.rank == 0]
        # Calculate the hypervolume
        self.cur_hyper_volume = hypervolume(self.ref_point, [g.fitness.values for g in non_dominated])

        # Calculate the sparsity
        self.cur_sparsity = sparsity([g.fitness.values for g in non_dominated])
        # Calculate the cardinality
        self.cur_cardinality = cardinality([g.fitness.values for g in non_dominated])
        # Log the metrics to wandb
        wandb.log({
            "eval/hypervolume": self.cur_hyper_volume,
            "eval/sparsity": self.cur_sparsity,
            "eval/cardinality": self.cur_cardinality,
        })
        front = wandb.Table(
            columns=["Objective {i}".format(i=i) for i in range(len(non_dominated[0].fitness.values))],
            data=[g.fitness.values for g in non_dominated]
        )
        wandb.log({"eval/front": front}, commit=False)

    def end_generation(self, config, population, species_set):
        ng = len(population)
        ns = len(species_set.species)
        print('Population of {0:d} members in {1:d} species'.format(ng, ns))
        print('Hyper-volume: {0:.3f}, Sparsity: {1:.3f}, Cardinality: {2:d}'.format(self.cur_hyper_volume, self.cur_sparsity, self.cur_cardinality))
        elapsed_time = time.time() - self.geration_start_time
        self.generation_times.append(elapsed_time)
        if len(self.generation_times) > 1:
            print('\n ****** Average generation time: {0:.3f} seconds ****** \n'.format(sum(self.generation_times)/len(self.generation_times)))
        else: 
            print('\n ****** Generation {0} took {1:.3f} seconds ****** \n'.format(self.generation, elapsed_time))