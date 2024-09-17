from neat.config import ConfigParameter, DefaultClassConfig
from itertools import count
from neat.species import Species
import math
import random

class NSGA2Reproduction:
    
    @classmethod
    def parse_config(cls, param_dict):
        """
        Same as original
        """
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('elitism', int, 0),
                                   ConfigParameter('survival_threshold', float, 0.2),
                                   ConfigParameter('min_species_size', int, 1)])


    def __init__(self, config, reporters, stagnation) -> None:
        """
        Same as original
        """
        self.reproduction_config = config
        self.reporters = reporters
        self.genome_indexer = count(1)
        self.stagnation = stagnation
        self.ancestors = {}
        # Track all existing populations and evaluate them
        self.parent_pop = []
        self.fronts = []
        self.parent_pop = {}
        self.parent_species = {}

    """
    Create num_genomes new genomes of the given type using the given configuration.
    """
    def create_new(self, genome_type, genome_config, num_genomes):
        """
        Same as original
        """
        new_genomes = {}
        for i in range(num_genomes):
            key = next(self.genome_indexer)
            g = genome_type(key)
            g.configure_new(genome_config)
            new_genomes[key] = g
            self.ancestors[key] = tuple()

        return new_genomes
    
    def fast_non_dominated_sort(self, population):
        F = {}
        S = {}
        n = {}
        for p in population.values():
            S[p.key] = []
            n[p.key] = 0
            if p.fitness is None:
                raise RuntimeError("Fitness not assigned to genome {}".format(p.key))
            for q in population.values():
                if p.key == q.key:
                    continue
                if q.fitness is None:
                    raise RuntimeError("Fitness not assigned to genome {}".format(q.key))
                # Compare the fitness of p and q
                if p.fitness.dominates(q.fitness):
                    # Add q to the set of solutions dominated by p
                    S[p.key].append(q.key)
                elif q.fitness.dominates(p.fitness):
                    # Increment the domination count of p
                    n[p.key] += 1
            # If p belongs to the first front
            if n[p.key] == 0:
                # Add p to the first front
                if not F.get(0):
                    F[0] = []
                F[0].append(p.key)
                p.fitness.rank = 0
    
        # Initialize the front counter
        i = 0
        while F[i]:
            Q = []
            for p in F[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        Q.append(q)
                        population[q].fitness.rank = -(i + 1)
            i += 1
            F[i] = Q
        return F
    
    def assing_crowding_distance(self, front, population):
        if len(front) == 0:
            return
        
        distances = [0] * len(front)
        nobj = len(population[front[0]].fitness.values)

        for m in range(nobj):
            front.sort(key=lambda x: population[x].fitness.values[m])
            distances[0] = float('inf')
            distances[-1] = float('inf')
            max_val = population[front[-1]].fitness.values[m]
            min_val = population[front[0]].fitness.values[m]
            if max_val == min_val:
                continue
            scale = max_val - min_val
            for i in range(1, len(front)-1):
                distances[i] += (population[front[i + 1]].fitness.values[m] - population[front[i - 1]].fitness.values[m])/scale


        for i, f in enumerate(front):
            population[f].fitness.crowding_dist = distances[i]
            
    
    def sort(self, species, generation, pop_size):
        # Filter out stagnated species, collect the set of non-stagnated species members
        remaining_species = {} # remaining species
        population = {}
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            # stagnant species: remove genomes from child population
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
                population = {id:g for id,g in population.items() if g not in stag_s.members}
            # non stagnant species: append species to parent species dictionary
            else:
                remaining_species[stag_sid] = stag_s

        # No genomes left.
        if not remaining_species:
            species.species = {}
            return {}
        

        if not remaining_species:
            self.reporters.complete_extinction()
            return self.create_new(species, generation, self.reproduction_config, pop_size)

        # Sort population using nsga2
        # First combine parent population with current population
        for p in self.parent_pop.values():
            population[p.key] = p
        for s in remaining_species.values():
            for g in s.members.values():
                population[g.key] = g
                

        # Merge child and parent species

        # Merge parent P(t) species and child (Qt) species,
        # so all non-stagnated genomes are covered by species.species
        species.species = remaining_species
        for id, sp in self.parent_species.items():
            if (id in species.species):
                species.species[id].members.update(sp.members)
            else:
                species.species[id] = sp

        F = self.fast_non_dominated_sort(population)

        self.parent_pop = {}
        i = 0
        while len(self.parent_pop) + len(F[i]) <= pop_size:
            self.assing_crowding_distance(F[i], population)
            for p in F[i]:
                self.parent_pop[p] = population[p]
            if len(self.parent_pop) == pop_size:
                break  # Stop if we have reached the required population size
            i += 1


        # If adding the next front exceeds pop_size, fill the remaining slots based on crowding distance
        if len(self.parent_pop) < pop_size:
            self.assing_crowding_distance(F[i], population)
            # Sort the individuals in the current front by their crowding distance in descending order
            F[i].sort(key=lambda x: population[x].fitness.crowding_dist, reverse=True)
            remaining_slots = pop_size - len(self.parent_pop)
            for p in F[i][:remaining_slots]:
                self.parent_pop[p] = population[p]

        # Sort population by rank and crowding distance
        new_pop = sorted(self.parent_pop.values(), key=lambda x: x.fitness, reverse=True)

        #for i, g in enumerate(new_pop):
        #    print(f"Genome {i} has rank {g.fitness.rank} and crowding distance {g.fitness.crowding_dist} and values {g.fitness.values}")

        pop_dict = {g.key:g for g in new_pop}

        ## NSGA-II : post step 2 : Clean Species
        # Remove the genomes that haven't passed the crowding-distance step
        # (The ones stagnated are already not on this dict)
        # Also rebuild SpeciesSet.genome_to_species
        species.genome_to_species = {}
        for _, sp in species.species.items():
            sp.members = {id:g for id,g in sp.members.items() if g.key in pop_dict.keys()}
            # map genome to species
            for id, g in sp.members.items():
                species.genome_to_species[id] = sp.key
        # Remove empty species
        species.species = {id:sp for id,sp in species.species.items() if len(sp.members) > 0}

        #self.parent_species should be a deepcopy of the species dictionary,
        # in order to avoid being modified by the species.speciate() method
        # the species in here are used to keep track of parent_genomes on next sort
        self.parent_species = {}
        for id, sp in species.species.items():
            self.parent_species[id] = Species(id, sp.created)
            self.parent_species[id].members = dict(sp.members)
            self.parent_species[id].representative = sp.representative

        

        return pop_dict



    def reproduce(self, config, species, pop_size, generation):
        # Here the members are already sorted
        new_population = {}
        for _, sp in species.species.items():
            # Sort species members by crowd distance
            members = list(sp.members.values())
            members.sort(key=lambda g: g.fitness, reverse=True)
            #for i, g in enumerate(members):
            #    print(f"Genome {i} has rank {g.fitness.rank} and crowding distance {g.fitness.crowding_dist} and values {g.fitness.values}")
            # Survival threshold: how many members should be used as parents
            repro_cutoff = int(math.ceil(self.reproduction_config.survival_threshold * len(members)))
            # Use at least two parents no matter what the threshold fraction result is.
            members = members[:max(repro_cutoff, 2)]
            # spawn the number of members on the species
            spawn = len(sp.members)
            for i in range(spawn):
                # pick two random parents
                parent_a = random.choice(members)
                parent_b = random.choice(members)
                # sexual reproduction
                # if a == b, it's asexual reproduction
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.configure_crossover(parent_a, parent_b, config.genome_config)
                child.mutate(config.genome_config)
                new_population[gid] = child

        return new_population