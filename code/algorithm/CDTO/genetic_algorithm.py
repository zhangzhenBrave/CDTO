import json
import os
import random
import sys
from multiprocessing import Pool
from itertools import repeat
from collections import Sequence

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from util import  PLOT_STYLE,get_log_dir,log
import seaborn as sns

from processor_monitoring import clear_processor_log, update_processor_log

sns.set(style=PLOT_STYLE)


from base import BaseOptimizer

from util import evaluate_placement, apply_placement, generate_random_placement, \
    get_device_assignment


def create_parent_selection_function(type, s=2):
    if type in ('linear', 'lin'):
        def linear(n_parents):
            return np.array([((2 - s) / n_parents) + ((2 * i * (s - 1)) / (n_parents * (n_parents - 1)))
                             for i in range(n_parents - 1, -1, -1)])

        return linear
    elif type in ('exponential', 'exp'):
        def exponential(n_parents):
            probs = np.array([1.0 - np.exp(-i) for i in range(n_parents - 1, -1, -1)])
            return probs / probs.sum()

        return exponential


def _evaluate(individual, task_dim, task_iner_priority, device_graph, task_unit, pipeline_batches=1, batches=1, simulator_comp_penalty=1,
              simulator_comm_penalty=1, device_memory_utilization=1):
    return 1 / evaluate_placement(apply_placement(task_dim, individual.placement),task_iner_priority, device_graph, task_unit,
                                  pipeline_batches=pipeline_batches, batches=batches,
                                  comp_penalty=simulator_comp_penalty, comm_penalty=simulator_comm_penalty,
                                  device_memory_utilization=device_memory_utilization)


def _calculate_binary_difference_diversity(population):
    return sum(
        int(g[0] != g[1]) for ind1 in population for ind2 in population for g in
        zip(ind1.placement, ind2.placement)) / (len(population[0].placement) * len(population) ** 2)


class GAOptimizer(BaseOptimizer):

    def __init__(self, mutation_rate=0.05, crossover_rate=0.8, crossover_type='one-point',
                 mutation_sharding_rate=0.2,
                 parent_selection_mechanism='rank', tournament_size=10,
                 parent_selection_function='linear', parent_selection_function_s=2,
                 population_size=10, generations=1, plot_fitness_history=False,
                 evolve_mutation_rate=False, elite_size=1, print_diversity=False,
                 min_mutation_rate=0.05, max_mutation_rate=0.9,
                 copy_mutation_rate=0, zone_mutation_rate=0,
                 benchmarking_population_size=100, benchmarking_generations=50,
                 benchmarking_function=None,
                 include_trivial_solutions_in_initialization=True,
                 checkpoint_period=-1,
                 monitor_processors=False,
                 allow_cpu=True, **kwargs):
        """
        Initializes the GA optimizer, setting important hyperparameters.
        :param mutation_rate: The rate at which mutation will be applied, set at the gene level.
        :param crossover_rate: The rate at which crossover will be applied.
        :param crossover_type: The type of crossover. ['uniform', 'one-point', 'n-point'] (n is any integer)
        :param parent_selection_function: The type of distribution function applied during parent selection.
                                          ['linear', 'exponential']
        """
        super().__init__(**kwargs)

        self.mutation_rate = mutation_rate
        self.copy_mutation_rate = copy_mutation_rate
        self.zone_mutation_rate = zone_mutation_rate
        self.crossover_rate = crossover_rate
        self.population_size = population_size
        self.elite_size = elite_size

        self.benchmarking_population_size = benchmarking_population_size
        self.benchmarking_generations = benchmarking_generations
        self.benchmarking_function = benchmarking_function

        self.mutation_sharding_rate = mutation_sharding_rate

        if crossover_type == 'uniform':
            self.crossover = 'uniform'
        elif crossover_type == 'one-point':
            self.crossover = 1
        elif crossover_type.endswith('-point'):
            self.crossover = int(crossover_type.split('-')[0])
        else:
            raise Exception('Invalid crossover type.')

        self.parent_selection_mechanism = parent_selection_mechanism

        self.tournament_size = tournament_size
        self.parent_selection_distribution = create_parent_selection_function(parent_selection_function,
                                                                              parent_selection_function_s)
        self.generations = generations
        self.evolve_mutation_rate = evolve_mutation_rate
        self.plot_fitness_history = plot_fitness_history
        self.print_diversity = print_diversity

        self.max_mutation_rate = max_mutation_rate
        self.min_mutation_rate = min_mutation_rate
        self.include_trivial_solutions_in_initialization = include_trivial_solutions_in_initialization
        self.allow_cpu = allow_cpu

        self.monitor_processors = monitor_processors

        self.worker_pool = None

        self.checkpoint_period = checkpoint_period
        if self.checkpoint_period > 0 and not os.path.exists(os.path.join(get_log_dir(), 'checkpoints')):
            os.makedirs(os.path.join(get_log_dir(), 'checkpoints'))

        if monitor_processors:
            clear_processor_log()

    def optimize(self, net_len, task_num,n_devices,task_iner_priority, device_graph, task_unit):
        #net_string=167, device_graph=50
        if self.n_threads > 1:
            self.worker_pool = Pool(self.n_threads)

        task_dim=[]
        for i in range(len(task_iner_priority)):
            task_dim.append(len(task_iner_priority[i]))

        def initialize(population_size):
            placements = []

            while len(placements) < population_size:
                placements.append(generate_random_placement( task_dim, n_devices, allow_device_0=self.allow_cpu))



            # print( placements)

            # if self.evolve_mutation_rate:
            #     return [Candidate(p,
            #                       min(max(random.normalvariate(self.mutation_rate, 0.1), self.min_mutation_rate),
            #                           self.max_mutation_rate),
            #                       min(max(random.normalvariate(self.zone_mutation_rate, 0.05), self.min_mutation_rate),
            #                           self.max_mutation_rate)
            #                       )
            #             for p in placements]

            return [Candidate(p) for p in placements]

        def evaluate(individual):
            return _evaluate(individual, task_dim, task_iner_priority, device_graph, task_unit,
                             pipeline_batches=self.pipeline_batches, batches=self.batches,
                             simulator_comm_penalty=self.simulator_comm_penalty,
                             simulator_comp_penalty=self.simulator_comp_penalty,
                             device_memory_utilization=self.device_memory_utilization)

        def rank(population, return_scores=False, benchmarking_function=None):
            # if self.worker_pool:
            #     fn_arg = zip(population, repeat(net_len), repeat(task_num), repeat(n_devices),
            #                  repeat(self.pipeline_batches), repeat(self.batches),
            #                  repeat(self.simulator_comp_penalty), repeat(self.simulator_comm_penalty),
            #                  repeat(self.device_memory_utilization))
            #     fitness_scores = self.worker_pool.starmap(_evaluate, fn_arg)
            # else:
            # print('3',population[7].placement)
            fitness_scores = list(map(evaluate, population))

            fitness_db = dict(zip(population, fitness_scores))

            if return_scores:
                fitness_scores = sorted(fitness_scores, key=lambda x: -x)
                return sorted(population, key=lambda x: -fitness_db[x]), fitness_scores
            else:
                return sorted(population, key=lambda x: -fitness_db[x])

        def select_parents(population, fitness_scores, n_parents=None):
            mating_pool = []
            if n_parents is None:
                n_parents = len(population)

            if self.parent_selection_mechanism == 'rank':
                prob_dist = self.parent_selection_distribution(n_parents)
                cum_dist = np.zeros(len(population))
                cum_sum = 0
                for i in range(len(population)):
                    cum_sum += prob_dist[i]
                    cum_dist[i] = cum_sum

                i = 0
                r = random.random() * 1 / n_parents
                while len(mating_pool) < n_parents:
                    while r <= cum_dist[i]:
                        mating_pool.append(population[i])
                        r += 1 / n_parents
                    i += 1
            elif self.parent_selection_mechanism == 'tournament':
                while len(mating_pool) < n_parents:
                    competitors = random.sample(tuple(zip(population, fitness_scores)), self.tournament_size)
                    winner = max(competitors, key=lambda x: x[1])[0]
                    mating_pool.append(winner)
            return mating_pool

        def crossover(parent1, parent2):
            if random.random() > self.crossover_rate:
                return parent1, parent2

            mutation_rate1, mutation_rate2 = parent1.mutation_rate, parent2.mutation_rate
            zone_mutation_rate1, zone_mutation_rate2 = parent1.zone_mutation_rate, parent2.zone_mutation_rate
            parent1, parent2 = parent1.placement, parent2.placement
            if self.crossover == 'uniform' or self.crossover >= len(parent1) - 1:
                child1, child2 = [], []

                for g in range(len(parent1)):
                    if random.random() > 0.5:
                        child1.append(parent1[g])
                        child2.append(parent2[g])
                    else:
                        child2.append(parent1[g])
                        child1.append(parent2[g])
                children = child1, child2
            else:
                crossover_points = []
                while len(crossover_points) < self.crossover:
                    new_point = random.randint(1, len(parent1) - 1)
                    if new_point not in crossover_points:
                        crossover_points.append(new_point)

                children = ([], [])
                parent_sel = int(random.random())
                crossover_points = [0] + crossover_points + [len(parent1)]
                for i in range(len(crossover_points) - 1):
                    children[parent_sel][crossover_points[i]:crossover_points[i + 1]] \
                        = parent1[crossover_points[i]:crossover_points[i + 1]]
                    children[(parent_sel + 1) % 2][crossover_points[i]:crossover_points[i + 1]] \
                        = parent2[crossover_points[i]:crossover_points[i + 1]]
                    parent_sel = (parent_sel + 1) % 2

            if self.evolve_mutation_rate:
                mix_rate = random.normalvariate(0.5, 0.1)
                mr1 = mutation_rate1 * mix_rate + mutation_rate2 * (1 - mix_rate)
                mr2 = mutation_rate2 * mix_rate + mutation_rate1 * (1 - mix_rate)
                zmr1 = zone_mutation_rate1 * mix_rate + zone_mutation_rate2 * (1 - mix_rate)
                zmr2 = zone_mutation_rate2 * mix_rate + zone_mutation_rate1 * (1 - mix_rate)
                children = Candidate(children[0], mr1, zmr1), Candidate(children[1], mr2, zmr2)
            else:
                children = Candidate(children[0]), Candidate(children[1])
            return children

        def recombine(mating_pool):
            assert len(mating_pool) % 2 == 0, "Mating pool must contain an equal number of parents"
            random.shuffle(mating_pool)
            children = []
            for i in range(0, len(mating_pool), 2):
                children.extend(crossover(mating_pool[i], mating_pool[i + 1]))
            return children

        def mutate_single_gene(gene):
            if random.random() < self.mutation_sharding_rate:
                if type(gene) != list:
                    if isinstance(gene, Sequence):
                        gene = list(gene)
                    else:
                        gene = [gene]

                if self.allow_cpu:
                    new_gene = list(set(gene + [random.randint(0, n_devices - 1)]))
                else:
                    new_gene = list(set(gene + [random.randint(1, n_devices - 1)]))
                if len(new_gene) == 1:
                    new_gene = new_gene[0]

                return new_gene

            if self.allow_cpu:
                return random.randint(0, n_devices - 1)
            return random.randint(1, n_devices - 1)

        def mutate(individual):
            if self.evolve_mutation_rate:
                mutation_rate = individual.mutation_rate
                zone_mutation_rate = individual.zone_mutation_rate
                if mutation_rate == 0:
                    new_mutation_rate = 0
                else:
                    new_mutation_rate = max(min(mutation_rate + random.normalvariate(0, 0.05), self.max_mutation_rate),
                                            self.min_mutation_rate)
                if zone_mutation_rate == 0:
                    new_zone_mutation_rate = 0
                else:
                    new_zone_mutation_rate = max(min(zone_mutation_rate + random.normalvariate(0, 0.02),
                                                     self.max_mutation_rate),
                                                 self.min_mutation_rate)
                placement = individual.placement

                if random.random() < self.zone_mutation_rate:
                    split1 = random.randint(0, len(placement) - 1)
                    split2 = split1 + min(np.random.geometric(0.2), len(placement) - split1)
                    dev = random.randint(0 if self.allow_cpu else 1, n_devices - 1)
                    placement = placement[:split1] + [dev] * (split2 - split1) + placement[split2:]
                else:
                    placement = [mutate_single_gene(g) if random.random() < new_mutation_rate else g
                                 for g in placement]

                return Candidate(placement, new_mutation_rate, new_zone_mutation_rate)
            else:
                placement = [mutate_single_gene(g) if random.random() < self.mutation_rate else g
                             for g in individual.placement]
                return Candidate(placement)

        def mutate_population(population):
            return [mutate(ind) for ind in population]

        def select_offspring(previous_generation_ranked, candidates, population_size=self.population_size):
            if self.elite_size:
                random.shuffle(candidates)
                return previous_generation_ranked[:self.elite_size] \
                       + candidates[:population_size - self.elite_size]
            return candidates

        pop = initialize(self.population_size)


        if self.plot_fitness_history:
            fitness_history = []

        if self.print_diversity:
            diversity_history = []

        if self.checkpoint_period > 0:
            with open(os.path.join(get_log_dir(), 'checkpoints/scores.csv'), 'w') as f:
                f.write('')

        if self.score_save_period:
            with open(os.path.join(get_log_dir(), 'time_history.csv'), 'w') as f:
                f.write('generation, time\n')

        def run_optimization(generations, population_size=self.population_size, benchmarking_function=None,
                             start_generation=0):

            nonlocal pop

            for i in tqdm(range(generations), disable=not self.verbose):


                ranked_pop, fitness_scores = rank(pop, return_scores=True, benchmarking_function=benchmarking_function)
                print("##############################")

                if self.checkpoint_period > 0 and i % self.checkpoint_period == 0:
                    best_solution_score = 1 / fitness_scores[0]

                    with open(os.path.join(get_log_dir(), 'checkpoints', 'scores.csv'), 'a') as f:
                        f.write(f'{i + start_generation}, {best_solution_score}\n')

                if self.plot_fitness_history:
                    fitness_history.append(1 / fitness_scores[0])


                mating_pool = select_parents(ranked_pop, fitness_scores)
                children = recombine(mating_pool)
                candidates = mutate_population(children)
                pop = select_offspring(ranked_pop, candidates, population_size=population_size)

                if self.verbose and ((i + 1) % int(self.verbose) == 0 or i == 0):
                    best_score = fitness_scores[0]
                    best_time = 1 / best_score
                    if self.print_diversity:
                        diversity = _calculate_binary_difference_diversity(ranked_pop)
                        diversity_history.append(diversity)
                        log(
                            f'[{i + 1}/{generations}] Best current time: {best_time:.2f}ms '
                            f'Diversity: {diversity:.4f}')
                    else:
                        log(f'[{i + 1}/{generations}] Best current time: {best_time:.2f}ms')

                if self.score_save_period and (i % self.score_save_period == 0 or i == generations - 1):
                    best_score = fitness_scores[0]
                    best_time = 1 / best_score
                    with open(os.path.join(get_log_dir(), 'time_history.csv'), 'a') as f:
                        f.write(f'{i + start_generation + 1}, {best_time}\n')

                if self.monitor_processors:
                    update_processor_log(step=i + start_generation)

        if self.verbose:
            log('Optimizing with simulator...')
        # print(10)
        run_optimization(self.generations)

        if self.benchmarking_generations and self.benchmarking_function:
            if self.verbose:
                log('Optimizing with benchmarking...')

            if self.benchmarking_population_size < self.population_size:
                ranked_pop = rank(pop)

                self.worker_pool.close()
                self.worker_pool = None

                pop = select_offspring(ranked_pop, pop, population_size=self.benchmarking_population_size)

            run_optimization(self.benchmarking_generations, benchmarking_function=self.benchmarking_function,
                             population_size=self.benchmarking_population_size, start_generation=self.generations)

        if self.plot_fitness_history:
            plt.plot(fitness_history)
            plt.title('Fitness')
            plt.savefig(os.path.join(get_log_dir(), 'fitness_history.pdf'), bb_inches='tight')
            plt.show()
            plt.close()

        if self.print_diversity:
            plt.plot(diversity_history)
            plt.title('Diversity')
            plt.savefig(os.path.join(get_log_dir(), 'diversity_history.pdf'), bb_inches='tight')
            plt.show()
            plt.close()
        ranked_pop, fitness_scores = rank(pop, return_scores=True)
        best_solution = 1/fitness_scores[0]
        best_placement=ranked_pop[0].placement



        return best_solution,best_placement


class Candidate:
    def __init__(self, placement, mutation_rate=0, zone_mutation_rate=0):
        self.placement = placement
        self.mutation_rate = mutation_rate
        self.zone_mutation_rate = zone_mutation_rate

    def __str__(self):
        return f'Placement: {self.placement}\t Mutation rate: {self.mutation_rate}\t ' \
               f'Zone mutation rate: {self.zone_mutation_rate}'
