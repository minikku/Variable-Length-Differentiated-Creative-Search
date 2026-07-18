"""GProp_d -- genetic neuro-evolution with structural operators (port of GProp_d.m)."""
from __future__ import annotations

import math
import numpy as np

from ..network import (
    initial_single_layer_network, random_hidden_size,
    mutation_operator, crossover_operator,
    addition_operator, elimination_operator, substitution_operator,
)
from .base import Individual, AlgoResult, pop_sort


def gprop_d(options, rng):
    npop = int(options["popsize"])
    max_iter = round(options["max_nfe"] / npop)
    opts = dict(options)

    pop = []
    for _ in range(npop):
        opts["hidn"] = random_hidden_size(opts, rng)
        chrom = initial_single_layer_network(opts, rng)
        fit, _, _ = options["fobj"](chrom)
        pop.append(Individual(chrom, fit))

    global_best = Individual(None, np.inf)
    convergence = []

    for _ in range(max_iter):
        pop1 = pop_sort(pop)
        # deep copies for the working/offspring population
        pop2 = [Individual(ind.chrom.copy(), ind.fitness) for ind in pop1]

        n_best = round(npop / 2)
        for i in range(1, n_best + 1):
            dst = n_best + i - 1            # 0-based offspring slot
            if dst >= npop:
                break
            if rng.random() < 0.8:
                pop2[dst].chrom = mutation_operator(pop2[i - 1].chrom, opts, 0.5, rng)
            if rng.random() < 0.5:
                pop2 = crossover_operator(pop2, rng)
            if rng.random() < 0.5:
                pop2[dst].chrom = addition_operator(pop2[i - 1].chrom, opts, rng)
            if rng.random() < 0.5:
                pop2[dst].chrom = elimination_operator(pop2[i - 1].chrom, opts, rng)
            if rng.random() < 0.5:
                pop2[dst].chrom = substitution_operator(pop2[i - 1].chrom, opts, rng)

        for ind in pop2:
            ind.fitness, _, _ = options["fobj"](ind.chrom)

        combined = pop1 + pop2
        combined = pop_sort(combined)
        pop = combined[:npop]

        if pop[0].fitness < global_best.fitness:
            global_best = Individual(pop[0].chrom.copy(), pop[0].fitness)
        convergence.append(global_best.fitness)

    return AlgoResult(global_best.fitness, global_best.chrom, convergence)
