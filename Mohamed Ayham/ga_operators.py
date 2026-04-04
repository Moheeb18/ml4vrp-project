import numpy as np
import random


# ---------------------------------------------------------------------------
# SELECTION
# ---------------------------------------------------------------------------

def tournament_selection(population, scores, tournament_size=4):
    """
    Tournament selection: pick `tournament_size` candidates at random,
    return the one with the lowest (best) fitness score.

    Args:
        population      : list of chromosomes
        scores          : list of fitness values (lower = better)
        tournament_size : number of candidates per tournament

    Returns:
        winner (list): the selected chromosome
    """
    candidates = random.sample(range(len(population)), tournament_size)
    best = min(candidates, key=lambda i: scores[i])
    return population[best][:]  # return a copy


# ---------------------------------------------------------------------------
# CROSSOVER
# ---------------------------------------------------------------------------

def order_crossover(parent1, parent2):
    """
    Order Crossover (OX) — the gold standard for permutation chromosomes.

    1. Copy a random sub-segment from parent1 into the child.
    2. Fill the remaining positions in order from parent2,
       skipping any customers already in the child.

    This always produces a valid permutation (no duplicates, no missing).

    Args:
        parent1, parent2 : list of customer indices

    Returns:
        child1, child2 (list): two offspring chromosomes
    """
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))

    def _ox(p1, p2):
        child = [None] * size
        child[a:b+1] = p1[a:b+1]
        segment_set = set(p1[a:b+1])
        pos = (b + 1) % size
        for gene in p2[b+1:] + p2[:b+1]:
            if gene not in segment_set:
                child[pos] = gene
                pos = (pos + 1) % size
        return child

    return _ox(parent1, parent2), _ox(parent2, parent1)


# ---------------------------------------------------------------------------
# MUTATION
# ---------------------------------------------------------------------------

def swap_mutation(chromosome, mutation_rate=0.02):
    """
    Swap Mutation: randomly swap two customers in the chromosome.

    Good for local exploration — moves a customer to a different position
    without breaking the permutation.
    """
    chrom = chromosome[:]
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(chrom)), 2)
        chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom


def inversion_mutation(chromosome, mutation_rate=0.02):
    """
    Inversion Mutation: reverse a random sub-segment of the chromosome.

    Good for escaping local optima — restructures a portion of the route
    order without losing any customer.
    """
    chrom = chromosome[:]
    if random.random() < mutation_rate:
        i, j = sorted(random.sample(range(len(chrom)), 2))
        chrom[i:j+1] = chrom[i:j+1][::-1]
    return chrom


def combined_mutation(chromosome, mutation_rate=0.02):
    """
    Apply both swap and inversion mutation independently.
    Using both gives a better balance of local and global search.
    """
    chrom = swap_mutation(chromosome, mutation_rate)
    chrom = inversion_mutation(chrom, mutation_rate)
    return chrom
