import numpy as np
import sys
import os
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent)
import src.routes_utilities
from src.routes_utilities import split_routes

def create_chromosome(num_customers):
    """
    Random chromosome: a shuffled permutation of customer indices [1..N].
    The depot (index 0) is never stored — split_routes adds it on decode.
    """
    customers = list(range(1, num_customers + 1))
    np.random.shuffle(customers)
    return customers


def nearest_neighbour_chromosome(coords, demands, capacity):
    """
    Build a chromosome using the Nearest Neighbour heuristic.

    Constructs routes greedily: from the depot, always visit the nearest
    unvisited customer that fits within the remaining capacity. When no
    customer fits, return to depot and start a new route.

    This produces a much better starting solution than random initialisation,
    which helps the GA converge faster and to better solutions.

    Args:
        coords   : np.array of node coordinates (row 0 = depot)
        demands  : np.array of demands (index 0 = depot)
        capacity : int, vehicle capacity

    Returns:
        chromosome (list): customer visit order as a flat permutation
    """
    num_customers = len(coords) - 1
    unvisited     = set(range(1, num_customers + 1))
    chromosome    = []

    while unvisited:
        current  = 0           # start at depot
        load     = 0
        route    = []

        while True:
            # Find nearest unvisited customer that fits capacity
            best_next = None
            best_dist = float("inf")

            for c in unvisited:
                if load + demands[c] <= capacity:
                    d = np.linalg.norm(coords[current] - coords[c])
                    if d < best_dist:
                        best_dist = d
                        best_next = c

            if best_next is None:
                break   # no customer fits — close this route

            route.append(best_next)
            unvisited.remove(best_next)
            load    += demands[best_next]
            current  = best_next

        chromosome.extend(route)

    return chromosome


def initialize_population(pop_size, num_customers, coords=None,
                           demands=None, capacity=None):
    """
    Create an initial population of pop_size chromosomes.

    If coords/demands/capacity are provided, seeds the population with
    nearest neighbour solutions (20% of pop) for a smarter start.
    The rest are random shuffles to maintain diversity.

    Args:
        pop_size     : number of chromosomes
        num_customers: number of customers (excluding depot)
        coords       : np.array (optional, enables NN seeding)
        demands      : np.array (optional)
        capacity     : int (optional)

    Returns:
        population (list of lists)
    """
    population = []

    # Seed with nearest neighbour solutions if coordinate data available
    if coords is not None and demands is not None and capacity is not None:
        nn_seeds = max(1, pop_size // 5)    # 20% NN-seeded
        for _ in range(nn_seeds):
            chrom = nearest_neighbour_chromosome(coords, demands, capacity)
            # Add slight variation so seeds aren't identical
            chrom = _perturb(chrom)
            population.append(chrom)

    # Fill remaining with random chromosomes
    while len(population) < pop_size:
        population.append(create_chromosome(num_customers))

    return population


def _perturb(chromosome, num_swaps=3):
    """Apply a few random swaps to a chromosome to create a unique variant."""
    chrom = chromosome[:]
    for _ in range(num_swaps):
        i, j = np.random.choice(len(chrom), size=2, replace=False)
        chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom


def decode_chromosome(chromosome, demands, capacity):
    """
    Decode a flat customer permutation into capacity-valid vehicle routes.
    Uses split_routes from routes_utilities.
    """
    return split_routes(chromosome, demands, capacity)


def chromosome_to_routes(chromosome, demands, capacity):
    """Convenience wrapper for decode_chromosome."""
    return decode_chromosome(chromosome, demands, capacity)
