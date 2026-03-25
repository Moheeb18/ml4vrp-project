import sys
import os
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent)
import src.routes_utilities
from src.routes_utilities import route_distance
from ga_representation import decode_chromosome

#from routes_utilities import route_distance

# Competition objective weight: penalise each extra vehicle heavily
VEHICLE_PENALTY = 1000


def fitness(chromosome, demands, capacity, distance_matrix):
    """
    Compute fitness using the ML4VRP competition objective function:

        fitness = VEHICLE_PENALTY * num_vehicles + total_distance

    Minimising vehicle count is the primary objective (worth 1000 distance
    units each), with total distance as the secondary objective.

    Args:
        chromosome      : list of customer indices (permutation)
        demands         : np.array of demands per node (index 0 = depot)
        capacity        : int, max vehicle capacity
        distance_matrix : 2D np.array of pairwise distances

    Returns:
        score (float): competition objective value (lower is better)
    """
    routes     = decode_chromosome(chromosome, demands, capacity)
    num_routes = len(routes)
    total_dist = sum(route_distance(r, distance_matrix) for r in routes)
    return VEHICLE_PENALTY * num_routes + total_dist


def fitness_from_routes(routes, distance_matrix):
    """
    Compute fitness directly from a list of already-decoded routes.
    Used after local search to avoid re-decoding.

    Args:
        routes          : list of routes (each [0, ..., 0])
        distance_matrix : 2D np.array

    Returns:
        score (float): competition objective value
    """
    num_routes = len(routes)
    total_dist = sum(route_distance(r, distance_matrix) for r in routes)
    return VEHICLE_PENALTY * num_routes + total_dist


def population_fitness(population, demands, capacity, distance_matrix):
    """
    Evaluate fitness for every chromosome in the population.

    Returns:
        scores (list of float): fitness values in same order as population
    """
    return [fitness(chrom, demands, capacity, distance_matrix)
            for chrom in population]
