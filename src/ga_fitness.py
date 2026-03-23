from ga_representation import decode_chromosome
from routes_utilities import route_distance


def fitness(chromosome, demands, capacity, distance_matrix):
    """
    Compute the fitness of a chromosome as the TOTAL distance across all routes.

    Lower is better. The chromosome is decoded into valid routes first,
    so capacity constraints are always satisfied (no penalty needed).

    Args:
        chromosome      : list of customer indices (permutation)
        demands         : np.array of demands per node (index 0 = depot)
        capacity        : int, max vehicle capacity
        distance_matrix : 2D np.array of pairwise distances

    Returns:
        total_dist (float): sum of distances across all vehicle routes
    """
    routes = decode_chromosome(chromosome, demands, capacity)
    total_dist = sum(route_distance(route, distance_matrix) for route in routes)
    return total_dist


def population_fitness(population, demands, capacity, distance_matrix):
    """
    Evaluate fitness for every chromosome in the population.

    Returns:
        scores (list of float): fitness value for each chromosome, same order
    """
    return [fitness(chrom, demands, capacity, distance_matrix)
            for chrom in population]
