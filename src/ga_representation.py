import numpy as np
from routes_utilities import split_routes


def create_chromosome(num_customers):
    """
    A chromosome is a random permutation of customer indices [1..num_customers].
    The depot (index 0) is never included — split_routes adds it automatically.
    """
    customers = list(range(1, num_customers + 1))
    np.random.shuffle(customers)
    return customers


def initialize_population(pop_size, num_customers):
    """
    Create an initial population of `pop_size` random chromosomes.
    """
    return [create_chromosome(num_customers) for _ in range(pop_size)]


def decode_chromosome(chromosome, demands, capacity):
    """
    Convert a flat customer permutation into a list of vehicle routes
    using the capacity-aware split_routes utility.

    Each route is of the form [0, c1, c2, ..., cn, 0] (depot-to-depot).

    Returns:
        routes (list of lists): valid vehicle routes
    """
    return split_routes(chromosome, demands, capacity)


def chromosome_to_routes(chromosome, demands, capacity):
    """
    Convenience wrapper — same as decode_chromosome.
    Returns the decoded routes for a given chromosome.
    """
    return decode_chromosome(chromosome, demands, capacity)
