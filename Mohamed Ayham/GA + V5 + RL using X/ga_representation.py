import numpy as np
from routes_utilities import split_routes


# ---------------------------------------------------------------------------
# CLARK-WRIGHT SAVINGS ALGORITHM
# ---------------------------------------------------------------------------

def savings_algorithm(coords, demands, capacity, distance_matrix):
    """
    Clark-Wright Savings Algorithm for CVRP.

    Builds an initial solution by greedily merging routes based on savings.
    A saving s(i,j) = d(depot,i) + d(depot,j) - d(i,j) represents how much
    distance is saved by visiting i and j on the same route instead of two
    separate depot-to-customer routes.

    Steps:
      1. Start with one route per customer: [depot → customer → depot]
      2. Compute savings for every pair of customers
      3. Sort savings descending
      4. Greedily merge pairs if the merged route is capacity-feasible
         and both customers are still at the ends of their respective routes

    This typically produces solutions 10-15% better than nearest-neighbour
    and gives the GA a much stronger starting point.

    Args:
        coords          : np.array of node coordinates (row 0 = depot)
        demands         : np.array of demands (index 0 = depot)
        capacity        : int, vehicle capacity
        distance_matrix : 2D np.array of pairwise distances

    Returns:
        routes (list of lists): feasible vehicle routes [0,...,0]
    """
    n = len(coords)
    depot = 0

    # Step 1 — one route per customer
    routes = [[depot, i, depot] for i in range(1, n)]
    route_load = [demands[i] for i in range(1, n)]

    # Index to find which route a customer belongs to
    customer_route = {i: i - 1 for i in range(1, n)}

    # Step 2 — compute all savings s(i,j)
    savings = []
    for i in range(1, n):
        for j in range(i + 1, n):
            s = (distance_matrix[depot][i]
                 + distance_matrix[depot][j]
                 - distance_matrix[i][j])
            savings.append((s, i, j))

    # Step 3 — sort descending
    savings.sort(key=lambda x: -x[0])

    # Step 4 — greedily merge
    for s, i, j in savings:
        ri = customer_route.get(i)
        rj = customer_route.get(j)

        if ri is None or rj is None:
            continue
        if ri == rj:
            continue   # already on the same route

        route_i = routes[ri]
        route_j = routes[rj]

        # i must be at the end of its route (just before depot)
        # j must be at the start of its route (just after depot)
        # OR vice versa — check both orientations

        i_at_end   = route_i[-2] == i
        j_at_start = route_j[1]  == j
        i_at_start = route_i[1]  == i
        j_at_end   = route_j[-2] == j

        merged = None

        if i_at_end and j_at_start:
            # Merge: route_i → route_j (drop one depot between them)
            merged = route_i[:-1] + route_j[1:]
        elif j_at_end and i_at_start:
            # Merge: route_j → route_i
            merged = route_j[:-1] + route_i[1:]
        else:
            continue   # neither orientation is valid

        # Check capacity
        new_load = route_load[ri] + route_load[rj]
        if new_load > capacity:
            continue

        # Accept merge — store in ri, mark rj as empty
        routes[ri]     = merged
        route_load[ri] = new_load
        routes[rj]     = []

        # Update customer_route for all customers now in ri
        for node in merged:
            if node != depot:
                customer_route[node] = ri

    # Filter out empty routes
    routes = [r for r in routes if len(r) > 2]
    return routes


def savings_to_chromosome(routes):
    """
    Convert a list of Clark-Wright routes into a flat chromosome permutation
    (strip depot nodes and concatenate customer sequences).
    """
    chromosome = []
    for route in routes:
        chromosome.extend([n for n in route if n != 0])
    return chromosome


# ---------------------------------------------------------------------------
# NEAREST NEIGHBOUR HEURISTIC
# ---------------------------------------------------------------------------

def nearest_neighbour_chromosome(coords, demands, capacity):
    """
    Greedy nearest-neighbour construction.
    From the depot, always visit the nearest unvisited feasible customer.
    """
    num_customers = len(coords) - 1
    unvisited     = set(range(1, num_customers + 1))
    chromosome    = []

    while unvisited:
        current = 0
        load    = 0
        route   = []

        while True:
            best_next = None
            best_dist = float("inf")
            for c in unvisited:
                if load + demands[c] <= capacity:
                    d = np.linalg.norm(coords[current] - coords[c])
                    if d < best_dist:
                        best_dist = d
                        best_next = c
            if best_next is None:
                break
            route.append(best_next)
            unvisited.remove(best_next)
            load    += demands[best_next]
            current  = best_next

        chromosome.extend(route)

    return chromosome


# ---------------------------------------------------------------------------
# POPULATION INITIALISATION
# ---------------------------------------------------------------------------

def _perturb(chromosome, num_swaps=3):
    """Apply a few random swaps to create a unique variant of a chromosome."""
    chrom = chromosome[:]
    for _ in range(num_swaps):
        i, j = np.random.choice(len(chrom), size=2, replace=False)
        chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom


def create_chromosome(num_customers):
    """Random permutation of all customer indices."""
    customers = list(range(1, num_customers + 1))
    np.random.shuffle(customers)
    return customers


def initialize_population(pop_size, num_customers, coords=None,
                           demands=None, capacity=None,
                           distance_matrix=None):
    """
    Create an initial population with a smart seeding strategy:

      - 20% Clark-Wright savings seeds  — best quality starting solutions
      - 20% Nearest-neighbour seeds     — fast greedy construction
      - 60% Random shuffles             — maintains diversity

    Clark-Wright seeds give the GA a significantly better starting point
    than random initialisation alone, reducing the number of generations
    needed to converge to a good solution.

    Args:
        pop_size        : number of chromosomes
        num_customers   : number of customers (excluding depot)
        coords          : np.array (optional, enables smart seeding)
        demands         : np.array (optional)
        capacity        : int (optional)
        distance_matrix : 2D np.array (optional, needed for Clark-Wright)

    Returns:
        population (list of lists)
    """
    population   = []
    smart_seeding = (coords is not None
                     and demands is not None
                     and capacity is not None
                     and distance_matrix is not None)

    if smart_seeding:
        cw_seeds = max(1, pop_size // 5)   # 20% Clark-Wright
        nn_seeds = max(1, pop_size // 5)   # 20% Nearest-Neighbour

        # Clark-Wright seeds
        cw_routes = savings_algorithm(coords, demands, capacity, distance_matrix)
        cw_chrom  = savings_to_chromosome(cw_routes)
        for _ in range(cw_seeds):
            population.append(_perturb(cw_chrom))

        # Nearest-Neighbour seeds
        nn_chrom = nearest_neighbour_chromosome(coords, demands, capacity)
        for _ in range(nn_seeds):
            population.append(_perturb(nn_chrom))

    # Fill remaining with random chromosomes
    while len(population) < pop_size:
        population.append(create_chromosome(num_customers))

    return population


# ---------------------------------------------------------------------------
# DECODE / ENCODE
# ---------------------------------------------------------------------------

def decode_chromosome(chromosome, demands, capacity):
    """Decode a flat permutation into capacity-valid routes via split_routes."""
    return split_routes(chromosome, demands, capacity)


def chromosome_to_routes(chromosome, demands, capacity):
    """Convenience wrapper for decode_chromosome."""
    return decode_chromosome(chromosome, demands, capacity)
