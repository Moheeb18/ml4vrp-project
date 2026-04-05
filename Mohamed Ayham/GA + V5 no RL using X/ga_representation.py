import numpy as np
import random
from routes_utilities import split_routes


# ---------------------------------------------------------------------------
# CLARK-WRIGHT SAVINGS CORE
# ---------------------------------------------------------------------------

def _build_savings(distance_matrix, n, depot=0,
                   demand_weight=0.0, demands=None, capacity=1,
                   noise=0.0):
    """
    Compute Clark-Wright savings for all customer pairs.

    savings(i,j) = d(depot,i) + d(depot,j) - d(i,j)

    Optional modifiers to produce diverse variants:
      demand_weight : float — blend in combined demand utilisation.
                      Higher values prefer merging routes that together
                      fill the vehicle closer to capacity.
      noise         : float — add uniform random noise scaled by this
                      fraction of the saving value. Breaks ties differently
                      across seeds and shuffles borderline merges.

    Returns list of (saving, i, j) sorted descending.
    """
    savings = []
    for i in range(1, n):
        for j in range(i + 1, n):
            s = (distance_matrix[depot][i]
                 + distance_matrix[depot][j]
                 - distance_matrix[i][j])

            # Demand-utilisation modifier
            if demand_weight > 0 and demands is not None:
                util = (demands[i] + demands[j]) / capacity
                s = s * (1 + demand_weight * util)

            # Noise modifier
            if noise > 0:
                s += random.uniform(-noise, noise) * abs(s)

            savings.append((s, i, j))

    savings.sort(key=lambda x: -x[0])
    return savings


def _apply_savings(savings, demands, capacity, n, depot=0,
                   capacity_pct=1.0):
    """
    Greedily merge routes using a pre-computed savings list.

    capacity_pct : float (0–1] — treat vehicle capacity as this fraction
                   of the real capacity during merging.  Values < 1 leave
                   headroom, forcing more conservative merges and producing
                   routes with fewer customers per vehicle, which gives the
                   GA more room to redistribute.
    """
    effective_cap = capacity * capacity_pct

    routes       = [[depot, i, depot] for i in range(1, n)]
    route_load   = [demands[i] for i in range(1, n)]
    cust_route   = {i: i - 1 for i in range(1, n)}

    for s, i, j in savings:
        ri = cust_route.get(i)
        rj = cust_route.get(j)
        if ri is None or rj is None or ri == rj:
            continue

        route_i = routes[ri]
        route_j = routes[rj]

        i_at_end   = route_i[-2] == i
        j_at_start = route_j[1]  == j
        i_at_start = route_i[1]  == i
        j_at_end   = route_j[-2] == j

        merged = None
        if i_at_end and j_at_start:
            merged = route_i[:-1] + route_j[1:]
        elif j_at_end and i_at_start:
            merged = route_j[:-1] + route_i[1:]
        else:
            continue

        new_load = route_load[ri] + route_load[rj]
        if new_load > effective_cap:
            continue

        routes[ri]     = merged
        route_load[ri] = new_load
        routes[rj]     = []
        for node in merged:
            if node != depot:
                cust_route[node] = ri

    return [r for r in routes if len(r) > 2]


def savings_algorithm(coords, demands, capacity, distance_matrix):
    """Standard Clark-Wright savings — reference implementation."""
    n       = len(coords)
    savings = _build_savings(distance_matrix, n)
    return _apply_savings(savings, demands, capacity, n)


# ---------------------------------------------------------------------------
# DIVERSE SEED GENERATORS
# Each function returns a flat chromosome (customer permutation).
# They all start from Clark-Wright but vary the savings formula,
# tie-breaking, merge priorities, or capacity assumptions so the
# population starts with genuinely different high-quality solutions.
# ---------------------------------------------------------------------------

def _routes_to_chrom(routes):
    chrom = []
    for r in routes:
        chrom.extend(n for n in r if n != 0)
    return chrom


def seed_standard(coords, demands, capacity, distance_matrix):
    """Standard Clark-Wright with no modifications."""
    routes = savings_algorithm(coords, demands, capacity, distance_matrix)
    return _routes_to_chrom(routes)


def seed_noisy(coords, demands, capacity, distance_matrix, noise=0.08):
    """
    Add small random noise to savings before sorting.
    Noise shuffles borderline merge decisions, producing a different
    route structure while still being guided by the savings principle.
    Each call produces a unique result due to randomness.
    """
    n       = len(coords)
    savings = _build_savings(distance_matrix, n, noise=noise)
    routes  = _apply_savings(savings, demands, capacity, n)
    return _routes_to_chrom(routes)


def seed_demand_weighted(coords, demands, capacity, distance_matrix,
                         demand_weight=0.5):
    """
    Weight savings by combined demand utilisation.
    Prefers merging pairs whose combined demand fills the vehicle well,
    producing fuller routes and potentially fewer vehicles.
    """
    n       = len(coords)
    savings = _build_savings(distance_matrix, n,
                             demand_weight=demand_weight,
                             demands=demands, capacity=capacity)
    routes  = _apply_savings(savings, demands, capacity, n)
    return _routes_to_chrom(routes)


def seed_conservative_capacity(coords, demands, capacity, distance_matrix,
                                cap_pct=0.85):
    """
    Use only 85% of vehicle capacity during merging.
    Produces shorter, more conservative routes.  The GA can then
    redistribute customers between these routes more freely.
    """
    n       = len(coords)
    savings = _build_savings(distance_matrix, n)
    routes  = _apply_savings(savings, demands, capacity, n,
                              capacity_pct=cap_pct)
    return _routes_to_chrom(routes)


def seed_reversed_customers(coords, demands, capacity, distance_matrix):
    """
    Process customers in reverse demand order before running savings.
    Changes which customer starts each route and therefore which merges
    are eligible first, producing a structurally different solution.
    """
    n       = len(coords)
    savings = _build_savings(distance_matrix, n)
    # Re-sort savings so high-demand customers are processed first
    # by reversing the tiebreak within equal-savings groups
    savings = sorted(savings,
                     key=lambda x: (-x[0],
                                    -(demands[x[1]] + demands[x[2]])))
    routes  = _apply_savings(savings, demands, capacity, n)
    return _routes_to_chrom(routes)


def seed_distance_penalised(coords, demands, capacity, distance_matrix):
    """
    Penalise savings by the direct distance between i and j.
    Discourages merging customers that are far apart, producing
    geographically tighter routes than standard CW.
    """
    n   = len(coords)
    depot = 0
    savings = []
    for i in range(1, n):
        for j in range(i + 1, n):
            raw_s = (distance_matrix[depot][i]
                     + distance_matrix[depot][j]
                     - distance_matrix[i][j])
            # Extra penalty for pairs that are far from each other
            penalty = 0.3 * distance_matrix[i][j]
            s = raw_s - penalty
            savings.append((s, i, j))
    savings.sort(key=lambda x: -x[0])
    routes = _apply_savings(savings, demands, capacity, n)
    return _routes_to_chrom(routes)


def seed_random_perturbation(coords, demands, capacity, distance_matrix,
                              num_swaps=None):
    """
    Start from standard CW then apply random swaps.
    num_swaps defaults to 5% of customers so the perturbation is
    meaningful but the solution is still CW-quality, not random.
    """
    base = seed_standard(coords, demands, capacity, distance_matrix)
    chrom = base[:]
    n_swaps = num_swaps or max(3, len(chrom) // 20)
    for _ in range(n_swaps):
        i, j = random.sample(range(len(chrom)), 2)
        chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom


# ---------------------------------------------------------------------------
# NEAREST NEIGHBOUR (fallback non-CW seed for diversity)
# ---------------------------------------------------------------------------

def nearest_neighbour_chromosome(coords, demands, capacity):
    """Greedy nearest-neighbour construction."""
    num_customers = len(coords) - 1
    unvisited     = set(range(1, num_customers + 1))
    chromosome    = []

    while unvisited:
        current = 0
        load    = 0
        route   = []
        while True:
            best_next, best_dist = None, float("inf")
            for c in unvisited:
                if load + demands[c] <= capacity:
                    d = np.linalg.norm(coords[current] - coords[c])
                    if d < best_dist:
                        best_dist, best_next = d, c
            if best_next is None:
                break
            route.append(best_next)
            unvisited.remove(best_next)
            load    += demands[best_next]
            current  = best_next
        chromosome.extend(route)

    return chromosome


def create_chromosome(num_customers):
    """Pure random permutation."""
    customers = list(range(1, num_customers + 1))
    np.random.shuffle(customers)
    return customers


# ---------------------------------------------------------------------------
# POPULATION INITIALISATION
# ---------------------------------------------------------------------------

# All diverse seed functions, each producing a structurally different
# Clark-Wright solution via a different savings formula or merge strategy.
_SEED_VARIANTS = [
    seed_standard,
    seed_demand_weighted,
    seed_conservative_capacity,
    seed_reversed_customers,
    seed_distance_penalised,
]


def initialize_population(pop_size, num_customers,
                           coords=None, demands=None,
                           capacity=None, distance_matrix=None):
    """
    Build an initial population with a diverse seeding strategy:

      Tier 1 — Diverse Clark-Wright seeds (40% of population)
        Each of the 5 CW variants produces a structurally different
        solution by varying the savings formula, tie-breaking rule,
        merge priority, or capacity assumption.  Each variant is then
        lightly perturbed (5% random swaps) to generate multiple unique
        chromosomes from the same base, ensuring no two seeds are identical.

      Tier 2 — Nearest-neighbour seeds (20% of population)
        Fast greedy construction from a different principle (proximity
        rather than savings), adding diversity the CW seeds cannot cover.

      Tier 3 — Random shuffles (40% of population)
        Maintains population diversity and prevents premature convergence.

    Args:
        pop_size        : total number of chromosomes
        num_customers   : number of customers (excluding depot)
        coords          : np.array (enables smart seeding)
        demands         : np.array
        capacity        : int
        distance_matrix : 2D np.array

    Returns:
        population (list of lists)
    """
    population    = []
    smart_seeding = (coords is not None and demands is not None
                     and capacity is not None and distance_matrix is not None)

    if smart_seeding:

        # ---- Tier 1: diverse Clark-Wright (40%) ---------------------------
        cw_budget = pop_size * 2 // 5          # 40%
        per_variant = max(1, cw_budget // len(_SEED_VARIANTS))

        for seed_fn in _SEED_VARIANTS:
            try:
                base = seed_fn(coords, demands, capacity, distance_matrix)
            except Exception:
                base = nearest_neighbour_chromosome(coords, demands, capacity)

            for k in range(per_variant):
                if k == 0:
                    # First copy: pure seed, no perturbation
                    population.append(base[:])
                else:
                    # Subsequent copies: lightly perturbed
                    chrom  = base[:]
                    n_swap = max(3, len(chrom) // 20)
                    for _ in range(n_swap):
                        i, j = random.sample(range(len(chrom)), 2)
                        chrom[i], chrom[j] = chrom[j], chrom[i]
                    population.append(chrom)

            # Also add one noisy variant per seed for extra variety
            try:
                noisy = seed_noisy(coords, demands, capacity, distance_matrix,
                                   noise=random.uniform(0.05, 0.15))
                population.append(noisy)
            except Exception:
                pass

        # ---- Tier 2: nearest-neighbour (20%) ------------------------------
        nn_budget = pop_size // 5
        nn_base   = nearest_neighbour_chromosome(coords, demands, capacity)
        for k in range(nn_budget):
            chrom = nn_base[:]
            if k > 0:
                n_swap = max(3, len(chrom) // 20)
                for _ in range(n_swap):
                    i, j = random.sample(range(len(chrom)), 2)
                    chrom[i], chrom[j] = chrom[j], chrom[i]
            population.append(chrom)

    # ---- Tier 3: random fill ----------------------------------------------
    while len(population) < pop_size:
        population.append(create_chromosome(num_customers))

    # Trim if over budget
    return population[:pop_size]


# ---------------------------------------------------------------------------
# DECODE / ENCODE
# ---------------------------------------------------------------------------

def decode_chromosome(chromosome, demands, capacity):
    """Decode flat permutation → capacity-valid routes via split_routes."""
    return split_routes(chromosome, demands, capacity)


def chromosome_to_routes(chromosome, demands, capacity):
    return decode_chromosome(chromosome, demands, capacity)
