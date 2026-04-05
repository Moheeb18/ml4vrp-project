from routes_utilities import route_distance
from ga_fitness import fitness_from_routes, VEHICLE_PENALTY


# ---------------------------------------------------------------------------
# INTRA-ROUTE 2-OPT
# ---------------------------------------------------------------------------

def two_opt_route(route, distance_matrix):
    """
    Apply 2-opt to a single route.
    Reverses sub-segments until no distance-reducing swap exists.
    """
    best      = route[:]
    best_dist = route_distance(best, distance_matrix)
    improved  = True

    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                new_dist  = route_distance(new_route, distance_matrix)
                if new_dist < best_dist - 1e-10:
                    best      = new_route
                    best_dist = new_dist
                    improved  = True
    return best


def two_opt_solution(routes, distance_matrix):
    """Apply 2-opt to every route independently."""
    improved = [two_opt_route(r, distance_matrix) for r in routes]
    total    = sum(route_distance(r, distance_matrix) for r in improved)
    return improved, total


# ---------------------------------------------------------------------------
# INTRA-ROUTE OR-OPT
# ---------------------------------------------------------------------------

def or_opt_route(route, distance_matrix):
    """
    Relocate individual customers within a single route.
    Tries every (customer, insertion_position) pair until no improvement.
    """
    best      = route[:]
    best_dist = route_distance(best, distance_matrix)
    improved  = True

    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            customer  = best[i]
            remaining = best[:i] + best[i+1:]
            for j in range(1, len(remaining)):
                candidate = remaining[:j] + [customer] + remaining[j:]
                cand_dist = route_distance(candidate, distance_matrix)
                if cand_dist < best_dist - 1e-10:
                    best      = candidate
                    best_dist = cand_dist
                    improved  = True
                    break
            if improved:
                break
    return best


def or_opt_solution(routes, distance_matrix):
    """Apply OR-opt to every route independently."""
    improved = [or_opt_route(r, distance_matrix) for r in routes]
    total    = sum(route_distance(r, distance_matrix) for r in improved)
    return improved, total


# ---------------------------------------------------------------------------
# INTER-ROUTE 2-OPT* (from Solver_with_2opstar.py)
# ---------------------------------------------------------------------------

def two_opt_star(routes, demands, capacity, distance_matrix):
    """
    2-opt* inter-route improvement operator.

    For every pair of routes, tries exchanging their suffixes at every
    possible split point. Unlike standard 2-opt which reverses a segment
    within one route, 2-opt* reconnects two routes by swapping their tails:

        Route 1: depot → ... → i | i+1 → ... → depot
        Route 2: depot → ... → j | j+1 → ... → depot

        After 2-opt*:
        New 1: depot → ... → i | j+1 → ... → depot
        New 2: depot → ... → j | i+1 → ... → depot

    This is a powerful inter-route operator because it can restructure
    large portions of two routes simultaneously. It is taken directly
    from Solver_with_2opstar.py and integrated here as part of the
    local search pipeline.

    Only accepts moves that:
      - Improve total distance
      - Keep both new routes capacity-feasible
      - Keep depot at start and end of each route

    Args:
        routes          : list of routes [0,...,0]
        demands         : np.array
        capacity        : int
        distance_matrix : 2D np.array

    Returns:
        routes (list): improved routes
        improved (bool): whether any move was accepted
    """
    improved     = False
    current_dist = sum(route_distance(r, distance_matrix) for r in routes)

    for r1 in range(len(routes)):
        for r2 in range(r1 + 1, len(routes)):
            route1 = routes[r1]
            route2 = routes[r2]

            if len(route1) <= 2 or len(route2) <= 2:
                continue

            for i in range(len(route1) - 1):
                for j in range(len(route2) - 1):
                    if i == 0 and j == 0:
                        continue
                    if i == len(route1) - 2 and j == len(route2) - 2:
                        continue

                    prefix1 = route1[:i + 1]
                    suffix1 = route1[i + 1:]
                    prefix2 = route2[:j + 1]
                    suffix2 = route2[j + 1:]

                    cand1 = prefix1 + suffix2
                    cand2 = prefix2 + suffix1

                    # Both must start and end at depot
                    if (cand1[0] != 0 or cand1[-1] != 0
                            or cand2[0] != 0 or cand2[-1] != 0):
                        continue

                    # Capacity check
                    load1 = sum(demands[c] for c in cand1 if c != 0)
                    load2 = sum(demands[c] for c in cand2 if c != 0)
                    if load1 > capacity or load2 > capacity:
                        continue

                    new_routes    = [r[:] for r in routes]
                    new_routes[r1] = cand1
                    new_routes[r2] = cand2
                    new_routes     = [r for r in new_routes if len(r) > 2]

                    new_dist = sum(route_distance(r, distance_matrix)
                                   for r in new_routes)

                    if new_dist < current_dist - 1e-10:
                        routes       = new_routes
                        current_dist = new_dist
                        improved     = True
                        return routes, improved   # restart after any improvement

    return routes, improved


def two_opt_star_solution(routes, demands, capacity, distance_matrix):
    """
    Apply 2-opt* repeatedly until no improvement can be found.
    """
    made_improvement = True
    while made_improvement:
        routes, made_improvement = two_opt_star(
            routes, demands, capacity, distance_matrix)
    total = sum(route_distance(r, distance_matrix) for r in routes)
    return routes, total


# ---------------------------------------------------------------------------
# INTER-ROUTE RELOCATION
# ---------------------------------------------------------------------------

def inter_route_relocate(routes, demands, capacity, distance_matrix):
    """
    Move one customer from one route to a better position in another route.
    Accepts move if competition objective (1000*NV + TD) improves.
    Removes empty routes automatically.
    """
    best_obj = fitness_from_routes(routes, distance_matrix)
    n_routes = len(routes)

    for r_from in range(n_routes):
        customers = [c for c in routes[r_from] if c != 0]

        for customer in customers:
            new_from = [0] + [c for c in customers if c != customer] + [0]

            for r_to in range(n_routes):
                if r_to == r_from:
                    continue
                to_load = sum(demands[c] for c in routes[r_to] if c != 0)
                if to_load + demands[customer] > capacity:
                    continue

                for ins_pos in range(1, len(routes[r_to])):
                    new_to    = (routes[r_to][:ins_pos]
                                 + [customer]
                                 + routes[r_to][ins_pos:])
                    candidate = []
                    for ri, r in enumerate(routes):
                        if ri == r_from:
                            candidate.append(new_from)
                        elif ri == r_to:
                            candidate.append(new_to)
                        else:
                            candidate.append(r)
                    candidate = [r for r in candidate if len(r) > 2]
                    cand_obj  = fitness_from_routes(candidate, distance_matrix)
                    if cand_obj < best_obj - 1e-10:
                        return candidate, True

    return routes, False


# ---------------------------------------------------------------------------
# ROUTE ELIMINATION
# ---------------------------------------------------------------------------

def route_elimination(routes, demands, capacity, distance_matrix):
    """
    Try to eliminate the smallest route by redistributing its customers
    into other routes using cheapest insertion. Each elimination saves
    1000 objective units (one fewer vehicle).
    """
    if len(routes) <= 1:
        return routes, False

    sorted_by_size = sorted(
        range(len(routes)), key=lambda i: len(routes[i]) - 2)

    for target_idx in sorted_by_size:
        target_custs = [c for c in routes[target_idx] if c != 0]
        working      = [r[:] for i, r in enumerate(routes) if i != target_idx]
        success      = True

        for customer in target_custs:
            best_cost = float("inf")
            best_r    = -1
            best_pos  = -1

            for ri, route in enumerate(working):
                load = sum(demands[c] for c in route if c != 0)
                if load + demands[customer] > capacity:
                    continue
                for pos in range(1, len(route)):
                    prev = route[pos - 1]
                    nxt  = route[pos]
                    cost = (distance_matrix[prev][customer]
                            + distance_matrix[customer][nxt]
                            - distance_matrix[prev][nxt])
                    if cost < best_cost:
                        best_cost = cost
                        best_r    = ri
                        best_pos  = pos

            if best_r == -1:
                success = False
                break

            working[best_r] = (working[best_r][:best_pos]
                               + [customer]
                               + working[best_r][best_pos:])

        if success:
            new_obj = fitness_from_routes(working, distance_matrix)
            old_obj = fitness_from_routes(routes, distance_matrix)
            if new_obj < old_obj - 1e-10:
                return working, True

    return routes, False


# ---------------------------------------------------------------------------
# INTER-ROUTE SWAP
# ---------------------------------------------------------------------------

def inter_route_swap(routes, demands, capacity, distance_matrix):
    """
    Swap one customer between two routes if objective improves and
    both capacity constraints still hold after the swap.
    """
    best_obj = fitness_from_routes(routes, distance_matrix)
    n_routes = len(routes)

    for r1 in range(n_routes):
        for r2 in range(r1 + 1, n_routes):
            custs1 = [c for c in routes[r1] if c != 0]
            custs2 = [c for c in routes[r2] if c != 0]
            load1  = sum(demands[c] for c in custs1)
            load2  = sum(demands[c] for c in custs2)

            for c1 in custs1:
                for c2 in custs2:
                    if (load1 - demands[c1] + demands[c2] > capacity or
                            load2 - demands[c2] + demands[c1] > capacity):
                        continue

                    new_r1    = [c2 if c == c1 else c for c in routes[r1]]
                    new_r2    = [c1 if c == c2 else c for c in routes[r2]]
                    candidate = [
                        new_r1 if i == r1 else (new_r2 if i == r2 else r)
                        for i, r in enumerate(routes)
                    ]
                    cand_obj = fitness_from_routes(candidate, distance_matrix)
                    if cand_obj < best_obj - 1e-10:
                        return candidate, True

    return routes, False


# ---------------------------------------------------------------------------
# FULL LOCAL SEARCH PIPELINE
# ---------------------------------------------------------------------------

def local_search(routes, distance_matrix, demands=None, capacity=None):
    """
    Full local search pipeline in pass order:
      1. 2-opt              — intra-route segment reversal
      2. OR-opt             — intra-route customer relocation
      3. 2-opt*             — inter-route suffix exchange (needs demands/capacity)
      4. Route elimination  — remove smallest route via cheapest insertion
      5. Inter-route relocation — move customers between routes
      6. Inter-route swap   — swap customers between routes

    All passes repeat until no operator can improve the objective further.
    """
    inter_route_enabled = (demands is not None and capacity is not None)
    prev_obj = float("inf")

    while True:

        # Intra-route passes
        routes, _ = two_opt_solution(routes, distance_matrix)
        routes, _ = or_opt_solution(routes, distance_matrix)

        if inter_route_enabled:
            # 2-opt* — powerful inter-route restructuring
            routes, _ = two_opt_star_solution(
                routes, demands, capacity, distance_matrix)

            # Route elimination — biggest gain per move (1000 per route)
            eliminated = True
            while eliminated:
                routes, eliminated = route_elimination(
                    routes, demands, capacity, distance_matrix)

            # Inter-route relocation
            moved = True
            while moved:
                routes, moved = inter_route_relocate(
                    routes, demands, capacity, distance_matrix)

            # Inter-route swap
            swapped = True
            while swapped:
                routes, swapped = inter_route_swap(
                    routes, demands, capacity, distance_matrix)

        current_obj = fitness_from_routes(routes, distance_matrix)
        if prev_obj - current_obj < 1e-10:
            break
        prev_obj = current_obj

    return routes, fitness_from_routes(routes, distance_matrix)


# ---------------------------------------------------------------------------
# CHROMOSOME RE-ENCODING
# ---------------------------------------------------------------------------

def routes_to_chromosome(routes):
    """Re-encode routes back into a flat chromosome permutation."""
    chromosome = []
    for route in routes:
        chromosome.extend([n for n in route if n != 0])
    return chromosome
