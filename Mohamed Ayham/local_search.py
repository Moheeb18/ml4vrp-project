import sys
import os
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent)
import src.routes_utilities
from src.routes_utilities import route_distance
from ga_fitness import fitness_from_routes, VEHICLE_PENALTY


# ---------------------------------------------------------------------------
# INTRA-ROUTE 2-OPT
# ---------------------------------------------------------------------------

def two_opt_route(route, distance_matrix):
    """
    Apply 2-opt improvement to a single route.
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
    """Apply 2-opt to every route in the solution independently."""
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
    """Apply OR-opt to every route in the solution independently."""
    improved = [or_opt_route(r, distance_matrix) for r in routes]
    total    = sum(route_distance(r, distance_matrix) for r in improved)
    return improved, total


# ---------------------------------------------------------------------------
# INTER-ROUTE RELOCATION
# Move one customer from one route to a different route if it improves
# the competition objective (1000*NV + TD). Crucially this can eliminate
# entire routes, saving 1000 units per route removed.
# ---------------------------------------------------------------------------

def inter_route_relocate(routes, demands, capacity, distance_matrix):
    """
    Try moving each customer from its current route to every position in
    every other route. Accept the move if it improves the competition
    objective (1000*NV + TD).

    When a route becomes empty after a relocation, it is removed — this
    reduces vehicle count by 1, saving 1000 objective units.

    Args:
        routes          : list of routes [0,...,0]
        demands         : np.array of demands
        capacity        : int, vehicle capacity
        distance_matrix : 2D np.array

    Returns:
        routes     (list) : improved routes
        improved   (bool) : whether any move was accepted
    """
    best_obj  = fitness_from_routes(routes, distance_matrix)
    improved  = False
    n_routes  = len(routes)

    for r_from in range(n_routes):
        route_from = routes[r_from]
        customers  = [c for c in route_from if c != 0]

        for cust_idx, customer in enumerate(customers):
            # Build route_from without this customer
            new_from = [0] + [c for c in customers if c != customer] + [0]

            for r_to in range(n_routes):
                if r_to == r_from:
                    continue

                route_to = routes[r_to]
                to_load  = sum(demands[c] for c in route_to if c != 0)

                if to_load + demands[customer] > capacity:
                    continue    # would violate capacity

                # Try inserting customer at every position in route_to
                for ins_pos in range(1, len(route_to)):
                    new_to = route_to[:ins_pos] + [customer] + route_to[ins_pos:]

                    # Build candidate full solution
                    candidate = []
                    for ri, r in enumerate(routes):
                        if ri == r_from:
                            candidate.append(new_from)
                        elif ri == r_to:
                            candidate.append(new_to)
                        else:
                            candidate.append(r)

                    # Remove empty routes (saves 1000 per route)
                    candidate = [r for r in candidate if len(r) > 2]

                    cand_obj = fitness_from_routes(candidate, distance_matrix)

                    if cand_obj < best_obj - 1e-10:
                        routes   = candidate
                        best_obj = cand_obj
                        improved = True
                        # Restart — route indices have changed
                        return routes, improved

    return routes, improved


# ---------------------------------------------------------------------------
# ROUTE ELIMINATION
# Actively try to eliminate the smallest route by redistributing its
# customers to other routes. Each eliminated route saves 1000 units.
# ---------------------------------------------------------------------------

def route_elimination(routes, demands, capacity, distance_matrix):
    """
    Try to eliminate the smallest route by inserting each of its customers
    into the best feasible position in any other route.

    If all customers can be redistributed, the route is removed.
    This directly targets the primary competition objective (min vehicles).

    Args:
        routes          : list of routes
        demands         : np.array
        capacity        : int
        distance_matrix : 2D np.array

    Returns:
        routes    (list): solution with route eliminated if possible
        eliminated (bool): whether a route was successfully removed
    """
    if len(routes) <= 1:
        return routes, False

    # Sort by number of customers — try to eliminate smallest route first
    sorted_by_size = sorted(
        range(len(routes)),
        key=lambda i: len(routes[i]) - 2    # exclude depot nodes
    )

    for target_idx in sorted_by_size:
        target_route  = routes[target_idx]
        target_custs  = [c for c in target_route if c != 0]
        other_routes  = [r[:] for i, r in enumerate(routes) if i != target_idx]

        # Try to insert every customer from target route into other routes
        working = [r[:] for r in other_routes]
        success = True

        for customer in target_custs:
            best_cost   = float("inf")
            best_r      = -1
            best_pos    = -1

            for ri, route in enumerate(working):
                load = sum(demands[c] for c in route if c != 0)
                if load + demands[customer] > capacity:
                    continue

                for pos in range(1, len(route)):
                    # Insertion cost delta (avoid recomputing full distance)
                    prev_node = route[pos - 1]
                    next_node = route[pos]
                    cost = (distance_matrix[prev_node][customer]
                            + distance_matrix[customer][next_node]
                            - distance_matrix[prev_node][next_node])
                    if cost < best_cost:
                        best_cost = cost
                        best_r    = ri
                        best_pos  = pos

            if best_r == -1:
                success = False
                break

            # Insert the customer
            working[best_r] = (working[best_r][:best_pos]
                               + [customer]
                               + working[best_r][best_pos:])

        if success:
            # Verify the elimination actually improves objective
            new_obj = fitness_from_routes(working, distance_matrix)
            old_obj = fitness_from_routes(routes, distance_matrix)
            if new_obj < old_obj - 1e-10:
                return working, True

    return routes, False


# ---------------------------------------------------------------------------
# INTER-ROUTE SWAP
# Swap one customer between two different routes if it improves objective
# and both capacity constraints remain satisfied.
# ---------------------------------------------------------------------------

def inter_route_swap(routes, demands, capacity, distance_matrix):
    """
    Try swapping one customer from route A with one customer from route B.
    Accept if the competition objective improves and capacities hold.
    """
    best_obj = fitness_from_routes(routes, distance_matrix)
    n_routes = len(routes)
    improved = False

    for r1 in range(n_routes):
        for r2 in range(r1 + 1, n_routes):
            custs1 = [c for c in routes[r1] if c != 0]
            custs2 = [c for c in routes[r2] if c != 0]
            load1  = sum(demands[c] for c in custs1)
            load2  = sum(demands[c] for c in custs2)

            for c1 in custs1:
                for c2 in custs2:
                    # Check capacity after swap
                    new_load1 = load1 - demands[c1] + demands[c2]
                    new_load2 = load2 - demands[c2] + demands[c1]

                    if new_load1 > capacity or new_load2 > capacity:
                        continue

                    new_r1 = [c2 if c == c1 else c for c in routes[r1]]
                    new_r2 = [c1 if c == c2 else c for c in routes[r2]]

                    candidate = [
                        new_r1 if i == r1 else (new_r2 if i == r2 else r)
                        for i, r in enumerate(routes)
                    ]

                    cand_obj = fitness_from_routes(candidate, distance_matrix)
                    if cand_obj < best_obj - 1e-10:
                        routes   = candidate
                        best_obj = cand_obj
                        improved = True
                        return routes, improved

    return routes, improved


# ---------------------------------------------------------------------------
# FULL LOCAL SEARCH PIPELINE
# ---------------------------------------------------------------------------

def local_search(routes, distance_matrix, demands=None, capacity=None):
    """
    Full local search pipeline combining all operators.

    Pass order (inner loop, repeated until stable):
      1. Intra-route 2-opt       — fix crossing paths within routes
      2. Intra-route OR-opt      — relocate customers within routes
      3. Inter-route relocation  — move customers between routes (needs demands/capacity)
      4. Inter-route swap        — swap customers between routes
      5. Route elimination       — try to remove the smallest route entirely

    Inter-route and elimination steps only run if demands and capacity are
    provided. This keeps the function backwards-compatible with calls that
    only do intra-route improvement.

    Args:
        routes          : list of routes
        distance_matrix : 2D np.array
        demands         : np.array (optional, enables inter-route ops)
        capacity        : int (optional)

    Returns:
        routes     (list) : locally optimised routes
        total_obj  (float): competition objective value after local search
    """
    inter_route_enabled = (demands is not None and capacity is not None)
    prev_obj = float("inf")

    while True:

        # -- Intra-route passes ---------------------------------------------
        routes, _ = two_opt_solution(routes, distance_matrix)
        routes, _ = or_opt_solution(routes, distance_matrix)

        # -- Inter-route passes (only if demands/capacity available) --------
        if inter_route_enabled:

            # Route elimination first — biggest possible gain (1000 per route)
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

        # -- Check convergence ----------------------------------------------
        current_obj = fitness_from_routes(routes, distance_matrix)
        if prev_obj - current_obj < 1e-10:
            break
        prev_obj = current_obj

    return routes, fitness_from_routes(routes, distance_matrix)


# ---------------------------------------------------------------------------
# CHROMOSOME RE-ENCODING
# ---------------------------------------------------------------------------

def routes_to_chromosome(routes):
    """
    Re-encode a list of routes back into a flat chromosome permutation.
    Strips the depot (node 0) and concatenates customer sequences.
    """
    chromosome = []
    for route in routes:
        chromosome.extend([n for n in route if n != 0])
    return chromosome
