from routes_utilities import route_distance


# ---------------------------------------------------------------------------
# INTRA-ROUTE 2-OPT
# ---------------------------------------------------------------------------

def two_opt_route(route, distance_matrix):
    """
    Apply 2-opt improvement to a single route.

    Repeatedly tries reversing every sub-segment [i+1 .. j] and keeps the
    change if it reduces the route distance. Runs until no improvement found.

    Args:
        route           : list of node indices, e.g. [0, 3, 7, 2, 0]
        distance_matrix : 2D np.array of pairwise distances

    Returns:
        best_route (list): improved route (same start/end depot)
    """
    best = route[:]
    best_dist = route_distance(best, distance_matrix)
    improved = True

    while improved:
        improved = False
        # Only iterate over the customer portion (exclude depots at ends)
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                # Reverse the segment between i and j
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                new_dist  = route_distance(new_route, distance_matrix)
                if new_dist < best_dist - 1e-10:
                    best      = new_route
                    best_dist = new_dist
                    improved  = True

    return best


def two_opt_solution(routes, distance_matrix):
    """
    Apply 2-opt to every route in a solution independently.

    Args:
        routes          : list of routes (each a list like [0, ..., 0])
        distance_matrix : 2D np.array

    Returns:
        improved_routes (list): all routes after 2-opt
        total_dist      (float): total distance after improvement
    """
    improved_routes = [two_opt_route(r, distance_matrix) for r in routes]
    total_dist = sum(route_distance(r, distance_matrix) for r in improved_routes)
    return improved_routes, total_dist


# ---------------------------------------------------------------------------
# OR-OPT  (move 1 customer to a better position within the same route)
# ---------------------------------------------------------------------------

def or_opt_route(route, distance_matrix):
    """
    OR-opt (1-customer relocation) within a single route.

    Tries moving each customer to every other position in the route and
    keeps the move if it reduces distance. Complements 2-opt well because
    it catches improvements that segment-reversal misses.

    Args:
        route           : list of node indices e.g. [0, 3, 7, 2, 0]
        distance_matrix : 2D np.array

    Returns:
        best_route (list): improved route
    """
    best = route[:]
    best_dist = route_distance(best, distance_matrix)
    improved = True

    while improved:
        improved = False
        for i in range(1, len(best) - 1):          # pick a customer to move
            customer = best[i]
            remaining = best[:i] + best[i+1:]      # route without that customer

            for j in range(1, len(remaining)):      # try every insertion point
                candidate = remaining[:j] + [customer] + remaining[j:]
                cand_dist = route_distance(candidate, distance_matrix)
                if cand_dist < best_dist - 1e-10:
                    best      = candidate
                    best_dist = cand_dist
                    improved  = True
                    break           # restart from new best

            if improved:
                break

    return best


def or_opt_solution(routes, distance_matrix):
    """
    Apply OR-opt to every route in a solution independently.
    """
    improved_routes = [or_opt_route(r, distance_matrix) for r in routes]
    total_dist = sum(route_distance(r, distance_matrix) for r in improved_routes)
    return improved_routes, total_dist


# ---------------------------------------------------------------------------
# COMBINED LOCAL SEARCH  (2-opt then OR-opt, repeated until stable)
# ---------------------------------------------------------------------------

def local_search(routes, distance_matrix):
    """
    Apply 2-opt and OR-opt in alternation until neither can improve further.

    This is the function called by the GA. Running both operators together
    consistently beats either alone.

    Args:
        routes          : list of routes (each [0, ..., 0])
        distance_matrix : 2D np.array

    Returns:
        routes     (list) : fully locally-optimised routes
        total_dist (float): total distance after local search
    """
    prev_dist = float("inf")

    while True:
        routes, dist_after_2opt  = two_opt_solution(routes, distance_matrix)
        routes, dist_after_oropt = or_opt_solution(routes, distance_matrix)

        # Stop when neither pass improved things
        if prev_dist - dist_after_oropt < 1e-10:
            break
        prev_dist = dist_after_oropt

    total_dist = sum(route_distance(r, distance_matrix) for r in routes)
    return routes, total_dist


# ---------------------------------------------------------------------------
# CHROMOSOME HELPERS
# ---------------------------------------------------------------------------

def routes_to_chromosome(routes):
    """
    Re-encode a list of routes back into a flat chromosome (permutation).

    Strips the depot (node 0) from each route and concatenates customers.
    Used after local search to put the improved solution back into the GA.

    Args:
        routes : list of routes e.g. [[0,3,7,0], [0,1,5,0]]

    Returns:
        chromosome (list): e.g. [3, 7, 1, 5]
    """
    chromosome = []
    for route in routes:
        chromosome.extend([node for node in route if node != 0])
    return chromosome
