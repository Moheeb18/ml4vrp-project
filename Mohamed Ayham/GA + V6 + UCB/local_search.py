from routes_utilities import route_distance
from ga_fitness import fitness_from_routes, VEHICLE_PENALTY
from Solver_V6 import (
    improve_routes,
    build_neighbor_lists,
    optimize_routes_neighbor_lists,
    two_opt,
    two_opt_star,
    total_distance,
    lns,
)


# ---------------------------------------------------------------------------
# LNS ITERATION BUDGET  (size-dependent to keep runtime sensible)
# ---------------------------------------------------------------------------

def _lns_budget(n, final=False):
    """
    Return (lns_iterations, remove_ratio, k) tuned to instance size n.

    When final=True the budget is tripled — the final pass runs only once
    so the extra cost is acceptable and meaningfully closes the gap on
    large instances. A-n32 (n<80) is unaffected in character; it just gets
    60 iters instead of 20, which completes in seconds anyway.

    During-loop (final=False):  Small=20, Medium=10, Large=5
    Final pass  (final=True):   Small=60, Medium=30, Large=15
    """
    multiplier = 3 if final else 1
    if n < 80:
        return 20 * multiplier, 0.15, 15
    elif n < 200:
        return 10 * multiplier, 0.15, 20
    else:
        return  5 * multiplier, 0.12, 20


# ---------------------------------------------------------------------------
# INTRA-ROUTE 2-OPT  (thin wrapper around Solver_V6)
# ---------------------------------------------------------------------------

def two_opt_solution(routes, distance_matrix):
    improved = [two_opt(r, distance_matrix) for r in routes]
    total    = sum(route_distance(r, distance_matrix) for r in improved)
    return improved, total


# ---------------------------------------------------------------------------
# INTRA-ROUTE OR-OPT
# ---------------------------------------------------------------------------

def or_opt_route(route, distance_matrix, chain_lengths=(1, 2, 3)):
    """
    OR-opt with chain lengths 1, 2, and 3.

    Tries relocating single customers (chain=1), pairs (chain=2), and
    triples (chain=3) to cheaper positions within the same route.
    Chain lengths 2 and 3 find improvements that single-customer moves
    miss entirely, especially on long routes with clustered customers.
    This is a strict superset of the old single-customer OR-opt — it
    can only improve or match the previous result, never worsen it.
    """
    best      = route[:]
    best_dist = route_distance(best, distance_matrix)
    improved  = True
    while improved:
        improved = False
        for length in chain_lengths:
            # Need at least length+2 nodes (depot + segment + depot)
            if len(best) < length + 3:
                continue
            for i in range(1, len(best) - length):
                segment   = best[i:i + length]
                remaining = best[:i] + best[i + length:]
                for j in range(1, len(remaining)):
                    candidate = remaining[:j] + segment + remaining[j:]
                    cand_dist = route_distance(candidate, distance_matrix)
                    if cand_dist < best_dist - 1e-10:
                        best      = candidate
                        best_dist = cand_dist
                        improved  = True
                        break
                if improved:
                    break
            if improved:
                break
    return best


def or_opt_solution(routes, distance_matrix):
    improved = [or_opt_route(r, distance_matrix) for r in routes]
    total    = sum(route_distance(r, distance_matrix) for r in improved)
    return improved, total


# ---------------------------------------------------------------------------
# ROUTE ELIMINATION
# ---------------------------------------------------------------------------

def route_elimination(routes, demands, capacity, distance_matrix):
    """
    Try to eliminate the smallest route via cheapest insertion into others.
    Saves VEHICLE_PENALTY objective units per route removed.
    """
    if len(routes) <= 1:
        return routes, False

    sorted_by_size = sorted(range(len(routes)), key=lambda i: len(routes[i]) - 2)

    for target_idx in sorted_by_size:
        target_custs = [c for c in routes[target_idx] if c != 0]
        working      = [r[:] for i, r in enumerate(routes) if i != target_idx]
        success      = True

        for customer in target_custs:
            best_cost, best_r, best_pos = float("inf"), -1, -1
            for ri, route in enumerate(working):
                load = sum(demands[c] for c in route if c != 0)
                if load + demands[customer] > capacity:
                    continue
                for pos in range(1, len(route)):
                    prev, nxt = route[pos - 1], route[pos]
                    cost = (distance_matrix[prev][customer]
                            + distance_matrix[customer][nxt]
                            - distance_matrix[prev][nxt])
                    if cost < best_cost:
                        best_cost, best_r, best_pos = cost, ri, pos
            if best_r == -1:
                success = False
                break
            working[best_r] = (working[best_r][:best_pos]
                               + [customer]
                               + working[best_r][best_pos:])

        if success:
            if fitness_from_routes(working, distance_matrix) < fitness_from_routes(routes, distance_matrix) - 1e-10:
                return working, True

    return routes, False


# ---------------------------------------------------------------------------
# INTER-ROUTE RELOCATION
# ---------------------------------------------------------------------------

def inter_route_relocate(routes, demands, capacity, distance_matrix):
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
                    new_to    = routes[r_to][:ins_pos] + [customer] + routes[r_to][ins_pos:]
                    candidate = []
                    for ri, r in enumerate(routes):
                        if ri == r_from:
                            candidate.append(new_from)
                        elif ri == r_to:
                            candidate.append(new_to)
                        else:
                            candidate.append(r)
                    candidate = [r for r in candidate if len(r) > 2]
                    if fitness_from_routes(candidate, distance_matrix) < best_obj - 1e-10:
                        return candidate, True

    return routes, False


# ---------------------------------------------------------------------------
# INTER-ROUTE SWAP
# ---------------------------------------------------------------------------

def inter_route_swap(routes, demands, capacity, distance_matrix):
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
                    candidate = [new_r1 if i == r1 else (new_r2 if i == r2 else r)
                                 for i, r in enumerate(routes)]
                    if fitness_from_routes(candidate, distance_matrix) < best_obj - 1e-10:
                        return candidate, True

    return routes, False


# ---------------------------------------------------------------------------
# FULL LOCAL SEARCH PIPELINE
# ---------------------------------------------------------------------------

def local_search(routes, distance_matrix, demands=None, capacity=None,
                 final=False):
    """
    Full local search pipeline using Solver_V6 operators.

    Pass order (repeated until stable):
      1. 2-opt              — intra-route (Solver_V6)
      2. OR-opt             — intra-route chains 1/2/3
      3. Neighbor-list relocate + swap + 2-opt loop  (Solver_V6)
      4. LNS               — destroy/repair, budget scaled by instance size
                             and tripled when final=True (end-of-run pass)
      5. 2-opt*            — inter-route suffix exchange (skipped for n>300)
      6. Route elimination  — remove smallest route via cheapest insertion
      7. Inter-route relocation + swap — fine-grained inter-route moves

    Args:
        final : bool — if True, use 3x LNS iteration budget. Set to True
                only for the end-of-run pass in ga_core. Keeps per-gen LS
                fast while allowing a thorough final improvement sweep.
    """
    inter_route_enabled = (demands is not None and capacity is not None)
    n        = len(distance_matrix)
    prev_obj = float("inf")

    # Size-dependent LNS budget — tripled on the final pass
    lns_iters, remove_ratio, k = _lns_budget(n, final=final)

    # Skip two_opt_star on very large instances (O(n^4) cost)
    run_two_opt_star = (n <= 300)

    while True:
        # -- Intra-route ---------------------------------------------------
        routes, _ = two_opt_solution(routes, distance_matrix)
        routes, _ = or_opt_solution(routes, distance_matrix)

        if inter_route_enabled:
            # Neighbor-list accelerated relocate + swap + intra 2-opt (V6)
            routes = optimize_routes_neighbor_lists(
                routes, demands, capacity, distance_matrix, k)

            # LNS: destroy/repair with regret insertion (replaces try_merge)
            routes = lns(
                routes, demands, capacity, distance_matrix,
                iterations=lns_iters,
                remove_ratio=remove_ratio,
                k=k,
            )

            # 2-opt* inter-route suffix exchange (skip for very large n)
            if run_two_opt_star:
                routes = two_opt_star(routes, demands, capacity, distance_matrix)

            # Route elimination: remove smallest route via cheapest insertion
            eliminated = True
            while eliminated:
                routes, eliminated = route_elimination(
                    routes, demands, capacity, distance_matrix)

            # Fine-grained inter-route moves
            moved = True
            while moved:
                routes, moved = inter_route_relocate(
                    routes, demands, capacity, distance_matrix)

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
# LIGHT LOCAL SEARCH  (used during GA loop every 25 generations)
# ---------------------------------------------------------------------------

def local_search_light(routes, distance_matrix, demands=None, capacity=None):
    """
    Fast local search for use inside the GA loop.

    Operators applied (all fast, no LNS, no 2-opt*):
      1. 2-opt              — intra-route
      2. OR-opt             — intra-route customer relocation
      3. Neighbor-list relocate + swap + intra 2-opt  (Solver_V6)
      4. Route elimination  — remove smallest route via cheapest insertion

    Runs in a small fraction of the time of the full local_search pipeline,
    allowing the GA to complete many real generations rather than spending
    all its time in local search.

    Args:
        routes          : list of routes
        distance_matrix : 2D np.array
        demands         : np.array (required for inter-route moves)
        capacity        : int (required for inter-route moves)

    Returns:
        routes  (list): improved routes
        obj     (float): competition objective after improvement
    """
    inter_route_enabled = (demands is not None and capacity is not None)
    n        = len(distance_matrix)
    k        = min(15, n - 1)
    prev_obj = float("inf")

    while True:
        routes, _ = two_opt_solution(routes, distance_matrix)
        routes, _ = or_opt_solution(routes, distance_matrix)

        if inter_route_enabled:
            routes = optimize_routes_neighbor_lists(
                routes, demands, capacity, distance_matrix, k)

            eliminated = True
            while eliminated:
                routes, eliminated = route_elimination(
                    routes, demands, capacity, distance_matrix)

        current_obj = fitness_from_routes(routes, distance_matrix)
        if prev_obj - current_obj < 1e-10:
            break
        prev_obj = current_obj

    return routes, fitness_from_routes(routes, distance_matrix)


# ---------------------------------------------------------------------------
# MEDIUM LOCAL SEARCH  (used during GA loop for medium/large instances)
# ---------------------------------------------------------------------------

def local_search_medium(routes, distance_matrix, demands=None, capacity=None,
                        lns_iters=5):
    """
    Medium-strength local search for use inside the GA loop on larger instances.

    Adds a short LNS pass on top of the light pipeline so the GA can
    actively merge routes during the loop — not just at the final pass.
    Without this, the vehicle count never drops on medium/large instances
    until the very end, leaving the objective high throughout the search.

    Operators applied:
      1. 2-opt              — intra-route
      2. OR-opt             — intra-route customer relocation
      3. Neighbor-list relocate + swap + intra 2-opt  (Solver_V6)
      4. LNS               — short destroy/repair (lns_iters iterations only)
      5. Route elimination  — remove smallest route via cheapest insertion

    No 2-opt* and no inter-route relocation/swap — those stay in the
    final full LS only.

    Args:
        routes          : list of routes
        distance_matrix : 2D np.array
        demands         : np.array
        capacity        : int
        lns_iters       : LNS iteration budget (3 for large, 5 for medium)

    Returns:
        routes  (list): improved routes
        obj     (float): competition objective after improvement
    """
    inter_route_enabled = (demands is not None and capacity is not None)
    n        = len(distance_matrix)
    k        = min(20, n - 1)
    prev_obj = float("inf")

    while True:
        routes, _ = two_opt_solution(routes, distance_matrix)
        routes, _ = or_opt_solution(routes, distance_matrix)

        if inter_route_enabled:
            routes = optimize_routes_neighbor_lists(
                routes, demands, capacity, distance_matrix, k)

            # Short LNS — enough to merge routes without taking minutes
            routes = lns(
                routes, demands, capacity, distance_matrix,
                iterations=lns_iters,
                remove_ratio=0.12,
                k=k,
            )

            eliminated = True
            while eliminated:
                routes, eliminated = route_elimination(
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
    chrom = []
    for route in routes:
        chrom.extend(n for n in route if n != 0)
    return chrom
