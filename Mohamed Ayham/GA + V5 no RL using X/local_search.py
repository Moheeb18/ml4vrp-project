from routes_utilities import route_distance
from ga_fitness import fitness_from_routes, VEHICLE_PENALTY
from Solver_v5 import (
    improve_routes,
    try_merge_routes,
    build_neighbor_lists,
    optimize_routes_neighbor_lists,
    two_opt,
    two_opt_star,
    total_distance,
)


# ---------------------------------------------------------------------------
# INTRA-ROUTE 2-OPT  (thin wrapper around Solver_v5)
# ---------------------------------------------------------------------------

def two_opt_solution(routes, distance_matrix):
    improved = [two_opt(r, distance_matrix) for r in routes]
    total    = sum(route_distance(r, distance_matrix) for r in improved)
    return improved, total


# ---------------------------------------------------------------------------
# INTRA-ROUTE OR-OPT
# ---------------------------------------------------------------------------

def or_opt_route(route, distance_matrix):
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
    improved = [or_opt_route(r, distance_matrix) for r in routes]
    total    = sum(route_distance(r, distance_matrix) for r in improved)
    return improved, total


# ---------------------------------------------------------------------------
# ROUTE ELIMINATION
# ---------------------------------------------------------------------------

def route_elimination(routes, demands, capacity, distance_matrix):
    """
    Try to eliminate the smallest route via cheapest insertion into others.
    Saves 1000 objective units per route removed.
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

def local_search(routes, distance_matrix, demands=None, capacity=None):
    """
    Full local search pipeline integrating Solver_v5 operators:

      Pass order (repeated until stable):
        1. 2-opt             — intra-route (Solver_v5)
        2. OR-opt            — intra-route customer relocation
        3. Neighbor-list relocate + swap  — fast inter-route (Solver_v5)
        4. 2-opt*            — inter-route suffix exchange (Solver_v5)
        5. try_merge_routes  — merge pairs of routes if competition
                               objective improves (Solver_v5, uses 1000*NV+TD)
        6. Route elimination — remove smallest route via cheapest insertion
        7. Inter-route relocation + swap — fine-grained inter-route moves

    try_merge_routes is the key addition from Solver_v5 — it evaluates
    the full competition objective (1000*NV + TD) when deciding whether
    to merge two routes, so it aggressively reduces vehicle count.
    """
    inter_route_enabled = (demands is not None and capacity is not None)
    prev_obj = float("inf")

    while True:
        # Intra-route
        routes, _ = two_opt_solution(routes, distance_matrix)
        routes, _ = or_opt_solution(routes, distance_matrix)

        if inter_route_enabled:
            # Neighbor-list accelerated relocate + swap (Solver_v5)
            n         = len(distance_matrix)
            k         = min(20, n - 1)
            nb_lists  = build_neighbor_lists(distance_matrix, k)
            routes    = optimize_routes_neighbor_lists(
                routes, demands, capacity, distance_matrix, k)

            # 2-opt* inter-route suffix exchange (Solver_v5)
            improved_star = True
            while improved_star:
                new_routes, improved_star = _two_opt_star_once(
                    routes, demands, capacity, distance_matrix)
                if improved_star:
                    routes = new_routes

            # try_merge_routes: uses competition objective to merge routes
            routes = try_merge_routes(
                routes, demands, capacity, distance_matrix,
                penalty_per_vehicle=VEHICLE_PENALTY)

            # Route elimination
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


def _two_opt_star_once(routes, demands, capacity, distance_matrix):
    """
    Single pass of 2-opt* from Solver_v5, returning (routes, improved).
    """
    current_dist = total_distance(routes, distance_matrix)

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
                    cand1 = route1[:i+1] + route2[j+1:]
                    cand2 = route2[:j+1] + route1[i+1:]
                    if (cand1[0] != 0 or cand1[-1] != 0
                            or cand2[0] != 0 or cand2[-1] != 0):
                        continue
                    l1 = sum(demands[c] for c in cand1 if c != 0)
                    l2 = sum(demands[c] for c in cand2 if c != 0)
                    if l1 > capacity or l2 > capacity:
                        continue
                    new_routes    = [r[:] for r in routes]
                    new_routes[r1] = cand1
                    new_routes[r2] = cand2
                    new_routes     = [r for r in new_routes if len(r) > 2]
                    new_dist = total_distance(new_routes, distance_matrix)
                    if new_dist < current_dist - 1e-10:
                        return new_routes, True

    return routes, False


# ---------------------------------------------------------------------------
# CHROMOSOME RE-ENCODING
# ---------------------------------------------------------------------------

def routes_to_chromosome(routes):
    """Re-encode routes back into a flat chromosome permutation."""
    chrom = []
    for route in routes:
        chrom.extend(n for n in route if n != 0)
    return chrom
