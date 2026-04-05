from routes_utilities import route_distance
from ga_fitness import fitness_from_routes, VEHICLE_PENALTY
from Solver_v5 import improve_routes, try_merge_routes, two_opt


def two_opt_solution(routes, distance_matrix):
    improved = [two_opt(r, distance_matrix) for r in routes]
    total = sum(route_distance(r, distance_matrix) for r in improved)
    return improved, total


def or_opt_route(route, distance_matrix):
    best = route[:]
    best_dist = route_distance(best, distance_matrix)
    improved = True

    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            customer = best[i]
            remaining = best[:i] + best[i + 1:]
            for j in range(1, len(remaining)):
                candidate = remaining[:j] + [customer] + remaining[j:]
                cand_dist = route_distance(candidate, distance_matrix)
                if cand_dist < best_dist - 1e-10:
                    best = candidate
                    best_dist = cand_dist
                    improved = True
                    break
            if improved:
                break

    return best


def or_opt_solution(routes, distance_matrix):
    improved = [or_opt_route(r, distance_matrix) for r in routes]
    total = sum(route_distance(r, distance_matrix) for r in improved)
    return improved, total


def route_elimination(routes, demands, capacity, distance_matrix):
    if len(routes) <= 1:
        return routes, False

    sorted_by_size = sorted(range(len(routes)), key=lambda i: len(routes[i]) - 2)

    for target_idx in sorted_by_size:
        target_customers = [c for c in routes[target_idx] if c != 0]
        working = [r[:] for i, r in enumerate(routes) if i != target_idx]
        success = True

        for customer in target_customers:
            best_cost = float("inf")
            best_r = -1
            best_pos = -1

            for ri, route in enumerate(working):
                load = sum(demands[c] for c in route if c != 0)
                if load + demands[customer] > capacity:
                    continue

                for pos in range(1, len(route)):
                    prev_node = route[pos - 1]
                    next_node = route[pos]
                    extra = (
                        distance_matrix[prev_node][customer]
                        + distance_matrix[customer][next_node]
                        - distance_matrix[prev_node][next_node]
                    )
                    if extra < best_cost:
                        best_cost = extra
                        best_r = ri
                        best_pos = pos

            if best_r == -1:
                success = False
                break

            working[best_r] = (
                working[best_r][:best_pos]
                + [customer]
                + working[best_r][best_pos:]
            )

        if success:
            if fitness_from_routes(working, distance_matrix) < fitness_from_routes(routes, distance_matrix) - 1e-10:
                return working, True

    return routes, False


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
                    new_to = routes[r_to][:ins_pos] + [customer] + routes[r_to][ins_pos:]
                    candidate = []

                    for ri, route in enumerate(routes):
                        if ri == r_from:
                            candidate.append(new_from)
                        elif ri == r_to:
                            candidate.append(new_to)
                        else:
                            candidate.append(route[:])

                    candidate = [r for r in candidate if len(r) > 2]

                    if fitness_from_routes(candidate, distance_matrix) < best_obj - 1e-10:
                        return candidate, True

    return routes, False


def inter_route_swap(routes, demands, capacity, distance_matrix):
    best_obj = fitness_from_routes(routes, distance_matrix)
    n_routes = len(routes)

    for r1 in range(n_routes):
        for r2 in range(r1 + 1, n_routes):
            custs1 = [c for c in routes[r1] if c != 0]
            custs2 = [c for c in routes[r2] if c != 0]
            load1 = sum(demands[c] for c in custs1)
            load2 = sum(demands[c] for c in custs2)

            for c1 in custs1:
                for c2 in custs2:
                    if load1 - demands[c1] + demands[c2] > capacity:
                        continue
                    if load2 - demands[c2] + demands[c1] > capacity:
                        continue

                    new_r1 = [c2 if c == c1 else c for c in routes[r1]]
                    new_r2 = [c1 if c == c2 else c for c in routes[r2]]
                    candidate = [
                        new_r1 if i == r1 else (new_r2 if i == r2 else route[:])
                        for i, route in enumerate(routes)
                    ]

                    if fitness_from_routes(candidate, distance_matrix) < best_obj - 1e-10:
                        return candidate, True

    return routes, False


def _solver_v5_params(distance_matrix):
    n = len(distance_matrix)

    if n < 80:
        return 20, 40, 0.15
    if n < 200:
        return 25, 60, 0.20
    return 30, 80, 0.25


def local_search(routes, distance_matrix, demands=None, capacity=None):
    """
    Main local improvement wrapper used by the GA.

    This uses Solver_v5.improve_routes(...) as the primary black-box
    improver, then keeps the extra route elimination / relocation / swap
    intensification already present in the GA project.
    """
    working_routes = [route[:] for route in routes]

    working_routes, _ = two_opt_solution(working_routes, distance_matrix)
    working_routes, _ = or_opt_solution(working_routes, distance_matrix)

    if demands is not None and capacity is not None:
        k, lns_iterations, remove_ratio = _solver_v5_params(distance_matrix)

        working_routes = improve_routes(
            working_routes,
            demands,
            capacity,
            distance_matrix,
            k=k,
            lns_iterations=lns_iterations,
            remove_ratio=remove_ratio,
        )

        improved = True
        while improved:
            improved = False

            candidate, changed = route_elimination(
                working_routes, demands, capacity, distance_matrix
            )
            if changed:
                working_routes = [r[:] for r in candidate]
                improved = True
                continue

            candidate, changed = inter_route_relocate(
                working_routes, demands, capacity, distance_matrix
            )
            if changed:
                working_routes = [r[:] for r in candidate]
                improved = True
                continue

            candidate, changed = inter_route_swap(
                working_routes, demands, capacity, distance_matrix
            )
            if changed:
                working_routes = [r[:] for r in candidate]
                improved = True

        working_routes = try_merge_routes(
            working_routes,
            demands,
            capacity,
            distance_matrix,
            penalty_per_vehicle=VEHICLE_PENALTY,
        )

        working_routes, _ = two_opt_solution(working_routes, distance_matrix)

    return working_routes, fitness_from_routes(working_routes, distance_matrix)


def routes_to_chromosome(routes):
    chrom = []
    for route in routes:
        chrom.extend(node for node in route if node != 0)
    return chrom
