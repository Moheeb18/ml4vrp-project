import random


def route_distance(route, distance_matrix):
    total = 0
    for i in range(len(route) - 1):
        total += distance_matrix[route[i]][route[i + 1]]
    return total


def total_distance(routes, distance_matrix):
    return sum(route_distance(route, distance_matrix) for route in routes)


def route_demand(route, demands):
    total = 0
    for node in route:
        if node != 0:
            total += demands[node]
    return total


def build_neighbor_lists(distance_matrix, k=20):
    n = len(distance_matrix)
    neighbor_lists = []

    for i in range(n):
        neighbors = [j for j in range(n) if j != i]
        neighbors.sort(key=lambda j: distance_matrix[i][j])
        neighbor_lists.append(neighbors[:k])

    return neighbor_lists


def build_position_map(routes):
    pos = {}
    for r_idx, route in enumerate(routes):
        for i, node in enumerate(route):
            pos[node] = (r_idx, i)
    return pos


def solve_vrp(coords, demands, capacity, distance_matrix):
    n = len(coords)
    unvisited = set(range(1, n))
    routes = []

    while unvisited:
        current = 0
        remaining_capacity = capacity
        route = [0]

        while True:
            candidates = [i for i in unvisited if demands[i] <= remaining_capacity]

            if not candidates:
                break

            next_node = min(candidates, key=lambda i: distance_matrix[current][i])

            route.append(next_node)
            unvisited.remove(next_node)
            remaining_capacity -= demands[next_node]
            current = next_node

        route.append(0)
        routes.append(route)

    return routes


def two_opt(route, distance_matrix):
    if len(route) <= 4:
        return route

    best = route[:]
    improved = True

    while improved:
        improved = False
        best_cost = route_distance(best, distance_matrix)

        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                if j - i == 1:
                    continue

                candidate = best[:]
                candidate[i:j] = reversed(candidate[i:j])
                candidate_cost = route_distance(candidate, distance_matrix)

                if candidate_cost < best_cost:
                    best = candidate
                    best_cost = candidate_cost
                    improved = True
                    break

            if improved:
                break

    return best


def relocate_with_neighbor_lists(routes, demands, capacity, distance_matrix, neighbor_lists):
    improved = True

    while improved:
        improved = False
        current_cost = total_distance(routes, distance_matrix)
        pos_map = build_position_map(routes)

        for customer in range(1, len(distance_matrix)):
            if customer not in pos_map:
                continue

            from_r, from_i = pos_map[customer]
            from_route = routes[from_r]

            if from_i == 0 or from_i == len(from_route) - 1:
                continue

            for neighbor in neighbor_lists[customer]:
                if neighbor == 0 or neighbor not in pos_map:
                    continue

                to_r, to_i = pos_map[neighbor]

                if from_r == to_r and (to_i == from_i or to_i == from_i - 1):
                    continue

                new_routes = [r[:] for r in routes]
                node = new_routes[from_r].pop(from_i)

                if from_r == to_r:
                    if to_i > from_i:
                        to_i -= 1
                    new_routes[to_r].insert(to_i + 1, node)
                else:
                    if route_demand(new_routes[to_r], demands) + demands[node] > capacity:
                        continue
                    new_routes[to_r].insert(to_i + 1, node)

                if len(new_routes[from_r]) <= 2:
                    new_routes.pop(from_r)

                new_cost = total_distance(new_routes, distance_matrix)

                if new_cost < current_cost:
                    routes = new_routes
                    improved = True
                    break

            if improved:
                break

    return routes


def swap_with_neighbor_lists(routes, demands, capacity, distance_matrix, neighbor_lists):
    improved = True

    while improved:
        improved = False
        current_cost = total_distance(routes, distance_matrix)
        pos_map = build_position_map(routes)

        for customer1 in range(1, len(distance_matrix)):
            if customer1 not in pos_map:
                continue

            r1, i1 = pos_map[customer1]
            if i1 == 0 or i1 == len(routes[r1]) - 1:
                continue

            for customer2 in neighbor_lists[customer1]:
                if customer2 == 0 or customer2 == customer1 or customer2 not in pos_map:
                    continue

                r2, i2 = pos_map[customer2]
                if i2 == 0 or i2 == len(routes[r2]) - 1:
                    continue

                if r1 == r2 and i1 == i2:
                    continue

                new_routes = [r[:] for r in routes]
                a = new_routes[r1][i1]
                b = new_routes[r2][i2]

                if r1 != r2:
                    new_demand_r1 = route_demand(new_routes[r1], demands) - demands[a] + demands[b]
                    new_demand_r2 = route_demand(new_routes[r2], demands) - demands[b] + demands[a]

                    if new_demand_r1 > capacity or new_demand_r2 > capacity:
                        continue

                new_routes[r1][i1], new_routes[r2][i2] = new_routes[r2][i2], new_routes[r1][i1]
                new_cost = total_distance(new_routes, distance_matrix)

                if new_cost < current_cost:
                    routes = new_routes
                    improved = True
                    break

            if improved:
                break

    return routes


def two_opt_star(routes, demands, capacity, distance_matrix):
    improved = True

    while improved:
        improved = False
        current_cost = total_distance(routes, distance_matrix)

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

                        if cand1[0] != 0 or cand1[-1] != 0 or cand2[0] != 0 or cand2[-1] != 0:
                            continue

                        demand1 = route_demand(cand1, demands)
                        demand2 = route_demand(cand2, demands)

                        if demand1 > capacity or demand2 > capacity:
                            continue

                        new_routes = [r[:] for r in routes]
                        new_routes[r1] = cand1
                        new_routes[r2] = cand2

                        new_routes = [r for r in new_routes if len(r) > 2]

                        new_cost = total_distance(new_routes, distance_matrix)

                        if new_cost < current_cost:
                            routes = new_routes
                            improved = True
                            break

                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return routes


def optimize_routes_neighbor_lists(routes, demands, capacity, distance_matrix, k=20):
    neighbor_lists = build_neighbor_lists(distance_matrix, k)

    improved = True
    while improved:
        improved = False
        old_cost = total_distance(routes, distance_matrix)

        routes = relocate_with_neighbor_lists(routes, demands, capacity, distance_matrix, neighbor_lists)
        routes = swap_with_neighbor_lists(routes, demands, capacity, distance_matrix, neighbor_lists)

        for i in range(len(routes)):
            routes[i] = two_opt(routes[i], distance_matrix)

        routes = two_opt_star(routes, demands, capacity, distance_matrix)

        new_cost = total_distance(routes, distance_matrix)

        if new_cost < old_cost:
            improved = True

    return routes


def destroy_solution(routes, distance_matrix, num_remove, mode="random"):
    customers = []

    for route in routes:
        for node in route[1:-1]:
            customers.append(node)

    if not customers:
        return [r[:] for r in routes], []

    num_remove = min(num_remove, len(customers))

    if mode == "random":
        removed_nodes = random.sample(customers, num_remove)

    elif mode == "related":
        seed = random.choice(customers)
        sorted_nodes = sorted(
            [node for node in customers if node != seed],
            key=lambda x: distance_matrix[seed][x]
        )
        removed_nodes = [seed] + sorted_nodes[:num_remove - 1]

    elif mode == "route":
        chosen_route_idx = random.randrange(len(routes))
        route_customers = routes[chosen_route_idx][1:-1]

        if len(route_customers) >= num_remove:
            removed_nodes = random.sample(route_customers, num_remove)
        else:
            removed_nodes = route_customers[:]
            remaining = num_remove - len(removed_nodes)
            others = [node for node in customers if node not in removed_nodes]
            removed_nodes += random.sample(others, min(remaining, len(others)))

    else:
        removed_nodes = random.sample(customers, num_remove)

    removed_set = set(removed_nodes)

    new_routes = []
    for route in routes:
        new_route = [0]
        for node in route[1:-1]:
            if node not in removed_set:
                new_route.append(node)
        new_route.append(0)

        if len(new_route) > 2:
            new_routes.append(new_route)

    return new_routes, removed_nodes


def insertion_cost(route, pos, node, distance_matrix):
    a = route[pos]
    b = route[pos + 1]
    return distance_matrix[a][node] + distance_matrix[node][b] - distance_matrix[a][b]


def best_insertion_position(route, node, distance_matrix):
    best_pos = None
    best_extra = float("inf")

    for i in range(len(route) - 1):
        extra = insertion_cost(route, i, node, distance_matrix)
        if extra < best_extra:
            best_extra = extra
            best_pos = i + 1

    return best_pos, best_extra


def best_insertion(routes, node, demands, capacity, distance_matrix):
    best_extra = float("inf")
    best_r = None
    best_pos = None

    for r_idx, route in enumerate(routes):
        if route_demand(route, demands) + demands[node] > capacity:
            continue

        pos, extra = best_insertion_position(route, node, distance_matrix)

        if extra < best_extra:
            best_extra = extra
            best_r = r_idx
            best_pos = pos

    if best_r is not None:
        routes[best_r].insert(best_pos, node)
    else:
        routes.append([0, node, 0])

    return routes


def regret_insertion_repair(routes, removed_nodes, demands, capacity, distance_matrix, regret_k=2):
    repaired = [r[:] for r in routes]
    remaining = removed_nodes[:]

    while remaining:
        best_node = None
        best_route_idx = None
        best_pos = None
        best_regret = -float("inf")
        best_first_cost = float("inf")

        for node in remaining:
            options = []

            for r_idx, route in enumerate(repaired):
                if route_demand(route, demands) + demands[node] > capacity:
                    continue

                pos, extra = best_insertion_position(route, node, distance_matrix)
                options.append((extra, r_idx, pos))

            if not options:
                options = [(0, None, None)]
            else:
                options.sort(key=lambda x: x[0])

            while len(options) < regret_k:
                options.append(options[-1])

            first_cost = options[0][0]
            regret = sum(opt[0] for opt in options[1:regret_k]) - (regret_k - 1) * first_cost

            if regret > best_regret or (regret == best_regret and first_cost < best_first_cost):
                best_regret = regret
                best_first_cost = first_cost
                best_node = node
                best_route_idx = options[0][1]
                best_pos = options[0][2]

        if best_node is None:
            break

        if best_route_idx is None:
            repaired.append([0, best_node, 0])
        else:
            repaired[best_route_idx].insert(best_pos, best_node)

        remaining.remove(best_node)

    return repaired


def repair_solution(routes, removed_nodes, demands, capacity, distance_matrix, method="regret"):
    if method == "greedy":
        repaired = [r[:] for r in routes]
        nodes = removed_nodes[:]
        random.shuffle(nodes)

        for node in nodes:
            repaired = best_insertion(repaired, node, demands, capacity, distance_matrix)

        return repaired

    return regret_insertion_repair(routes, removed_nodes, demands, capacity, distance_matrix, regret_k=2)


def lns(routes, demands, capacity, distance_matrix, iterations=50, remove_ratio=0.15, k=20):
    best_routes = [r[:] for r in routes]
    best_cost = total_distance(best_routes, distance_matrix)

    current_routes = [r[:] for r in routes]
    current_cost = total_distance(current_routes, distance_matrix)

    num_customers = sum(len(route) - 2 for route in routes)
    if num_customers == 0:
        return best_routes

    no_improve = 0

    for it in range(iterations):
        num_remove = max(2, int(num_customers * remove_ratio))

        if no_improve < 10:
            destroy_mode = random.choice(["random", "related"])
        else:
            destroy_mode = random.choice(["random", "related", "route"])

        partial_routes, removed_nodes = destroy_solution(
            current_routes,
            distance_matrix,
            num_remove,
            mode=destroy_mode
        )

        if not removed_nodes:
            continue

        repaired_routes = repair_solution(
            partial_routes,
            removed_nodes,
            demands,
            capacity,
            distance_matrix,
            method="regret"
        )

        repaired_routes = optimize_routes_neighbor_lists(
            repaired_routes,
            demands,
            capacity,
            distance_matrix,
            k
        )

        repaired_cost = total_distance(repaired_routes, distance_matrix)

        accept = False

        if repaired_cost < current_cost:
            accept = True
        else:
            tolerance = 0.01 + 0.02 * (1 - it / iterations)
            if repaired_cost <= current_cost * (1 + tolerance):
                if random.random() < 0.20:
                    accept = True

        if accept:
            current_routes = [r[:] for r in repaired_routes]
            current_cost = repaired_cost

        if repaired_cost < best_cost:
            best_routes = [r[:] for r in repaired_routes]
            best_cost = repaired_cost
            no_improve = 0
        else:
            no_improve += 1

    return best_routes


def improve_routes(routes, demands, capacity, distance_matrix, k=25, lns_iterations=60, remove_ratio=0.20):
    routes = [r[:] for r in routes]

    for i in range(len(routes)):
        routes[i] = two_opt(routes[i], distance_matrix)

    routes = optimize_routes_neighbor_lists(routes, demands, capacity, distance_matrix, k)
    routes = lns(routes, demands, capacity, distance_matrix, iterations=lns_iterations, remove_ratio=remove_ratio, k=k)

    return routes


def solve_vrp_improved(coords, demands, capacity, distance_matrix, k=25, lns_iterations=60, remove_ratio=0.20):
    routes = solve_vrp(coords, demands, capacity, distance_matrix)
    routes = improve_routes(routes, demands, capacity, distance_matrix, k, lns_iterations, remove_ratio)
    return routes