def swap(routes, demands, capacity, distance_matrix):
    improved = True

    while improved:
        improved = False

        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):

                route_i = routes[i]
                route_j = routes[j]

                for a in range(1, len(route_i) - 1):
                    for b in range(1, len(route_j) - 1):

                        node_i = route_i[a]
                        node_j = route_j[b]

                        load_i = sum(demands[n] for n in route_i if n != 0)
                        load_j = sum(demands[n] for n in route_j if n != 0)

                        new_load_i = load_i - demands[node_i] + demands[node_j]
                        new_load_j = load_j - demands[node_j] + demands[node_i]

                        if new_load_i > capacity or new_load_j > capacity:
                            continue

                        prev_i = route_i[a - 1]
                        next_i = route_i[a + 1]

                        prev_j = route_j[b - 1]
                        next_j = route_j[b + 1]

                        old_cost = (
                            distance_matrix[prev_i][node_i] +
                            distance_matrix[node_i][next_i] +
                            distance_matrix[prev_j][node_j] +
                            distance_matrix[node_j][next_j]
                        )

                        new_cost = (
                            distance_matrix[prev_i][node_j] +
                            distance_matrix[node_j][next_i] +
                            distance_matrix[prev_j][node_i] +
                            distance_matrix[node_i][next_j]
                        )

                        if new_cost < old_cost:
                            route_i[a], route_j[b] = route_j[b], route_i[a]
                            improved = True
                            break

                    if improved:
                        break

                if improved:
                    break

            if improved:
                break

    return routes