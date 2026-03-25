def relocate(routes, demands, capacity, distance_matrix):
    improved = True

    while improved:
        improved = False

        for i in range(len(routes)):
            for j in range(len(routes)):
                if i == j:
                    continue

                route_i = routes[i]
                route_j = routes[j]

                for idx in range(1, len(route_i) - 1):
                    node = route_i[idx]

                    load_j = sum(demands[n] for n in route_j if n != 0)
                    if load_j + demands[node] > capacity:
                        continue

                    prev_i = route_i[idx - 1]
                    next_i = route_i[idx + 1]

                    remove_cost = distance_matrix[prev_i][node] + distance_matrix[node][next_i] - distance_matrix[prev_i][next_i]

                    for pos in range(1, len(route_j)):
                        prev_j = route_j[pos - 1]
                        next_j = route_j[pos]

                        add_cost = distance_matrix[prev_j][node] + distance_matrix[node][next_j] - distance_matrix[prev_j][next_j]

                        if add_cost < remove_cost:
                            routes[i] = route_i[:idx] + route_i[idx+1:]
                            routes[j] = route_j[:pos] + [node] + route_j[pos:]
                            improved = True
                            break

                    if improved:
                        break

                if improved:
                    break

            if improved:
                break

    return routes