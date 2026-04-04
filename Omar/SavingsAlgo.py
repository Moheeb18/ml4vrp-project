import random

def savings_algorithm(coords, demands, capacity, distance_matrix):
    n = len(coords)
    depot = 0

    routes = []
    route_loads = []

    for i in range(1, n):
        routes.append([depot, i, depot])
        route_loads.append(demands[i])

    savings = []
    for i in range(1, n):
        for j in range(i+1, n):
            s = distance_matrix[depot][i] + distance_matrix[depot][j] - distance_matrix[i][j]
            savings.append((s, i, j))

    savings.sort(key=lambda x: x[0] + random.uniform(-1e-6, 1e-6), reverse=True)

    def find_route(customer):
        for idx, route in enumerate(routes):
            if customer in route[1:-1]:
                return idx
        return None

    for s, i, j in savings:
        ri = find_route(i)
        rj = find_route(j)

        if ri is None or rj is None or ri == rj:
            continue

        route_i = routes[ri]
        route_j = routes[rj]

        if route_i[-2] == i and route_j[1] == j:
            if route_loads[ri] + route_loads[rj] <= capacity:
                new_route = route_i[:-1] + route_j[1:]
            else:
                continue
        elif route_i[1] == i and route_j[-2] == j:
            if route_loads[ri] + route_loads[rj] <= capacity:
                new_route = route_j[:-1] + route_i[1:]
            else:
                continue
        else:
            continue

        routes[ri] = new_route
        route_loads[ri] += route_loads[rj]

        del routes[rj]
        del route_loads[rj]

    return routes